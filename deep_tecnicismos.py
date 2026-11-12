"""FASE 1 (E1): detección de tecnicismos con deep learning.

- BiGRU etiquetador con distintos embeddings (aleatorio, Word2Vec propio, Word2Vec con dataset
  externo, spaCy), congelados o fine-tuneados.
- Variantes con AUTOATENCION sobre la salida de la BiGRU y con embedding de categoria gramatical (POS).
- Preentrenamiento: la BiGRU se preentrena con el dataset externo y se ajusta despues con train
  (con su control: mismo vocabulario ampliado sin preentrenar).
- BERTIN (RoBERTa en español) para clasificación de tokens, con el codificador congelado o fine-tuneado.
Mismos fragmentos, mismas etiquetas y mismas métricas que en el modelo shallow. Cada red se entrena con
varias semillas y se informa la media y la desviación típica.
"""
import copy
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import (VOCAB_MAX, VOCAB_MIN_FREQ, TAG_HIDDEN, DROPOUT, BATCH, LR, TAG_EPOCAS, PACIENCIA, CLIP,
                    TAG_UMBRALES, MODELO_CONTEXTUAL, BERT_LR, BERT_EPOCAS, BERT_BATCH, BERT_MAX_LEN,
                    POS_DIM, N_CABEZAS_ATT, SEMILLAS, PRE_EPOCAS_TAG)
from embeddings import construir_embedding, fijar_semilla, DEV, PAD, UNK
from metricas import agregar_semillas, ic_bootstrap_f1
from tecnicismos import metricas_tokens, id_pos, N_POS

VARIANTES_TAG = {
    "BiGRU, embeddings aleatorios":        dict(tipo="gru", emb="aleatorio", congelar=False),
    "BiGRU, Word2Vec propio congelado":    dict(tipo="gru", emb="w2v", congelar=True),
    "BiGRU, Word2Vec propio fine-tuneado": dict(tipo="gru", emb="w2v", congelar=False),
    "BiGRU, spaCy congelado":              dict(tipo="gru", emb="spacy", congelar=True),
    "BiGRU, spaCy fine-tuneado":           dict(tipo="gru", emb="spacy", congelar=False),
    # --- atencion y rasgos no textuales (POS) ---
    "BiGRU+autoatención, Word2Vec propio fine-tuneado":
        dict(tipo="gru", emb="w2v", congelar=False, atencion=True),
    "BiGRU+autoatención+POS, Word2Vec propio fine-tuneado":
        dict(tipo="gru", emb="w2v", congelar=False, atencion=True, pos=True),
    # --- dataset externo (solo se ejecutan con --externo y el dataset conectado en preentreno.py) ---
    "BiGRU, Word2Vec externo fine-tuneado":
        dict(tipo="gru", emb="w2v_ext", congelar=False, requiere_ext=True),
    "BiGRU, vocabulario ampliado, sin preentreno (control)":
        dict(tipo="gru", emb="w2v", congelar=False, vocab_ext=True, requiere_ext=True),
    "BiGRU, preentreno en dataset externo + fine-tuning":
        dict(tipo="gru", emb="w2v", congelar=False, vocab_ext=True, preentreno=True, requiere_ext=True),
    # --- contextuales ---
    "BERTIN congelado (solo capa lineal)": dict(tipo="bert", congelar=True),
    "BERTIN fine-tuneado":                 dict(tipo="bert", congelar=False),
}
STD_KEYS = ["f1", "f1_val", "accuracy", "precision", "recall"]


# ------------------------------------------------------------------ utilidades comunes
def _metricas_desde_probs(chunks, probs, umbral):
    y = np.concatenate([c["y"] for c in chunks])
    p = np.concatenate(probs)
    return metricas_tokens(y, (p >= umbral).astype(int))


def _elegir_umbral(chunks, probs):
    """Umbral de decisión que maximiza el F1 en VALIDACIÓN (hiperparámetro del deep)."""
    return max(TAG_UMBRALES, key=lambda u: _metricas_desde_probs(chunks, probs, u)["f1"])


def _resultado(nombre, cfg, val, test, pv, pt, epocas, n_param, t0, cobertura, umbral):
    m_va = _metricas_desde_probs(val, pv, umbral)
    m_te = _metricas_desde_probs(test, pt, umbral)
    rep = {"aleatorio": "Embeddings aleatorios", "w2v": "Word2Vec propio (E2)",
           "w2v_ext": "Word2Vec propio + dataset externo",
           "spacy": "spaCy es_core_news_md", None: "BERTIN (contextual)"}[cfg.get("emb")]
    if cfg.get("emb") != "aleatorio":
        rep += " congelado" if cfg["congelar"] else " fine-tuneado"
    if cfg.get("preentreno"):
        rep += " + preentreno externo"
    y = np.concatenate([c["y"] for c in test])
    ev = y != -100
    pred = (np.concatenate(pt) >= umbral).astype(int)
    return {"modelo": nombre, "familia": "deep", "representacion": rep,
            "umbral": umbral, "epocas": epocas,
            **{f"{k}_val": v for k, v in m_va.items()}, **m_te,
            "f1_ic95_s0": ic_bootstrap_f1(y[ev], pred[ev]),
            "parametros_entrenables": n_param, "tiempo_s": round(time.time() - t0),
            "cobertura_emb": cobertura}


# ------------------------------------------------------------------ BiGRU
def construir_vocab_tag(train_chunks, extra_chunks=None):
    """Vocabulario con las palabras de train (y, si se da, las del dataset externo)."""
    c = Counter(w.lower() for ch in train_chunks for w in ch["words"])
    if extra_chunks:
        c.update(w.lower() for ch in extra_chunks for w in ch["words"])
    itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + [w for w, f in c.most_common(VOCAB_MAX) if f >= VOCAB_MIN_FREQ]
    return itos, {w: i for i, w in enumerate(itos)}


class EtiquetadorGRU(nn.Module):
    def __init__(self, emb, atencion=False, n_pos=0):
        super().__init__()
        self.emb = emb
        self.pos_emb = nn.Embedding(n_pos, POS_DIM, padding_idx=0) if n_pos else None
        self.rnn = nn.GRU(emb.embedding_dim + (POS_DIM if n_pos else 0), TAG_HIDDEN,
                          batch_first=True, bidirectional=True)
        self.att = (nn.MultiheadAttention(2 * TAG_HIDDEN, N_CABEZAS_ATT, dropout=DROPOUT, batch_first=True)
                    if atencion else None)
        self.norm = nn.LayerNorm(2 * TAG_HIDDEN) if atencion else None
        self.drop = nn.Dropout(DROPOUT)
        self.out = nn.Linear(2 * TAG_HIDDEN, 2)

    def forward(self, x, lens, pos=None):
        e = self.drop(self.emb(x))
        if self.pos_emb is not None:
            e = torch.cat([e, self.drop(self.pos_emb(pos))], dim=-1)
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        if self.att is not None:     # autoatencion con residual: cada palabra mira a todo el fragmento
            a, _ = self.att(h, h, h, key_padding_mask=(x == PAD), need_weights=False)
            h = self.norm(h + self.drop(a))
        return self.out(self.drop(h))


def _tensores(chunks, idx, stoi):
    L = max(len(chunks[i]["words"]) for i in idx)
    x = torch.zeros(len(idx), L, dtype=torch.long)
    pos = torch.zeros(len(idx), L, dtype=torch.long)
    y = torch.full((len(idx), L), -100, dtype=torch.long)
    for b, i in enumerate(idx):
        n = len(chunks[i]["words"])
        x[b, :n] = torch.tensor([stoi.get(w.lower(), UNK) for w in chunks[i]["words"]])
        pos[b, :n] = torch.tensor([id_pos(p) for p in chunks[i]["pos"]])
        y[b, :n] = torch.tensor(chunks[i]["y"])
    return x.to(DEV), y.to(DEV), pos.to(DEV)


def _probs_gru(modelo, chunks, stoi, batch=64):
    modelo.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(chunks), batch):
            idx = list(range(i, min(i + batch, len(chunks))))
            x, _, pos = _tensores(chunks, idx, stoi)
            lens = (x != PAD).sum(1).clamp(min=1)
            p = F.softmax(modelo(x, lens, pos), dim=-1)[..., 1].cpu().numpy()
            probs += [p[b, :len(chunks[j]["words"])] for b, j in enumerate(idx)]
    return probs


def _epocas_gru(modelo, params, train, val, stoi, epocas, paciencia, etiqueta):
    """Bucle de entrenamiento. Con val=None (preentrenamiento) no hay early stopping y se devuelve
    el estado final; con val se devuelve el estado con mejor F1 de validación."""
    opt = torch.optim.Adam(params, lr=LR)
    mejor, mejor_estado, sin_mejora, hechas = -1.0, None, 0, 0
    for ep in range(1, epocas + 1):
        modelo.train()
        orden = list(range(len(train)))
        random.shuffle(orden)
        total = 0.0
        for i in range(0, len(orden), BATCH):
            x, y, pos = _tensores(train, orden[i:i + BATCH], stoi)
            lens = (x != PAD).sum(1).clamp(min=1)
            loss = F.cross_entropy(modelo(x, lens, pos).reshape(-1, 2), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), CLIP)
            opt.step()
            total += float(loss)
        hechas = ep
        media = total / max(len(orden) / BATCH, 1)
        if val is None:
            print(f"[{etiqueta}] época {ep:02d} | loss {media:.4f}")
            continue
        f1 = _metricas_desde_probs(val, _probs_gru(modelo, val, stoi), 0.5)["f1"]
        print(f"[{etiqueta}] época {ep:02d} | loss {media:.4f} | F1 val {f1:.4f}")
        if f1 > mejor + 1e-4:
            mejor, mejor_estado, sin_mejora = f1, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                print("Early stopping.")
                break
    if val is None:
        return copy.deepcopy(modelo.state_dict()), hechas
    return mejor_estado, hechas


def entrenar_gru(nombre, cfg, itos, stoi, train, val, test, modelo_w2v, nlp, epocas, semilla,
                 ext_chunks=None, w2v_ext=None):
    fijar_semilla(semilla)
    emb, cobertura = construir_embedding(cfg["emb"], itos, cfg["congelar"], modelo_w2v, nlp, w2v_ext)
    modelo = EtiquetadorGRU(emb, cfg.get("atencion", False), N_POS if cfg.get("pos") else 0).to(DEV)
    params = [p for p in modelo.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)
    print(f"\n--- {nombre} | semilla {semilla} | parámetros entrenables: {n_param:,} | cobertura embedding: "
          f"{'-' if cobertura is None else f'{100 * cobertura:.1f}%'} ---")
    t0 = time.time()
    if cfg.get("preentreno"):
        estado, _ = _epocas_gru(modelo, params, ext_chunks, None, stoi, PRE_EPOCAS_TAG, 0, "preentreno")
        modelo.load_state_dict(estado)
    estado, hechas = _epocas_gru(modelo, params, train, val, stoi, epocas, PACIENCIA, nombre)
    modelo.load_state_dict(estado)
    pv, pt = _probs_gru(modelo, val, stoi), _probs_gru(modelo, test, stoi)
    return _resultado(nombre, cfg, val, test, pv, pt, hechas, n_param, t0, cobertura, _elegir_umbral(val, pv)), pt


# ------------------------------------------------------------------ BERTIN
def _codificar_bert(tok, chunks, idx):
    enc = tok([chunks[i]["words"] for i in idx], is_split_into_words=True, truncation=True,
              max_length=BERT_MAX_LEN, padding=True, return_tensors="pt")
    y = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
    primeros = []   # (fila, posición subtoken, índice de palabra): solo el primer subtoken de cada palabra
    for b, i in enumerate(idx):
        prev = None
        for pos, w in enumerate(enc.word_ids(b)):
            if w is not None and w != prev:
                y[b, pos] = chunks[i]["y"][w]
                primeros.append((b, pos, w))
            prev = w
    return {k: v.to(DEV) for k, v in enc.items()}, y.to(DEV), primeros


def _probs_bert(modelo, tok, chunks, batch=32):
    modelo.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(chunks), batch):
            idx = list(range(i, min(i + batch, len(chunks))))
            enc, _, primeros = _codificar_bert(tok, chunks, idx)
            p = F.softmax(modelo(**enc).logits, dim=-1)[..., 1].cpu().numpy()
            por_fila = [np.zeros(len(chunks[j]["words"])) for j in idx]   # palabras truncadas -> prob 0
            for b, pos, w in primeros:
                por_fila[b][w] = p[b, pos]
            probs += por_fila
    return probs


def entrenar_bert(nombre, cfg, train, val, test, epocas, semilla):
    from transformers import RobertaForTokenClassification, RobertaTokenizerFast
    fijar_semilla(semilla)
    tok = RobertaTokenizerFast.from_pretrained(MODELO_CONTEXTUAL, add_prefix_space=True)
    modelo = RobertaForTokenClassification.from_pretrained(MODELO_CONTEXTUAL, num_labels=2,
                                                           use_safetensors=False).to(DEV)
    if cfg["congelar"]:
        modelo.roberta.requires_grad_(False)
    params = [p for p in modelo.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)
    print(f"\n--- {nombre} | semilla {semilla} | parámetros entrenables: {n_param:,} ---")
    opt = torch.optim.AdamW(params, lr=BERT_LR if not cfg["congelar"] else 1e-3, weight_decay=0.01)
    t0, mejor, mejor_estado = time.time(), -1.0, None
    for ep in range(1, epocas + 1):
        modelo.train()
        if cfg["congelar"]:
            modelo.roberta.eval()
        orden = list(range(len(train)))
        random.shuffle(orden)
        for i in range(0, len(orden), BERT_BATCH):
            enc, y, _ = _codificar_bert(tok, train, orden[i:i + BERT_BATCH])
            loss = F.cross_entropy(modelo(**enc).logits.reshape(-1, 2), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
        f1 = _metricas_desde_probs(val, _probs_bert(modelo, tok, val), 0.5)["f1"]
        print(f"época {ep} | F1 val {f1:.4f}")
        if f1 > mejor:
            mejor, mejor_estado = f1, {k: v.detach().clone() for k, v in modelo.state_dict().items()}
    modelo.load_state_dict(mejor_estado)
    pv, pt = _probs_bert(modelo, tok, val), _probs_bert(modelo, tok, test)
    return _resultado(nombre, cfg, val, test, pv, pt, epocas, n_param, t0, None, _elegir_umbral(val, pv)), pt


# ------------------------------------------------------------------ orquestación
def ejecutar_deep_tecnicismos(train_chunks, val_chunks, test_chunks, modelo_w2v, nlp, variantes=None,
                              con_bert=True, epocas=None, semillas=None, ext_chunks=None, w2v_ext=None):
    """Entrena cada variante con todas las semillas. Devuelve (resultados agregados, predicciones):
    predicciones[nombre] = {'probs': probabilidades de test (semilla 0), 'umbral'} para el análisis de errores."""
    print("\n=== FASE 1 · DEEP LEARNING (tecnicismos) ===")
    semillas = semillas or SEMILLAS[:1]
    itos, stoi = construir_vocab_tag(train_chunks)
    itos_x, stoi_x = construir_vocab_tag(train_chunks, ext_chunks) if ext_chunks else (None, None)
    print("Vocabulario:", len(itos), "| fragmentos train/val/test:", len(train_chunks), len(val_chunks),
          len(test_chunks), "| semillas:", semillas)
    resultados, preds = [], {}
    for nombre, cfg in VARIANTES_TAG.items():
        if (variantes and nombre not in variantes) or (cfg["tipo"] == "bert" and not con_bert):
            continue
        if cfg.get("requiere_ext") and (ext_chunks is None or w2v_ext is None):
            print(f"\n(se omite '{nombre}': no hay dataset externo conectado, ver preentreno.py)")
            continue
        runs = []
        for k, s in enumerate(semillas):
            if cfg["tipo"] == "gru":
                v_itos, v_stoi = (itos_x, stoi_x) if cfg.get("vocab_ext") else (itos, stoi)
                r, pt = entrenar_gru(nombre, cfg, v_itos, v_stoi, train_chunks, val_chunks, test_chunks,
                                     modelo_w2v, nlp, epocas or TAG_EPOCAS, s, ext_chunks, w2v_ext)
            else:
                r, pt = entrenar_bert(nombre, cfg, train_chunks, val_chunks, test_chunks,
                                      epocas or BERT_EPOCAS, s)
            runs.append(r)
            if k == 0:
                preds[nombre] = {"probs": np.concatenate(pt), "umbral": r["umbral"]}
        agg = agregar_semillas(runs, STD_KEYS)
        agg["hiperparametros"] = (f"umbral={np.median([r['umbral'] for r in runs]):.2f}, "
                                  f"épocas={int(np.median([r['epocas'] for r in runs]))}, semillas={len(runs)}")
        resultados.append(agg)
        print(f"{nombre}: F1 val={agg['f1_val']:.3f}  F1 test={agg['f1']:.3f} ± {agg['f1_std']:.3f}")
    return resultados, preds
