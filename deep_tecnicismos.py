"""FASE 1 (E1): detección de tecnicismos con deep learning.

- BiGRU etiquetador con distintos embeddings (aleatorio, Word2Vec propio, spaCy), congelados o fine-tuneados.
- BERTIN (RoBERTa en español) para clasificación de tokens, con el codificador congelado o fine-tuneado.
Mismos fragmentos, mismas etiquetas y mismas métricas que en el modelo shallow.
"""
import copy
import time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import (VOCAB_MAX, VOCAB_MIN_FREQ, TAG_HIDDEN, DROPOUT, BATCH, LR, TAG_EPOCAS, PACIENCIA, CLIP,
                    TAG_UMBRALES, MODELO_CONTEXTUAL, BERT_LR, BERT_EPOCAS, BERT_BATCH, BERT_MAX_LEN)
from embeddings import construir_embedding, fijar_semilla, DEV, PAD, UNK
from tecnicismos import metricas_tokens

VARIANTES_TAG = {
    "BiGRU, embeddings aleatorios":        dict(tipo="gru", emb="aleatorio", congelar=False),
    "BiGRU, Word2Vec propio congelado":    dict(tipo="gru", emb="w2v", congelar=True),
    "BiGRU, Word2Vec propio fine-tuneado": dict(tipo="gru", emb="w2v", congelar=False),
    "BiGRU, spaCy congelado":              dict(tipo="gru", emb="spacy", congelar=True),
    "BiGRU, spaCy fine-tuneado":           dict(tipo="gru", emb="spacy", congelar=False),
    "BERTIN congelado (solo capa lineal)": dict(tipo="bert", congelar=True),
    "BERTIN fine-tuneado":                 dict(tipo="bert", congelar=False),
}


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
    rep = {"aleatorio": "Embeddings aleatorios", "w2v": "Word2Vec propio",
           "spacy": "spaCy es_core_news_md", None: "BERTIN (contextual)"}[cfg.get("emb")]
    if cfg.get("emb") != "aleatorio":
        rep += " congelado" if cfg["congelar"] else " fine-tuneado"
    return {"modelo": nombre, "familia": "deep", "representacion": rep,
            "hiperparametros": f"umbral={umbral}, épocas={epocas}",
            **{f"{k}_val": v for k, v in m_va.items()}, **m_te,
            "parametros_entrenables": n_param, "tiempo_s": round(time.time() - t0),
            "cobertura_emb": cobertura}


# ------------------------------------------------------------------ BiGRU
def construir_vocab_tag(train_chunks):
    c = Counter(w.lower() for ch in train_chunks for w in ch["words"])
    itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + [w for w, f in c.most_common(VOCAB_MAX) if f >= VOCAB_MIN_FREQ]
    return itos, {w: i for i, w in enumerate(itos)}


class EtiquetadorGRU(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = emb
        self.rnn = nn.GRU(emb.embedding_dim, TAG_HIDDEN, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(DROPOUT)
        self.out = nn.Linear(2 * TAG_HIDDEN, 2)

    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        return self.out(self.drop(h))


def _tensores(chunks, idx, stoi):
    L = max(len(chunks[i]["words"]) for i in idx)
    x = torch.zeros(len(idx), L, dtype=torch.long)
    y = torch.full((len(idx), L), -100, dtype=torch.long)
    for b, i in enumerate(idx):
        ids = [stoi.get(w.lower(), UNK) for w in chunks[i]["words"]]
        x[b, :len(ids)] = torch.tensor(ids)
        y[b, :len(ids)] = torch.tensor(chunks[i]["y"])
    return x.to(DEV), y.to(DEV)


def _probs_gru(modelo, chunks, stoi, batch=64):
    modelo.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(chunks), batch):
            idx = list(range(i, min(i + batch, len(chunks))))
            x, _ = _tensores(chunks, idx, stoi)
            p = F.softmax(modelo(x, (x != PAD).sum(1).clamp(min=1)), dim=-1)[..., 1].cpu().numpy()
            probs += [p[b, :len(chunks[j]["words"])] for b, j in enumerate(idx)]
    return probs


def entrenar_gru(nombre, cfg, itos, stoi, train, val, test, modelo_w2v, nlp, epocas):
    import random
    fijar_semilla()
    emb, cobertura = construir_embedding(cfg["emb"], itos, cfg["congelar"], modelo_w2v, nlp)
    modelo = EtiquetadorGRU(emb).to(DEV)
    params = [p for p in modelo.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)
    print(f"\n--- {nombre} | parámetros entrenables: {n_param:,} | cobertura embedding: "
          f"{'-' if cobertura is None else f'{100 * cobertura:.1f}%'} ---")
    opt = torch.optim.Adam(params, lr=LR)
    t0, mejor, mejor_estado, sin_mejora, hechas = time.time(), -1.0, None, 0, 0
    for ep in range(1, epocas + 1):
        modelo.train()
        orden = list(range(len(train)))
        random.shuffle(orden)
        total = 0.0
        for i in range(0, len(orden), BATCH):
            idx = orden[i:i + BATCH]
            x, y = _tensores(train, idx, stoi)
            loss = F.cross_entropy(modelo(x, (x != PAD).sum(1).clamp(min=1)).reshape(-1, 2),
                                   y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), CLIP)
            opt.step()
            total += float(loss)
        f1 = _metricas_desde_probs(val, _probs_gru(modelo, val, stoi), 0.5)["f1"]
        hechas = ep
        print(f"época {ep:02d} | loss {total / max(len(orden) / BATCH, 1):.4f} | F1 val {f1:.4f}")
        if f1 > mejor + 1e-4:
            mejor, mejor_estado, sin_mejora = f1, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                print("Early stopping.")
                break
    modelo.load_state_dict(mejor_estado)
    pv, pt = _probs_gru(modelo, val, stoi), _probs_gru(modelo, test, stoi)
    return _resultado(nombre, cfg, val, test, pv, pt, hechas, n_param, t0, cobertura, _elegir_umbral(val, pv))


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


def entrenar_bert(nombre, cfg, train, val, test, epocas):
    import random
    from transformers import RobertaForTokenClassification, RobertaTokenizerFast
    fijar_semilla()
    tok = RobertaTokenizerFast.from_pretrained(MODELO_CONTEXTUAL, add_prefix_space=True)
    modelo = RobertaForTokenClassification.from_pretrained(MODELO_CONTEXTUAL, num_labels=2,
                                                           use_safetensors=False).to(DEV)
    if cfg["congelar"]:
        modelo.roberta.requires_grad_(False)
    params = [p for p in modelo.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)
    print(f"\n--- {nombre} | parámetros entrenables: {n_param:,} ---")
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
    return _resultado(nombre, cfg, val, test, pv, pt, epocas, n_param, t0, None, _elegir_umbral(val, pv))


# ------------------------------------------------------------------ orquestación
def ejecutar_deep_tecnicismos(train_chunks, val_chunks, test_chunks, modelo_w2v, nlp, variantes=None,
                              con_bert=True, epocas=None):
    print("\n=== FASE 1 · DEEP LEARNING (tecnicismos) ===")
    itos, stoi = construir_vocab_tag(train_chunks)
    print("Vocabulario:", len(itos), "| fragmentos train/val/test:", len(train_chunks), len(val_chunks), len(test_chunks))
    resultados = []
    for nombre, cfg in VARIANTES_TAG.items():
        if (variantes and nombre not in variantes) or (cfg["tipo"] == "bert" and not con_bert):
            continue
        if cfg["tipo"] == "gru":
            resultados.append(entrenar_gru(nombre, cfg, itos, stoi, train_chunks, val_chunks, test_chunks,
                                           modelo_w2v, nlp, epocas or TAG_EPOCAS))
        else:
            resultados.append(entrenar_bert(nombre, cfg, train_chunks, val_chunks, test_chunks, epocas or BERT_EPOCAS))
        r = resultados[-1]
        print(f"{nombre}: F1 val={r['f1_val']:.3f}  F1 test={r['f1']:.3f}")
    return resultados
