"""FASE 3 (simplificación): seq2seq GRU (codificador bidireccional + decodificador con atención de Bahdanau).

Mejoras sobre la primera iteración:
  - Mecanismo de COPIA (pointer-generator, See et al. 2017): el decodificador puede copiar palabras
    del texto original, lo que ayuda con cifras, fechas, nombres propios y palabras fuera de vocabulario.
  - Decodificación con BEAM SEARCH, bloqueo de n-gramas repetidos y normalización por longitud.
  - Búsqueda de hiperparámetros sobre validación (opcional, --buscar-hp).
  - Varias semillas por variante (media y desviación típica).
  - Preentrenamiento con un dataset externo (con su control de vocabulario ampliado sin preentrenar).
  - Los vectores preentrenados de spaCy se recentran y reescalan (ver embeddings.py).
"""
import copy
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import (CARPETA_E3, VOCAB_MAX, VOCAB_MIN_FREQ, S2S_MAX_SRC, S2S_MAX_TGT, S2S_HIDDEN, DROPOUT,
                    BATCH, LR, S2S_EPOCAS, PACIENCIA, CLIP, SEMILLAS, GRID_S2S, HP_EPOCAS,
                    NO_REPETIR_NGRAMA, LONG_ALPHA, PRE_EPOCAS_S2S)
from embeddings import construir_embedding, fijar_semilla, DEV, PAD, UNK, BOS, EOS
from metricas import bleu, bleu_ic, agregar_semillas

VARIANTES = {
    "GRU+att, embeddings aleatorios":       dict(emb="aleatorio", congelar=False, atencion=True),
    "GRU+att, Word2Vec propio congelado":   dict(emb="w2v", congelar=True, atencion=True),
    "GRU+att, Word2Vec propio fine-tuneado": dict(emb="w2v", congelar=False, atencion=True),
    "GRU+att, spaCy congelado":             dict(emb="spacy", congelar=True, atencion=True),
    "GRU+att, spaCy fine-tuneado":          dict(emb="spacy", congelar=False, atencion=True),
    "GRU sin atención, Word2Vec propio fine-tuneado": dict(emb="w2v", congelar=False, atencion=False),
    # --- mecanismo de copia ---
    "GRU+att+copia, Word2Vec propio fine-tuneado": dict(emb="w2v", congelar=False, atencion=True, copia=True),
    # --- dataset externo (solo con --externo y el dataset conectado en preentreno.py) ---
    "GRU+att, Word2Vec externo fine-tuneado":
        dict(emb="w2v_ext", congelar=False, atencion=True, requiere_ext=True),
    "GRU+att, vocabulario ampliado, sin preentreno (control)":
        dict(emb="w2v", congelar=False, atencion=True, vocab_ext=True, requiere_ext=True),
    "GRU+att, preentreno en dataset externo + fine-tuning":
        dict(emb="w2v", congelar=False, atencion=True, vocab_ext=True, preentreno=True, requiere_ext=True),
}
HP_DEFECTO = dict(hidden=S2S_HIDDEN, dropout=DROPOUT, lr=LR)
STD_KEYS = ["bleu_val", "bleu_test", "bleu_test_beam"]


# ------------------------------------------------------------------ datos
def _toks(registro):
    return [t.lower() for t in registro["tokens"]]


def construir_vocab(train, extra=None):
    """Vocabulario con train (original y simplificado) y, si se da, el dataset externo."""
    c = Counter()
    for df in (train, extra):
        if df is None:
            continue
        for col in ("procesado_original", "procesado_simplificado"):
            for r in df[col]:
                c.update(_toks(r))
    itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + \
           [w for w, f in c.most_common(VOCAB_MAX) if f >= VOCAB_MIN_FREQ]
    return itos, {w: i for i, w in enumerate(itos)}


class Datos:
    """Pares (original, simplificado) ya convertidos a ids.
    Con copia=True se construye ademas el vocabulario ampliado POR FRASE: las palabras del original
    que no estan en el vocabulario reciben ids V, V+1, ... (src_ext/tgt_ext) para poder copiarlas."""
    def __init__(self, df, stoi, copia=False):
        self.copia = copia
        self.refs = df["answer"].tolist()
        V = len(stoi)
        self.src, self.tgt, self.src_ext, self.tgt_ext, self.oov = [], [], [], [], []
        for o, s in zip(df["procesado_original"], df["procesado_simplificado"]):
            st = _toks(o)[:S2S_MAX_SRC] or ["<unk>"]
            tt = _toks(s)[:S2S_MAX_TGT - 2]
            self.src.append([stoi.get(t, UNK) for t in st])
            self.tgt.append([BOS] + [stoi.get(t, UNK) for t in tt] + [EOS])
            if copia:
                oov_ids = {}
                ext = []
                for t in st:
                    if t in stoi:
                        ext.append(stoi[t])
                    else:
                        if t not in oov_ids:
                            oov_ids[t] = V + len(oov_ids)
                        ext.append(oov_ids[t])
                self.src_ext.append(ext)
                self.tgt_ext.append([BOS] + [stoi[t] if t in stoi else oov_ids.get(t, UNK) for t in tt] + [EOS])
                self.oov.append(list(oov_ids))      # el orden de insercion = orden de los ids

    def __len__(self):
        return len(self.src)


def _pad(seqs):
    t = torch.zeros(len(seqs), max(len(s) for s in seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, :len(s)] = torch.tensor(s)
    return t


def lotes(d, batch, barajar):
    idx = list(range(len(d)))
    if barajar:
        random.shuffle(idx)
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        lote = {"idx": b, "src": _pad([d.src[j] for j in b]).to(DEV), "tgt": _pad([d.tgt[j] for j in b]).to(DEV)}
        if d.copia:
            lote["src_ext"] = _pad([d.src_ext[j] for j in b]).to(DEV)
            lote["tgt_ext"] = _pad([d.tgt_ext[j] for j in b]).to(DEV)
            lote["n_oov"] = max(len(d.oov[j]) for j in b)
        yield lote


# ------------------------------------------------------------------ modelo
class Codificador(nn.Module):
    def __init__(self, emb, hid, dropout):
        super().__init__()
        self.emb = emb
        self.rnn = nn.GRU(emb.embedding_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hid, hid)
        self.drop = nn.Dropout(dropout)

    def forward(self, src, lens):
        x = self.drop(self.emb(src))
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        h0 = torch.tanh(self.fc(torch.cat([h[-2], h[-1]], dim=1)))
        return out, h0


class Atencion(nn.Module):
    """Atención aditiva (Bahdanau et al., 2015)."""
    def __init__(self, hid):
        super().__init__()
        self.W = nn.Linear(hid, hid, bias=False)
        self.U = nn.Linear(2 * hid, hid, bias=False)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, h, enc_out, enc_proj, mask):
        e = self.v(torch.tanh(self.W(h).unsqueeze(1) + enc_proj)).squeeze(2)
        a = F.softmax(e.masked_fill(~mask, -1e9), dim=1)
        return torch.bmm(a.unsqueeze(1), enc_out).squeeze(1), a


class Decodificador(nn.Module):
    def __init__(self, emb, hid, n_vocab, dropout, usar_atencion, copia=False):
        super().__init__()
        self.emb = emb
        d = emb.embedding_dim
        self.att = Atencion(hid) if usar_atencion else None
        self.rnn = nn.GRUCell(d + 2 * hid, hid)
        self.out = nn.Linear(hid + 2 * hid + d, n_vocab)
        self.gen = nn.Linear(hid + 2 * hid + d, 1) if copia else None    # p_gen: generar vs copiar
        self.drop = nn.Dropout(dropout)

    def paso(self, y_prev, h, enc_out, enc_proj, mask, ctx_fijo):
        e = self.drop(self.emb(y_prev))
        if self.att is not None:
            ctx, a = self.att(h, enc_out, enc_proj, mask)
        else:
            ctx, a = ctx_fijo, None
        h = self.rnn(torch.cat([e, ctx], dim=1), h)
        feat = torch.cat([h, ctx, e], dim=1)
        logits = self.out(self.drop(feat))
        p_gen = torch.sigmoid(self.gen(self.drop(feat))) if self.gen is not None else None
        return logits, h, a, p_gen


def _prohibidos(hist, n):
    """Palabras que cerrarian un n-grama ya generado (para no repetir)."""
    if n < 2 or len(hist) < n - 1:
        return ()
    pref = tuple(hist[len(hist) - (n - 1):])
    return {hist[i + n - 1] for i in range(len(hist) - n + 1) if tuple(hist[i:i + n - 1]) == pref}


class Seq2Seq(nn.Module):
    def __init__(self, emb, n_vocab, usar_atencion, hidden=S2S_HIDDEN, dropout=DROPOUT, copia=False):
        super().__init__()
        if copia and not usar_atencion:
            raise ValueError("El mecanismo de copia necesita atencion.")
        self.copia, self.V = copia, n_vocab
        self.enc = Codificador(emb, hidden, dropout)
        self.dec = Decodificador(emb, hidden, n_vocab, dropout, usar_atencion, copia)  # embedding compartido

    def _codificar(self, src):
        lens = (src != PAD).sum(1).clamp(min=1)
        mask = src != PAD
        enc_out, h = self.enc(src, lens)
        # Contexto fijo (solo se usa en la variante sin atención): media de los estados del codificador
        ctx_fijo = (enc_out * mask.unsqueeze(2)).sum(1) / lens.unsqueeze(1)
        proj = self.dec.att.U(enc_out) if self.dec.att is not None else None
        return enc_out, h, mask, proj, ctx_fijo

    def _dist(self, logits, a, p_gen, src_ext, n_oov):
        """Distribucion sobre el vocabulario (y, con copia, sobre vocabulario + palabras del original)."""
        pv = F.softmax(logits, dim=-1)
        if not self.copia:
            return pv
        if n_oov:
            pv = F.pad(pv, (0, n_oov))
        pc = torch.zeros_like(pv).scatter_add(1, src_ext, a)
        return p_gen * pv + (1 - p_gen) * pc

    def forward(self, b):
        """Teacher forcing. b: lote (ver lotes). Devuelve (loss_suma, nº tokens, nº aciertos)."""
        tgt = b["tgt"]
        gold_all = b["tgt_ext"] if self.copia else tgt
        enc_out, h, mask, proj, ctx = self._codificar(b["src"])
        loss, n, ok = 0.0, 0, 0
        for t in range(tgt.size(1) - 1):
            logits, h, a, pg = self.dec.paso(tgt[:, t], h, enc_out, proj, mask, ctx)
            gold = gold_all[:, t + 1]
            if self.copia:
                p = self._dist(logits, a, pg, b["src_ext"], b["n_oov"])
                nll = -torch.log(p.gather(1, gold.unsqueeze(1)).squeeze(1) + 1e-12)
                loss = loss + (nll * (gold != PAD)).sum()
                pred = p.argmax(1)
            else:
                loss = loss + F.cross_entropy(logits, gold, ignore_index=PAD, reduction="sum")
                pred = logits.argmax(1)
            with torch.no_grad():
                m = gold != PAD
                n += int(m.sum())
                ok += int(((pred == gold) & m).sum())
        return loss, n, ok

    @torch.no_grad()
    def generar(self, b, max_len=S2S_MAX_TGT, no_repeat=NO_REPETIR_NGRAMA):
        """Decodificación voraz. <pad>, <unk> y <bos> no se generan; se bloquean n-gramas repetidos."""
        src = b["src"]
        enc_out, h, mask, proj, ctx = self._codificar(src)
        B = src.size(0)
        y_in = torch.full((B,), BOS, dtype=torch.long, device=src.device)
        salida = torch.zeros(B, max_len, dtype=torch.long, device=src.device)
        fin = torch.zeros(B, dtype=torch.bool, device=src.device)
        hist = [[] for _ in range(B)]
        for t in range(max_len):
            logits, h, a, pg = self.dec.paso(y_in, h, enc_out, proj, mask, ctx)
            lp = torch.log(self._dist(logits, a, pg, b.get("src_ext"), b.get("n_oov", 0)) + 1e-12)
            lp[:, [PAD, UNK, BOS]] = -1e9
            if no_repeat:
                for r in range(B):
                    if not bool(fin[r]):
                        for w in _prohibidos(hist[r], no_repeat):
                            lp[r, w] = -1e9
            y = lp.argmax(1).masked_fill(fin, PAD)
            salida[:, t] = y
            for r, w in enumerate(y.tolist()):
                hist[r].append(w)
            fin |= y == EOS
            if bool(fin.all()):
                break
            y_in = torch.where(y >= self.V, torch.full_like(y, UNK), y)   # palabras copiadas -> <unk> como entrada
        return salida

    @torch.no_grad()
    def generar_beam(self, b, K, max_len=S2S_MAX_TGT, no_repeat=NO_REPETIR_NGRAMA, alpha=LONG_ALPHA):
        """Beam search de UNA frase (lote de tamaño 1). Devuelve la lista de ids de la mejor hipotesis."""
        enc_out, h, mask, proj, ctx = self._codificar(b["src"])
        enc_out = enc_out.expand(K, -1, -1).contiguous()
        mask, ctx, h = mask.expand(K, -1).contiguous(), ctx.expand(K, -1).contiguous(), h.expand(K, -1).contiguous()
        if proj is not None:
            proj = proj.expand(K, -1, -1).contiguous()
        src_ext = b["src_ext"].expand(K, -1).contiguous() if self.copia else None
        n_oov = b["n_oov"] if self.copia else 0
        seqs = [[] for _ in range(K)]
        scores = torch.full((K,), -1e9, device=DEV)
        scores[0] = 0.0                                   # al principio solo vive una hipotesis
        y_in = torch.full((K,), BOS, dtype=torch.long, device=DEV)
        terminadas = []
        for t in range(max_len):
            logits, h, a, pg = self.dec.paso(y_in, h, enc_out, proj, mask, ctx)
            lp = torch.log(self._dist(logits, a, pg, src_ext, n_oov) + 1e-12)
            lp[:, [PAD, UNK, BOS]] = -1e9
            if no_repeat:
                for k in range(K):
                    for w in _prohibidos(seqs[k], no_repeat):
                        lp[k, w] = -1e9
            Vp = lp.size(1)
            top_s, top_i = (scores.unsqueeze(1) + lp).reshape(-1).topk(min(2 * K, K * Vp))
            nuevas, n_scores, n_idx = [], [], []
            for s, i in zip(top_s.tolist(), top_i.tolist()):
                if s < -1e8:
                    continue
                k, w = divmod(i, Vp)
                if w == EOS:
                    terminadas.append((s / ((len(seqs[k]) + 1) ** alpha), seqs[k]))
                else:
                    nuevas.append(seqs[k] + [w])
                    n_scores.append(s)
                    n_idx.append(k)
                if len(nuevas) == K:
                    break
            if not nuevas or len(terminadas) >= K:
                break
            while len(nuevas) < K:                       # relleno para mantener K hipotesis
                nuevas.append(nuevas[0])
                n_scores.append(-1e9)
                n_idx.append(n_idx[0])
            seqs = nuevas
            scores = torch.tensor(n_scores, device=DEV)
            h = h[torch.tensor(n_idx, device=DEV)]
            y_in = torch.tensor([w if w < self.V else UNK for w in (s[-1] for s in seqs)],
                                dtype=torch.long, device=DEV)
        if not terminadas:
            terminadas = [(scores[i].item() / ((len(seqs[i]) + 1) ** alpha), seqs[i]) for i in range(K)]
        return max(terminadas, key=lambda x: x[0])[1]


# ------------------------------------------------------------------ entrenamiento / evaluación
def _epoca(modelo, datos, opt=None):
    modelo.train(opt is not None)
    L, N, OK = 0.0, 0, 0
    with torch.set_grad_enabled(opt is not None):
        for b in lotes(datos, BATCH, barajar=opt is not None):
            loss, n, ok = modelo(b)
            if opt is not None:
                opt.zero_grad()
                (loss / n).backward()
                nn.utils.clip_grad_norm_(modelo.parameters(), CLIP)
                opt.step()
            L += float(loss)
            N += n
            OK += ok
    return L / N, OK / N


def _a_texto(ids, itos, oov=None):
    pal = []
    for i in ids:
        if i == EOS or i == PAD:
            break
        if i < len(itos):
            pal.append(itos[i])
        else:                                            # palabra copiada del original (fuera de vocabulario)
            j = i - len(itos)
            pal.append(oov[j] if oov and j < len(oov) else "<unk>")
    return " ".join(pal)


def traducir(modelo, d, itos, beam=0, batch=64):
    """Textos generados para todos los ejemplos de d. beam>1 -> beam search; si no, voraz."""
    modelo.eval()
    hip = []
    if beam and beam > 1:
        for b in lotes(d, 1, False):
            ids = modelo.generar_beam(b, beam)
            hip.append(_a_texto(ids, itos, d.oov[b["idx"][0]] if d.copia else None))
    else:
        for b in lotes(d, batch, False):
            sal = modelo.generar(b)
            for r, j in enumerate(b["idx"]):
                hip.append(_a_texto(sal[r].tolist(), itos, d.oov[j] if d.copia else None))
    return hip


def _bucle(modelo, params, d_train, d_val, epocas, lr, paciencia, etiqueta):
    """Entrenamiento. Con d_val=None (preentrenamiento) no hay early stopping y se devuelve el estado final."""
    opt = torch.optim.Adam(params, lr=lr)
    mejor, mejor_estado, sin_mejora, hechas = float("inf"), None, 0, 0
    for ep in range(1, epocas + 1):
        l_tr, a_tr = _epoca(modelo, d_train, opt)
        hechas = ep
        if d_val is None:
            print(f"[{etiqueta}] época {ep:02d} | train loss {l_tr:.3f} acc {a_tr:.3f}")
            continue
        l_va, a_va = _epoca(modelo, d_val)
        print(f"[{etiqueta}] época {ep:02d} | train loss {l_tr:.3f} acc {a_tr:.3f} | val loss {l_va:.3f} acc {a_va:.3f}")
        if l_va < mejor - 1e-4:
            mejor, mejor_estado, sin_mejora = l_va, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                print("Early stopping.")
                break
    if d_val is None:
        return copy.deepcopy(modelo.state_dict()), None, hechas
    return mejor_estado, mejor, hechas


def entrenar_variante(nombre, cfg, itos, datos, modelo_w2v, nlp, epocas, hp, semilla, w2v_ext=None,
                      datos_ext=None, beam=0, evaluar=True, guardar=False):
    """Entrena una variante con una semilla. Devuelve (resultado, hipotesis voraces de test,
    hipotesis con beam de test o None)."""
    fijar_semilla(semilla)
    t0 = time.time()
    emb, cobertura = construir_embedding(cfg["emb"], itos, cfg["congelar"], modelo_w2v, nlp, w2v_ext)
    modelo = Seq2Seq(emb, len(itos), cfg["atencion"], hp["hidden"], hp["dropout"], cfg.get("copia", False)).to(DEV)
    params = [p for p in modelo.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"\n--- {nombre} | semilla {semilla} | hidden={hp['hidden']} dropout={hp['dropout']} lr={hp['lr']} | "
          f"parámetros entrenables: {n_train:,} | cobertura embedding: "
          f"{'-' if cobertura is None else f'{100 * cobertura:.1f}%'} ---")
    if cfg.get("preentreno"):
        estado, _, _ = _bucle(modelo, params, datos_ext, None, PRE_EPOCAS_S2S, hp["lr"], 0, "preentreno")
        modelo.load_state_dict(estado)
    estado, mejor, hechas = _bucle(modelo, params, datos["train"], datos["val"], epocas, hp["lr"], PACIENCIA, nombre)
    modelo.load_state_dict(estado)
    if not evaluar:
        return {"loss_val": mejor, "epocas": hechas}, None, None
    if guardar:
        (CARPETA_E3 / "modelos").mkdir(parents=True, exist_ok=True)
        nombre_f = nombre.replace(" ", "_").replace(",", "").replace("+", "").replace("(", "").replace(")", "")
        torch.save(estado, CARPETA_E3 / "modelos" / (nombre_f + ".pt"))

    hip_val = traducir(modelo, datos["val"], itos)
    hip_te = traducir(modelo, datos["test"], itos)
    hip_beam = traducir(modelo, datos["test"], itos, beam=beam) if beam and beam > 1 else None
    _, a_te = _epoca(modelo, datos["test"])
    rep = {"aleatorio": "Embeddings aleatorios", "w2v": "Word2Vec propio (E2)",
           "w2v_ext": "Word2Vec propio + dataset externo",
           "spacy": "spaCy es_core_news_md"}[cfg["emb"]]
    if cfg["emb"] != "aleatorio":
        rep += " congelado" if cfg["congelar"] else " fine-tuneado"
    if cfg.get("preentreno"):
        rep += " + preentreno externo"
    res = {"modelo": nombre, "familia": "deep", "representacion": rep,
           "bleu_val": bleu(hip_val, datos["val"].refs), "bleu_test": bleu(hip_te, datos["test"].refs),
           "loss_val": mejor, "acc_token_test": a_te, "epocas": hechas,
           "parametros_entrenables": n_train, "tiempo_s": round(time.time() - t0), "cobertura_emb": cobertura}
    if hip_beam is not None:
        res["bleu_test_beam"] = bleu(hip_beam, datos["test"].refs)
    print(f"BLEU val={res['bleu_val']:.2f}  test={res['bleu_test']:.2f}"
          + (f"  test(beam {beam})={res['bleu_test_beam']:.2f}" if hip_beam is not None else ""))
    return res, hip_te, hip_beam


def buscar_hiperparametros(itos, datos, modelo_w2v, nlp, epocas):
    """Rejilla (hidden x dropout x lr) sobre la variante GRU+att con Word2Vec fine-tuneado.
    Se elige la combinacion con menor loss de VALIDACION (comparable entre configuraciones)."""
    print("\n=== FASE 3 · BÚSQUEDA DE HIPERPARÁMETROS (validación) ===")
    base = dict(emb="w2v", congelar=False, atencion=True)
    filas, mejor = [], None
    for hidden in GRID_S2S["hidden"]:
        for dropout in GRID_S2S["dropout"]:
            for lr in GRID_S2S["lr"]:
                hp = dict(hidden=hidden, dropout=dropout, lr=lr)
                r, _, _ = entrenar_variante(f"HP h={hidden} d={dropout} lr={lr}", base, itos, datos, modelo_w2v,
                                            nlp, epocas, hp, SEMILLAS[0], evaluar=False)
                filas.append({**hp, "loss_val": round(r["loss_val"], 4), "epocas": r["epocas"]})
                if mejor is None or r["loss_val"] < mejor[0]:
                    mejor = (r["loss_val"], hp)
    CARPETA_E3.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(CARPETA_E3 / "grid_seq2seq.csv", index=False)
    print(f"Mejor combinación: {mejor[1]} (loss val {mejor[0]:.3f}). Se usará en todas las variantes.")
    return mejor[1]


def ejecutar_deep(train, val, test, modelo_w2v, nlp, variantes=None, epocas=None, opc=None):
    """Devuelve (resultados agregados, salidas de test por sistema, hiperparametros usados).
    salidas[nombre] = {'greedy': [...], 'beam': [...] o None} (semilla 0)."""
    print("\n=== DEEP LEARNING (seq2seq GRU + atención) ===")
    opc = opc or {}
    semillas = opc.get("semillas") or SEMILLAS[:1]
    beam = opc.get("beam", 0)
    externo = opc.get("externo")
    ext_df = externo["df"] if externo else None
    w2v_ext = externo["w2v"] if externo else None

    itos, stoi = construir_vocab(train)
    itos_x, stoi_x = construir_vocab(train, ext_df) if ext_df is not None else (None, None)
    print("Vocabulario compartido:", len(itos), "| semillas:", semillas, "| beam:", beam or "no")
    cache = {}

    def datos_de(vocab_ext, copia):
        clave = (vocab_ext, copia)
        if clave not in cache:
            st = stoi_x if vocab_ext else stoi
            cache[clave] = {"train": Datos(train, st, copia), "val": Datos(val, st, copia),
                            "test": Datos(test, st, copia)}
        return cache[clave]

    hp = dict(HP_DEFECTO)
    if opc.get("buscar_hp"):
        hp = buscar_hiperparametros(itos, datos_de(False, False), modelo_w2v, nlp,
                                    min(HP_EPOCAS, epocas or HP_EPOCAS))
    resultados, salidas = [], {}
    for nombre, cfg in VARIANTES.items():
        if variantes and nombre not in variantes:
            continue
        if cfg.get("requiere_ext") and ext_df is None:
            print(f"\n(se omite '{nombre}': no hay dataset externo conectado, ver preentreno.py)")
            continue
        vx, copia = cfg.get("vocab_ext", False), cfg.get("copia", False)
        d = datos_de(vx, copia)
        v_itos = itos_x if vx else itos
        datos_ext = Datos(ext_df, stoi_x if vx else stoi, copia) if cfg.get("preentreno") else None
        runs = []
        for k, s in enumerate(semillas):
            r, hip_te, hip_beam = entrenar_variante(nombre, cfg, v_itos, d, modelo_w2v, nlp, epocas or S2S_EPOCAS,
                                                    hp, s, w2v_ext, datos_ext, beam, guardar=(k == 0))
            runs.append(r)
            if k == 0:
                salidas[nombre] = {"greedy": hip_te, "beam": hip_beam}
        agg = agregar_semillas(runs, STD_KEYS)
        agg["hiperparametros"] = f"hidden={hp['hidden']}, dropout={hp['dropout']}, lr={hp['lr']}"
        mejor_hip = salidas[nombre]["beam"] or salidas[nombre]["greedy"]
        agg["bleu_test_ic"] = bleu_ic(mejor_hip, d["test"].refs)
        resultados.append(agg)
    return resultados, salidas, hp
