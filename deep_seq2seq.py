"""Deep learning: seq2seq GRU (codificador bidireccional + decodificador con atención de Bahdanau).

Se comparan distintos embeddings de entrada (aleatorio, Word2Vec propio del E2, vectores de
spaCy), congelados o fine-tuneados, y se incluye una ablación sin atención.
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

from config import (CARPETA_E3, VOCAB_MAX, VOCAB_MIN_FREQ, S2S_MAX_SRC, S2S_MAX_TGT, S2S_HIDDEN, DROPOUT,
                    BATCH, LR, S2S_EPOCAS, PACIENCIA, CLIP)
from embeddings import construir_embedding, fijar_semilla, DEV, PAD, UNK, BOS, EOS
from metricas import bleu

VARIANTES = {
    "GRU+att, embeddings aleatorios":       dict(emb="aleatorio", congelar=False, atencion=True),
    "GRU+att, Word2Vec propio congelado":   dict(emb="w2v", congelar=True, atencion=True),
    "GRU+att, Word2Vec propio fine-tuneado": dict(emb="w2v", congelar=False, atencion=True),
    "GRU+att, spaCy congelado":             dict(emb="spacy", congelar=True, atencion=True),
    "GRU+att, spaCy fine-tuneado":          dict(emb="spacy", congelar=False, atencion=True),
    "GRU sin atención, Word2Vec propio fine-tuneado": dict(emb="w2v", congelar=False, atencion=False),
}


# ------------------------------------------------------------------ datos
def _toks(registro):
    return [t.lower() for t in registro["tokens"]]


def construir_vocab(train):
    c = Counter()
    for col in ("procesado_original", "procesado_simplificado"):
        for r in train[col]:
            c.update(_toks(r))
    itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + \
           [w for w, f in c.most_common(VOCAB_MAX) if f >= VOCAB_MIN_FREQ]
    return itos, {w: i for i, w in enumerate(itos)}


def _ids(tokens, stoi, maximo):
    ids = [stoi.get(t, UNK) for t in tokens[:maximo]]
    return ids or [UNK]


def preparar(df, stoi):
    src = [_ids(_toks(r), stoi, S2S_MAX_SRC) for r in df["procesado_original"]]
    tgt = [[BOS] + _ids(_toks(r), stoi, S2S_MAX_TGT - 2) + [EOS] for r in df["procesado_simplificado"]]
    return src, tgt, df["answer"].tolist()


def _pad(seqs):
    t = torch.zeros(len(seqs), max(len(s) for s in seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, :len(s)] = torch.tensor(s)
    return t


def lotes(src, tgt, batch, barajar):
    idx = list(range(len(src)))
    if barajar:
        random.shuffle(idx)
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        yield b, _pad([src[j] for j in b]).to(DEV), _pad([tgt[j] for j in b]).to(DEV)


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
    def __init__(self, emb, hid, n_vocab, dropout, usar_atencion):
        super().__init__()
        self.emb = emb
        d = emb.embedding_dim
        self.att = Atencion(hid) if usar_atencion else None
        self.rnn = nn.GRUCell(d + 2 * hid, hid)
        self.out = nn.Linear(hid + 2 * hid + d, n_vocab)
        self.drop = nn.Dropout(dropout)

    def paso(self, y_prev, h, enc_out, enc_proj, mask, ctx_fijo):
        e = self.drop(self.emb(y_prev))
        ctx = self.att(h, enc_out, enc_proj, mask)[0] if self.att is not None else ctx_fijo
        h = self.rnn(torch.cat([e, ctx], dim=1), h)
        return self.out(self.drop(torch.cat([h, ctx, e], dim=1))), h


class Seq2Seq(nn.Module):
    def __init__(self, emb, n_vocab, usar_atencion):
        super().__init__()
        self.enc = Codificador(emb, S2S_HIDDEN, DROPOUT)
        self.dec = Decodificador(emb, S2S_HIDDEN, n_vocab, DROPOUT, usar_atencion)  # embedding compartido

    def _codificar(self, src):
        lens = (src != PAD).sum(1).clamp(min=1)
        mask = src != PAD
        enc_out, h = self.enc(src, lens)
        # Contexto fijo (solo se usa en la variante sin atención): media de los estados del codificador
        ctx_fijo = (enc_out * mask.unsqueeze(2)).sum(1) / lens.unsqueeze(1)
        proj = self.dec.att.U(enc_out) if self.dec.att is not None else None
        return enc_out, h, mask, proj, ctx_fijo

    def forward(self, src, tgt):
        """Teacher forcing. Devuelve (loss_suma, nº tokens, nº aciertos)."""
        enc_out, h, mask, proj, ctx = self._codificar(src)
        loss, n, ok = 0.0, 0, 0
        for t in range(tgt.size(1) - 1):
            logits, h = self.dec.paso(tgt[:, t], h, enc_out, proj, mask, ctx)
            gold = tgt[:, t + 1]
            loss = loss + F.cross_entropy(logits, gold, ignore_index=PAD, reduction="sum")
            with torch.no_grad():
                m = gold != PAD
                n += int(m.sum())
                ok += int(((logits.argmax(1) == gold) & m).sum())
        return loss, n, ok

    @torch.no_grad()
    def generar(self, src, max_len=S2S_MAX_TGT):
        """Decodificación voraz (greedy). <pad>, <unk> y <bos> no se pueden generar."""
        enc_out, h, mask, proj, ctx = self._codificar(src)
        B = src.size(0)
        y = torch.full((B,), BOS, dtype=torch.long, device=src.device)
        salida = torch.zeros(B, max_len, dtype=torch.long, device=src.device)
        fin = torch.zeros(B, dtype=torch.bool, device=src.device)
        for t in range(max_len):
            logits, h = self.dec.paso(y, h, enc_out, proj, mask, ctx)
            logits[:, [PAD, UNK, BOS]] = -1e9
            y = logits.argmax(1).masked_fill(fin, PAD)
            salida[:, t] = y
            fin |= y == EOS
            if bool(fin.all()):
                break
        return salida


# ------------------------------------------------------------------ entrenamiento / evaluación
def _epoca(modelo, datos, opt=None):
    src, tgt = datos
    modelo.train(opt is not None)
    L, N, OK = 0.0, 0, 0
    with torch.set_grad_enabled(opt is not None):
        for _, s, t in lotes(src, tgt, BATCH, barajar=opt is not None):
            loss, n, ok = modelo(s, t)
            if opt is not None:
                opt.zero_grad()
                (loss / n).backward()
                nn.utils.clip_grad_norm_(modelo.parameters(), CLIP)
                opt.step()
            L += float(loss)
            N += n
            OK += ok
    return L / N, OK / N


def _a_texto(ids, itos):
    pal = []
    for i in ids:
        if i == EOS or i == PAD:
            break
        pal.append(itos[i])
    return " ".join(pal)


def traducir(modelo, src, itos, batch=64):
    modelo.eval()
    hip = []
    for i in range(0, len(src), batch):
        s = _pad(src[i:i + batch]).to(DEV)
        hip += [_a_texto(fila.tolist(), itos) for fila in modelo.generar(s)]
    return hip


def entrenar_variante(nombre, cfg, itos, datos, modelo_w2v, nlp, epocas):
    fijar_semilla()
    emb, cobertura = construir_embedding(cfg["emb"], itos, cfg["congelar"], modelo_w2v, nlp)
    modelo = Seq2Seq(emb, len(itos), cfg["atencion"]).to(DEV)
    params = [p for p in modelo.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"\n--- {nombre} | parámetros entrenables: {n_train:,} | "
          f"cobertura embedding: {'-' if cobertura is None else f'{100 * cobertura:.1f}%'} ---")
    opt = torch.optim.Adam(params, lr=LR)

    mejor, mejor_estado, sin_mejora, t0, hechas = float("inf"), None, 0, time.time(), 0
    for ep in range(1, epocas + 1):
        l_tr, a_tr = _epoca(modelo, (datos["train"][0], datos["train"][1]), opt)
        l_va, a_va = _epoca(modelo, (datos["val"][0], datos["val"][1]))
        hechas = ep
        print(f"época {ep:02d} | train loss {l_tr:.3f} acc {a_tr:.3f} | val loss {l_va:.3f} acc {a_va:.3f}")
        if l_va < mejor - 1e-4:
            mejor, mejor_estado, sin_mejora = l_va, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                print("Early stopping.")
                break
    modelo.load_state_dict(mejor_estado)
    (CARPETA_E3 / "modelos").mkdir(parents=True, exist_ok=True)
    torch.save(mejor_estado, CARPETA_E3 / "modelos" / (nombre.replace(" ", "_").replace(",", "")
                                                          .replace("+", "") + ".pt"))

    hip_val = traducir(modelo, datos["val"][0], itos)
    hip_te = traducir(modelo, datos["test"][0], itos)
    l_te, a_te = _epoca(modelo, (datos["test"][0], datos["test"][1]))
    res = {"modelo": nombre, "familia": "deep",
           "representacion": {"aleatorio": "Embeddings aleatorios", "w2v": "Word2Vec propio (E2)",
                              "spacy": "spaCy es_core_news_md"}[cfg["emb"]]
                             + (" congelado" if cfg["congelar"] and cfg["emb"] != "aleatorio" else "")
                             + (" fine-tuneado" if not cfg["congelar"] and cfg["emb"] != "aleatorio" else ""),
           "bleu_val": bleu(hip_val, datos["val"][2]), "bleu_test": bleu(hip_te, datos["test"][2]),
           "loss_val": mejor, "acc_token_test": a_te, "epocas": hechas,
           "parametros_entrenables": n_train, "tiempo_s": round(time.time() - t0),
           "cobertura_emb": cobertura}
    print(f"BLEU val={res['bleu_val']:.2f}  test={res['bleu_test']:.2f}")
    return res, hip_te[:5]


def ejecutar_deep(train, val, test, modelo_w2v, nlp, variantes=None, epocas=None):
    print("\n=== DEEP LEARNING (seq2seq GRU + atención) ===")
    itos, stoi = construir_vocab(train)
    print("Vocabulario compartido:", len(itos))
    datos = {"train": preparar(train, stoi), "val": preparar(val, stoi), "test": preparar(test, stoi)}
    resultados, muestras = [], {}
    for nombre, cfg in VARIANTES.items():
        if variantes and nombre not in variantes:
            continue
        r, m = entrenar_variante(nombre, cfg, itos, datos, modelo_w2v, nlp, epocas or S2S_EPOCAS)
        resultados.append(r)
        muestras[nombre] = m
    return resultados, muestras

