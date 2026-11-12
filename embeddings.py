"""Embeddings que se comparan en el E3 (fase 1 y fase 3): aleatorios, Word2Vec propio (E2) y spaCy.
Reutilizan en memoria el modelo Word2Vec y el modelo de spaCy que ya se han cargado en el E2."""
import random
import numpy as np
import torch
import torch.nn as nn
from config import SEED, EMB_ALEATORIO, NORMALIZAR_EMB, STD_EMB_OBJETIVO

PAD, UNK, BOS, EOS = 0, 1, 2, 3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semilla(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _matriz_con_aleatorios(itos, buscar, dim):
    """Fila = vector preentrenado si existe; si no, aleatorio con la misma escala."""
    M = np.zeros((len(itos), dim), dtype=np.float32)
    hay = np.zeros(len(itos), dtype=bool)
    for i, w in enumerate(itos):
        if i < 4:
            continue
        v = buscar(w)
        if v is not None:
            M[i], hay[i] = v, True
    escala = float(M[hay].std()) if hay.any() else 0.1
    faltan = ~hay
    faltan[:4] = False
    M[faltan] = np.random.RandomState(SEED).normal(0, escala, (int(faltan.sum()), dim))
    M[PAD] = 0
    return M, hay[4:].mean()


def matriz_w2v(itos, modelo_w2v):
    wv = modelo_w2v.wv    # entrenado en el E2 con los tokens en minusculas
    return _matriz_con_aleatorios(itos, lambda w: wv[w] if w in wv else None, wv.vector_size)


def matriz_spacy(itos, nlp):
    def buscar(w):
        lex = nlp.vocab[w]
        return lex.vector if lex.has_vector else None
    return _matriz_con_aleatorios(itos, buscar, nlp.vocab.vectors_length)


def _diagnostico(M, tipo):
    """Escala y diversidad de los vectores (ayuda a entender por que unos embeddings rinden peor)."""
    filas = M[4:]
    unicas = len({f.tobytes() for f in filas})
    print(f"[emb {tipo}] std={filas.std():.3f} | filas distintas: {unicas}/{len(filas)} "
          f"({100 * unicas / max(len(filas), 1):.1f} %)")


def _normalizar(M, tipo):
    """Recentra y reescala los vectores preentrenados a una escala comun (STD_EMB_OBJETIVO).
    Los vectores de spaCy tienen otra escala que los demas y eso puede dificultar el entrenamiento."""
    sub = M[4:]
    M[4:] = (sub - sub.mean(axis=0)) / (sub.std() + 1e-8) * STD_EMB_OBJETIVO
    M[PAD] = 0
    print(f"[emb {tipo}] reescalado a std={STD_EMB_OBJETIVO}")
    return M


def construir_embedding(tipo, itos, congelar, modelo_w2v, nlp, modelo_w2v_ext=None):
    """Devuelve (nn.Embedding, cobertura). cobertura = % de palabras del vocabulario con vector previo.
    tipo: aleatorio | w2v (propio, E2) | w2v_ext (propio + dataset externo) | spacy."""
    if tipo == "aleatorio":
        return nn.Embedding(len(itos), EMB_ALEATORIO, padding_idx=PAD), None
    if tipo == "w2v":
        M, cobertura = matriz_w2v(itos, modelo_w2v)
    elif tipo == "w2v_ext":
        if modelo_w2v_ext is None:
            raise ValueError("Se pidio w2v_ext pero no hay Word2Vec entrenado con el dataset externo.")
        M, cobertura = matriz_w2v(itos, modelo_w2v_ext)
    else:
        M, cobertura = matriz_spacy(itos, nlp)
    _diagnostico(M, tipo)
    if NORMALIZAR_EMB.get(tipo):
        M = _normalizar(M, tipo)
    return nn.Embedding.from_pretrained(torch.tensor(M), freeze=congelar, padding_idx=PAD), cobertura
