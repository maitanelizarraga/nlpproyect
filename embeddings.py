"""Embeddings que se comparan en el E3 (fase 1 y fase 3): aleatorios, Word2Vec propio y spaCy.
Reutilizan en memoria el modelo Word2Vec y el modelo de spaCy que ya se han cargado en el E2."""
import random
import numpy as np
import torch
import torch.nn as nn
from config import SEED, EMB_ALEATORIO

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


def construir_embedding(tipo, itos, congelar, modelo_w2v, nlp):
    """Devuelve (nn.Embedding, cobertura). cobertura = % de palabras del vocabulario con vector previo."""
    if tipo == "aleatorio":
        return nn.Embedding(len(itos), EMB_ALEATORIO, padding_idx=PAD), None
    M, cobertura = matriz_w2v(itos, modelo_w2v) if tipo == "w2v" else matriz_spacy(itos, nlp)
    return nn.Embedding.from_pretrained(torch.tensor(M), freeze=congelar, padding_idx=PAD), cobertura
