"""FASE 3 (simplificacion): puntos de comparacion y orquestacion. El modelo sequence to sequence
esta en deep_seq2seq.py (se importa solo si se piden las redes)."""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from config import CARPETA_E3
from metricas import bleu


def _copia(train, df):
    return df["question"].tolist()


def _vecino(train, df):
    """Devuelve la simplificación del texto de train más parecido (TF-IDF, coseno)."""
    tf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = tf.fit_transform(train["question"])
    Xq = tf.transform(df["question"])
    idx = np.concatenate([linear_kernel(Xq[i:i + 256], Xtr).argmax(axis=1)
                          for i in range(0, Xq.shape[0], 256)])
    return train["answer"].iloc[idx].tolist()


def ejecutar_baselines(train, val, test):
    resultados, muestras = [], {}
    for nombre, fn in [("Baseline: copiar el original", _copia),
                       ("Baseline: vecino más cercano (TF-IDF)", _vecino)]:
        resultados.append({"modelo": nombre, "familia": "baseline",
                           "representacion": "-" if "copiar" in nombre else "TF-IDF (dispersa)",
                           "bleu_val": bleu(fn(train, val), val["answer"]),
                           "bleu_test": bleu(fn(train, test), test["answer"])})
        muestras[nombre] = fn(train, test)[:5]
    return resultados, muestras


def ejecutar_fase3(train, val, test, modelo_w2v, nlp, con_deep=True, epocas=None, variantes=None):
    """FASE 3 completa: puntos de comparacion (copiar, texto mas parecido) + seq2seq con atencion (BLEU)."""
    resultados, muestras = ejecutar_baselines(train, val, test)
    if con_deep:
        from deep_seq2seq import ejecutar_deep   # import tardio: torch solo si hace falta
        r, m = ejecutar_deep(train, val, test, modelo_w2v, nlp, variantes, epocas)
        resultados += r
        muestras.update(m)
    CARPETA_E3.mkdir(parents=True, exist_ok=True)
    with open(CARPETA_E3 / "muestras_test.json", "w", encoding="utf-8") as fh:
        json.dump({"originales": test["question"].head(5).tolist(),
                   "referencias": test["answer"].head(5).tolist(), "salidas": muestras},
                  fh, ensure_ascii=False, indent=1)
    return resultados
