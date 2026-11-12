"""Analisis de errores y validacion de las etiquetas debiles (rigor del E3).

- analisis_fase1: donde falla cada clasificador (por categoria gramatical, frecuencia del lema en
  train y longitud de la palabra) y que lemas son los errores mas repetidos.
- analisis_fase3: propiedades de las salidas del simplificador (longitud, repeticiones, si conserva
  cifras y entidades) frente a la referencia.
- muestra_validacion_etiquetas: CSV para etiquetar a mano una muestra y medir cuanto se parece la
  etiqueta automatica ("el lema desaparece en la simplificacion") a la nocion humana de "palabra
  compleja". Se rellena a mano y se evalua con validar_etiquetas.py.
"""
import random
import re

import numpy as np
import pandas as pd

from config import CARPETA_E3


def _prf(y, p):
    y, p = np.asarray(y).astype(bool), np.asarray(p).astype(bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return round(prec, 3), round(rec, 3), round(f1, 3)


# ------------------------------------------------------------------ fase 1
def analisis_fase1(modelos, freq_train):
    """modelos: {nombre: DataFrame con columnas palabra, lema, pos, y, pred (solo tokens evaluables)}.
    freq_train: Counter de lemas (en minusculas) de los textos originales de TRAIN."""
    filas, errores = [], []
    for nombre, df in modelos.items():
        df = df.copy()
        df["freq"] = df["lema"].map(lambda l: freq_train.get(l, 0))
        df["frecuencia_lema_train"] = pd.cut(df["freq"], [-1, 0, 2, 9, 49, 10 ** 9],
                                             labels=["no vista", "1-2", "3-9", "10-49", "50+"])
        df["longitud_palabra"] = pd.cut(df["palabra"].str.len(), [0, 4, 7, 10, 1000],
                                        labels=["1-4", "5-7", "8-10", "11+"])
        for dim, col in [("categoria_gramatical", "pos"), ("frecuencia_lema_train", "frecuencia_lema_train"),
                         ("longitud_palabra", "longitud_palabra")]:
            for valor, g in df.groupby(col, observed=True):
                p, r, f1 = _prf(g["y"], g["pred"])
                filas.append({"modelo": nombre, "dimension": dim, "valor": str(valor), "n": len(g),
                              "pct_complejas": round(100 * g["y"].mean(), 1),
                              "precision": p, "recall": r, "f1": f1})
        totales = df["lema"].value_counts()
        for tipo, sub in [("falso_negativo", df[(df["y"] == 1) & (df["pred"] == 0)]),
                          ("falso_positivo", df[(df["y"] == 0) & (df["pred"] == 1)])]:
            for lema, n in sub["lema"].value_counts().head(20).items():
                errores.append({"modelo": nombre, "tipo": tipo, "lema": lema, "n_errores": int(n),
                                "n_apariciones": int(totales[lema])})
    CARPETA_E3.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(CARPETA_E3 / "analisis_fase1_dimensiones.csv", index=False)
    pd.DataFrame(errores).to_csv(CARPETA_E3 / "analisis_fase1_top_errores.csv", index=False)
    print(f"Analisis de errores de la fase 1 guardado ({len(modelos)} modelos).")


# ------------------------------------------------------------------ fase 3
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _repeticion_trigramas(texto):
    t = texto.lower().split()
    tri = list(zip(t, t[1:], t[2:]))
    return 100 * (1 - len(set(tri)) / len(tri)) if tri else 0.0


def analisis_fase3(test, salidas):
    """salidas: {sistema: lista de textos generados para 'test'}. Se anade la referencia como techo."""
    fuentes = test["question"].tolist()
    refs = test["answer"].tolist()
    entidades = [[e["texto"].lower() for e in o["entidades"]] for o in test["procesado_original"]]
    sistemas = dict(salidas)
    sistemas["Referencia (GPT-3.5 del corpus)"] = refs
    filas = []
    for nombre, hips in sistemas.items():
        long_h, ratio, rep, nums, ents = [], [], [], [], []
        for fuente, ref, hip, es in zip(fuentes, refs, hips, entidades):
            h = str(hip).lower()
            long_h.append(len(h.split()))
            ratio.append(len(h.split()) / max(len(ref.split()), 1))
            rep.append(_repeticion_trigramas(h))
            cifras = set(_NUM.findall(fuente))
            if cifras:
                nums.append(np.mean([c in h for c in cifras]))
            if es:
                ents.append(np.mean([e in h for e in set(es)]))
        filas.append({"sistema": nombre, "palabras_media": round(float(np.mean(long_h)), 1),
                      "ratio_longitud_vs_ref": round(float(np.mean(ratio)), 2),
                      "trigramas_repetidos_pct": round(float(np.mean(rep)), 2),
                      "cifras_conservadas_pct": round(100 * float(np.mean(nums)), 1) if nums else None,
                      "entidades_conservadas_pct": round(100 * float(np.mean(ents)), 1) if ents else None})
    CARPETA_E3.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(CARPETA_E3 / "analisis_fase3_salidas.csv", index=False)

    rng = random.Random(0)
    idx = rng.sample(range(len(test)), min(6, len(test)))
    md = ["# Ejemplos de salidas (test)\n"]
    for i in idx:
        md.append(f"## Texto {i}\n\n**Original:** {fuentes[i]}\n\n**Referencia:** {refs[i]}\n")
        for nombre, hips in salidas.items():
            md.append(f"- **{nombre}:** {hips[i]}")
        md.append("")
    (CARPETA_E3 / "ejemplos_fase3.md").write_text("\n".join(md), encoding="utf-8")
    print("Analisis de salidas de la fase 3 guardado (analisis_fase3_salidas.csv, ejemplos_fase3.md).")


# ------------------------------------------------------------------ validacion de etiquetas
def muestra_validacion_etiquetas(df, n=200, seed=0):
    """CSV con n tokens al azar de 'df' (necesita y_tokens) para etiquetarlos a mano.
    Columna 'etiqueta_manual': 1 = palabra que un ciudadano sin formacion juridica tendria
    dificultad para entender o que conviene reescribir; 0 = palabra sencilla."""
    cand = [(d, i) for d, ys in enumerate(df["y_tokens"]) for i, y in enumerate(ys) if y != -100]
    muestra = random.Random(seed).sample(cand, min(n, len(cand)))
    ids = df["document_id"].tolist()
    filas = []
    for d, i in muestra:
        o = df["procesado_original"].iloc[d]
        toks = o["tokens"]
        filas.append({"document_id": ids[d], "palabra": toks[i], "lema": o["lemas"][i],
                      "contexto": " ".join(toks[max(0, i - 6):i]) + " [[" + toks[i] + "]] "
                                  + " ".join(toks[i + 1:i + 7]),
                      "etiqueta_automatica": df["y_tokens"].iloc[d][i], "etiqueta_manual": ""})
    CARPETA_E3.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_E3 / "muestra_validacion_etiquetas.csv"
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"Muestra para validar etiquetas a mano: {ruta} ({len(filas)} tokens)")
