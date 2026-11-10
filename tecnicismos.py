"""FASE 1 (E1): detección de palabras complejas / tecnicismos con shallow learning.

Etiquetado débil: un token del texto original (solo tokens alfabéticos) es 'complejo' (1) si su
lema NO aparece en el texto simplificado correspondiente, y 'conservado' (0) si aparece.
Es decir, se aprende qué palabras suele eliminar o reescribir el simplificador. Es una
aproximación (una palabra puede desaparecer porque se omite la frase y no porque sea difícil).

Clasificador por token con representación dispersa: cada token se describe con una "bolsa" de
rasgos (palabra, lema, prefijo, sufijos, vecinos) vectorizada con BoW o TF-IDF, más rasgos
no textuales (longitud, posición relativa, mayúsculas).
"""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import CARPETA_E3, VENTANAS, GRID_C, GRID_CLASS_WEIGHT, MAX_CHUNK, SEED
from metricas import metricas_clasificacion


# ------------------------------------------------------------------ etiquetas
def etiquetas_documento(orig, simp):
    """Lista con 1/0 por token del original y -100 para los tokens no evaluables (no alfabéticos)."""
    lemas_simp = {l.lower() for l in simp["lemas"]}
    y = []
    for tok, lem in zip(orig["tokens"], orig["lemas"]):
        y.append(-100 if not tok.isalpha() else int(lem.lower() not in lemas_simp))
    return y


def anadir_etiquetas(df):
    df = df.copy()
    df["y_tokens"] = [etiquetas_documento(o, s) for o, s in
                      zip(df["procesado_original"], df["procesado_simplificado"])]
    return df


def trocear(df, maximo=MAX_CHUNK):
    """Fragmentos de <= maximo tokens: dict(words, lemmas, y). Se usan en los modelos deep."""
    trozos = []
    for o, y in zip(df["procesado_original"], df["y_tokens"]):
        for i in range(0, len(o["tokens"]), maximo):
            trozos.append({"words": o["tokens"][i:i + maximo], "lemmas": o["lemas"][i:i + maximo],
                           "y": y[i:i + maximo]})
    return trozos


def metricas_tokens(y_real, y_pred):
    """Métricas sobre los tokens evaluables (y != -100)."""
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    m = y_real != -100
    return metricas_clasificacion(y_real[m], y_pred[m])


# ------------------------------------------------------------------ rasgos
def _rasgos_token(toks, lems, i, ventana):
    t = toks[i].lower()
    f = [f"T_{t}", f"L_{lems[i].lower()}", f"S3_{t[-3:]}", f"S4_{t[-4:]}", f"P3_{t[:3]}"]
    for k in range(1, ventana + 1):
        f.append(f"P{k}_{toks[i - k].lower() if i - k >= 0 else '<s>'}")
        f.append(f"N{k}_{toks[i + k].lower() if i + k < len(toks) else '</s>'}")
    return " ".join(f)


def construir_tokens(df, ventana):
    """Una fila por token alfabético: texto de rasgos, rasgos numéricos, etiqueta y lema."""
    textos, num, y, lemas = [], [], [], []
    for o, ys in zip(df["procesado_original"], df["y_tokens"]):
        toks, lems, n = o["tokens"], o["lemas"], len(o["tokens"])
        for i, (t, yi) in enumerate(zip(toks, ys)):
            if yi == -100:
                continue
            textos.append(_rasgos_token(toks, lems, i, ventana))
            num.append([min(len(t), 20) / 20, i / max(n - 1, 1), float(t.istitle()), float(t.isupper())])
            y.append(yi)
            lemas.append(lems[i].lower())
    return textos, csr_matrix(np.asarray(num, dtype=float)), np.asarray(y), lemas


CONFIGURACIONES = {
    "BoW + Regresión logística": {
        "vec": lambda: CountVectorizer(lowercase=False, token_pattern=r"\S+", min_df=2),
        "clf": lambda C, cw: LogisticRegression(C=C, class_weight=cw, max_iter=2000, random_state=SEED),
        "representacion": "Bag of Words (dispersa)"},
    "TF-IDF + SVM lineal": {
        "vec": lambda: TfidfVectorizer(lowercase=False, token_pattern=r"\S+", min_df=2, sublinear_tf=True),
        "clf": lambda C, cw: LinearSVC(C=C, class_weight=cw, random_state=SEED),
        "representacion": "TF-IDF (dispersa)"},
}


def baseline_lema(train_tok, val_tok, test_tok):
    """Etiqueta mayoritaria por lema (lo visto en train). Lema no visto -> 0."""
    acum = defaultdict(list)
    for l, yi in zip(train_tok[3], train_tok[2]):
        acum[l].append(yi)
    mapa = {l: int(np.mean(v) >= 0.5) for l, v in acum.items()}
    pred = lambda d: np.array([mapa.get(l, 0) for l in d[3]])
    return pred(val_tok), pred(test_tok)


def ejecutar_shallow_tecnicismos(train, val, test):
    print("\n=== FASE 1 · SHALLOW LEARNING (tecnicismos, representación dispersa) ===")
    datos = {v: [construir_tokens(d, v) for d in (train, val, test)] for v in VENTANAS}
    ytr = datos[VENTANAS[0]][0][2]
    print(f"Tokens evaluables: train={len(ytr)} val={len(datos[VENTANAS[0]][1][2])} "
          f"test={len(datos[VENTANAS[0]][2][2])} | % complejos train={100 * ytr.mean():.1f}")

    resultados, grid, mejor_global = [], [], None
    tr0, va0, te0 = datos[VENTANAS[0]]
    pv, pt = baseline_lema(tr0, va0, te0)
    resultados.append({"modelo": "Baseline: etiqueta mayoritaria por lema", "familia": "baseline",
                       "representacion": "-", **{f"{k}_val": v for k, v in metricas_clasificacion(va0[2], pv).items()},
                       **metricas_clasificacion(te0[2], pt)})
    # Referencia trivial: todo "complejo"
    resultados.append({"modelo": "Baseline: todo complejo", "familia": "baseline", "representacion": "-",
                       **{f"{k}_val": v for k, v in metricas_clasificacion(va0[2], np.ones_like(va0[2])).items()},
                       **metricas_clasificacion(te0[2], np.ones_like(te0[2]))})

    for nombre, cfg in CONFIGURACIONES.items():
        print(f"\n--- {nombre}: búsqueda de hiperparámetros sobre VALIDACIÓN ---")
        mejor = None
        for v in VENTANAS:
            (tr_t, tr_n, tr_y, _), (va_t, va_n, va_y, _), _ = datos[v]
            vec = cfg["vec"]()
            Xtr = hstack([vec.fit_transform(tr_t), tr_n]).tocsr()
            Xva = hstack([vec.transform(va_t), va_n]).tocsr()
            for C in GRID_C:
                for cw in GRID_CLASS_WEIGHT:
                    clf = cfg["clf"](C, cw).fit(Xtr, tr_y)
                    f1 = metricas_clasificacion(va_y, clf.predict(Xva))["f1"]
                    grid.append({"modelo": nombre, "ventana": v, "C": C, "class_weight": str(cw),
                                 "f1_val": round(f1, 4)})
                    if mejor is None or f1 > mejor[0]:
                        mejor = (f1, v, C, cw, vec, clf)
        f1v, v, C, cw, vec, clf = mejor
        print(f"Mejor: ventana={v}, C={C}, class_weight={cw} (F1 val={f1v:.4f})")
        (_, _, _, _), (va_t, va_n, va_y, _), (te_t, te_n, te_y, te_l) = datos[v]
        pred_va = clf.predict(hstack([vec.transform(va_t), va_n]).tocsr())
        pred_te = clf.predict(hstack([vec.transform(te_t), te_n]).tocsr())
        resultados.append({"modelo": nombre, "familia": "shallow", "representacion": cfg["representacion"],
                           "hiperparametros": f"ventana={v}, C={C}, class_weight={cw}",
                           **{f"{k}_val": x for k, x in metricas_clasificacion(va_y, pred_va).items()},
                           **metricas_clasificacion(te_y, pred_te)})
        if mejor_global is None or f1v > mejor_global[0]:
            mejor_global = (f1v, nombre, te_l, pred_te)

    # Puente con la fase 2 (glosario): lemas que el mejor modelo marca como complejos en test
    _, nombre, lemas_te, pred_te = mejor_global
    cnt = Counter(l for l, p in zip(lemas_te, pred_te) if p == 1)
    CARPETA_E3.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cnt.most_common(200), columns=["lema", "apariciones_marcado_complejo"]) \
        .to_csv(CARPETA_E3 / "candidatos_glosario.csv", index=False)
    pd.DataFrame(grid).to_csv(CARPETA_E3 / "grid_shallow_tecnicismos.csv", index=False)
    return resultados


def ejecutar_fase1(train, val, test, modelo_w2v, nlp, con_deep=True, con_bert=True, epocas=None, variantes=None):
    """FASE 1 completa: etiquetas + shallow learning + redes (GRU bidireccional y RoBERTa).
    train/val/test son los DataFrames del E2 (con procesado_original y procesado_simplificado)."""
    train, val, test = (anadir_etiquetas(d) for d in (train, val, test))
    resultados = ejecutar_shallow_tecnicismos(train, val, test)
    if con_deep:
        from deep_tecnicismos import ejecutar_deep_tecnicismos   # import tardio: torch solo si hace falta
        resultados += ejecutar_deep_tecnicismos(trocear(train), trocear(val), trocear(test),
                                                modelo_w2v, nlp, variantes, con_bert, epocas)
    return resultados
