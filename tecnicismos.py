"""FASE 1 (E1): detección de palabras complejas / tecnicismos.

Etiquetado débil: un token del texto original (solo tokens alfabéticos) es 'complejo' (1) si su
lema NO aparece en el texto simplificado correspondiente, y 'conservado' (0) si aparece.
Es decir, se aprende qué palabras suele eliminar o reescribir el simplificador. Es una
aproximación (una palabra puede desaparecer porque se omite la frase y no porque sea difícil);
por eso se genera una muestra para validarla a mano (analisis.muestra_validacion_etiquetas).

Clasificador por token con representación dispersa: cada token se describe con una "bolsa" de
rasgos vectorizada con BoW o TF-IDF, más rasgos no textuales. Se comparan dos conjuntos de rasgos:
  - base:       palabra, lema, prefijo, sufijos, vecinos + 4 numéricos (longitud, posición, mayúsculas)
  - extendido:  base + categoría gramatical (propia y de los vecinos), tipo de entidad, stopword
                y frecuencia del lema en train (rasgos de otra naturaleza que el texto)
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from analisis import analisis_fase1, muestra_validacion_etiquetas
from config import (CARPETA_E3, VENTANAS, GRID_C, GRID_CLASS_WEIGHT, GRID_RASGOS, MAX_CHUNK, SEED)
from metricas import metricas_clasificacion, ic_bootstrap_f1

UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
        "PUNCT", "SCONJ", "SYM", "VERB", "X"]
POS_ID = {p: i + 1 for i, p in enumerate(UPOS)}     # 0 = relleno
N_POS = len(UPOS) + 2                               # relleno + categorias + desconocida
POS_CONTENIDO = {"NOUN", "VERB", "ADJ", "ADV"}      # las que tiene sentido proponer para el glosario


def id_pos(p):
    return POS_ID.get(p, len(UPOS) + 1)


# ------------------------------------------------------------------ rasgos de spaCy
def asegurar_rasgos(df, nlp):
    """Garantiza que procesado_original tiene pos/stop/ent. Si el E2 se guardó sin ellos, se
    recalculan con spaCy sobre el mismo texto (se comprueba que los tokens coinciden)."""
    if all("pos" in r for r in df["procesado_original"]):
        return df
    print("procesado_original sin POS/stop/ent: recalculando con spaCy (solo la primera vez)...")
    df = df.copy()
    docs = nlp.pipe(df["question"].astype(str).tolist(), batch_size=32)
    nuevos = []
    for r, doc in zip(df["procesado_original"], docs):
        toks = [t for t in doc if not t.is_space]
        if [t.text for t in toks] != r["tokens"]:
            raise ValueError("Los tokens recalculados no coinciden con los guardados en el E2: "
                             "vuelve a ejecutar el E2 (python main.py --solo-e2).")
        r2 = dict(r)
        r2["pos"] = [t.pos_ for t in toks]
        r2["stop"] = [bool(t.is_stop) for t in toks]
        r2["ent"] = [t.ent_type_ or "O" for t in toks]
        nuevos.append(r2)
    df["procesado_original"] = nuevos
    return df


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
    """Fragmentos de <= maximo tokens: dict(words, lemmas, pos, y). Se usan en los modelos deep."""
    trozos = []
    for o, y in zip(df["procesado_original"], df["y_tokens"]):
        pos = o.get("pos") or ["X"] * len(o["tokens"])
        for i in range(0, len(o["tokens"]), maximo):
            trozos.append({"words": o["tokens"][i:i + maximo], "lemmas": o["lemas"][i:i + maximo],
                           "pos": pos[i:i + maximo], "y": y[i:i + maximo]})
    return trozos


def info_chunks(chunks):
    """(info, y) aplanados en el orden de los fragmentos: info = [(palabra, lema, pos)]."""
    info, ys = [], []
    for c in chunks:
        for w, l, p, y in zip(c["words"], c["lemmas"], c["pos"], c["y"]):
            info.append((w, l.lower(), p))
            ys.append(y)
    return info, np.asarray(ys)


def metricas_tokens(y_real, y_pred):
    """Métricas sobre los tokens evaluables (y != -100)."""
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    m = y_real != -100
    return metricas_clasificacion(y_real[m], y_pred[m])


def frecuencias_lemas(df):
    """Frecuencia de cada lema (minúsculas) en los textos ORIGINALES. Se calcula solo con train."""
    c = Counter()
    for o in df["procesado_original"]:
        c.update(l.lower() for l in o["lemas"])
    return c


# ------------------------------------------------------------------ rasgos del shallow
def _rasgos_token(toks, lems, i, ventana, o=None, extendido=False):
    t = toks[i].lower()
    f = [f"T_{t}", f"L_{lems[i].lower()}", f"S3_{t[-3:]}", f"S4_{t[-4:]}", f"P3_{t[:3]}"]
    for k in range(1, ventana + 1):
        f.append(f"P{k}_{toks[i - k].lower() if i - k >= 0 else '<s>'}")
        f.append(f"N{k}_{toks[i + k].lower() if i + k < len(toks) else '</s>'}")
    if extendido:
        pos = o["pos"]
        f.append(f"POS_{pos[i]}")
        for k in range(1, ventana + 1):
            f.append(f"PPOS{k}_{pos[i - k] if i - k >= 0 else '<s>'}")
            f.append(f"NPOS{k}_{pos[i + k] if i + k < len(toks) else '</s>'}")
        if o["ent"][i] != "O":
            f.append(f"ENT_{o['ent'][i]}")
        if o["stop"][i]:
            f.append("STOP")
    return " ".join(f)


def construir_tokens(df, ventana, extendido=False, freq=None):
    """Una fila por token alfabético. Devuelve dict con: textos (rasgos), num (rasgos numéricos),
    y (etiqueta), lemas e info [(palabra, lema, pos)] para el análisis de errores."""
    textos, num, y, lemas, info = [], [], [], [], []
    max_log = np.log1p(max(freq.values())) if (extendido and freq) else 1.0
    for o, ys in zip(df["procesado_original"], df["y_tokens"]):
        toks, lems, n = o["tokens"], o["lemas"], len(o["tokens"])
        pos = o.get("pos")
        for i, (t, yi) in enumerate(zip(toks, ys)):
            if yi == -100:
                continue
            textos.append(_rasgos_token(toks, lems, i, ventana, o, extendido))
            fila = [min(len(t), 20) / 20, i / max(n - 1, 1), float(t.istitle()), float(t.isupper())]
            if extendido:
                fila += [np.log1p(freq.get(lems[i].lower(), 0)) / max_log, min(len(lems[i]), 20) / 20,
                         float(o["stop"][i]), float(o["ent"][i] != "O")]
            num.append(fila)
            y.append(yi)
            lemas.append(lems[i].lower())
            info.append((t, lems[i].lower(), pos[i] if pos else "X"))
    return {"textos": textos, "num": csr_matrix(np.asarray(num, dtype=float)), "y": np.asarray(y),
            "lemas": lemas, "info": info}


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
    for l, yi in zip(train_tok["lemas"], train_tok["y"]):
        acum[l].append(yi)
    mapa = {l: int(np.mean(v) >= 0.5) for l, v in acum.items()}
    pred = lambda d: np.array([mapa.get(l, 0) for l in d["lemas"]])
    return pred(val_tok), pred(test_tok)


def _matriz(vec, d, ajustar=False):
    X = vec.fit_transform(d["textos"]) if ajustar else vec.transform(d["textos"])
    return hstack([X, d["num"]]).tocsr()


def _fila(nombre, familia, representacion, hip, va_y, pred_va, te_y, pred_te):
    return {"modelo": nombre, "familia": familia, "representacion": representacion, "hiperparametros": hip,
            **{f"{k}_val": x for k, x in metricas_clasificacion(va_y, pred_va).items()},
            **metricas_clasificacion(te_y, pred_te),
            "f1_ic95_s0": ic_bootstrap_f1(te_y, pred_te), "n_semillas": 1}


def ejecutar_shallow_tecnicismos(train, val, test, extra=None):
    """Shallow con rejilla sobre validación. extra: DataFrame externo ya etiquetado (preentrenamiento
    en el sentido shallow: se reajusta el mejor modelo con train + datos externos)."""
    print("\n=== FASE 1 · SHALLOW LEARNING (tecnicismos, representación dispersa) ===")
    freq = frecuencias_lemas(train)
    datos = {}
    for v in VENTANAS:
        for r in GRID_RASGOS:
            datos[(v, r)] = [construir_tokens(d, v, r == "extendido", freq) for d in (train, val, test)]
    tr0, va0, te0 = datos[(VENTANAS[0], GRID_RASGOS[0])]
    print(f"Tokens evaluables: train={len(tr0['y'])} val={len(va0['y'])} test={len(te0['y'])} "
          f"| % complejos train={100 * tr0['y'].mean():.1f}")

    resultados, grid, grupos = [], [], []
    pv, pt = baseline_lema(tr0, va0, te0)
    resultados.append({"modelo": "Baseline: etiqueta mayoritaria por lema", "familia": "baseline",
                       "representacion": "-", **{f"{k}_val": v for k, v in metricas_clasificacion(va0["y"], pv).items()},
                       **metricas_clasificacion(te0["y"], pt), "f1_ic95_s0": ic_bootstrap_f1(te0["y"], pt)})
    resultados.append({"modelo": "Baseline: todo complejo", "familia": "baseline", "representacion": "-",
                       **{f"{k}_val": v for k, v in metricas_clasificacion(va0["y"], np.ones_like(va0["y"])).items()},
                       **metricas_clasificacion(te0["y"], np.ones_like(te0["y"])),
                       "f1_ic95_s0": ic_bootstrap_f1(te0["y"], np.ones_like(te0["y"]))})

    for nombre, cfg in CONFIGURACIONES.items():
        for r in GRID_RASGOS:
            print(f"\n--- {nombre} · rasgos {r}: búsqueda de hiperparámetros sobre VALIDACIÓN ---")
            mejor = None
            for v in VENTANAS:
                tr, va, _ = datos[(v, r)]
                vec = cfg["vec"]()
                Xtr, Xva = _matriz(vec, tr, True), _matriz(vec, va)
                for C in GRID_C:
                    for cw in GRID_CLASS_WEIGHT:
                        clf = cfg["clf"](C, cw).fit(Xtr, tr["y"])
                        f1 = metricas_clasificacion(va["y"], clf.predict(Xva))["f1"]
                        grid.append({"modelo": nombre, "rasgos": r, "ventana": v, "C": C,
                                     "class_weight": str(cw), "f1_val": round(f1, 4)})
                        if mejor is None or f1 > mejor[0]:
                            mejor = (f1, v, C, cw, vec, clf)
            f1v, v, C, cw, vec, clf = mejor
            print(f"Mejor: ventana={v}, C={C}, class_weight={cw} (F1 val={f1v:.4f})")
            _, va, te = datos[(v, r)]
            pred_va, pred_te = clf.predict(_matriz(vec, va)), clf.predict(_matriz(vec, te))
            hip = f"ventana={v}, C={C}, class_weight={cw}"
            resultados.append(_fila(f"{nombre} · rasgos {r}", "shallow", cfg["representacion"], hip,
                                    va["y"], pred_va, te["y"], pred_te))
            grupos.append({"nombre": f"{nombre} · rasgos {r}", "f1v": f1v, "te": te, "pred_te": pred_te})

            if extra is not None:
                ex = construir_tokens(extra, v, r == "extendido", freq)
                tr = datos[(v, r)][0]
                vec2 = cfg["vec"]()
                X2 = hstack([vec2.fit_transform(tr["textos"] + ex["textos"]),
                             csr_matrix(np.vstack([tr["num"].toarray(), ex["num"].toarray()]))]).tocsr()
                clf2 = cfg["clf"](C, cw).fit(X2, np.concatenate([tr["y"], ex["y"]]))
                p_va, p_te = clf2.predict(_matriz(vec2, va)), clf2.predict(_matriz(vec2, te))
                resultados.append(_fila(f"{nombre} · rasgos {r} + datos externos", "shallow",
                                        cfg["representacion"], f"{hip}; +{len(ex['y'])} tokens externos",
                                        va["y"], p_va, te["y"], p_te))

    mejor_g = max(grupos, key=lambda g: g["f1v"])
    _guardar_candidatos_glosario(mejor_g["te"], mejor_g["pred_te"])
    CARPETA_E3.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid).to_csv(CARPETA_E3 / "grid_shallow_tecnicismos.csv", index=False)
    payload = {"freq": freq, "mejor": {"nombre": mejor_g["nombre"], "info": mejor_g["te"]["info"],
                                       "y": mejor_g["te"]["y"], "pred": mejor_g["pred_te"]}}
    return resultados, payload


def _guardar_candidatos_glosario(te, pred):
    """Puente con la fase 2 (glosario DPEJ): lemas de categoría léxica (nombre, verbo, adjetivo,
    adverbio) que el mejor modelo marca como complejos en test. Se filtran las palabras gramaticales."""
    marcados, totales, pos_por_lema = Counter(), Counter(), defaultdict(Counter)
    for (palabra, lema, pos), p in zip(te["info"], pred):
        totales[lema] += 1
        pos_por_lema[lema][pos] += 1
        if p == 1:
            marcados[lema] += 1
    filas = []
    for lema, n in marcados.most_common():
        pos = pos_por_lema[lema].most_common(1)[0][0]
        if n >= 2 and pos in POS_CONTENIDO and len(lema) >= 4:
            filas.append({"lema": lema, "pos_mas_frecuente": pos, "apariciones_marcado_complejo": n,
                          "apariciones_totales": totales[lema], "proporcion_marcado": round(n / totales[lema], 2)})
    CARPETA_E3.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas[:200]).to_csv(CARPETA_E3 / "candidatos_glosario.csv", index=False)


def _df_analisis(info, y, pred):
    m = np.asarray(y) != -100
    df = pd.DataFrame(info, columns=["palabra", "lema", "pos"])[m]
    df["y"], df["pred"] = np.asarray(y)[m], np.asarray(pred)[m]
    return df.reset_index(drop=True)


def ejecutar_fase1(train, val, test, modelo_w2v, nlp, con_deep=True, con_bert=True, epocas=None,
                   variantes=None, opc=None):
    """FASE 1 completa: etiquetas + shallow learning + redes (GRU bidireccional con/sin atención y
    RoBERTa) + análisis de errores. opc: dict con 'semillas' y 'externo' (ver preentreno.py)."""
    opc = opc or {}
    externo = opc.get("externo")
    train, val, test = (anadir_etiquetas(asegurar_rasgos(d, nlp)) for d in (train, val, test))
    ext_df = anadir_etiquetas(asegurar_rasgos(externo["df"], nlp)) if externo else None
    muestra_validacion_etiquetas(test)

    resultados, payload = ejecutar_shallow_tecnicismos(train, val, test, ext_df)
    a_analizar = {payload["mejor"]["nombre"]: _df_analisis(payload["mejor"]["info"], payload["mejor"]["y"],
                                                          payload["mejor"]["pred"])}
    if con_deep:
        from deep_tecnicismos import ejecutar_deep_tecnicismos   # import tardio: torch solo si hace falta
        info_te, y_te = info_chunks(trocear(test))
        res_d, preds = ejecutar_deep_tecnicismos(
            trocear(train), trocear(val), trocear(test), modelo_w2v, nlp, variantes, con_bert, epocas,
            semillas=opc.get("semillas"), ext_chunks=trocear(ext_df) if ext_df is not None else None,
            w2v_ext=externo["w2v"] if externo else None)
        resultados += res_d
        mejores = sorted([r for r in res_d], key=lambda r: -r["f1_val"])[:2]
        for r in mejores:
            p = preds[r["modelo"]]
            a_analizar[r["modelo"]] = _df_analisis(info_te, y_te, (p["probs"] >= p["umbral"]).astype(int))
    analisis_fase1(a_analizar, payload["freq"])
    return resultados
