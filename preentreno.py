"""Preentrenamiento con un dataset externo.

TODO (vosotras): conectar el dataset en `cargar_dataset_externo()` (es lo unico que hay que tocar).

El dataset debe ser de PARES (texto complejo, texto simplificado) en español, porque se usa igual
que el corpus propio:
  - Fase 1 (tecnicismos): se etiquetan sus tokens con la misma regla debil y se usa para
      * preentrenar la BiGRU (y despues ajustarla con train),
      * reajustar los modelos shallow con train + externo.
  - Fase 3 (simplificacion): se usa para preentrenar el seq2seq (y despues ajustarlo con train).
  - Ambas: se entrena un Word2Vec con train + externo (variante "Word2Vec externo").

Garantias para no inflar los resultados:
  - se eliminan del externo los textos cuyo original coincida con algun texto de val o test,
  - el Word2Vec solo ve train + externo (nunca val ni test),
  - cada modelo con preentreno tiene un control con el mismo vocabulario pero sin preentreno.
"""
import joblib
import pandas as pd

from config import CARPETA_E3, EXTERNO_MAX, SEED
from data_loader import quitar_prefijo


# --------------------------------------------------------------------------- CONECTAR AQUI
def cargar_dataset_externo():
    df = pd.read_csv("alia_legal_1500_lenguaje_claro.csv")   # si da error, probad sep=";" o encoding="utf-8-sig"
    print("Columnas del dataset externo:", list(df.columns))
    return df.rename(columns={"NOMBRE_COL_ORIGINAL": "question",
                              "NOMBRE_COL_SIMPLIFICADO": "answer"})[["question", "answer"]]
# -----------------------------------------------------------------------------------------


def _limpiar(df, excluir):
    faltan = {"question", "answer"} - set(df.columns)
    if faltan:
        raise ValueError(f"El dataset externo debe tener las columnas 'question' y 'answer' (faltan {sorted(faltan)}).")
    n0 = len(df)
    df = df.dropna(subset=["question", "answer"]).copy()
    df["question"] = df["question"].astype(str).map(quitar_prefijo)
    df["answer"] = df["answer"].astype(str).str.strip()
    palabras = df["question"].str.split().str.len()
    df = df[(palabras >= 5) & (palabras <= 400) & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["question", "answer"])
    n_fuga = int(df["question"].isin(excluir).sum())
    df = df[~df["question"].isin(excluir)]
    if len(df) > EXTERNO_MAX:
        df = df.sample(EXTERNO_MAX, random_state=SEED)
    df = df.reset_index(drop=True)
    df.insert(0, "document_id", range(-len(df), 0))     # ids negativos: no chocan con los del corpus
    print(f"Dataset externo: {n0} filas -> {len(df)} tras limpiar "
          f"(eliminadas {n_fuga} que coinciden con val/test; maximo {EXTERNO_MAX})")
    return df


def entrenar_word2vec_externo(df_train, df_ext):
    """Word2Vec con train (original + simplificado) + externo. Mismos parametros que el del E2."""
    from gensim.models import Word2Vec
    from modelo_word2vec import normalizar_tokens
    corpus = []
    for d in (df_train, df_ext):
        for col in ("procesado_original", "procesado_simplificado"):
            corpus += [normalizar_tokens(r["tokens"]) for r in d[col]]
    modelo = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=2, workers=4, sg=1,
                      epochs=20, seed=SEED)
    CARPETA_E3.mkdir(parents=True, exist_ok=True)
    modelo.save(str(CARPETA_E3 / "word2vec_externo.model"))
    print("Word2Vec externo: vocabulario", len(modelo.wv), "| frases", len(corpus))
    return modelo


def preparar_externo(nlp, df_corpus, df_train):
    """Carga, limpia y procesa con spaCy el dataset externo (con cache) y entrena el Word2Vec externo.
    df_corpus: corpus completo (se usa para excluir lo que este en val/test). Devuelve
    {'df': externo procesado, 'w2v': Word2Vec} o None si no hay dataset conectado."""
    df = cargar_dataset_externo()
    if df is None:
        print("\n[preentreno] No hay dataset externo conectado (preentreno.cargar_dataset_externo devuelve None): "
              "se omiten las variantes externas.")
        return None
    cache = CARPETA_E3 / "externo_procesado.joblib"
    if cache.exists():
        print(f"[preentreno] usando cache {cache} (borrala si cambias el dataset externo)")
        procesado = joblib.load(cache)
    else:
        from nlp_processor import procesar_textos
        evaluacion = set(df_corpus[df_corpus["split"] != "train"]["question"])
        procesado = procesar_textos(_limpiar(df, evaluacion), nlp)
        CARPETA_E3.mkdir(parents=True, exist_ok=True)
        joblib.dump(procesado, cache)
    return {"df": procesado, "w2v": entrenar_word2vec_externo(df_train, procesado)}
