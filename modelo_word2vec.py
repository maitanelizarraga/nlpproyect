import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from config import SEED, CARPETA_SALIDA


def entrenar_word2vec(df_train):
    print("\n--- Entrenando Word2Vec ---")
    # Extraer solo la lista de tokens (textos originales de la partición de entrenamiento)
    corpus_tokens = [registro["tokens"] for registro in df_train["procesado_original"]]

    modelo = Word2Vec(
        sentences=corpus_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=20,
        seed=SEED,
    )

    modelo.save(str(CARPETA_SALIDA / "word2vec_juridico.model"))
    print("Vocabulario aprendido Word2Vec:", len(modelo.wv))
    return modelo


def _contar_oov(df, esta_en_vocabulario):
    """Cuenta tokens totales y OOV de un DataFrame ya procesado.
    Devuelve (total, oov, Counter de palabras OOV)."""
    from collections import Counter
    total, oov, palabras = 0, 0, Counter()
    for registro in df["procesado_original"]:
        for palabra in registro["tokens"]:
            total += 1
            if not esta_en_vocabulario(palabra):
                oov += 1
                palabras[palabra] += 1
    return total, oov, palabras


def analizar_palabras_desconocidas(df, modelo_w2v):
    """OOV del Word2Vec propio, global y por partición (requiere columna 'split')."""
    print("\n--- Analizando palabras desconocidas (OOV) con Word2Vec propio ---")
    wv = modelo_w2v.wv
    en_vocab = lambda p: p in wv or p.lower() in wv

    filas, resultados = [], {}
    grupos = [("global", df)] + [(s, df[df["split"] == s]) for s in ["train", "validation", "test"]]
    for nombre, parte in grupos:
        total, oov, palabras = _contar_oov(parte, en_vocab)
        pct = 100 * oov / total if total else 0.0
        resultados[nombre] = (total, oov, pct)
        filas.append({"particion": nombre, "tokens": total, "oov": oov, "pct_oov": round(pct, 2)})
        print(f"{nombre:>10}: {total} tokens, {oov} OOV ({pct:.2f}%)")
        if nombre == "global":
            palabras_global = palabras

    pd.DataFrame(filas).to_csv(CARPETA_SALIDA / "oov_word2vec_propio.csv", index=False)
    pd.DataFrame(palabras_global.most_common(200), columns=["token", "apariciones"]) \
        .to_csv(CARPETA_SALIDA / "oov_top200_word2vec_propio.csv", index=False)
    return resultados


def analizar_oov_preentrenado(df, nlp):
    """OOV con los vectores estáticos preentrenados que incluye es_core_news_md.
    Sirve como referencia para decidir si compensa entrenar un Word2Vec propio."""
    print("\n--- Analizando OOV con vectores preentrenados de spaCy ---")
    en_vocab = lambda p: nlp.vocab[p].has_vector or nlp.vocab[p.lower()].has_vector
    total, oov, _ = _contar_oov(df, en_vocab)
    pct = 100 * oov / total if total else 0.0
    print(f"spaCy {nlp.meta['name']}: {total} tokens, {oov} OOV ({pct:.2f}%)")
    return total, oov, pct


def vectorizar_documentos_w2v(df, modelo_w2v):
    """Un vector por documento: media de los vectores de sus tokens conocidos.
    Documentos sin ningún token conocido -> vector de ceros."""
    print("\n--- Vectores de documento con Word2Vec (media) ---")
    wv = modelo_w2v.wv
    dim = wv.vector_size
    matriz = np.zeros((len(df), dim), dtype=np.float32)
    for i, registro in enumerate(df["procesado_original"]):
        vecs = [wv[p] if p in wv else wv[p.lower()]
                for p in registro["tokens"] if p in wv or p.lower() in wv]
        if vecs:
            matriz[i] = np.mean(vecs, axis=0)
    np.savez_compressed(
        CARPETA_SALIDA / "embeddings_word2vec_originales.npz",
        document_id=df["document_id"].to_numpy(),
        embeddings=matriz,
    )
    print("Guardado. Dimensiones:", matriz.shape)
    return matriz
