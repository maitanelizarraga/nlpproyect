from config import CARPETA_SALIDA
from data_loader import cargar_dataset, split_dataset
from nlp_processor import cargar_spacy, procesar_textos
from vectorizacion_clasica import crear_bag_of_words, crear_tfidf
from modelo_word2vec import (
    entrenar_word2vec,
    analizar_palabras_desconocidas,
    analizar_oov_preentrenado,
    vectorizar_documentos_w2v,
)
from modelo_roberta import generar_embeddings_contextuales
from exportador import exportar_corpus


def main():
    # 1. Preparar carpeta
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    # 2. Cargar y dividir
    df = cargar_dataset()
    train_df, val_df, test_df = split_dataset(df)

    # 3. Procesamiento lingüístico (spaCy)
    nlp = cargar_spacy()
    df_procesado = procesar_textos(df, nlp)

    # Marcamos la partición de cada documento (queda guardada en CSV/JSONL)
    df_procesado["split"] = "train"
    df_procesado.loc[df_procesado["document_id"].isin(val_df["document_id"]), "split"] = "validation"
    df_procesado.loc[df_procesado["document_id"].isin(test_df["document_id"]), "split"] = "test"

    train_p = df_procesado[df_procesado["split"] == "train"].copy()
    val_p = df_procesado[df_procesado["split"] == "validation"].copy()
    test_p = df_procesado[df_procesado["split"] == "test"].copy()

    # Textos originales (question) para vectorizar
    textos_train = train_p["question"].astype(str).tolist()
    textos_val = val_p["question"].astype(str).tolist()
    textos_test = test_p["question"].astype(str).tolist()

    # 4. Vectorización clásica (ajustada solo con train)
    crear_bag_of_words(textos_train, textos_test, textos_val)
    crear_tfidf(textos_train, textos_test, textos_val)

    # 5. Embeddings estáticos (Word2Vec propio, entrenado solo con train)
    modelo_w2v = entrenar_word2vec(train_p)
    analizar_oov_preentrenado(df_procesado, nlp)            # referencia: vectores preentrenados
    analizar_palabras_desconocidas(df_procesado, modelo_w2v)  # OOV propio: global y por partición
    vectorizar_documentos_w2v(df_procesado, modelo_w2v)     # un vector (media) por documento

    # 6. Embeddings contextuales (RoBERTa)
    generar_embeddings_contextuales(df_procesado)

    # 7. Guardar todo
    exportar_corpus(df_procesado)

    print("\n¡Pipeline finalizado con éxito!")


if __name__ == "__main__":
    main()
