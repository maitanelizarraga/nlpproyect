import argparse
import pandas as pd

from config import CARPETA_SALIDA, SEMILLAS, BEAM
from data_loader import cargar_dataset, split_dataset, tiene_prefijo, reparar_prefijo_procesado
from nlp_processor import cargar_spacy, procesar_textos
from vectorizacion_clasica import crear_bag_of_words, crear_tfidf
from modelo_word2vec import (
    entrenar_word2vec,
    analizar_palabras_desconocidas,
    analizar_oov_preentrenado,
    vectorizar_documentos_w2v,
)
from modelo_roberta import generar_embeddings_contextuales
from exportador import exportar_corpus, cargar_corpus_exportado, exportar_tabla

COLUMNAS_FASE1 = ["modelo", "familia", "representacion", "accuracy", "precision", "recall", "f1", "f1_std",
                  "f1_ic95_s0", "f1_val", "n_semillas", "hiperparametros"]
COLUMNAS_FASE3 = ["modelo", "familia", "representacion", "bleu_val", "bleu_test", "bleu_test_std",
                  "bleu_test_beam", "bleu_test_ic", "cobertura_emb", "epocas", "n_semillas"]


def pipeline_e2():
    """E2: procesamiento y representacion de datos (igual que antes)."""
    # 1. Preparar carpeta
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    # 2. Cargar y dividir
    df = cargar_dataset()
    train_df, val_df, test_df = split_dataset(df)

    # 3. Procesamiento linguistico (spaCy)
    nlp = cargar_spacy()
    df_procesado = procesar_textos(df, nlp)

    # Marcamos la particion de cada documento (queda guardada en CSV/JSONL)
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

    # 4. Vectorizacion clasica (ajustada solo con train)
    crear_bag_of_words(textos_train, textos_test, textos_val)
    crear_tfidf(textos_train, textos_test, textos_val)

    # 5. Embeddings estaticos (Word2Vec propio, entrenado solo con train: question + answer)
    modelo_w2v = entrenar_word2vec(train_p)
    analizar_oov_preentrenado(df_procesado, nlp)              # referencia: vectores preentrenados
    analizar_palabras_desconocidas(df_procesado, modelo_w2v)  # OOV propio: global y por particion
    vectorizar_documentos_w2v(df_procesado, modelo_w2v)       # un vector (media) por documento

    # 6. Embeddings contextuales (RoBERTa)
    generar_embeddings_contextuales(df_procesado)

    # 7. Guardar todo
    exportar_corpus(df_procesado)
    return df_procesado, nlp, modelo_w2v


def cargar_e2_guardado():
    """Para ejecutar solo el E3: recupera lo que el E2 ya guardo (corpus, Word2Vec) sin recalcularlo."""
    from gensim.models import Word2Vec
    print("\n--- Cargando resultados del E2 ya guardados ---")
    df_procesado = cargar_corpus_exportado()
    if tiene_prefijo(df_procesado):
        # El E2 guardado es anterior a la correccion de datos: se quita el prefijo del prompt sin repetir
        # spaCy y se reentrena el Word2Vec (rapido) porque su vocabulario incluia ese prefijo.
        df_procesado, n = reparar_prefijo_procesado(df_procesado)
        print(f"\n[correccion de datos] prefijo del prompt eliminado en {n} textos del E2 guardado.")
        print("Se reentrena el Word2Vec con los textos corregidos. Para regenerar TODOS los ficheros del E2 "
              "(BoW, TF-IDF, RoBERTa, OOV...) ejecuta: python main.py --solo-e2")
        modelo_w2v = entrenar_word2vec(df_procesado[df_procesado["split"] == "train"])
    else:
        modelo_w2v = Word2Vec.load(str(CARPETA_SALIDA / "word2vec_juridico.model"))
    return df_procesado, cargar_spacy(), modelo_w2v


def pipeline_e3(df_procesado, nlp, modelo_w2v, args):
    """E3: iteracion corregida. Fase 1 (palabras complejas, clasificacion) y fase 3 (simplificacion, BLEU),
    con varias semillas, analisis de errores y, opcionalmente, preentrenamiento con un dataset externo."""
    if tiene_prefijo(df_procesado):
        raise SystemExit("El corpus aun contiene el prefijo 'Escribe el siguiente texto en lenguaje claro:'. "
                         "Ejecuta primero el E2 (python main.py --solo-e2) o usa --solo-e3 para repararlo.")
    train = df_procesado[df_procesado["split"] == "train"].reset_index(drop=True)
    val = df_procesado[df_procesado["split"] == "validation"].reset_index(drop=True)
    test = df_procesado[df_procesado["split"] == "test"].reset_index(drop=True)
    epocas, sufijo = None, ""
    semillas = SEMILLAS[:max(1, min(args.semillas, len(SEMILLAS)))]
    if args.rapido:                      # prueba de humo: subconjunto pequeno, 2 epocas y 1 semilla
        train, val, test, epocas, sufijo = train.head(200), val.head(60), test.head(60), 2, "_rapido"
        semillas = semillas[:1]

    externo = None
    if args.externo:
        from preentreno import preparar_externo
        externo = preparar_externo(nlp, df_procesado, train)
    opc = {"semillas": semillas, "beam": args.beam, "buscar_hp": args.buscar_hp, "externo": externo}
    print(f"\n[E3] semillas={semillas} | beam={args.beam or 'no'} | busqueda HP={args.buscar_hp} | "
          f"dataset externo={'si' if externo else 'no'}")

    if args.fase in ("1", "todo"):
        from tecnicismos import ejecutar_fase1
        res = ejecutar_fase1(train, val, test, modelo_w2v, nlp, con_deep=not args.sin_deep,
                             con_bert=not args.sin_bert, epocas=epocas, variantes=args.variantes, opc=opc)
        exportar_tabla(pd.DataFrame(res), "tabla_fase1_tecnicismos", COLUMNAS_FASE1, sufijo)

    if args.fase in ("3", "todo"):
        from simplificacion import ejecutar_fase3
        res = ejecutar_fase3(train, val, test, modelo_w2v, nlp, con_deep=not args.sin_deep,
                             epocas=epocas, variantes=args.variantes, opc=opc)
        exportar_tabla(pd.DataFrame(res), "tabla_fase3_simplificacion", COLUMNAS_FASE3, sufijo)


def main():
    ap = argparse.ArgumentParser(description="Pipeline NLP: E2 (datos y representaciones) + E3 (modelos)")
    ap.add_argument("--solo-e2", action="store_true", help="ejecutar solo el E2")
    ap.add_argument("--solo-e3", action="store_true", help="ejecutar solo el E3, usando lo que el E2 ya guardo")
    ap.add_argument("--fase", choices=["1", "3", "todo"], default="todo", help="fase del E3 a ejecutar")
    ap.add_argument("--sin-deep", action="store_true", help="E3 sin redes neuronales (solo shallow y puntos de comparacion)")
    ap.add_argument("--sin-bert", action="store_true", help="E3 sin RoBERTa (util si no hay GPU)")
    ap.add_argument("--rapido", action="store_true", help="E3: prueba de humo con pocos textos y 2 epocas")
    ap.add_argument("--variantes", nargs="*", help="E3: nombres exactos de las variantes de redes a ejecutar")
    ap.add_argument("--semillas", type=int, default=len(SEMILLAS),
                    help=f"E3: numero de semillas por red (1-{len(SEMILLAS)}); se informa media y desviacion")
    ap.add_argument("--beam", type=int, default=BEAM, help="E3 fase 3: tamano del haz (0 = solo decodificacion voraz)")
    ap.add_argument("--buscar-hp", action="store_true",
                    help="E3 fase 3: busqueda de hiperparametros del seq2seq sobre validacion (lenta)")
    ap.add_argument("--externo", action="store_true",
                    help="E3: usar el dataset externo conectado en preentreno.py (preentreno y Word2Vec externo)")
    args = ap.parse_args()

    if args.solo_e3:
        df_procesado, nlp, modelo_w2v = cargar_e2_guardado()
    else:
        df_procesado, nlp, modelo_w2v = pipeline_e2()
        print("\n¡Pipeline E2 finalizado con éxito!")
    if not args.solo_e2:
        pipeline_e3(df_procesado, nlp, modelo_w2v, args)
        print("\n¡Pipeline E3 finalizado con éxito!")


if __name__ == "__main__":
    main()
