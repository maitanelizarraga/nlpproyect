import json
from gensim.models import Word2Vec
from config import SEED, CARPETA_SALIDA

def entrenar_word2vec(df_train):
    print("\n--- Entrenando Word2Vec ---")
    # Extraer solo la lista de tokens
    corpus_tokens = [registro["tokens"] for registro in df_train["procesado_original"]]

    # Configurar y entrenar el modelo
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
    
    # Guardar
    modelo.save(str(CARPETA_SALIDA / "word2vec_juridico.model"))
    print("Vocabulario aprendido Word2Vec:", len(modelo.wv))
    
    return modelo

def analizar_palabras_desconocidas(df, modelo_w2v):
    print("\n--- Analizando palabras desconocidas (OOV) ---")
    total_tokens = 0
    tokens_desconocidos = 0
    palabras_desconocidas = set()

    for registro in df["procesado_original"]:
        for palabra in registro["tokens"]:
            total_tokens += 1
            
            # Buscar la palabra tal cual o en minúscula
            if palabra not in modelo_w2v.wv and palabra.lower() not in modelo_w2v.wv:
                tokens_desconocidos += 1
                palabras_desconocidas.add(palabra)

    porcentaje = 100 * tokens_desconocidos / total_tokens if total_tokens else 0.0
    
    print("Total de tokens:", total_tokens)
    print("Tokens no representados:", tokens_desconocidos)
    print(f"Porcentaje OOV: {porcentaje:.2f}%")