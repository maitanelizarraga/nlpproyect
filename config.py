from pathlib import Path

# Configuraciones básicas
SEED = 123
SEED_SPLIT = 42  # semilla de train_test_split (se mantiene 42 para reproducir los resultados ya obtenidos)
CARPETA_SALIDA = Path("resultados_e2")
ARCHIVO_LOCAL = Path("lenguaje_claro_dataset.csv")

# Configuraciones de modelos
MODELO_SPACY = "es_core_news_md"
MODELO_CONTEXTUAL = "bertin-project/bertin-roberta-base-spanish"
MAX_TOKENS_ROBERTA = 512
BATCH_SIZE_ROBERTA = 8
# ======================= E3: PRIMERA ITERACION =======================
CARPETA_E3 = Path("resultados_e3")

# --- Fase 1 (palabras complejas) con shallow learning: rejilla de hiper-parametros ---
VENTANAS = [0, 2]                  # palabras vecinas que se usan como contexto (0 = ninguna)
GRID_C = [0.1, 1.0, 10.0]
GRID_CLASS_WEIGHT = [None, "balanced"]

# --- Ajustes comunes de las redes neuronales (fase 1 y fase 3) ---
VOCAB_MAX = 8000                   # palabras del vocabulario (solo train)
VOCAB_MIN_FREQ = 2
EMB_ALEATORIO = 100                # dimension de los embeddings aleatorios (= Word2Vec propio)
DROPOUT = 0.3
BATCH = 32
LR = 1e-3
PACIENCIA = 3                      # early stopping
CLIP = 1.0

# --- Fase 1 con redes: GRU bidireccional y RoBERTa ---
MAX_CHUNK = 100                    # los textos se trocean en fragmentos de 100 palabras
TAG_HIDDEN = 128
TAG_EPOCAS = 15
TAG_UMBRALES = [round(0.1 + 0.05 * i, 2) for i in range(17)]   # 0.10 ... 0.90 (se elige en validacion)
BERT_LR = 3e-5
BERT_EPOCAS = 3
BERT_BATCH = 16
BERT_MAX_LEN = 320

# --- Fase 3 (simplificacion): sequence to sequence con atencion ---
S2S_MAX_SRC = 200
S2S_MAX_TGT = 200
S2S_HIDDEN = 256
S2S_EPOCAS = 20
