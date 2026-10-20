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