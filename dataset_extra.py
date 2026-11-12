# pip install datasets pandas
from datasets import load_dataset
from itertools import islice
import pandas as pd

# 1. Cargar el dataset en modo streaming (no descarga todo el corpus)
dataset = load_dataset(
    "SINAI/ALIA-es-legal-administrative",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

# 2. Quedarnos solo con las primeras 1500 filas y con la columna "text"
N = 1500
rows = [{"text": ejemplo["text"]} for ejemplo in islice(dataset, N)]

# 3. Convertir a DataFrame y exportar a CSV
df = pd.DataFrame(rows)

# Verificación rápida antes de exportar
print(f"Filas obtenidas: {len(df)}")
print(f"Columnas: {list(df.columns)}")
print(df.head(3))

# 4. Guardar como CSV (UTF-8, sin índice)
df.to_csv("alia_legal_1500.csv", index=False, encoding="utf-8")

print("✅ CSV generado: alia_legal_1500.csv")