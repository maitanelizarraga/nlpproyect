# pip install datasets pandas
from datasets import load_dataset
from itertools import islice
import pandas as pd
import csv

# 1. Cargar el dataset en modo streaming (no descarga todo)
dataset = load_dataset(
    "Ramitha/spanish-legal-data",
    split="train",
    streaming=True,
)

# 2. Quedarnos solo con las primeras N filas y la columna "Data"
N = 1500
rows = [{"text": ejemplo["Data"]} for ejemplo in islice(dataset, N)]

# 3. Convertir a DataFrame
df = pd.DataFrame(rows)

# 4. Normalizar el texto: colapsar saltos de línea y espacios múltiples
#    Así cada fila queda en una sola línea y se ve bien en Excel/CSV
df["text"] = (
    df["text"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 5. Verificación rápida
print(f"Filas obtenidas: {len(df)}")
print(f"Columnas: {list(df.columns)}")
print(df.head(3))

# 6. Guardar CSV (UTF-8 con BOM para Excel, comillas en todos los campos)
df.to_csv(
    "alia_legal_1500.csv",
    index=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_ALL,
)

print("✅ CSV generado: alia_legal_1500.csv")