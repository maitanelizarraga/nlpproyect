from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import SEED_SPLIT

def cargar_dataset():
    print("\n CARGAMOS EL DATASET")

    dataset = load_dataset("somosnlp/lenguaje-claro-dataset")
    if "train" not in dataset:
        raise ValueError("El dataset descargado no contiene la partición 'train'.")
    df = dataset["train"].to_pandas()

    columnas_necesarias = {"question", "answer"}
    faltan = columnas_necesarias - set(df.columns)
    if faltan:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(faltan)}. "
            f"Columnas encontradas: {df.columns.tolist()}"
        )

    print("\n--- Análisis inicial ---")
    print("Dimensiones iniciales:", df.shape)
    print("Columnas:", df.columns.tolist())
    print("\nValores nulos por columna:\n", df.isnull().sum())
    print("\nFilas duplicadas:", int(df.duplicated().sum()))

    n_inicial = len(df)
    # Eliminar filas que no tienen texto original o simplificado.
    df = df.dropna(subset=["question", "answer"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer"] != "")].copy()

    # Se eliminan duplicados exactos de los pares de texto para no repetir ejemplos.
    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
    df.insert(0, "document_id", np.arange(len(df), dtype=int))

    df["longitud_original"] = df["question"].str.split().str.len()
    df["longitud_simplificada"] = df["answer"].str.split().str.len()

    print("\nDimensiones tras limpieza:", df.shape)
    print("Registros eliminados en la limpieza:", n_inicial - len(df))
    print("Longitud media del texto original:",
          round(float(df["longitud_original"].mean()), 2))
    print("Longitud media del texto simplificado:",
          round(float(df["longitud_simplificada"].mean()), 2))

    return df

def split_dataset(df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=SEED_SPLIT):
    print("\n DIVISIÓN DE DATASET:")
    # Primero separamos train del resto (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=random_state,
        shuffle=True
    )

    # Ahora dividimos el resto en val y test
    # Proporción relativa de val dentro de temp_df
    val_ratio = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        random_state=random_state,
        shuffle=True
    )
    print("Longitud de train:", len(train_df))
    print("Longitud de val:", len(val_df))
    print("Longitud de test:", len(test_df))

    return train_df, val_df, test_df