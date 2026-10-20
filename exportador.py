import json
import pandas as pd
import numpy as np
from config import CARPETA_SALIDA

COLUMNAS_ANIDADAS = ["procesado_original", "procesado_simplificado"]


def limpiar_dato(valor):
    """Asegura que el dato sea compatible con el formato JSON."""
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return float(valor)
    if isinstance(valor, (dict, list, tuple)):
        return valor
    if pd.isna(valor):
        return None
    return valor


def exportar_corpus(df):
    print("\n--- Exportando resultados ---")

    # 1. CSV: las columnas anidadas (dict) se guardan como JSON válido (comillas dobles),
    #    de forma que puedan recuperarse con json.loads al leer el fichero.
    df_csv = df.copy()
    for col in COLUMNAS_ANIDADAS:
        df_csv[col] = df_csv[col].apply(lambda v: json.dumps(v, ensure_ascii=False))
    ruta_csv = CARPETA_SALIDA / "corpus_procesado.csv"
    df_csv.to_csv(ruta_csv, index=False, encoding="utf-8")

    # 2. JSONL: un registro por línea, con las estructuras anidadas intactas
    ruta_jsonl = CARPETA_SALIDA / "corpus_procesado.jsonl"
    with open(ruta_jsonl, "w", encoding="utf-8") as archivo:
        for registro in df.to_dict(orient="records"):
            registro_limpio = {clave: limpiar_dato(valor) for clave, valor in registro.items()}
            archivo.write(json.dumps(registro_limpio, ensure_ascii=False) + "\n")

    print("Archivos exportados a:", CARPETA_SALIDA.resolve())
