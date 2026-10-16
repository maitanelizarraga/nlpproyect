import json
import pandas as pd
import numpy as np
from config import CARPETA_SALIDA

def limpiar_dato(valor):
    """Asegura que el dato sea compatible con el formato JSON."""
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return float(valor)
    if pd.isna(valor) and not isinstance(valor, (dict, list, tuple)):
        return None
    return valor

def exportar_corpus(df):
    print("\n--- Exportando resultados ---")
    
    # 1. Guardar en formato tabla (CSV)
    ruta_csv = CARPETA_SALIDA / "corpus_procesado.csv"
    df.to_csv(ruta_csv, index=False)

    # 2. Guardar en formato lista de diccionarios (JSONL)
    ruta_jsonl = CARPETA_SALIDA / "corpus_procesado.jsonl"
    with open(ruta_jsonl, "w", encoding="utf-8") as archivo:
        for registro in df.to_dict(orient="records"):
            registro_limpio = {clave: limpiar_dato(valor) for clave, valor in registro.items()}
            archivo.write(json.dumps(registro_limpio, ensure_ascii=False) + "\n")

    print("Archivos exportados a:", CARPETA_SALIDA.resolve())