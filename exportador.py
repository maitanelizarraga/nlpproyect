import json
import pandas as pd
import numpy as np
from config import CARPETA_SALIDA, CARPETA_E3

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


def cargar_corpus_exportado():
    """Lee corpus_procesado.csv (lo que guarda exportar_corpus) y recupera las columnas anidadas.
    Sirve para ejecutar solo el E3 sin repetir el procesamiento del E2."""
    df = pd.read_csv(CARPETA_SALIDA / "corpus_procesado.csv")
    for col in COLUMNAS_ANIDADAS:
        df[col] = df[col].apply(json.loads)
    return df


def _tabla_markdown(df, columnas):
    columnas = [c for c in columnas if c in df.columns]

    def formato(valor, col):
        if isinstance(valor, str):
            return valor
        if pd.isna(valor):
            return "-"
        if col.startswith("bleu"):
            return f"{valor:.2f}"
        return f"{valor:.3f}" if isinstance(valor, float) else str(valor)

    filas = [[formato(v, c) for v, c in zip(fila, columnas)] for fila in df[columnas].values]
    return "\n".join(["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)] +
                     ["| " + " | ".join(f) + " |" for f in filas])


def exportar_tabla(df, nombre, columnas, sufijo=""):
    """Guarda la tabla comparativa del E3 en CSV y en Markdown (y la imprime)."""
    CARPETA_E3.mkdir(parents=True, exist_ok=True)
    df.to_csv(CARPETA_E3 / f"{nombre}{sufijo}.csv", index=False)
    md = _tabla_markdown(df, columnas)
    (CARPETA_E3 / f"{nombre}{sufijo}.md").write_text(md, encoding="utf-8")
    print(f"\n### {nombre}\n{md}")
