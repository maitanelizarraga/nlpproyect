import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import SEED_SPLIT, PREFIJO_REGEX

_PREFIJO = re.compile(PREFIJO_REGEX, re.IGNORECASE)


def quitar_prefijo(texto):
    """Elimina la instruccion del prompt original ("Escribe el siguiente texto en lenguaje claro:")
    que aparece al principio de TODOS los textos 'question' del corpus. No es parte del texto
    administrativo: si se deja, contamina las etiquetas, el vocabulario y el glosario."""
    return _PREFIJO.sub("", str(texto), count=1).strip()


def reparar_prefijo_procesado(df):
    """Para corpus YA procesados con spaCy (E2 antiguo): quita el prefijo de 'question' y de las
    listas de tokens/lemas/POS... de 'procesado_original', sin volver a ejecutar spaCy.
    Devuelve (df, n_reparados). Los textos sin prefijo no se tocan."""
    df = df.copy()
    prefijo = ["escribe", "el", "siguiente", "texto", "en", "lenguaje", "claro"]
    nuevos, n_rep = [], 0
    for o in df["procesado_original"]:
        toks = [t.lower() for t in o["tokens"][:8]]
        o2 = dict(o)
        if toks[:7] == prefijo:                                   # "claro" y ":" por separado
            n, resto = (8 if len(toks) > 7 and toks[7] == ":" else 7), None
        elif toks[:6] == prefijo[:6] and len(toks) > 6 and toks[6].startswith("claro:"):
            n, resto = 7, o["tokens"][6][6:]                      # spaCy pego "claro:" al texto siguiente
        else:
            nuevos.append(o)
            continue
        for clave in ("tokens", "lemas", "pos", "stop", "ent"):
            if clave in o2 and len(o2[clave]) == len(o["tokens"]):
                o2[clave] = o2[clave][n:]
        if resto:                                                 # queda un resto pegado: se repone como token
            o2["tokens"] = [resto] + o2["tokens"]
            if "lemas" in o2:
                o2["lemas"] = [resto.lower()] + o2["lemas"]
            for clave, valor in (("pos", "X"), ("stop", False), ("ent", "O")):
                if clave in o2:
                    o2[clave] = [valor] + o2[clave]
        if o2.get("oraciones"):
            o2["oraciones"] = [quitar_prefijo(o2["oraciones"][0])] + o2["oraciones"][1:]
        nuevos.append(o2)
        n_rep += 1
    df["procesado_original"] = nuevos
    df["question"] = df["question"].map(quitar_prefijo)
    if "tokens_original" in df:
        df["tokens_original"] = [len(o["tokens"]) for o in nuevos]
    if "longitud_original" in df:
        df["longitud_original"] = df["question"].str.split().str.len()
    return df, n_rep


def tiene_prefijo(df):
    """Numero de textos 'question' que todavia empiezan con la instruccion del prompt."""
    return int(df["question"].astype(str).map(lambda t: bool(_PREFIJO.match(t))).sum())


def cargar_dataset():
    print("\n CARGAMOS EL DATASET")

    from datasets import load_dataset   # import tardio: solo hace falta al descargar el corpus
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
    # CORRECCION: se quita la instruccion del prompt que precede a todos los textos originales.
    n_prefijo = tiene_prefijo(df)
    df["question"] = df["question"].map(quitar_prefijo)
    print(f"Prefijo de instruccion eliminado en {n_prefijo} de {len(df)} textos originales")
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