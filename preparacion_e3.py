"""Cada 'question' del corpus empieza con la instruccion 'Escribe el siguiente texto en lenguaje claro:'.
Esas palabras estan en TODOS los originales y en NINGUN simplificado, asi que la fase 1 las etiquetaria
siempre como complejas (y los modelos las acertarian sin esfuerzo). Se quitan y se vuelve a procesar con spaCy."""
from config import PREFIJO_INSTRUCCION
from nlp_processor import procesar_textos


def quitar_instruccion(df, nlp):
    df = df.copy()
    df["question"] = df["question"].str.replace(PREFIJO_INSTRUCCION, "", regex=True)
    quedan = int(df["question"].str.contains("lenguaje claro", case=False).sum())
    print(f"Instruccion inicial eliminada. Textos que aun contienen 'lenguaje claro': {quedan} de {len(df)}")
    return procesar_textos(df, nlp)
