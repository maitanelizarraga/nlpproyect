"""Ejecuta TODO el proyecto con un solo comando:  python ejecutar_todo.py

Pasos (automaticos):
  1. E2: si no hay E2 guardado, o es el antiguo (con el prefijo del prompt), se ejecuta/regenera entero.
     Si ya esta bien, se reutiliza.
  2. Prueba de humo del E3 (200 textos, 2 epocas): si algo falla, falla en minutos y no tras horas.
  3. E3 completo: fase 1 + fase 3, 3 semillas, beam search, analisis de errores, y variantes con el
     dataset externo si lo habeis conectado en preentreno.py (si no, se omiten solas).

Se ajusta cambiando las constantes de aqui abajo (no hace falta ningun argumento).
"""
from types import SimpleNamespace

import pandas as pd

from config import CARPETA_SALIDA, SEMILLAS, BEAM
from data_loader import tiene_prefijo
from main import pipeline_e2, cargar_e2_guardado, pipeline_e3

# ----------------------------------------------------------------- ajustes
PRUEBA_DE_HUMO = True        # ejecutar antes una version rapida del E3 para detectar errores pronto
REGENERAR_E2_SI_ANTIGUO = True   # si el E2 guardado tiene el prefijo del prompt, rehacerlo entero
BUSCAR_HIPERPARAMETROS = False   # busqueda de hiperparametros del seq2seq (anade horas)
USAR_DATASET_EXTERNO = True      # usa preentreno.py si hay dataset conectado; si no, se omite solo
USAR_BERT = True                 # False si no hay GPU
# -------------------------------------------------------------------------


def _e2_guardado_valido():
    ruta = CARPETA_SALIDA / "corpus_procesado.csv"
    if not ruta.exists():
        return False
    if REGENERAR_E2_SI_ANTIGUO and tiene_prefijo(pd.read_csv(ruta, usecols=["question"])):
        print("[todo] El E2 guardado es anterior a la correccion de datos: se regenera entero.")
        return False
    return True


def _args(rapido):
    return SimpleNamespace(fase="todo", sin_deep=False, sin_bert=not USAR_BERT, rapido=rapido, variantes=None,
                           semillas=len(SEMILLAS), beam=BEAM, buscar_hp=BUSCAR_HIPERPARAMETROS and not rapido,
                           externo=USAR_DATASET_EXTERNO)


def main():
    if _e2_guardado_valido():
        print("[todo] Reutilizando el E2 guardado.")
        df, nlp, w2v = cargar_e2_guardado()
    else:
        df, nlp, w2v = pipeline_e2()
        print("\n[todo] E2 terminado.")
    if PRUEBA_DE_HUMO:
        print("\n[todo] Prueba de humo del E3 (rapida)...")
        pipeline_e3(df, nlp, w2v, _args(rapido=True))
        print("\n[todo] Prueba de humo OK. Lanzando el E3 completo...")
    pipeline_e3(df, nlp, w2v, _args(rapido=False))
    print("\n[todo] ¡Todo terminado! Resultados en resultados_e2/ y resultados_e3/")


if __name__ == "__main__":
    main()
