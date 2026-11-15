# Simplificación de textos jurídico-administrativos con PLN (E2 + E3)

Proyecto de *Fundamentos del Procesamiento de Lenguaje Natural* (Ziortza Astigarraga y Maitane Lizarraga).
Corpus: [`somosnlp/lenguaje-claro-dataset`](https://huggingface.co/datasets/somosnlp/lenguaje-claro-dataset)
(pares *texto administrativo → lenguaje claro*, español).

- **E2**: procesamiento (spaCy) y representaciones (BoW, TF-IDF, Word2Vec propio, RoBERTa).
- **E3**: primera iteración de resolución, con dos tareas:
  - **Fase 1 – palabras complejas** (clasificación por token: ¿esta palabra desaparece en la versión clara?). Métricas: accuracy, precision, recall, F1.
  - **Fase 3 – simplificación** (secuencia a secuencia). Métrica: BLEU.

La fase 2 del E1 (glosario DPEJ) solo está iniciada: `candidatos_glosario.csv` lista los lemas candidatos.

## Ejecución rápida

```bash
uv sync                       # o: pip install -e .   (Python >= 3.12)
python ejecutar_todo.py       # TODO: E2 si hace falta + prueba de humo + E3 completo + verificación + tablas
```

Opciones útiles de `ejecutar_todo.py`:

| Opción | Efecto |
|---|---|
| `--sin-bert` | no entrena BERTIN/RoBERTa (si no hay GPU) |
| `--sin-humo` | salta la prueba de humo |
| `--sin-hp` | no hace la búsqueda de hiperparámetros del seq2seq |
| `--sin-externo` | no ejecuta las variantes con dataset externo |
| `--semillas N` / `--beam N` | semillas por red (1–3) / tamaño del haz (0 = voraz) |
| `--fase 1\|3\|todo` | solo una fase del E3 |
| `--forzar-e2` | regenera el E2 aunque ya sea válido |
| `--solo-verificar` | no entrena: comprueba `resultados_e3/` y regenera las tablas |

**Qué hace `ejecutar_todo.py`** (cada paso en un proceso aparte con `main.py`):

1. Comprueba librerías, modelo de spaCy y GPU.
2. Reutiliza el E2 guardado o lo regenera si es antiguo (con el prefijo del prompt) o le faltan campos.
3. Archiva `resultados_e3/` anterior en `resultados_e3_antiguo_<fecha>/` para no mezclar ejecuciones.
4. **Prueba de humo** (200 textos, 2 épocas) escribiendo en `resultados_e3_rapido/`, que se borra al terminar.
5. **E3 completo** en `resultados_e3/`.
6. **Verifica** los resultados (columnas esperadas, 615 textos de test, variantes presentes, sin restos de la prueba de humo).
7. Genera `resultados_e3/tablas_para_informe.md` y `resultados_e3/ejecucion.json`.

Si algo falla se aborta con el motivo; si la verificación falla, el programa termina con código 1.

> Tiempos: el E3 completo entrena 10 variantes seq2seq (fase 3) y 12 variantes de redes (fase 1), cada una con 3 semillas, más beam search.
> Con GPU son horas; con CPU, no es viable con BERTIN. Cada paso se puede ejecutar por separado con `main.py`.

## Uso manual (`main.py`)

```bash
python main.py --solo-e2                          # E2 completo (descarga el corpus de Hugging Face)
python main.py --solo-e3                          # E3 con el E2 guardado
python main.py --solo-e3 --fase 1 --sin-deep      # fase 1 solo con shallow (minutos, sin GPU)
python main.py --solo-e3 --externo --buscar-hp    # con dataset externo y búsqueda de hiperparámetros
python main.py --solo-e3 --rapido                 # prueba de humo (200 textos, 2 épocas, 1 semilla)
```

Otras opciones: `--sin-bert`, `--semillas N`, `--beam N`, `--variantes "nombre exacto" ...`.
Para que la prueba de humo no pise los resultados reales, ejecútala con `E3_SALIDA=resultados_e3_rapido`
(lo hace `ejecutar_todo.py` automáticamente).

## Qué requisito de E3 cubre cada parte

| Requisito de E3 | Dónde |
|---|---|
| 2 técnicas shallow con representación dispersa | `tecnicismos.py`: BoW + regresión logística, TF-IDF + SVM lineal (rejilla sobre validación) |
| 2 técnicas deep comparando embeddings y fine-tuning | `deep_tecnicismos.py` (BiGRU, BERTIN) y `deep_seq2seq.py` (GRU+atención): aleatorio, Word2Vec propio, spaCy, BERTIN; congelado vs fine-tuneado |
| Métricas: accuracy, recall, precisión, F1 / BLEU | `metricas.py`; tablas `tabla_fase1_tecnicismos.*` y `tabla_fase3_simplificacion.*` |
| Justificar hiperparámetros | `grid_shallow_tecnicismos.csv`, `grid_seq2seq.csv`, umbrales y épocas elegidos en validación (ver documento E3) |
| Atención | autoatención en el etiquetador (fase 1); atención de Bahdanau y mecanismo de copia en el seq2seq (fase 3); ablación sin atención |
| Features no solo texto | rasgos `extendido` (POS, entidad, stopword, frecuencia del lema) y BiGRU+POS |
| Preentrenar con datasets parecidos | `preentreno.py` + `alia_legal_1500_lenguaje_claro.csv`: Word2Vec externo, BiGRU/seq2seq preentrenados y shallow reajustado, cada uno con su control sin preentreno |

## Estructura del código

| Archivo | Contenido |
|---|---|
| `config.py` | todos los ajustes (semillas, rejillas, tamaños, épocas, dataset externo, carpeta de salida) |
| `data_loader.py` | carga, limpieza (**quita el prefijo del prompt**) y división 70/15/15 |
| `nlp_processor.py` | spaCy: tokens, oraciones, lemas, POS, stopword, entidades |
| `vectorizacion_clasica.py` | BoW y TF-IDF (E2) |
| `modelo_word2vec.py`, `modelo_roberta.py` | Word2Vec propio y RoBERTa (E2) |
| `embeddings.py` | embeddings comparados en el E3 (aleatorio, Word2Vec, spaCy reescalado, Word2Vec externo) |
| `tecnicismos.py`, `deep_tecnicismos.py` | fase 1 (shallow y deep) |
| `simplificacion.py`, `deep_seq2seq.py` | fase 3 (baselines y seq2seq con atención, copia y beam) |
| `preentreno.py` | carga y limpieza del dataset externo, Word2Vec externo |
| `metricas.py`, `analisis.py` | métricas, IC bootstrap, análisis de errores |
| `validar_etiquetas.py` | evalúa las etiquetas débiles frente a una muestra etiquetada a mano |
| `exportador.py` | guardado de corpus y tablas |
| `main.py`, `ejecutar_todo.py` | orquestación |

## Salidas

`resultados_e2/`: corpus procesado (CSV/JSONL), BoW/TF-IDF, Word2Vec, embeddings, OOV.

`resultados_e3/`:

- `tabla_fase1_tecnicismos.{csv,md}`, `tabla_fase3_simplificacion.{csv,md}`: tablas comparativas (media de semillas, `*_std`, IC 95 % de la semilla 0).
- `tablas_para_informe.md`: las tablas anteriores más rejillas y análisis, listas para pegar en el documento.
- `grid_shallow_tecnicismos.csv`, `grid_seq2seq.csv`: rejillas de hiperparámetros.
- `analisis_fase1_dimensiones.csv`, `analisis_fase1_top_errores.csv`: dónde falla cada clasificador.
- `analisis_fase3_salidas.csv`, `ejemplos_fase3.md`, `salidas_test_completas.json`: calidad de las salidas del simplificador.
- `candidatos_glosario.csv`: lemas candidatos para el glosario (fase 2).
- `muestra_validacion_etiquetas.csv`: 200 tokens para etiquetar a mano.
- `ejecucion.json`: ajustes, versiones y tiempos de la ejecución.
- `modelos/`: pesos de la semilla 0 de cada red.

## Validación manual de las etiquetas (pendiente de hacer a mano)

Las etiquetas de la fase 1 son **débiles** (“el lema del original no aparece en la versión clara”). Para medir cuánto se
parecen a la idea humana de *palabra difícil*:

1. Abrir `resultados_e3/muestra_validacion_etiquetas.csv` y rellenar `etiqueta_manual` (1 = palabra que un ciudadano sin formación jurídica tendría dificultad para entender o que conviene reescribir; 0 = palabra sencilla). Idealmente por dos personas, guardando cada copia con su nombre.
2. `python validar_etiquetas.py resultados_e3/muestra_validacion_etiquetas.csv [otra_anotadora.csv]` imprime accuracy, precisión, recall, F1 y κ de Cohen (y el acuerdo entre anotadoras).

## Dataset externo

`config.ARCHIVO_EXTERNO` apunta a `alia_legal_1500_lenguaje_claro.csv` (279 pares, 274 tras limpiar). Es del mismo dominio y tarea,
pero pequeño y con respuestas a veces resumidas, así que **no cabe esperar grandes mejoras**: el experimento sirve para medir
si ayuda. Garantías anti-fuga (en `preentreno.py`): se eliminan del externo los textos que coincidan con val/test, el Word2Vec
externo solo ve train + externo, y cada modelo preentrenado tiene un control con el mismo vocabulario sin preentreno.
Para usar otro dataset, cambiar `ARCHIVO_EXTERNO` o editar `cargar_dataset_externo()`; si se cambia, borrar `resultados_e3/externo_procesado.joblib`.

## Reproducibilidad

- Partición 70/15/15 con `SEED_SPLIT = 42`; semillas de las redes `[123, 7, 2024]` (se informa media ± desviación).
- Vocabularios, vectorizadores y Word2Vec se ajustan **solo con train**; los hiperparámetros se eligen en validación y el test se usa una sola vez.
- Todos los sistemas se evalúan con el mismo BLEU (`sacrebleu`, texto en minúsculas con puntuación separada).

## Limitaciones conocidas

- Las etiquetas débiles recogen lo que reescribe GPT-3.5 (también palabras corrientes como *usuario* o *tener*), no solo tecnicismos.
- BLEU frente a una referencia generada por GPT-3.5 premia copiar el original y no mide fidelidad legal; el análisis de `analisis_fase3_salidas.csv` (cifras y entidades conservadas) lo complementa.
- La búsqueda de hiperparámetros del seq2seq se hace solo sobre una variante y se reutiliza en las demás.
- Los textos del corpus fueron generados con IA: no usar para decisiones legales.

Licencia: MIT (ver `LICENSE`).
