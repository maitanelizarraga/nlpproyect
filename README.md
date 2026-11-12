# nlpproyect — E3 (versión corregida)

Pipeline de simplificación de textos administrativos/jurídicos. E2 = datos y representaciones,
E3 = modelos (fase 1: palabras complejas; fase 3: simplificación).

## Qué se ha corregido / añadido respecto a la primera iteración del E3

| Punto | Cambio | Dónde |
|---|---|---|
| Error de datos | Todos los `question` empezaban con *"Escribe el siguiente texto en lenguaje claro:"* (instrucción del prompt). Se elimina al cargar el corpus; si el E2 guardado es antiguo, `--solo-e3` lo repara solo (sin repetir spaCy) y reentrena el Word2Vec. | `data_loader.py`, `main.py` |
| Preentrenamiento | Hook para conectar un dataset externo de pares (complejo, simple). Con él: Word2Vec externo, BiGRU preentrenada, seq2seq preentrenado, shallow reajustado, **cada uno con su control sin preentreno**. | `preentreno.py` |
| Rasgos no textuales (shallow) | Dos conjuntos de rasgos: `base` y `extendido` (POS propio y de vecinos, entidad, stopword, frecuencia del lema en train). Se comparan en la tabla. | `tecnicismos.py` |
| Fase 3 | Mecanismo de copia (pointer-generator), beam search + bloqueo de trigramas repetidos, búsqueda de hiperparámetros, reescalado de los vectores de spaCy y diagnóstico de embeddings. | `deep_seq2seq.py`, `embeddings.py` |
| Rigor | 3 semillas por red (media ± desviación), IC 95 % por bootstrap (F1 y BLEU), análisis de errores, validación manual de las etiquetas débiles. | `metricas.py`, `analisis.py`, `validar_etiquetas.py` |
| Atención en la fase 1 | Variantes BiGRU + autoatención y BiGRU + autoatención + POS. | `deep_tecnicismos.py` |

## Uso

**Un solo comando (hace todo solo):** `python ejecutar_todo.py` (ajustes en las constantes de su cabecera).

Opciones manuales con `main.py`:

```
python main.py                                  # E2 + E3 (regenera todo el E2 con los datos corregidos)
python main.py --solo-e3                        # E3 con el E2 guardado (lo repara si es el antiguo)
python main.py --solo-e3 --rapido               # prueba de humo (200 textos, 2 épocas, 1 semilla)
python main.py --solo-e3 --fase 1 --sin-deep    # fase 1 solo con shallow (minutos, sin GPU)
python main.py --solo-e3 --semillas 1           # una sola semilla (más rápido)
python main.py --solo-e3 --beam 0               # fase 3 solo con decodificación voraz
python main.py --solo-e3 --buscar-hp            # + búsqueda de hiperparámetros del seq2seq (lenta)
python main.py --solo-e3 --externo              # + variantes con el dataset externo
python validar_etiquetas.py resultados_e3/muestra_validacion_etiquetas.csv
```

**Recomendado la primera vez:** `python main.py --solo-e2` para regenerar todos los ficheros del E2
(BoW, TF-IDF, OOV, RoBERTa...) con el corpus corregido y que las cifras del informe E2 coincidan.
Las cifras del informe del E2 (longitudes medias, vocabulario, OOV) cambiarán al quitar el prefijo.

## Conectar el dataset externo

En `preentreno.py`, rellenar `cargar_dataset_externo()` para que devuelva un DataFrame con las
columnas `question` (original/complejo) y `answer` (simplificado). Después ejecutar con `--externo`.
Si cambiáis de dataset, borrad `resultados_e3/externo_procesado.joblib` (caché del procesado).
Se eliminan automáticamente los textos del externo que coincidan con val/test.

## Salidas en `resultados_e3/`

- `tabla_fase1_tecnicismos.{csv,md}`, `tabla_fase3_simplificacion.{csv,md}`: tablas comparativas
  (media sobre semillas, `*_std`, IC 95 % de la semilla 0).
- `grid_shallow_tecnicismos.csv`, `grid_seq2seq.csv`: rejillas de hiperparámetros.
- `analisis_fase1_dimensiones.csv` (F1 por categoría gramatical, frecuencia del lema y longitud),
  `analisis_fase1_top_errores.csv` (lemas más fallados).
- `analisis_fase3_salidas.csv` (longitud, trigramas repetidos, % de cifras y entidades conservadas,
  incluida la propia referencia), `ejemplos_fase3.md`, `salidas_test_completas.json`.
- `candidatos_glosario.csv`: lemas léxicos (nombre/verbo/adjetivo/adverbio) marcados como complejos,
  con su proporción, para la fase 2 (glosario DPEJ).
- `muestra_validacion_etiquetas.csv`: 200 tokens para etiquetar a mano (ver `validar_etiquetas.py`).

## Notas y límites

- Las etiquetas de la fase 1 son débiles ("el lema desaparece en la simplificación"); la muestra manual
  sirve para cuantificar cuánto se parecen a la noción humana de palabra compleja.
- `bleu_test_ic` usa las salidas con beam si se activó; los IC remuestrean tokens/frases sueltos, así
  que son algo optimistas.
- Los textos de referencia (GPT-3.5) no conservan todas las cifras ni entidades del original
  (ver `analisis_fase3_salidas.csv`, fila "Referencia"): BLEU frente a ellos premia también esas omisiones.
