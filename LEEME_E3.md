# Proyecto NLP: E2 + E3 integrados

Sustituye/añade estos archivos en la carpeta del E2 (los demás del E2 no cambian:
data_loader.py, nlp_processor.py, vectorizacion_clasica.py, modelo_roberta.py).

Modificados: config.py, main.py, exportador.py, modelo_word2vec.py (corregido: question+answer, minúsculas)
Nuevos: metricas.py, embeddings.py, tecnicismos.py, deep_tecnicismos.py, simplificacion.py, deep_seq2seq.py

Dependencias extra: pip install sacrebleu   (el resto ya se usa en el E2: torch, transformers, gensim, spacy, scikit-learn)

Uso (un solo main):
  python main.py                              # E2 completo + E3 completo
  python main.py --solo-e2                    # solo E2 (necesario la primera vez: regenera el Word2Vec corregido)
  python main.py --solo-e3 --rapido           # prueba de humo del E3 (usa lo que el E2 ya guardó)
  python main.py --solo-e3 --fase 1 --sin-deep   # fase 1 solo con shallow (minutos, sin GPU)
  python main.py --solo-e3 --sin-bert         # E3 sin RoBERTa
  python main.py --solo-e3                    # E3 completo (GPU recomendable)
Salidas del E3 en resultados_e3/.
