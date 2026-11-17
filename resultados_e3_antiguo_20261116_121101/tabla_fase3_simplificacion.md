| modelo | familia | representacion | bleu_val | bleu_test | cobertura_emb | epocas |
|---|---|---|---|---|---|---|
| Baseline: copiar el original | baseline | - | 23.94 | 24.09 | - | - |
| Baseline: vecino más cercano (TF-IDF) | baseline | TF-IDF (dispersa) | 20.22 | 19.25 | - | - |
| GRU+att, embeddings aleatorios | deep | Embeddings aleatorios | 15.87 | 15.04 | - | 20.000 |
| GRU+att, Word2Vec propio congelado | deep | Word2Vec propio (E2) congelado | 17.42 | 15.89 | 1.000 | 20.000 |
| GRU+att, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 19.52 | 17.69 | 1.000 | 20.000 |
| GRU+att, spaCy congelado | deep | spaCy es_core_news_md congelado | 11.05 | 10.26 | 0.960 | 13.000 |
| GRU+att, spaCy fine-tuneado | deep | spaCy es_core_news_md fine-tuneado | 11.26 | 10.50 | 0.960 | 13.000 |
| GRU sin atención, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 13.93 | 12.43 | 1.000 | 20.000 |