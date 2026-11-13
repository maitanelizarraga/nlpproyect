| modelo | familia | representacion | bleu_val | bleu_test | cobertura_emb | epocas |
|---|---|---|---|---|---|---|
| Baseline: copiar el original | baseline | - | 23.94 | 24.09 | - | - |
| Baseline: vecino más cercano (TF-IDF) | baseline | TF-IDF (dispersa) | 20.22 | 19.25 | - | - |
| GRU+att, embeddings aleatorios | deep | Embeddings aleatorios | 15.87 | 15.04 | - | 20.000 |
| GRU+att, Word2Vec propio congelado | deep | Word2Vec propio congelado | 18.42 | 16.85 | 1.000 | 20.000 |
| GRU+att, Word2Vec propio fine-tuneado | deep | Word2Vec propio fine-tuneado | 20.37 | 19.09 | 1.000 | 20.000 |
| GRU+att, spaCy congelado | deep | spaCy es_core_news_md congelado | 11.05 | 10.26 | 0.960 | 13.000 |
| GRU+att, spaCy fine-tuneado | deep | spaCy es_core_news_md fine-tuneado | 11.26 | 10.50 | 0.960 | 13.000 |
| GRU sin atención, Word2Vec propio fine-tuneado | deep | Word2Vec propio fine-tuneado | 12.61 | 11.26 | 1.000 | 20.000 |