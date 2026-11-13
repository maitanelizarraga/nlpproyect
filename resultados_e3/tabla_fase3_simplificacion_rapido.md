| modelo | familia | representacion | bleu_val | bleu_test | bleu_test_std | bleu_test_beam | bleu_test_ic | cobertura_emb | epocas | n_semillas |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline: copiar el original | baseline | - | 24.26 | 30.09 | - | - | [25.02, 34.61] | - | - | 1 |
| Baseline: vecino más cercano (TF-IDF) | baseline | TF-IDF (dispersa) | 13.42 | 9.41 | - | - | [6.42, 13.14] | - | - | 1 |
| GRU+att, embeddings aleatorios | deep | Embeddings aleatorios | 0.01 | 0.02 | 0.00 | 0.00 | [0.00, 0.00] | - | 2.000 | 1 |
| GRU+att, Word2Vec propio congelado | deep | Word2Vec propio (E2) congelado | 0.10 | 0.19 | 0.00 | 0.20 | [0.10, 0.26] | 1.000 | 2.000 | 1 |
| GRU+att, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 0.10 | 0.19 | 0.00 | 0.24 | [0.10, 0.42] | 1.000 | 2.000 | 1 |
| GRU+att, spaCy congelado | deep | spaCy es_core_news_md congelado | 0.73 | 0.66 | 0.00 | 1.44 | [0.80, 1.98] | 0.964 | 2.000 | 1 |
| GRU+att, spaCy fine-tuneado | deep | spaCy es_core_news_md fine-tuneado | 0.72 | 0.80 | 0.00 | 1.16 | [0.69, 1.75] | 0.964 | 2.000 | 1 |
| GRU sin atención, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 0.08 | 0.18 | 0.00 | 0.19 | [0.13, 0.23] | 1.000 | 2.000 | 1 |
| GRU+att+copia, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 1.20 | 0.71 | 0.00 | 0.65 | [0.51, 0.79] | 1.000 | 2.000 | 1 |