| modelo | familia | representacion | bleu_val | bleu_test | bleu_test_std | bleu_test_beam | bleu_test_ic | cobertura_emb | epocas | n_semillas |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline: copiar el original | baseline | - | 26.36 | 26.47 | - | - | [25.12, 27.78] | - | - | 1 |
| Baseline: vecino más cercano (TF-IDF) | baseline | TF-IDF (dispersa) | 20.23 | 19.30 | - | - | [17.59, 20.80] | - | - | 1 |
| GRU+att+copia, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 24.41 | 23.36 | 0.00 | 24.75 | [23.38, 26.42] | 1.000 | 20.000 | 1 |