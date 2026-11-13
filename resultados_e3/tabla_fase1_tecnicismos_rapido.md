| modelo | familia | representacion | accuracy | precision | recall | f1 | f1_std | f1_ic95_s0 | f1_val | n_semillas | hiperparametros |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline: etiqueta mayoritaria por lema | baseline | - | 0.760 | 0.687 | 0.439 | 0.536 | - | [0.511, 0.561] | 0.584 | - | - |
| Baseline: todo complejo | baseline | - | 0.315 | 0.315 | 1.000 | 0.479 | - | [0.460, 0.497] | 0.514 | - | - |
| BoW + Regresión logística · rasgos base | shallow | Bag of Words (dispersa) | 0.768 | 0.596 | 0.818 | 0.689 | - | [0.669, 0.709] | 0.729 | 1.000 | ventana=2, C=0.1, class_weight=balanced |
| BoW + Regresión logística · rasgos extendido | shallow | Bag of Words (dispersa) | 0.779 | 0.611 | 0.819 | 0.700 | - | [0.680, 0.720] | 0.728 | 1.000 | ventana=2, C=0.1, class_weight=balanced |
| TF-IDF + SVM lineal · rasgos base | shallow | TF-IDF (dispersa) | 0.777 | 0.610 | 0.808 | 0.695 | - | [0.674, 0.716] | 0.726 | 1.000 | ventana=2, C=0.1, class_weight=balanced |
| TF-IDF + SVM lineal · rasgos extendido | shallow | TF-IDF (dispersa) | 0.783 | 0.617 | 0.819 | 0.704 | - | [0.683, 0.723] | 0.727 | 1.000 | ventana=2, C=0.1, class_weight=balanced |
| BiGRU, embeddings aleatorios | deep | Embeddings aleatorios | 0.682 | 0.497 | 0.817 | 0.618 | 0.000 | [0.600, 0.637] | 0.627 | 1.000 | umbral=0.25, épocas=2, semillas=1 |
| BiGRU, Word2Vec propio congelado | deep | Word2Vec propio (E2) congelado | 0.657 | 0.476 | 0.872 | 0.616 | 0.000 | [0.595, 0.632] | 0.617 | 1.000 | umbral=0.30, épocas=2, semillas=1 |
| BiGRU, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 0.663 | 0.481 | 0.872 | 0.620 | 0.000 | [0.598, 0.637] | 0.626 | 1.000 | umbral=0.30, épocas=2, semillas=1 |
| BiGRU, spaCy congelado | deep | spaCy es_core_news_md congelado | 0.676 | 0.492 | 0.853 | 0.624 | 0.000 | [0.603, 0.643] | 0.640 | 1.000 | umbral=0.35, épocas=2, semillas=1 |
| BiGRU, spaCy fine-tuneado | deep | spaCy es_core_news_md fine-tuneado | 0.682 | 0.497 | 0.864 | 0.631 | 0.000 | [0.611, 0.649] | 0.649 | 1.000 | umbral=0.35, épocas=2, semillas=1 |
| BiGRU+autoatención, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 0.708 | 0.523 | 0.842 | 0.645 | 0.000 | [0.626, 0.662] | 0.658 | 1.000 | umbral=0.20, épocas=2, semillas=1 |
| BiGRU+autoatención+POS, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 0.771 | 0.616 | 0.719 | 0.664 | 0.000 | [0.641, 0.685] | 0.677 | 1.000 | umbral=0.35, épocas=2, semillas=1 |
| BERTIN congelado (solo capa lineal) | deep | BERTIN (contextual) congelado | 0.644 | 0.461 | 0.778 | 0.579 | 0.000 | [0.561, 0.603] | 0.597 | 1.000 | umbral=0.25, épocas=2, semillas=1 |
| BERTIN fine-tuneado | deep | BERTIN (contextual) fine-tuneado | 0.804 | 0.672 | 0.740 | 0.704 | 0.000 | [0.684, 0.724] | 0.731 | 1.000 | umbral=0.40, épocas=2, semillas=1 |