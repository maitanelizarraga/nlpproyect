| modelo | familia | representacion | accuracy | precision | recall | f1 | f1_std | f1_ic95_s0 | f1_val | n_semillas | hiperparametros |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline: etiqueta mayoritaria por lema | baseline | - | 0.805 | 0.738 | 0.625 | 0.677 | - | [0.671, 0.684] | 0.685 | - | - |
| Baseline: todo complejo | baseline | - | 0.328 | 0.328 | 1.000 | 0.494 | - | [0.488, 0.499] | 0.495 | - | - |
| BoW + Regresión logística · rasgos base | shallow | Bag of Words (dispersa) | 0.815 | 0.684 | 0.810 | 0.742 | - | [0.736, 0.747] | 0.749 | 1.000 | ventana=2, C=1.0, class_weight=balanced |
| BoW + Regresión logística · rasgos extendido | shallow | Bag of Words (dispersa) | 0.817 | 0.686 | 0.813 | 0.744 | - | [0.738, 0.750] | 0.750 | 1.000 | ventana=2, C=1.0, class_weight=balanced |
| TF-IDF + SVM lineal · rasgos base | shallow | TF-IDF (dispersa) | 0.813 | 0.676 | 0.828 | 0.744 | - | [0.738, 0.749] | 0.751 | 1.000 | ventana=2, C=0.1, class_weight=balanced |
| TF-IDF + SVM lineal · rasgos extendido | shallow | TF-IDF (dispersa) | 0.814 | 0.676 | 0.833 | 0.746 | - | [0.741, 0.751] | 0.750 | 1.000 | ventana=2, C=0.1, class_weight=balanced |
| BiGRU, embeddings aleatorios | deep | Embeddings aleatorios | 0.813 | 0.687 | 0.792 | 0.735 | 0.002 | [0.732, 0.743] | 0.743 | 3.000 | umbral=0.35, épocas=15, semillas=3 |
| BiGRU, Word2Vec propio congelado | deep | Word2Vec propio (E2) congelado | 0.802 | 0.665 | 0.801 | 0.727 | 0.004 | [0.723, 0.734] | 0.734 | 3.000 | umbral=0.45, épocas=15, semillas=3 |
| BiGRU, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 0.819 | 0.697 | 0.790 | 0.741 | 0.003 | [0.732, 0.743] | 0.747 | 3.000 | umbral=0.40, épocas=10, semillas=3 |
| BiGRU, spaCy congelado | deep | spaCy es_core_news_md congelado | 0.808 | 0.677 | 0.790 | 0.729 | 0.006 | [0.718, 0.728] | 0.736 | 3.000 | umbral=0.45, épocas=10, semillas=3 |
| BiGRU, spaCy fine-tuneado | deep | spaCy es_core_news_md fine-tuneado | 0.821 | 0.698 | 0.799 | 0.745 | 0.001 | [0.739, 0.749] | 0.750 | 3.000 | umbral=0.40, épocas=10, semillas=3 |
| BiGRU+autoatención, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 0.824 | 0.708 | 0.786 | 0.745 | 0.000 | [0.740, 0.751] | 0.748 | 3.000 | umbral=0.40, épocas=11, semillas=3 |
| BiGRU+autoatención+POS, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 0.829 | 0.724 | 0.770 | 0.746 | 0.005 | [0.745, 0.756] | 0.752 | 3.000 | umbral=0.45, épocas=9, semillas=3 |
| BERTIN congelado (solo capa lineal) | deep | BERTIN (contextual) congelado | 0.757 | 0.604 | 0.757 | 0.672 | 0.001 | [0.667, 0.679] | 0.670 | 3.000 | umbral=0.35, épocas=3, semillas=3 |
| BERTIN fine-tuneado | deep | BERTIN (contextual) fine-tuneado | 0.848 | 0.755 | 0.794 | 0.774 | 0.003 | [0.769, 0.778] | 0.783 | 3.000 | umbral=0.45, épocas=3, semillas=3 |