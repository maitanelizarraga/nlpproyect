| modelo | familia | representacion | accuracy | precision | recall | f1 | f1_val | hiperparametros |
|---|---|---|---|---|---|---|---|---|
| Baseline: etiqueta mayoritaria por lema | baseline | - | 0.805 | 0.738 | 0.625 | 0.677 | 0.685 | - |
| Baseline: todo complejo | baseline | - | 0.328 | 0.328 | 1.000 | 0.494 | 0.495 | - |
| BoW + Regresión logística | shallow | Bag of Words (dispersa) | 0.815 | 0.684 | 0.810 | 0.742 | 0.749 | ventana=2, C=1.0, class_weight=balanced |
| TF-IDF + SVM lineal | shallow | TF-IDF (dispersa) | 0.813 | 0.676 | 0.828 | 0.744 | 0.751 | ventana=2, C=0.1, class_weight=balanced |
| BiGRU, embeddings aleatorios | deep | Embeddings aleatorios | 0.816 | 0.691 | 0.792 | 0.738 | 0.746 | umbral=0.35, épocas=15 |
| BiGRU, Word2Vec propio congelado | deep | Word2Vec propio congelado | 0.800 | 0.657 | 0.816 | 0.728 | 0.735 | umbral=0.4, épocas=15 |
| BiGRU, Word2Vec propio fine-tuneado | deep | Word2Vec propio fine-tuneado | 0.816 | 0.687 | 0.804 | 0.741 | 0.741 | umbral=0.4, épocas=8 |
| BiGRU, spaCy congelado | deep | spaCy es_core_news_md congelado | 0.816 | 0.697 | 0.778 | 0.735 | 0.746 | umbral=0.4, épocas=11 |
| BiGRU, spaCy fine-tuneado | deep | spaCy es_core_news_md fine-tuneado | 0.829 | 0.737 | 0.743 | 0.740 | 0.750 | umbral=0.45, épocas=11 |
| BERTIN congelado (solo capa lineal) | deep | BERTIN (contextual) congelado | 0.758 | 0.604 | 0.759 | 0.672 | 0.671 | umbral=0.35, épocas=3 |
| BERTIN fine-tuneado | deep | BERTIN (contextual) fine-tuneado | 0.849 | 0.761 | 0.788 | 0.774 | 0.782 | umbral=0.45, épocas=3 |