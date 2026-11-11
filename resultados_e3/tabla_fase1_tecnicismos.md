| modelo | familia | representacion | accuracy | precision | recall | f1 | f1_val | hiperparametros |
|---|---|---|---|---|---|---|---|---|
| Baseline: etiqueta mayoritaria por lema | baseline | - | 0.819 | 0.787 | 0.690 | 0.735 | 0.743 | - |
| Baseline: todo complejo | baseline | - | 0.365 | 0.365 | 1.000 | 0.535 | 0.536 | - |
| BoW + Regresión logística | shallow | Bag of Words (dispersa) | 0.834 | 0.749 | 0.821 | 0.783 | 0.792 | ventana=2, C=1.0, class_weight=balanced |
| TF-IDF + SVM lineal | shallow | TF-IDF (dispersa) | 0.833 | 0.741 | 0.835 | 0.785 | 0.791 | ventana=2, C=0.1, class_weight=balanced |
| BiGRU, embeddings aleatorios | deep | Embeddings aleatorios | 0.832 | 0.754 | 0.801 | 0.777 | 0.783 | umbral=0.45, épocas=15 |
| BiGRU, Word2Vec propio congelado | deep | Word2Vec propio (E2) congelado | 0.827 | 0.743 | 0.803 | 0.772 | 0.783 | umbral=0.45, épocas=15 |
| BiGRU, Word2Vec propio fine-tuneado | deep | Word2Vec propio (E2) fine-tuneado | 0.834 | 0.745 | 0.827 | 0.784 | 0.788 | umbral=0.4, épocas=9 |
| BiGRU, spaCy congelado | deep | spaCy es_core_news_md congelado | 0.838 | 0.766 | 0.801 | 0.783 | 0.789 | umbral=0.45, épocas=15 |
| BiGRU, spaCy fine-tuneado | deep | spaCy es_core_news_md fine-tuneado | 0.841 | 0.765 | 0.814 | 0.789 | 0.793 | umbral=0.45, épocas=15 |
| BERTIN congelado (solo capa lineal) | deep | BERTIN (contextual) congelado | 0.776 | 0.664 | 0.783 | 0.719 | 0.719 | umbral=0.4, épocas=3 |
| BERTIN fine-tuneado | deep | BERTIN (contextual) fine-tuneado | 0.857 | 0.779 | 0.849 | 0.813 | 0.821 | umbral=0.4, épocas=3 |