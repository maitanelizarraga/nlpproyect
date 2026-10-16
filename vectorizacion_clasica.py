import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import save_npz
from config import CARPETA_SALIDA

def crear_bag_of_words(textos_train, textos_test, textos_val):
    print("\n--- Entrenando Bag of Words ---")
    min_df = 2 if len(textos_train) >= 10 else 1
    
    # 1. Configurar el modelo
    bow = CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=min_df)
    
    # 2. Entrenar y transformar
    X_bow_train = bow.fit_transform(textos_train)
    X_bow_test = bow.transform(textos_test)
    X_bow_val = bow.transform(textos_val)
    
    # 3. Guardar modelo y matrices
    joblib.dump(bow, CARPETA_SALIDA / "vectorizador_bow.joblib")
    save_npz(CARPETA_SALIDA / "bow_train.npz", X_bow_train)
    save_npz(CARPETA_SALIDA / "bow_test.npz", X_bow_test)
    save_npz(CARPETA_SALIDA / "bow_validation.npz", X_bow_val)
    
    print("Vocabulario BoW:", len(bow.vocabulary_))
    return bow

def crear_tfidf(textos_train, textos_test, textos_val):
    print("\n--- Entrenando TF-IDF ---")
    min_df = 2 if len(textos_train) >= 10 else 1
    
    # 1. Configurar el modelo
    tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
    
    # 2. Entrenar y transformar
    X_tfidf_train = tfidf.fit_transform(textos_train)
    X_tfidf_test = tfidf.transform(textos_test)
    X_tfidf_val = tfidf.transform(textos_val)
    
    # 3. Guardar modelo y matrices
    joblib.dump(tfidf, CARPETA_SALIDA / "vectorizador_tfidf.joblib")
    save_npz(CARPETA_SALIDA / "tfidf_train.npz", X_tfidf_train)
    save_npz(CARPETA_SALIDA / "tfidf_test.npz", X_tfidf_test)
    save_npz(CARPETA_SALIDA / "tfidf_validation.npz", X_tfidf_val)
    
    print("Vocabulario TF-IDF:", len(tfidf.vocabulary_))
    return tfidf