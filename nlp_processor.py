import spacy
from config import MODELO_SPACY

#el spacy analiza y entiende la estructura gramatical del texto
def cargar_spacy():
    print("\n CARGAR SPACY:")
    #Carga el modelo español
    nlp = spacy.load(MODELO_SPACY)
    print("modelo cargado:", MODELO_SPACY)
    return nlp
   
def procesar_textos(df, nlp):
    print("\n PROCESAR TEXTOS:")
    """
    Tokeniza, separa en oraciones, obtiene lemas y entidades.
    Se procesan tanto los textos originales (question) como los simplificados (answer).
    """
    df = df.copy()
    originales = df["question"].tolist()
    simplificados = df["answer"].tolist()

    def extraer(doc):
        return {
            "tokens": [t.text for t in doc if not t.is_space],
            "oraciones": [s.text for s in doc.sents],
            "lemas": [t.lemma_ for t in doc if not t.is_space],
            "entidades": [
                {"texto": ent.text, "tipo": ent.label_}
                for ent in doc.ents
            ],
        }

    print("Procesando textos con spaCy...")
    docs_originales = list(nlp.pipe(originales, batch_size=32))
    docs_simplificados = list(nlp.pipe(simplificados, batch_size=32))

    df["procesado_original"] = [extraer(doc) for doc in docs_originales]
    df["procesado_simplificado"] = [extraer(doc) for doc in docs_simplificados]
    # Longitud en tokens de spaCy (comparable con el vocabulario y el análisis OOV)
    df["tokens_original"] = [len(p["tokens"]) for p in df["procesado_original"]]
    df["tokens_simplificado"] = [len(p["tokens"]) for p in df["procesado_simplificado"]]
    print("Media de tokens spaCy (original):", round(float(df["tokens_original"].mean()), 2))
    print("Media de tokens spaCy (simplificado):", round(float(df["tokens_simplificado"].mean()), 2))
    print("procesado")
    return df