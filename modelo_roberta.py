import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
from config import (
    MODELO_CONTEXTUAL,
    BATCH_SIZE_ROBERTA,
    MAX_TOKENS_ROBERTA,
    CARPETA_SALIDA,
)

def procesar_lote_textos(textos, tokenizador, modelo, dispositivo):
    """Procesa un grupo de textos y devuelve sus vectores matemáticos."""
    entradas = tokenizador(
        textos,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS_ROBERTA,
    )
    
    entradas = {k: v.to(dispositivo) for k, v in entradas.items()}
    
    with torch.no_grad():
        salida = modelo(**entradas).last_hidden_state

    mascara = entradas["attention_mask"].unsqueeze(-1).to(salida.dtype)
    vectores = (salida * mascara).sum(dim=1) / mascara.sum(dim=1).clamp(min=1)
    
    return vectores.cpu().numpy()

def generar_embeddings_contextuales(df):
    print("\n--- Generando embeddings con RoBERTa ---")
    
    config_roberta = RobertaConfig.from_pretrained(MODELO_CONTEXTUAL)
    tokenizador = RobertaTokenizerFast.from_pretrained(MODELO_CONTEXTUAL)
    
    # <--- CLAVE: use_safetensors=False obliga a descargar e interpretar pytorch_model.bin
    modelo = RobertaModel.from_pretrained(
        MODELO_CONTEXTUAL, 
        config=config_roberta, 
        use_safetensors=False
    )
    
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(dispositivo)
    modelo.eval()
    
    textos_originales = df["question"].astype(str).tolist()
    todos_los_vectores = []

    for inicio in range(0, len(textos_originales), BATCH_SIZE_ROBERTA):
        lote = textos_originales[inicio:inicio + BATCH_SIZE_ROBERTA]
        vectores_lote = procesar_lote_textos(lote, tokenizador, modelo, dispositivo)
        todos_los_vectores.append(vectores_lote)
        print(f"Procesados {inicio + len(lote)} / {len(textos_originales)} textos")

    matriz_final = np.concatenate(todos_los_vectores, axis=0).astype(np.float32)
    
    np.savez_compressed(
        CARPETA_SALIDA / "embeddings_roberta_originales.npz",
        document_id=df["document_id"].to_numpy(),
        embeddings=matriz_final,
    )
    print("Guardado con éxito. Dimensiones:", matriz_final.shape)