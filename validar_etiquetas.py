"""Valida las etiquetas debiles de la fase 1 con una muestra etiquetada a mano.

1) Ejecutar la fase 1: genera resultados_e3/muestra_validacion_etiquetas.csv
2) Rellenar la columna 'etiqueta_manual' (1 = palabra dificil para un ciudadano / conviene reescribirla,
   0 = palabra sencilla). Idealmente por dos personas (guardar cada copia con su nombre) para medir
   tambien el acuerdo entre anotadoras.
3) python validar_etiquetas.py resultados_e3/muestra_validacion_etiquetas.csv [otra_anotadora.csv]
"""
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score


def cargar(ruta):
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    df["etiqueta_manual"] = pd.to_numeric(df["etiqueta_manual"], errors="coerce")
    return df.dropna(subset=["etiqueta_manual"]).astype({"etiqueta_manual": int})


def main(rutas):
    a = cargar(rutas[0])
    print(f"Tokens etiquetados a mano: {len(a)} | % complejos (manual): {100 * a['etiqueta_manual'].mean():.1f} "
          f"| % complejos (automatica): {100 * a['etiqueta_automatica'].mean():.1f}")
    p, r, f1, _ = precision_recall_fscore_support(a["etiqueta_manual"], a["etiqueta_automatica"],
                                                  average="binary", pos_label=1, zero_division=0)
    print(f"Etiqueta automatica frente a la manual -> accuracy={accuracy_score(a['etiqueta_manual'], a['etiqueta_automatica']):.3f} "
          f"precision={p:.3f} recall={r:.3f} F1={f1:.3f} "
          f"kappa={cohen_kappa_score(a['etiqueta_manual'], a['etiqueta_automatica']):.3f}")
    if len(rutas) > 1:
        b = cargar(rutas[1])
        m = a.merge(b, on=["document_id", "palabra", "contexto"], suffixes=("_a", "_b"))
        print(f"Acuerdo entre anotadoras ({len(m)} tokens comunes): "
              f"kappa={cohen_kappa_score(m['etiqueta_manual_a'], m['etiqueta_manual_b']):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
