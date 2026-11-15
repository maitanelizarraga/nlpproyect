"""Ejecuta TODO el proyecto (E2 + E3) con un solo comando:

    python ejecutar_todo.py                 # lo recomendado: humo + E3 completo + verificacion + tablas
    python ejecutar_todo.py --sin-bert      # sin RoBERTa/BERTIN (si no hay GPU)
    python ejecutar_todo.py --solo-verificar  # no entrena: solo comprueba resultados_e3/ y regenera las tablas

Que hace, en orden (cada paso se ejecuta en un proceso aparte con main.py):
  0. Comprueba el entorno (librerias, modelo de spaCy, GPU).
  1. E2: si no hay E2 guardado, o es el antiguo (con el prefijo del prompt) o le faltan POS/stop/ent,
     lo regenera entero (python main.py --solo-e2). Si esta bien, lo reutiliza.
  2. Archiva resultados_e3/ anterior en resultados_e3_antiguo_<fecha>/ para que NO se mezclen
     resultados de ejecuciones distintas (la causa de que antes hubiera tablas desfasadas).
  3. PRUEBA DE HUMO (200 textos, 2 epocas, 1 semilla) escribiendo en resultados_e3_rapido/ (no pisa nada).
     Si falla, se aborta en minutos y no tras horas.
  4. E3 COMPLETO en resultados_e3/: fase 1 + fase 3, 3 semillas, beam search, busqueda de hiperparametros
     del seq2seq y variantes con el dataset externo (config.ARCHIVO_EXTERNO).
  5. VERIFICA que los resultados son coherentes (columnas nuevas, 615 textos de test, filas esperadas...).
  6. Genera resultados_e3/tablas_para_informe.md (tablas listas para pegar en el documento E3) y
     resultados_e3/ejecucion.json (ajustes, librerias y tiempos).
"""
import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
os.chdir(RAIZ)                      # config.py usa rutas relativas
sys.path.insert(0, str(RAIZ))

from config import CARPETA_SALIDA, ARCHIVO_EXTERNO, SEMILLAS, BEAM   # noqa: E402

E3 = Path("resultados_e3")
E3_HUMO = Path("resultados_e3_rapido")
PY = sys.executable
TIEMPOS = {}


# ------------------------------------------------------------------ utilidades
def paso(titulo):
    print("\n" + "=" * 100 + f"\n[todo] {titulo}\n" + "=" * 100, flush=True)


def correr(args, env_extra=None, titulo=""):
    """Ejecuta 'python main.py <args>' en un proceso aparte. Aborta si falla."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", **(env_extra or {}))
    t0 = time.time()
    print(f"[todo] $ python main.py {' '.join(args)}", flush=True)
    r = subprocess.run([PY, "main.py", *args], env=env, cwd=RAIZ)
    TIEMPOS[titulo or " ".join(args)] = round(time.time() - t0)
    if r.returncode != 0:
        sys.exit(f"\n[todo] ERROR: 'main.py {' '.join(args)}' termino con codigo {r.returncode}. "
                 f"Revisa el mensaje de arriba. Nada mas se ha ejecutado.")


# ------------------------------------------------------------------ paso 0: entorno
def comprobar_entorno(args):
    paso("0. Comprobando el entorno")
    faltan = []
    for m in ["pandas", "numpy", "sklearn", "scipy", "joblib", "gensim", "spacy", "torch", "sacrebleu",
              "transformers"]:
        try:
            importlib.import_module(m)
        except Exception:
            faltan.append(m)
    if faltan:
        sys.exit(f"[todo] Faltan librerias: {', '.join(faltan)}. Instalalas con 'uv sync' o 'pip install -e .'")
    import spacy
    import torch
    from config import MODELO_SPACY
    if not spacy.util.is_package(MODELO_SPACY):
        sys.exit(f"[todo] Falta el modelo de spaCy. Instalalo: python -m spacy download {MODELO_SPACY}")
    gpu = torch.cuda.is_available()
    print(f"Python {platform.python_version()} | torch {torch.__version__} | GPU: "
          f"{torch.cuda.get_device_name(0) if gpu else 'NO (se usara CPU: el E3 completo puede tardar muchas horas)'}")
    if not gpu and not args.sin_bert:
        print("[todo] AVISO: sin GPU, BERTIN fine-tuneado es muy lento. Considera --sin-bert.")


# ------------------------------------------------------------------ paso 1: E2
def e2_valido():
    ruta = CARPETA_SALIDA / "corpus_procesado.csv"
    if not ruta.exists():
        return False, "no hay corpus_procesado.csv"
    from data_loader import tiene_prefijo
    if tiene_prefijo(pd.read_csv(ruta, usecols=["question"])):
        return False, "el corpus guardado contiene el prefijo del prompt (E2 antiguo)"
    primera = json.loads(pd.read_csv(ruta, nrows=1)["procesado_original"].iloc[0])
    if not {"pos", "stop", "ent"} <= set(primera):
        return False, "al procesado le faltan pos/stop/ent"
    for f in ["word2vec_juridico.model", "embeddings_roberta_originales.npz", "vectorizador_bow.joblib",
              "vectorizador_tfidf.joblib"]:
        if not (CARPETA_SALIDA / f).exists():
            return False, f"falta {CARPETA_SALIDA / f}"
    return True, "ok"


def paso_e2(args):
    paso("1. E2 (datos y representaciones)")
    ok, motivo = e2_valido()
    if ok and not args.forzar_e2:
        print("[todo] E2 guardado correcto: se reutiliza.")
        return
    print(f"[todo] Se regenera el E2 ({'forzado' if args.forzar_e2 else motivo}). Necesita internet "
          f"(Hugging Face) y varios minutos.")
    correr(["--solo-e2"], titulo="E2")
    ok, motivo = e2_valido()
    if not ok:
        sys.exit(f"[todo] El E2 regenerado no es valido: {motivo}")


# ------------------------------------------------------------------ paso 2: archivar
def archivar_resultados():
    paso("2. Archivando resultados_e3/ anteriores")
    if E3.exists() and any(E3.iterdir()):
        destino = Path(f"resultados_e3_antiguo_{datetime.now():%Y%m%d_%H%M%S}")
        E3.rename(destino)
        print(f"[todo] resultados_e3/ anterior movido a {destino}/ (puedes borrarlo).")
    else:
        print("[todo] No habia resultados anteriores.")
    if E3_HUMO.exists():
        shutil.rmtree(E3_HUMO)


# ------------------------------------------------------------------ pasos 3 y 4: E3
def externo_conectado():
    if not Path(ARCHIVO_EXTERNO).exists():
        return False
    from preentreno import cargar_dataset_externo
    return cargar_dataset_externo() is not None


def opciones_e3(args, usar_externo, rapido):
    o = ["--solo-e3", "--fase", args.fase]
    if args.sin_bert:
        o.append("--sin-bert")
    if usar_externo:
        o.append("--externo")
    if not args.sin_hp and args.fase in ("3", "todo"):
        o.append("--buscar-hp")
    if rapido:
        o.append("--rapido")
    else:
        o += ["--semillas", str(args.semillas), "--beam", str(args.beam)]
    return o


def paso_humo(args, usar_externo):
    if args.sin_humo:
        return
    paso("3. Prueba de humo (200 textos, 2 epocas) -> resultados_e3_rapido/")
    correr(opciones_e3(args, usar_externo, True), env_extra={"E3_SALIDA": str(E3_HUMO)}, titulo="humo")
    shutil.rmtree(E3_HUMO, ignore_errors=True)
    print("[todo] Prueba de humo OK (carpeta temporal borrada).")


def paso_completo(args, usar_externo):
    paso("4. E3 completo -> resultados_e3/  (puede tardar horas)")
    correr(opciones_e3(args, usar_externo, False), titulo="E3 completo")


# ------------------------------------------------------------------ paso 5: verificacion
def verificar(args, usar_externo):
    paso("5. Verificando resultados_e3/")
    chk = []

    def anadir(ok, msg):
        chk.append((ok, msg))
        print(("  [OK]  " if ok else "  [FALLO] ") + msg)

    def leer(nombre):
        ruta = E3 / nombre
        return pd.read_csv(ruta) if ruta.exists() else None

    n_test = int((pd.read_csv(CARPETA_SALIDA / "corpus_procesado.csv", usecols=["split"])["split"] == "test").sum())

    if args.fase in ("1", "todo"):
        t = leer("tabla_fase1_tecnicismos.csv")
        anadir(t is not None, "existe tabla_fase1_tecnicismos.csv")
        if t is not None:
            anadir({"f1_std", "f1_ic95_s0", "n_semillas"} <= set(t.columns), "fase 1: columnas f1_std, IC95 y n_semillas")
            deep = t[t["familia"] == "deep"]
            anadir(len(deep) > 0 and (deep["n_semillas"] == args.semillas).all(),
                   f"fase 1: todas las redes entrenadas con {args.semillas} semillas")
            anadir(t["modelo"].str.contains("autoatención").any(), "fase 1: variantes con atencion")
            anadir(t["modelo"].str.contains("POS").any() and t["modelo"].str.contains("extendido").any(),
                   "fase 1: features no textuales (POS / rasgos extendidos)")
            if not args.sin_bert:
                anadir(t["modelo"].str.contains("BERTIN fine-tuneado").any(), "fase 1: BERTIN fine-tuneado")
            if usar_externo:
                anadir(t["modelo"].str.contains("externo").any(), "fase 1: variantes con dataset externo")
        for f in ["grid_shallow_tecnicismos.csv", "candidatos_glosario.csv", "analisis_fase1_dimensiones.csv",
                  "analisis_fase1_top_errores.csv", "muestra_validacion_etiquetas.csv"]:
            anadir((E3 / f).exists(), f"existe {f}")

    if args.fase in ("3", "todo"):
        t = leer("tabla_fase3_simplificacion.csv")
        anadir(t is not None, "existe tabla_fase3_simplificacion.csv")
        if t is not None:
            anadir({"bleu_test_ic", "n_semillas"} <= set(t.columns), "fase 3: columnas bleu_test_ic y n_semillas")
            if args.beam:
                anadir("bleu_test_beam" in t.columns and t["bleu_test_beam"].notna().any(), "fase 3: BLEU con beam search")
            anadir(t["modelo"].str.contains("copia").any(), "fase 3: variante con mecanismo de copia")
            anadir(t["modelo"].str.contains("sin atención").any(), "fase 3: ablacion sin atencion")
            if usar_externo:
                anadir(t["modelo"].str.contains("externo").any(), "fase 3: variantes con dataset externo")
            copia = t[t["modelo"] == "Baseline: copiar el original"]
            if len(copia):
                c = float(copia["bleu_test"].iloc[0])
                anadir(abs(c - 24.09) > 0.5,
                       f"fase 3: baseline 'copiar' = {c:.2f} (24.09 era el valor CON el prefijo del prompt)")
        salidas = E3 / "salidas_test_completas.json"
        if salidas.exists():
            n = len(json.load(open(salidas, encoding="utf-8"))["originales"])
            anadir(n == n_test, f"salidas_test_completas.json tiene {n} textos de test (esperado {n_test}; "
                                f"60 = prueba de humo)")
        else:
            anadir(False, "existe salidas_test_completas.json")
        for f in ["analisis_fase3_salidas.csv", "ejemplos_fase3.md"]:
            anadir((E3 / f).exists(), f"existe {f}")
        if not args.sin_hp:
            anadir((E3 / "grid_seq2seq.csv").exists(), "existe grid_seq2seq.csv (busqueda de hiperparametros)")

    rap = [p.name for p in E3.glob("*_rapido*")] if E3.exists() else []
    anadir(not rap, "resultados_e3/ no contiene ficheros de la prueba de humo" + (f" ({rap})" if rap else ""))

    m = E3 / "muestra_validacion_etiquetas.csv"
    if m.exists():
        d = pd.read_csv(m, encoding="utf-8-sig")
        hechas = int(pd.to_numeric(d["etiqueta_manual"], errors="coerce").notna().sum())
        print(f"  [INFO] validacion manual de etiquetas: {hechas}/{len(d)} tokens etiquetados "
              f"({'pendiente: ver README' if hechas == 0 else 'ejecuta validar_etiquetas.py'})")

    fallos = [m for ok, m in chk if not ok]
    if fallos:
        print(f"\n[todo] {len(fallos)} comprobacion(es) FALLARON. Corrigelas antes de entregar:")
        for m in fallos:
            print("   -", m)
    else:
        print("\n[todo] Todas las comprobaciones han pasado.")
    return not fallos


# ------------------------------------------------------------------ paso 6: tablas para el informe
def _md(df, formatos):
    cols = [c for c in formatos if c in df.columns]

    def f(v, c):
        if isinstance(v, str):
            return v.replace("|", "/")
        if pd.isna(v):
            return "-"
        return formatos[c].format(v)

    filas = [[f(v, c) for v, c in zip(r, cols)] for r in df[cols].values]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in filas])


def generar_tablas(args):
    paso("6. Generando tablas para el informe")
    partes = [f"# Tablas del E3 (generadas el {datetime.now():%Y-%m-%d %H:%M})\n"]
    t1 = E3 / "tabla_fase1_tecnicismos.csv"
    if t1.exists():
        d = pd.read_csv(t1)
        partes += ["## Fase 1 · palabras complejas (test; deep = media de semillas)\n",
                   _md(d, {"modelo": "", "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
                           "f1": "{:.3f}", "f1_std": "{:.3f}", "f1_ic95_s0": "{}", "f1_val": "{:.3f}",
                           "hiperparametros": ""}), ""]
    g1 = E3 / "grid_shallow_tecnicismos.csv"
    if g1.exists():
        g = pd.read_csv(g1, keep_default_na=False)
        cols = [c for c in ["modelo", "rasgos"] if c in g.columns]
        mejor = g.loc[g.groupby(cols)["f1_val"].idxmax()]
        partes += ["## Fase 1 · mejor combinacion de la rejilla shallow (F1 en validacion)\n",
                   _md(mejor, {"modelo": "", "rasgos": "", "ventana": "{}", "C": "{}", "class_weight": "", "f1_val": "{:.4f}"}), ""]
    t3 = E3 / "tabla_fase3_simplificacion.csv"
    if t3.exists():
        d = pd.read_csv(t3)
        partes += ["## Fase 3 · simplificacion (BLEU, 0-100)\n",
                   _md(d, {"modelo": "", "bleu_val": "{:.2f}", "bleu_test": "{:.2f}", "bleu_test_std": "{:.2f}",
                           "bleu_test_beam": "{:.2f}", "bleu_test_ic": "{}", "cobertura_emb": "{:.3f}",
                           "epocas": "{:.0f}", "n_semillas": "{:.0f}", "hiperparametros": ""}), ""]
    a3 = E3 / "analisis_fase3_salidas.csv"
    if a3.exists():
        partes += ["## Fase 3 · propiedades de las salidas (test)\n",
                   _md(pd.read_csv(a3), {"sistema": "", "palabras_media": "{:.1f}", "ratio_longitud_vs_ref": "{:.2f}",
                                         "trigramas_repetidos_pct": "{:.2f}", "cifras_conservadas_pct": "{:.1f}",
                                         "entidades_conservadas_pct": "{:.1f}"}), ""]
    g3 = E3 / "grid_seq2seq.csv"
    if g3.exists():
        partes += ["## Fase 3 · busqueda de hiperparametros del seq2seq (loss de validacion)\n",
                   _md(pd.read_csv(g3).sort_values("loss_val"),
                       {"hidden": "{}", "dropout": "{}", "lr": "{}", "loss_val": "{:.4f}", "epocas": "{}"}), ""]
    (E3 / "tablas_para_informe.md").write_text("\n".join(partes), encoding="utf-8")
    print(f"[todo] Escrito {E3 / 'tablas_para_informe.md'}")

    try:
        import torch
        torch_v, gpu = torch.__version__, torch.cuda.is_available()
    except Exception:
        torch_v, gpu = None, None
    meta = {"fecha": datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
            "torch": torch_v, "gpu": gpu,
            "ajustes": {k: v for k, v in vars(args).items()}, "semillas": SEMILLAS[:args.semillas],
            "tiempos_s": TIEMPOS}
    (E3 / "ejecucion.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Ejecuta E2 + E3 completo con comprobaciones")
    ap.add_argument("--fase", choices=["1", "3", "todo"], default="todo")
    ap.add_argument("--sin-humo", action="store_true", help="saltar la prueba de humo")
    ap.add_argument("--sin-bert", action="store_true", help="sin RoBERTa/BERTIN (si no hay GPU)")
    ap.add_argument("--sin-hp", action="store_true", help="sin busqueda de hiperparametros del seq2seq")
    ap.add_argument("--sin-externo", action="store_true", help="sin las variantes con dataset externo")
    ap.add_argument("--semillas", type=int, default=len(SEMILLAS), help="semillas por red (1-%d)" % len(SEMILLAS))
    ap.add_argument("--beam", type=int, default=BEAM, help="tamano del haz (0 = solo voraz)")
    ap.add_argument("--forzar-e2", action="store_true", help="regenerar el E2 aunque sea valido")
    ap.add_argument("--no-archivar", action="store_true", help="no mover resultados_e3/ anteriores")
    ap.add_argument("--solo-verificar", action="store_true", help="no entrena: verifica y genera tablas")
    args = ap.parse_args()
    args.semillas = max(1, min(args.semillas, len(SEMILLAS)))

    usar_externo = (not args.sin_externo) and externo_conectado()
    if not args.sin_externo and not usar_externo:
        print(f"[todo] AVISO: no se encuentra {ARCHIVO_EXTERNO}; se omiten las variantes con dataset externo.")

    if not args.solo_verificar:
        t0 = time.time()
        comprobar_entorno(args)
        paso_e2(args)
        if not args.no_archivar:
            archivar_resultados()
        paso_humo(args, usar_externo)
        paso_completo(args, usar_externo)
        print(f"\n[todo] Entrenamiento terminado en {(time.time() - t0) / 60:.1f} min")
    ok = verificar(args, usar_externo)
    generar_tablas(args)
    print("\n[todo] Resultados en resultados_e2/ y resultados_e3/. "
          "Siguiente paso: pegar resultados_e3/tablas_para_informe.md en el documento E3.")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
