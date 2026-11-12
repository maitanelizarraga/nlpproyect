import re
import numpy as np
from config import BOOTSTRAP_N
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def normalizar(texto):
    """Minúsculas y signos de puntuación separados. Se aplica igual a hipótesis y referencias
    para que BLEU sea comparable entre TODOS los sistemas (shallow y deep)."""
    return " ".join(re.findall(r"\w+|[^\w\s]", str(texto).lower()))


def bleu(hipotesis, referencias):
    """BLEU de corpus (sacrebleu, 0-100) con los textos ya normalizados."""
    from sacrebleu.metrics import BLEU
    hip = [normalizar(h) for h in hipotesis]
    ref = [normalizar(r) for r in referencias]
    return BLEU(tokenize="none").corpus_score(hip, [ref]).score


def metricas_clasificacion(y_real, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_real, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
    }


# ------------------------------------------------------------------ rigor experimental
def agregar_semillas(runs, std_keys=()):
    """Combina los resultados de varias semillas: media de cada metrica numerica y, para las claves
    de std_keys, desviacion tipica (ddof=1) en '<clave>_std'."""
    out = dict(runs[0])
    for k, v in runs[0].items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            vals = np.array([r[k] for r in runs if r.get(k) is not None], dtype=float)
            out[k] = float(vals.mean())
            if k in std_keys:
                out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    out["n_semillas"] = len(runs)
    return out


def ic_bootstrap_f1(y_real, y_pred, n=BOOTSTRAP_N, seed=0):
    """Intervalo de confianza al 95 % del F1 (clase 1) por bootstrap sobre tokens.
    Nota: remuestrea tokens sueltos (ignora que pertenecen a documentos), asi que es algo optimista."""
    y = np.asarray(y_real).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    rng, f1s = np.random.default_rng(seed), []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yt, pt = y[idx], p[idx]
        tp, fp, fn = (yt & pt).sum(), (~yt & pt).sum(), (yt & ~pt).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def bleu_ic(hipotesis, referencias, n=BOOTSTRAP_N, seed=0):
    """Intervalo de confianza al 95 % del BLEU por bootstrap sobre frases (de test)."""
    from sacrebleu.metrics import BLEU
    hip = [normalizar(h) for h in hipotesis]
    ref = [normalizar(r) for r in referencias]
    rng, medida, vals = np.random.default_rng(seed), BLEU(tokenize="none"), []
    for _ in range(n):
        idx = rng.integers(0, len(hip), len(hip))
        vals.append(medida.corpus_score([hip[i] for i in idx], [[ref[i] for i in idx]]).score)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"[{lo:.2f}, {hi:.2f}]"
