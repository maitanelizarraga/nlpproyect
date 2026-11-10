import re
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
