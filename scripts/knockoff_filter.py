import numpy as np
from scipy import stats


def knockoff_threshold(scores, q):
    """
    tau = min{ c > 0 : (#{s_i <= -c} + 1) / max(#{s_i >= c}, 1) <= q }

    Returns np.inf if no valid threshold exists (nothing is selected).
    Candidates are sorted ascending so we return the most liberal (lowest) c
    that still controls FDR, maximising power.
    """
    scores = np.asarray(scores, dtype=float)
    abs_scores = np.abs(scores)
    candidates = np.unique(abs_scores[abs_scores > 0])
    candidates = np.sort(candidates)  # ascending order

    for c in candidates:
        n_neg = int(np.sum(scores <= -c))
        n_pos = int(np.sum(scores >= c))
        if (n_neg + 1) / max(n_pos, 1) <= q:
            return float(c)

    return np.inf


def knockoff_select(scores, q):
    """Return indices of texts declared as human-written (s_i > tau)."""
    tau = knockoff_threshold(scores, q)
    if np.isinf(tau):
        return np.array([], dtype=int)
    return np.where(np.asarray(scores, dtype=float) > tau)[0]


def symmetry_check(null_scores):
    """
    Characterise symmetry of the null (AI-text) score distribution.
    Under Assumption 1, the null scores should be symmetric around 0.
    KS test compares the empirical distribution of s_i to that of -s_i.
    A high p-value indicates we cannot reject symmetry.
    """
    s = np.asarray(null_scores, dtype=float)
    ks_stat, ks_pval = stats.ks_2samp(s, -s)
    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "median": float(np.median(s)),
        "fraction_positive": float(np.mean(s > 0)),
    }


def fdr_power(scores, labels, q):
    """
    Apply knockoff filter at target FDR level q and return empirical metrics.

    scores : list/array of signed statistics (larger => more likely human)
    labels : 1 = human, 0 = AI
    q      : target FDR level in (0, 1)
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    tau = knockoff_threshold(scores, q)

    if np.isinf(tau):
        return {
            "target_fdr": q,
            "actual_fdr": 0.0,
            "power": 0.0,
            "n_selected": 0,
            "n_false_discoveries": 0,
            "threshold": float("inf"),
        }

    selected = np.where(scores > tau)[0]
    n_false = int(np.sum(labels[selected] == 0))
    n_true  = int(np.sum(labels[selected] == 1))
    n_human = int(np.sum(labels == 1))

    return {
        "target_fdr": q,
        "actual_fdr": n_false / max(len(selected), 1),
        "power": n_true / max(n_human, 1),
        "n_selected": len(selected),
        "n_false_discoveries": n_false,
        "threshold": float(tau),
    }
