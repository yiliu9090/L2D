"""
Apply the knockoff filter to pre-computed detection scores.

Supports L2D, ImBD, and Likelihood results files.  L2D/ImBD outputs already
contain per-text signed statistics f(T_i, R_i).  Likelihood outputs contain
raw log-likelihoods; if a 'signed_predictions' key is present (written by
detect_likelihood.py --knockoff) those are used; otherwise the raw scores are
negated (AI text has higher likelihood, so -g(T) is positively correlated with
being human).

Sections produced:
  4.1  Symmetry Condition  -- KS test on null (AI) score distribution
  4.2  Empirical FDR Control -- actual FDR and power at each target level q

Usage examples:
  # L2D (signed stats already in file):
  python scripts/detect_knockoff.py \\
      --results_file exp_diverse/results/AcademicResearch_Llama-3-70B.l2d.json \\
      --method l2d \\
      --output_file exp_diverse/results/AcademicResearch_Llama-3-70B.knockoff_l2d.json

  # Likelihood with rewrite-based signed stats (run detect_likelihood.py --knockoff first):
  python scripts/detect_knockoff.py \\
      --results_file exp_diverse/results/AcademicResearch_Llama-3-70B.likelihood_knockoff.json \\
      --method likelihood \\
      --output_file exp_diverse/results/AcademicResearch_Llama-3-70B.knockoff_likelihood.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from knockoff_filter import symmetry_check, fdr_power


def extract_scores(data, method):
    """
    Return (scores, labels) where scores[i] is the signed knockoff statistic
    and labels[i] is 1 (human) or 0 (AI).

    Convention: larger score => more likely human.

    For L2D / ImBD the output already stores signed stats in
    predictions['real'] (human) and predictions['samples'] (AI).

    For Likelihood we prefer the 'signed_predictions' key written by
    detect_likelihood.py --knockoff (g(R_i) - g(T_i)).  Falling back to
    negating the raw log-likelihood is an approximation that omits the rewrite
    comparison and may weaken the symmetry guarantee.
    """
    if method in ("l2d", "imbd"):
        if "signed_predictions" in data:
            # Prefer proper knockoff stats W_i = f(T_i) - f(R_i) if available
            real    = data["signed_predictions"]["real"]
            samples = data["signed_predictions"]["samples"]
        else:
            real    = data["predictions"]["real"]
            samples = data["predictions"]["samples"]

    elif method == "likelihood":
        if "signed_predictions" in data:
            real    = data["signed_predictions"]["real"]
            samples = data["signed_predictions"]["samples"]
        else:
            # Fallback: negate raw log-likelihood (higher => AI => negate for human)
            print("[warn] 'signed_predictions' not found; negating raw likelihood scores. "
                  "Re-run detect_likelihood.py --knockoff for rewrite-based statistics.")
            real    = [-s for s in data["predictions"]["real"]]
            samples = [-s for s in data["predictions"]["samples"]]
    else:
        raise ValueError(f"Unknown method: {method}")

    scores = real + samples
    labels = [1] * len(real) + [0] * len(samples)
    return scores, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True,
                        help="JSON output from detect_l2d.py / detect_ImBD.py / detect_likelihood.py")
    parser.add_argument("--method", required=True,
                        choices=["l2d", "imbd", "likelihood"],
                        help="Detection method used to produce the results file")
    parser.add_argument("--output_file", required=True,
                        help="Where to save knockoff experiment results")
    parser.add_argument("--q_levels", nargs="+", type=float,
                        default=[0.05, 0.1, 0.2, 0.3, 0.5],
                        help="Target FDR levels q to evaluate")
    args = parser.parse_args()

    with open(args.results_file) as f:
        data = json.load(f)

    scores, labels = extract_scores(data, args.method)
    null_scores = [s for s, l in zip(scores, labels) if l == 0]

    sym = symmetry_check(null_scores)
    results = {
        "method":  args.method,
        "n_human": int(sum(labels)),
        "n_ai":    int(len(labels) - sum(labels)),
        "n_total": int(len(labels)),
        # Section 4.1: Symmetry Condition
        "symmetry": sym,
        # Section 4.2: Empirical FDR Control
        "fdr_control": {},
    }

    print(f"\n=== {args.method.upper()} Knockoff Experiment ===")
    print(f"Corpus: {results['n_human']} human + {results['n_ai']} AI = {results['n_total']} texts")
    print(f"\n--- Section 4.1: Symmetry Condition (null / AI-text statistics) ---")
    print(f"  mean={sym['mean']:.4f}  std={sym['std']:.4f}  "
          f"frac_positive={sym['fraction_positive']:.3f}  "
          f"KS p-value={sym['ks_pvalue']:.4f}")

    print(f"\n--- Section 4.2: Empirical FDR Control ---")
    print(f"{'Target FDR':>12} {'Actual FDR':>12} {'Power':>8} {'Selected':>10} {'Threshold':>12}")
    print("-" * 58)

    for q in sorted(args.q_levels):
        r = fdr_power(scores, labels, q)
        results["fdr_control"][str(q)] = r
        fdr_flag = " *" if r["actual_fdr"] > q + 1e-9 else "  "
        print(f"{q:>12.2f} {r['actual_fdr']:>12.3f}{fdr_flag} {r['power']:>8.3f} "
              f"{r['n_selected']:>10} {r['threshold']:>12.4f}")

    print("  (* marks violations of FDR guarantee -- should not appear)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output_file}")
