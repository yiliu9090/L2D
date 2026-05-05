# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import tqdm
import argparse
import json
from utils import load_data
from model import load_tokenizer, load_model
from metrics import get_roc_metrics, get_precision_recall_metrics

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).mean()
    log_likelihood = torch.nan_to_num(log_likelihood, nan=-1e4)
    
    return log_likelihood.item()

def score_text(scoring_tokenizer, scoring_model, text, device):
    """Return log-likelihood of a single text string."""
    tokenized = scoring_tokenizer(
        text, return_tensors="pt", padding=True, return_token_type_ids=False
    ).to(device)
    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits = scoring_model(**tokenized).logits[:, :-1]
    return get_likelihood(logits, labels)


def load_rewrite_file(rewrite_file, dataset_file, regen_number=4):
    """Load rewrite JSON; auto-derive path from dataset path if not given."""
    if not rewrite_file:
        rewrite_file = (
            dataset_file.replace("/data/", "/results/")
            + f".rewrite_{regen_number}.json"
        )
    with open(rewrite_file) as f:
        return json.load(f)


def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    name = 'likelihood'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Knockoff mode: compute signed statistic g(R_i) - g(T_i) per text.
    # Under the null (AI text), T_i and R_i are exchangeable, so the
    # statistic is symmetric around 0.  For human text, g(R_i) > g(T_i)
    # because the LLM rewrite R_i has higher likelihood than the original,
    # giving a positive signal.  These signed stats are stored in
    # 'signed_predictions' and consumed by detect_knockoff.py.
    # ------------------------------------------------------------------
    if args.knockoff:
        rewrite_data = load_rewrite_file(args.rewrite_file, args.dataset_file,
                                         args.regen_number)
        signed_original, signed_sampled = [], []
        for idx in tqdm.tqdm(range(n_samples), desc="Computing knockoff likelihood stats"):
            g_orig = score_text(scoring_tokenizer, scoring_model,
                                data["original"][idx], args.device)
            g_samp = score_text(scoring_tokenizer, scoring_model,
                                data["sampled"][idx], args.device)
            g_rewrite_orig = score_text(scoring_tokenizer, scoring_model,
                                        rewrite_data[idx]["rewrite_original"][0], args.device)
            g_rewrite_samp = score_text(scoring_tokenizer, scoring_model,
                                        rewrite_data[idx]["rewrite_sampled"][0], args.device)
            # g(R) - g(T): positive for human texts, ~0 for AI texts
            signed_original.append(g_rewrite_orig - g_orig)
            signed_sampled.append(g_rewrite_samp - g_samp)

        signed_predictions = {"real": signed_original, "samples": signed_sampled}
        fpr, tpr, roc_auc = get_roc_metrics(signed_original, signed_sampled)
        p, r, pr_auc = get_precision_recall_metrics(signed_original, signed_sampled)
        print(f"Knockoff likelihood signed-stat ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

        results_file = f"{args.output_file}.likelihood_knockoff.json"
        results = {
            "name": "likelihood_knockoff",
            "info": {"n_samples": n_samples},
            # signed_predictions used by detect_knockoff.py
            "signed_predictions": signed_predictions,
            # also expose raw signed stats as predictions for compatibility
            "predictions": signed_predictions,
            "metrics": {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr},
            "pr_metrics": {"pr_auc": pr_auc, "precision": p, "recall": r},
        }
        with open(results_file, "w") as fout:
            json.dump(results, fout, indent=2)
            print(f"Knockoff likelihood results written into {results_file}")
        return

    # ------------------------------------------------------------------
    # Standard mode: per-text log-likelihood (used for ROC AUC baseline).
    # ------------------------------------------------------------------
    eval_results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        original_crit = score_text(scoring_tokenizer, scoring_model,
                                   original_text, args.device)
        sampled_crit  = score_text(scoring_tokenizer, scoring_model,
                                   sampled_text,  args.device)
        eval_results.append({"original_crit": original_crit, "sampled_crit": sampled_crit})

    predictions = {'real': [x["original_crit"] for x in eval_results],
                   'samples': [x["sampled_crit"] for x in eval_results]}
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    results_file = f'{args.output_file}.{name}.json'
    results = {
        'name': f'{name}_threshold',
        'info': {'n_samples': n_samples},
        'predictions': predictions,
        'raw_results': eval_results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'loss': 1 - pr_auc
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout, indent=2)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt2")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_gpt2")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    # Knockoff mode: compute g(R_i) - g(T_i) signed statistics
    parser.add_argument('--knockoff', action='store_true',
                        help='Compute rewrite-based signed stats for knockoff filter')
    parser.add_argument('--rewrite_file', type=str, default='',
                        help='Path to .rewrite_N.json (auto-derived from dataset_file if empty)')
    parser.add_argument('--regen_number', type=int, default=4,
                        help='Rewrite count suffix used when auto-deriving rewrite_file path')
    args = parser.parse_args()

    experiment(args)