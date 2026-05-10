from ImBD.dataset import CustomDataset_rewrite
from ImBD.spo import ComputeScore
from ImBD.engine import run
import torch
from torch.utils.data import Subset
import argparse
import numpy as np
import random
import json
import os
import tqdm as tqdm_module

from utils import load_data
from metrics import get_roc_metrics, get_precision_recall_metrics


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_rewrite_file(rewrite_file, eval_dataset, regen_number):
    if not rewrite_file:
        rewrite_file = (
            eval_dataset.replace("/data/", "/results/")
            + f".rewrite_{regen_number}.json"
        )
    with open(rewrite_file) as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('-a', type=int, default=1, help="accumulation steps")
    parser.add_argument('--task_name', type=str, default="ai_detection_500")
    parser.add_argument('--epochs', type=int, default=2, help="finetuning epochs")
    parser.add_argument('-ebt', action="store_true", help="Evaluate model before tuning")
    parser.add_argument('--datanum', type=int, default=500, help="num of training data")
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--eval_after_train', action="store_true")
    parser.add_argument('--SPOtrained', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--from_pretrained', type=str)
    parser.add_argument('--save_trained', action="store_true")
    parser.add_argument('--eval_dataset', type=str, default="./exp_prompt/data/xsum_gpt-4o_rewrite")
    parser.add_argument('--output_file', type=str, default="./exp_prompt/results/")
    parser.add_argument('--base_model', type=str, default="gemma-9b-instruct")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--train_dataset', type=str, default='./exp_prompt/data/squad_gpt-4o_rewrite')
    parser.add_argument('--device', type=str, default='cuda')
    # Knockoff mode: compute W_i = f(T_i) - f(R_i) signed statistics
    parser.add_argument('--knockoff', action='store_true',
                        help='Compute signed knockoff stats f(T_i)-f(R_i) instead of raw scores')
    parser.add_argument('--rewrite_file', type=str, default='',
                        help='Path to .rewrite_N.json (auto-derived from eval_dataset if empty)')
    parser.add_argument('--regen_number', type=int, default=4,
                        help='Rewrite count suffix used when auto-deriving rewrite_file path')
    args = parser.parse_args()

    set_seed(args.seed)
    SPOtrained = True if args.SPOtrained == "True" else False
    print(f"Running with args: {args}")

    # Load scoring model (needed for all modes)
    model = ComputeScore(
        args.base_model, args.base_model,
        SPOtrained=SPOtrained, SPO_beta=args.beta,
        cache_dir=args.cache_dir,
    )
    if args.from_pretrained:
        print(f"Loading ckpt from {args.from_pretrained}...")
        model.from_pretrained(args.from_pretrained)

    # ------------------------------------------------------------------
    # Knockoff mode: compute W_i = f(T_i) - f(R_i) per text.
    # Under the null (AI text), T_i and R_i are exchangeable, so W_i is
    # symmetric around 0.  For human text, f(T_i) > f(R_i) since the SPO
    # model scores the original human text higher than its LLM rewrite.
    # ------------------------------------------------------------------
    if args.knockoff:
        data = load_data(args.eval_dataset)
        n_samples = len(data["sampled"])
        rewrite_data = load_rewrite_file(args.rewrite_file, args.eval_dataset, args.regen_number)

        model.eval()
        signed_original, signed_sampled = [], []
        raw_records = {"original": [], "sampled": []}
        for idx in tqdm_module.tqdm(range(n_samples), desc="Computing IMBD knockoff stats"):
            f_T_orig = model.score_text(data["original"][idx])
            f_R_orig = model.score_text(rewrite_data[idx]["rewrite_original"][0])
            signed_original.append(f_T_orig - f_R_orig)
            raw_records["original"].append({"f_T": f_T_orig, "f_R": f_R_orig})

            f_T_samp = model.score_text(data["sampled"][idx])
            f_R_samp = model.score_text(rewrite_data[idx]["rewrite_sampled"][0])
            signed_sampled.append(f_T_samp - f_R_samp)
            raw_records["sampled"].append({"f_T": f_T_samp, "f_R": f_R_samp})

        signed_predictions = {"real": signed_original, "samples": signed_sampled}
        fpr, tpr, roc_auc = get_roc_metrics(signed_original, signed_sampled)
        p, r, pr_auc = get_precision_recall_metrics(signed_original, signed_sampled)
        print(f"IMBD knockoff ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

        results_file = f"{args.output_file}.imbd_knockoff.json"
        results = {
            "name": "imbd_knockoff",
            "info": {"n_samples": n_samples},
            "signed_predictions": signed_predictions,
            "predictions": signed_predictions,
            # raw_records: per-text f(T_i) and f(R_i) before differencing
            # "original" = human texts, "sampled" = AI texts
            "raw_records": raw_records,
            "metrics": {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr},
            "pr_metrics": {"pr_auc": pr_auc, "precision": p, "recall": r},
        }
        os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
        with open(results_file, "w") as fout:
            json.dump(results, fout, indent=2)
        print(f"IMBD knockoff results written into {results_file}")
        exit(0)

    # ------------------------------------------------------------------
    # Standard training / eval mode
    # ------------------------------------------------------------------
    train_data = CustomDataset_rewrite(data_json_dir=args.train_dataset)
    val_data = CustomDataset_rewrite(data_json_dir=args.eval_dataset)

    subset_indices = torch.randperm(len(train_data))[:args.datanum]
    train_subset = Subset(train_data, subset_indices)
    print(len(train_subset))
    print(len(val_data))

    run(
        model,
        [train_subset, val_data],
        DEVICE=args.device,
        ckpt_dir=f"./scripts/ImBD/ckpt/{args.task_name}_spo_lr_{args.lr}_beta_{args.beta}_a_{args.a}",
        args=args,
    )
