

import argparse
import os
import random

import numpy as np
import torch

from base_model import BaseModel
from load_data import DataLoader
from utils import select_gpu


class Options:
    pass


DATASET_CONFIG = {
    "family": (0.0036, 0.9990, 0.000017, 48, 5, 0.29, "relu", 20, 400),
    "umls": (0.0012, 0.9980, 0.000140, 64, 5, 0.01, "tanh", 10, 100),
    "WN18RR": (0.0030, 0.9940, 0.000140, 64, 5, 0.02, "idd", 50, 1000),
    "fb15k-237": (0.0009, 0.9938, 0.000080, 48, 5, 0.0391, "idd", 6, 400),
    "nell": (0.0011, 0.9938, 0.000089, 128, 64, 0.2593, "idd", 10, 400),
    "YAGO": (0.0010, 0.9950, 0.000100, 64, 2, 0.1946, "relu", 5, 200),
}


DATASET_RUN_CONFIG = {
    "family": {
        "epochs": 130,
        "eval_interval": 1,
        "fact_ratio": 0.80,
        "topk": 3007,
        "layers": 8,
        "batch_size": 8,
        "ppr_alpha": 0.25,
        "ppr_iterations": 20,
        "eval_path_samples": 1,
        "gpu": 0,
    },
    "umls": {
        "epochs": 130,
        "eval_interval": 1,
        "fact_ratio": 0.90,
        "topk": 135,
        "layers": 5,
        "batch_size": 16,
        "ppr_alpha": 0.2,
        "ppr_iterations": 20,
        "eval_path_samples": 1,
        "fm_max_weight": 0.08,
        "gpu": 0,
    },
    "WN18RR": {
        "epochs": 130,
        "eval_interval": 1,
        "fact_ratio": 0.80,
        "topk": 0.17,
        "layers": 8,
        "batch_size": 24,
        "ppr_alpha": 0.12,
        "ppr_iterations": 20,
        "eval_path_samples": 1,
        "fm_max_weight": 0.08,
        "gpu": 0,
    },
    "fb15k-237": {
        "epochs": 130,
        "eval_interval": 1,
        "fact_ratio": 0.99,
        "topk": 0.2,
        "layers": 7,
        "batch_size": 8,
        "ppr_alpha": 0.15,
        "ppr_iterations": 20,
        "eval_path_samples": 1,
        "gpu": 0,
        "fm_warmup": 5,
        "fm_ramp": 3,
        "fm_max_weight": 0.08,
        "remove_1hop_edges": True,
    },
    "nell": {
        "epochs": 130,
        "eval_interval": 1,
        "fact_ratio": 0.99,
        "topk": 0.05,
        "layers": 6,
        "batch_size": 8,
        "ppr_alpha": 0.12,
        "ppr_iterations": 20,
        "eval_path_samples": 1,
        "gpu": 0,
    },
    "YAGO": {
        "epochs": 130,
        "eval_interval": 1,
        "fact_ratio": 0.995,
        "topk": 0.01,
        "layers": 8,
        "batch_size": 16,
        "ppr_alpha": 0.15,
        "ppr_iterations": 20,
        "eval_path_samples": 1,
        "gpu": 0,
        "fm_warmup": 5,
        "fm_ramp": 3,
        "fm_max_weight": 0.12,
    },
}


RUNTIME_DEFAULTS = {
    "epochs": 130,
    "eval_interval": 1,
    "layers": 6,
    "topk": -1,
    "batch_size": -1,
    "fact_ratio": 0.96,
    "remove_1hop_edges": False,
    "ppr_alpha": 0.15,
    "ppr_iterations": 20,
    "fm_warmup": 0,
    "fm_ramp": 0,
    "fm_max_weight": 1.0,
    "eval_path_samples": 1,
    "gpu": -1,
}


def dataset_name(data_path):
    return os.path.basename(os.path.normpath(data_path))






def parse_args():
    parser = argparse.ArgumentParser(
        description="Candidate denoising VAE conditioning uniform-to-answer score FM"
    )
    parser.add_argument("data_path_pos", nargs="?", help="dataset path")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--topk", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--fact_ratio", type=float, default=None)
    parser.add_argument(
        "--remove_1hop_edges",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--ppr_alpha", type=float, default=None)
    parser.add_argument("--ppr_iterations", type=int, default=None)

    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--con_weight", type=float, default=1.0)
    parser.add_argument("--vae_rec_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument("--fm_weight", type=float, default=1.0)
    parser.add_argument("--fm_warmup", type=int, default=None)
    parser.add_argument("--fm_ramp", type=int, default=None)
    parser.add_argument("--fm_max_weight", type=float, default=None)
    parser.add_argument("--con_temperature", type=float, default=1.0)
    parser.add_argument("--con_negatives", type=int, default=64)
    parser.add_argument("--ode_steps", type=int, default=4)
    parser.add_argument("--eval_path_samples", type=int, default=None)
    parser.add_argument("--feature_mask_rate", type=float, default=0.30)
    parser.add_argument("--fm_source_power", type=float, default=2.0)
    parser.add_argument("--fm_source_fraction", type=float, default=0.50)
    parser.add_argument("--target_negative_mass", type=float, default=0.05)
    parser.add_argument("--unreached_score", type=float, default=-1e9)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--eval_split", choices=("valid", "test", "both"), default="both"
    )
    args = parser.parse_args()

    if args.data_path_pos and args.data_path:
        if os.path.normpath(args.data_path_pos) != os.path.normpath(args.data_path):
            parser.error("use either positional data_path or --data_path, not both")
    args.data_path = args.data_path or args.data_path_pos or "./data/WN18RR/"
    name = dataset_name(args.data_path)
    dataset_defaults = DATASET_RUN_CONFIG.get(name, {})
    for key, fallback in RUNTIME_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, dataset_defaults.get(key, fallback))
    if args.con_temperature <= 0:
        parser.error("--con_temperature must be positive")
    if args.ode_steps < 1 or args.eval_path_samples < 1 or args.layers < 1:
        parser.error("ODE steps, path samples, and layers must be positive")
    if args.con_negatives < 1:
        parser.error("--con_negatives must be positive")
    if not 0.0 < args.fact_ratio < 1.0:
        parser.error("--fact_ratio must be between zero and one")
    if not 0.0 < args.ppr_alpha < 1.0:
        parser.error("--ppr_alpha must be between zero and one")
    if args.ppr_iterations < 1:
        parser.error("--ppr_iterations must be positive")
    if not 0.0 <= args.feature_mask_rate < 1.0:
        parser.error("--feature_mask_rate must be in [0, 1)")
    if args.fm_source_power <= 0.0:
        parser.error("--fm_source_power must be positive")
    if not 0.0 <= args.fm_source_fraction <= 1.0:
        parser.error("--fm_source_fraction must be in [0, 1]")
    if not 0.0 < args.target_negative_mass < 0.5:
        parser.error("--target_negative_mass must be in (0, 0.5)")
    if args.fm_warmup < 0 or args.fm_ramp < 0:
        parser.error("--fm_warmup and --fm_ramp must be non-negative")
    return args


def build_options(args, loader):
    name = dataset_name(args.data_path)
    if name not in DATASET_CONFIG:
        raise ValueError(f"no configuration for dataset: {name}")
    lr, decay, weight_decay, hidden, attn, dropout, act, batch, topk = DATASET_CONFIG[name]
    opts = Options()
    opts.lr = lr
    opts.decay_rate = decay
    opts.lamb = weight_decay
    opts.hidden_dim = hidden
    opts.attn_dim = attn
    opts.dropout = dropout
    opts.act = act
    opts.n_batch = batch if args.batch_size <= 0 else args.batch_size
    opts.topk = topk if args.topk <= 0 else args.topk
    opts.n_layer = args.layers
    opts.seed = args.seed
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.device = torch.device(f"cuda:{args.gpu}")
    for key in (
        "rec_weight",
        "con_weight",
        "vae_rec_weight",
        "kl_weight",
        "fm_weight",
        "fm_warmup",
        "fm_ramp",
        "fm_max_weight",
        "con_temperature",
        "con_negatives",
        "ode_steps",
        "eval_path_samples",
        "feature_mask_rate",
        "fm_source_power",
        "fm_source_fraction",
        "target_negative_mass",
        "unreached_score",
        "grad_clip",
    ):
        setattr(opts, key, getattr(args, key))
    return opts


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("this implementation requires CUDA and torch_scatter")
    if args.gpu < 0:
        args.gpu = select_gpu()
    torch.cuda.set_device(args.gpu)

    name = dataset_name(args.data_path)
    preliminary = DATASET_CONFIG.get(name)
    if preliminary is None:
        raise ValueError(f"no configuration for dataset: {name}")
    default_batch = preliminary[7] if args.batch_size <= 0 else args.batch_size
    loader = DataLoader(
        args.data_path,
        n_batch=default_batch,
        fact_ratio=args.fact_ratio,
        remove_1hop_edges=args.remove_1hop_edges,
        ppr_alpha=args.ppr_alpha,
        ppr_iterations=args.ppr_iterations,
    )
    opts = build_options(args, loader)
    model = BaseModel(opts, loader)


    run_tag = f"candidate_vae_condscorefm_nbf{opts.n_layer}"
    if args.remove_1hop_edges:
        run_tag += "_remove1hop"
    checkpoint = args.checkpoint or os.path.join(
        args.data_path, "saveModel", f"{run_tag}_best.pt"
    )
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)


    if args.eval_only:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        saved = torch.load(checkpoint, map_location=f"cuda:{args.gpu}")
        model.model.load_state_dict(saved["model_state_dict"], strict=True)
        parts = []
        if args.eval_split in ("valid", "both"):
            metrics = model.evaluate_split("valid")
            parts.append(model.format_metrics("VALID", metrics))
        if args.eval_split in ("test", "both"):
            metrics = model.evaluate_split("test")
            parts.append(model.format_metrics("TEST", metrics))
        print("\t".join(parts))
        return

    best_mrr = -1.0
    best_epoch = -1
    best_valid = ""
    start_epoch = 1
    if args.resume:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"checkpoint not found for --resume: {checkpoint}")
        saved = torch.load(checkpoint, map_location=f"cuda:{args.gpu}")
        model.model.load_state_dict(saved["model_state_dict"], strict=True)
        if "optimizer_state_dict" in saved:
            model.optimizer.load_state_dict(saved["optimizer_state_dict"])
        if "scheduler_state_dict" in saved:
            model.scheduler.load_state_dict(saved["scheduler_state_dict"])
        best_epoch = int(saved.get("epoch", 0))
        best_mrr = float(saved.get("valid_mrr", -1.0))
        best_valid = str(saved.get("valid_str", ""))
        start_epoch = best_epoch + 1

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = model.train_batch(epoch)
        if epoch % args.eval_interval != 0:
            print(f"[EPOCH {epoch}] LOSS:{train_loss:.4f}")
            continue
        valid_metrics = model.evaluate_split("valid")
        valid_str = model.format_metrics("VALID", valid_metrics)
        epoch_line = f"[EPOCH {epoch}] LOSS:{train_loss:.4f}\t{valid_str}"
        print(epoch_line)
        if valid_metrics[0] > best_mrr:
            best_mrr = valid_metrics[0]
            best_epoch = epoch
            best_valid = valid_str
            torch.save(
                {
                    "model_state_dict": model.model.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),
                    "scheduler_state_dict": model.scheduler.state_dict(),
                    "epoch": epoch,
                    "valid_mrr": best_mrr,
                    "valid_str": valid_str,
                    "loss_weights": dict(model.loss_weights),
                },
                checkpoint,
            )

    if best_epoch < 0:
        raise RuntimeError("no validation checkpoint was produced")
    saved = torch.load(checkpoint, map_location=f"cuda:{args.gpu}")
    model.model.load_state_dict(saved["model_state_dict"], strict=True)
    test_metrics = model.evaluate_split("test")
    test_str = model.format_metrics("TEST", test_metrics)
    print(f"[FINAL BEST EPOCH {best_epoch}] {best_valid}\t{test_str}")


if __name__ == "__main__":
    main()
