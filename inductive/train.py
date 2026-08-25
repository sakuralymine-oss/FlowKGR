

import argparse
import os
import random

import numpy as np
import torch

from base_model import BaseModel
from load_data import DataLoader, InductiveDataLoader
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
    
    
    
    
    "WN18RR_v1": (0.0058, 0.9912, 0.000023, 64, 3, 0.2, "idd", 50, 0.8, 7),
    "WN18RR_v2": (0.0021, 0.9968, 0.000018, 64, 3, 0.4237, "relu", 20, 700, 5),
    "WN18RR_v3": (0.0030, 0.9940, 0.000035, 128, 8, 0.205, "relu", 24, 1000, 7),
    "WN18RR_v4": (0.0020, 0.9940, 0.000140, 64, 5, 0.1555, "idd", 24, 500, 6),
    "fb237_v1": (0.0010, 0.9950, 0.000081, 48, 5, 0.30, "relu", 20, 500, 3),
    "fb237_v2": (0.0087, 0.9937, 0.000025, 16, 5, 0.3265, "relu", 20, 200, 5),
    "fb237_v3": (0.0079, 0.9934, 0.000187, 48, 5, 0.4632, "relu", 10, 200, 7),
    "fb237_v4": (0.0010, 0.9950, 0.000186, 48, 5, 0.35, "relu", 20, 500, 7),
    "nell_v1": (0.0046, 0.9902, 0.000220, 32, 5, 0.3268, "relu", 20, 0.99, 6),
    "nell_v2": (0.0049, 0.9948, 0.000072, 16, 5, 0.3247, "relu", 100, 400, 3),
    "nell_v3": (0.0090, 0.9986, 0.000298, 16, 3, 0.1336, "relu", 75, 500, 4),
    "nell_v4": (0.0011, 0.9938, 0.000089, 128, 64, 0.20, "idd", 100, 200, 4),
}


DATASET_RUN_CONFIG = {
    "family": {
        "epochs": 250,
        "eval_interval": 1,
        "fact_ratio": 0.80,
        "topk": 3007,
        "layers": 3,
        "batch_size": 8,
        "ppr_alpha": 0.25,
        "ppr_iterations": 20,
        "eval_path_samples": 1,
        "gpu": 0,
    },
    "umls": {
        "epochs": 300,
        "eval_interval": 1,
        "fact_ratio": 0.90,
        "topk": 135,
        "layers": 6,
        "batch_size": 16,
        "ppr_alpha": 0.2,
        "ppr_iterations": 20,
        "eval_path_samples": 1,
        "fm_max_weight": 0.08,
        "gpu": 0,
    },
    "WN18RR": {
        "epochs": 40,
        "eval_interval": 1,
        "fact_ratio": 0.96,
        "topk": 0.99,
        "layers": 7,
        "batch_size": 24,
        "ppr_alpha": 0.15,
        "ppr_iterations": 30,
        "eval_path_samples": 1,
        "fm_max_weight": 0.08,
        "gpu": 0,
    },
    "fb15k-237": {
        "epochs": 40,
        "eval_interval": 1,
        "fact_ratio": 0.99,
        "topk": 0.07,
        "layers": 3,
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
        "epochs": 40,
        "eval_interval": 1,
        "fact_ratio": 0.99,
        "topk": 0.03,
        "layers": 3,
        "batch_size": 8,
        "ppr_alpha": 0.12,
        "ppr_iterations": 20,
        "eval_path_samples": 1,
        "gpu": 0,
        "fm_warmup": 5,
        "fm_ramp": 5,
        "fm_max_weight": 0.12,
    },
    "YAGO": {
        "epochs": 150,
        "eval_interval": 1,
        "fact_ratio": 0.995,
        "topk": 0.01,
        "layers": 3,
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
    "epochs": 80,
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


def base_dataset_name(name):
    
    if name.startswith("WN18RR_v"):
        return "WN18RR"
    if name.startswith("fb237_v"):
        return "fb15k-237"
    if name.startswith("nell_v"):
        return "nell"
    return name


def config_dataset_name(name):
    
    return name if name in DATASET_CONFIG else base_dataset_name(name)






def parse_args():
    parser = argparse.ArgumentParser(
        description="Candidate denoising VAE conditioning uniform-to-answer score FM"
    )
    parser.add_argument("data_path_pos", nargs="?", help="dataset path")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
        "--inductive",
        action="store_true",
        help=(
            "fully inductive GraIL/DiffusionE protocol; expects data_path and "
            "data_path+'_ind' with disjoint entity sets"
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--topk", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--fact_ratio", type=float, default=None)
    parser.add_argument(
        "--remove_1hop_edges",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--ppr_alpha", type=float, default=None)
    parser.add_argument("--ppr_iterations", type=int, default=None)
    parser.add_argument(
        "--local_relation_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "add a query-gated ID-free relation profile of each candidate's "
            "observed local neighborhood"
        ),
    )

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
    parser.add_argument("--con_hard_warmup_fraction", type=float, default=0.20)
    parser.add_argument("--con_hard_ramp_fraction", type=float, default=0.30)
    parser.add_argument("--con_hard_final_ratio", type=float, default=0.50)
    parser.add_argument("--top10_loss_weight", type=float, default=0.0)
    parser.add_argument("--top10_margin", type=float, default=0.10)
    parser.add_argument("--top10_warmup_fraction", type=float, default=0.20)
    parser.add_argument("--top10_ramp_fraction", type=float, default=0.20)
    parser.add_argument("--ode_steps", type=int, default=4)
    parser.add_argument("--eval_path_samples", type=int, default=None)
    parser.add_argument(
        "--ema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="evaluate and checkpoint an exponential moving average of model weights",
    )
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--feature_mask_rate", type=float, default=0.30)
    parser.add_argument(
        "--mask_rate_exp",
        action="store_true",
        help=(
            "parameter-study mode for feature_mask_rate; only appends the "
            "mask rate to checkpoint/result names so different rates do not "
            "overwrite each other"
        ),
    )
    parser.add_argument("--fm_source_power", type=float, default=2.0)
    parser.add_argument("--fm_source_fraction", type=float, default=0.50)
    parser.add_argument("--target_negative_mass", type=float, default=0.05)
    parser.add_argument(
        "--deterministic_vae",
        action="store_true",
        help="strict deterministic denoising encoder: direct z output, no sampling or KL",
    )
    parser.add_argument(
        "--static_scorer",
        action="store_true",
        help="replace ODE/FM with one vector-field evaluation at X=0 and tau=0",
    )
    parser.add_argument("--disable_fm", action="store_true")
    parser.add_argument(
        "--ablate_direct_predictor",
        action="store_true",
        help=(
            "formal w/o-FM ablation: replace score-space FM/ODE with a "
            "parameter-matched set-aware direct endpoint predictor trained "
            "against the same answer target"
        ),
    )
    parser.add_argument(
        "--ablate_plain_encoder",
        action="store_true",
        help=(
            "formal w/o-VAE ablation: replace the candidate VAE with a "
            "deterministic encoder using the same candidate inputs"
        ),
    )
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
    eval_path_samples_was_explicit = args.eval_path_samples is not None
    name = dataset_name(args.data_path)
    config_name = config_dataset_name(name)
    dataset_defaults = dict(DATASET_RUN_CONFIG.get(base_dataset_name(name), {}))
    split_config = DATASET_CONFIG.get(config_name)
    if args.inductive and split_config is not None and len(split_config) == 10:
        dataset_defaults.update(
            layers=split_config[9],
            batch_size=split_config[7],
            topk=split_config[8],
        )
    for key, fallback in RUNTIME_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, dataset_defaults.get(key, fallback))
    if args.inductive:
        args.remove_1hop_edges = False
        if not eval_path_samples_was_explicit:
            args.eval_path_samples = 8
    if args.ema is None:
        args.ema = bool(args.inductive)
    if not args.inductive:
        
        
        args.con_hard_warmup_fraction = 0.0
        args.con_hard_ramp_fraction = 0.0
        args.con_hard_final_ratio = 1.0

    if args.dropout is not None and not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.con_temperature <= 0:
        parser.error("--con_temperature must be positive")
    if args.ode_steps < 1 or args.eval_path_samples < 1 or args.layers < 1:
        parser.error("ODE steps, path samples, and layers must be positive")
    if args.con_negatives < 1:
        parser.error("--con_negatives must be positive")
    if not 0.0 <= args.con_hard_warmup_fraction <= 1.0:
        parser.error("--con_hard_warmup_fraction must be in [0, 1]")
    if not 0.0 <= args.con_hard_ramp_fraction <= 1.0:
        parser.error("--con_hard_ramp_fraction must be in [0, 1]")
    if args.con_hard_warmup_fraction + args.con_hard_ramp_fraction > 1.0:
        parser.error("hard-negative warmup and ramp fractions must sum to at most 1")
    if not 0.0 <= args.con_hard_final_ratio <= 1.0:
        parser.error("--con_hard_final_ratio must be in [0, 1]")
    if args.top10_loss_weight < 0.0:
        parser.error("--top10_loss_weight must be non-negative")
    if args.top10_margin < 0.0:
        parser.error("--top10_margin must be non-negative")
    if not 0.0 <= args.top10_warmup_fraction <= 1.0:
        parser.error("--top10_warmup_fraction must be in [0, 1]")
    if not 0.0 <= args.top10_ramp_fraction <= 1.0:
        parser.error("--top10_ramp_fraction must be in [0, 1]")
    if args.top10_warmup_fraction + args.top10_ramp_fraction > 1.0:
        parser.error("top10 warmup and ramp fractions must sum to at most 1")
    if not 0.0 < args.ema_decay < 1.0:
        parser.error("--ema_decay must be in (0, 1)")
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
    if args.static_scorer and args.disable_fm:
        parser.error("--static_scorer and --disable_fm cannot be used together")
    if args.ablate_direct_predictor and (args.static_scorer or args.disable_fm):
        parser.error(
            "--ablate_direct_predictor cannot be combined with "
            "--static_scorer or --disable_fm"
        )
    if args.ablate_direct_predictor and args.deterministic_vae:
        parser.error(
            "run one formal ablation at a time; do not combine "
            "--ablate_direct_predictor with --deterministic_vae"
        )
    if args.ablate_plain_encoder and (
        args.ablate_direct_predictor
        or args.deterministic_vae
        or args.static_scorer
        or args.disable_fm
    ):
        parser.error(
            "run one formal ablation at a time; do not combine "
            "--ablate_plain_encoder with another ablation flag"
        )
    if args.static_scorer and args.deterministic_vae:
        parser.error(
            "run one formal ablation at a time; do not combine "
            "--static_scorer with --deterministic_vae"
        )
    if args.local_relation_context and not args.inductive:
        parser.error("--local_relation_context currently requires --inductive")
    return args






def build_options(args, loader):
    name = dataset_name(args.data_path)
    config_name = config_dataset_name(name)
    if config_name not in DATASET_CONFIG:
        raise ValueError(f"no configuration for dataset: {name}")
    config = DATASET_CONFIG[config_name]
    lr, decay, weight_decay, hidden, attn, dropout, act, batch, topk = config[:9]
    opts = Options()
    opts.lr = lr
    opts.decay_rate = decay
    opts.lamb = weight_decay
    opts.hidden_dim = hidden
    opts.attn_dim = attn
    opts.dropout = dropout if args.dropout is None else args.dropout
    opts.act = act
    opts.n_batch = batch if args.batch_size <= 0 else args.batch_size
    opts.topk = topk if args.topk <= 0 else args.topk
    opts.n_layer = args.layers
    opts.epochs = args.epochs
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
        "con_hard_warmup_fraction",
        "con_hard_ramp_fraction",
        "con_hard_final_ratio",
        "top10_loss_weight",
        "top10_margin",
        "top10_warmup_fraction",
        "top10_ramp_fraction",
        "ode_steps",
        "eval_path_samples",
        "ema",
        "ema_decay",
        "feature_mask_rate",
        "fm_source_power",
        "fm_source_fraction",
        "target_negative_mass",
        "deterministic_vae",
        "static_scorer",
        "disable_fm",
        "ablate_direct_predictor",
        "ablate_plain_encoder",
        "inductive",
        "local_relation_context",
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
    config_name = config_dataset_name(name)
    preliminary = DATASET_CONFIG.get(config_name)
    if preliminary is None:
        raise ValueError(f"no configuration for dataset: {name}")
    default_batch = preliminary[7] if args.batch_size <= 0 else args.batch_size
    if args.inductive:
        loader = InductiveDataLoader(
            args.data_path,
            n_batch=default_batch,
            ppr_alpha=args.ppr_alpha,
            ppr_iterations=args.ppr_iterations,
        )
    else:
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
    if args.inductive:
        run_tag += "_inductive"
        run_tag += "_qrelgru_multidepth"
    if args.inductive:
        run_tag += "_negcurr"
        weight_tag = f"{args.top10_loss_weight:.2f}".replace(".", "p")
        margin_tag = f"{args.top10_margin:.2f}".replace(".", "p")
        run_tag += f"_h10b{weight_tag}m{margin_tag}"
    if args.local_relation_context:
        run_tag += "_localrel"
    if args.ema:
        ema_tag = f"{args.ema_decay:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        run_tag += f"_ema{ema_tag}_mc{args.eval_path_samples}"
    if args.dropout is not None:
        dropout_tag = f"{args.dropout:.2f}".replace(".", "p")
        run_tag += f"_drop{dropout_tag}"
    if args.mask_rate_exp:
        mask_tag = f"{args.feature_mask_rate:.2f}".replace(".", "p")
        run_tag += f"_mask{mask_tag}"
    if args.deterministic_vae:
        run_tag += "_detDenoise"
    if args.static_scorer:
        run_tag += "_static"
    if args.disable_fm:
        run_tag += "_noFM"
    if args.ablate_direct_predictor:
        run_tag += "_directPredictor"
    if args.ablate_plain_encoder:
        run_tag += "_plainEncoder"
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
        model.load_checkpoint_weights(saved, resume=False)
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
        model.load_checkpoint_weights(saved, resume=True)
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
            checkpoint_payload = {
                    "optimizer_state_dict": model.optimizer.state_dict(),
                    "scheduler_state_dict": model.scheduler.state_dict(),
                    "epoch": epoch,
                    "valid_mrr": best_mrr,
                    "valid_str": valid_str,
                    "loss_weights": dict(model.loss_weights),
                }
            checkpoint_payload.update(model.checkpoint_weights())
            torch.save(checkpoint_payload, checkpoint)

    if best_epoch < 0:
        raise RuntimeError("no validation checkpoint was produced")
    saved = torch.load(checkpoint, map_location=f"cuda:{args.gpu}")
    model.load_checkpoint_weights(saved, resume=True)
    test_metrics = model.evaluate_split("test")
    test_str = model.format_metrics("TEST", test_metrics)
    print(f"[FINAL BEST EPOCH {best_epoch}] {best_valid}\t{test_str}")


if __name__ == "__main__":
    main()
