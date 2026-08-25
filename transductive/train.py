

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


def _parse_int_list(text):
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("expected a comma-separated list of positive integers")
    return values


def _parse_float_list(text):
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("expected a comma-separated list of numbers")
    return values


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
    parser.add_argument(
        "--fm_condition_hidden",
        action="store_true",
        help=(
            "condition score FM directly on the raw candidate GNN state h_i; "
            "bypasses the VAE, query-context input, and entity-state input"
        ),
    )
    parser.add_argument(
        "--analysis_all",
        action="store_true",
        help="run all five checkpoint-only analysis experiments and exit",
    )
    parser.add_argument(
        "--analysis_paper_all",
        action="store_true",
        help=(
            "run the five paper-facing analyses added for latent states, "
            "flow difficulty, incomplete-evidence robustness, efficiency, "
            "and case study; training behavior is unchanged"
        ),
    )
    parser.add_argument("--analyze_latent_states", action="store_true")
    parser.add_argument("--analyze_flow_difficulty", action="store_true")
    parser.add_argument("--analyze_robustness", action="store_true")
    parser.add_argument("--analyze_efficiency", action="store_true")
    parser.add_argument("--analyze_case_study", action="store_true")
    parser.add_argument("--analyze_multi_answer_case", action="store_true")
    parser.add_argument("--analyze_vae_sources", action="store_true")
    parser.add_argument("--multi_case_topk", type=int, default=10)
    parser.add_argument(
        "--multi_case_query_index",
        type=int,
        default=None,
        help="optional forward multi-answer dataset query index",
    )
    parser.add_argument("--analyze_path_samples", action="store_true")
    parser.add_argument("--analyze_latent", action="store_true")
    parser.add_argument("--analyze_flow", action="store_true")
    parser.add_argument("--analyze_ode_steps", action="store_true")
    parser.add_argument("--analyze_candidate_budget", action="store_true")
    parser.add_argument(
        "--analysis_split", choices=("valid", "test"), default="test"
    )
    parser.add_argument(
        "--path_sample_values", type=str, default="1,2,4,8"
    )
    parser.add_argument(
        "--flow_tau_values", type=str, default="0,0.25,0.5,0.75,1"
    )
    parser.add_argument(
        "--ode_step_values", type=str, default="1,2,4,8"
    )
    parser.add_argument(
        "--candidate_topk_values", type=str, default="0.03,0.05,0.07,0.10"
    )
    parser.add_argument("--analysis_file", type=str, default=None)
    parser.add_argument("--latent_sample_count", type=int, default=8)
    parser.add_argument("--latent_analysis_topk", type=int, default=10)
    parser.add_argument("--difficulty_max_depth", type=int, default=8)
    parser.add_argument(
        "--robustness_drop_values", type=str, default="0,0.1,0.2,0.3,0.4"
    )
    parser.add_argument(
        "--robustness_seeds", type=str, default="1234,2234,3234"
    )
    parser.add_argument("--case_sample_count", type=int, default=3)
    parser.add_argument("--case_topk", type=int, default=5)
    parser.add_argument(
        "--case_query_indices",
        type=str,
        default=None,
        help=(
            "optional comma-separated dataset query indices; when omitted, "
            "one representative single-answer and one multi-answer forward "
            "query are selected by median deterministic FM gain"
        ),
    )
    parser.add_argument("--unreached_score", type=float, default=-1e9)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--perf_file", type=str, default=None)
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
    if args.fm_condition_hidden:
        
        
        args.eval_path_samples = 1

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
    if args.latent_sample_count < 2:
        parser.error("--latent_sample_count must be at least two")
    if args.multi_case_topk < 1:
        parser.error("--multi_case_topk must be positive")
    if args.latent_analysis_topk < 1 or args.case_topk < 1:
        parser.error("analysis top-k values must be positive")
    if args.difficulty_max_depth < 1 or args.case_sample_count < 1:
        parser.error("difficulty depth and case sample count must be positive")
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
    if args.fm_condition_hidden and (
        args.ablate_plain_encoder
        or args.ablate_direct_predictor
        or args.deterministic_vae
        or args.static_scorer
        or args.disable_fm
    ):
        parser.error(
            "run one formal ablation at a time; do not combine "
            "--fm_condition_hidden with another ablation flag"
        )
    if args.static_scorer and args.deterministic_vae:
        parser.error(
            "run one formal ablation at a time; do not combine "
            "--static_scorer with --deterministic_vae"
        )
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
        "deterministic_vae",
        "static_scorer",
        "disable_fm",
        "ablate_direct_predictor",
        "ablate_plain_encoder",
        "fm_condition_hidden",
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

    if any(
        (
            args.analysis_all,
            args.analysis_paper_all,
            args.analyze_path_samples,
            args.analyze_latent,
            args.analyze_flow,
            args.analyze_ode_steps,
            args.analyze_candidate_budget,
            args.analyze_latent_states,
            args.analyze_flow_difficulty,
            args.analyze_robustness,
            args.analyze_efficiency,
            args.analyze_case_study,
            args.analyze_multi_answer_case,
            args.analyze_vae_sources,
        )
    ):
        raise ValueError("diagnostic modes have been removed")

    run_tag = f"candidate_vae_condscorefm_nbf{opts.n_layer}"
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
    if args.fm_condition_hidden:
        run_tag += "_hCondition"
    if args.remove_1hop_edges:
        run_tag += "_remove1hop"
    checkpoint = args.checkpoint or os.path.join(
        args.data_path, "saveModel", f"{run_tag}_best.pt"
    )
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    perf_file = args.perf_file or os.path.join(
        project_root, "results", name, f"{run_tag}_results.txt"
    )
    os.makedirs(os.path.dirname(os.path.abspath(perf_file)), exist_ok=True)

    config = (
        f"dataset={name} lr={opts.lr:.4g} decay={opts.decay_rate:.4g} "
        f"wd={opts.lamb:.6g} dim={opts.hidden_dim} attn={opts.attn_dim} "
        f"layers={opts.n_layer} batch={opts.n_batch} dropout={opts.dropout:.4f} "
        f"act={opts.act} one_shot_ppr_topk={opts.topk} "
        f"fact_ratio={args.fact_ratio:.4f} remove_1hop_edges={args.remove_1hop_edges} "
        f"ppr_alpha={args.ppr_alpha:.3f} ppr_iterations={args.ppr_iterations} "
        f"loss={('rank+con+score_fm' if (opts.ablate_plain_encoder or opts.fm_condition_hidden) else 'rank+con+vae_rec+' + ('kl+direct_endpoint' if opts.ablate_direct_predictor else ('kl+static_score' if opts.static_scorer else 'kl+score_fm')))} "
        f"weights=({opts.rec_weight},{opts.con_weight},{opts.vae_rec_weight},"
        f"{opts.kl_weight},{opts.fm_weight}) "
        f"mask={opts.feature_mask_rate:.2f} fm_source_power={opts.fm_source_power:.2f} "
        f"fm_source_fraction={opts.fm_source_fraction:.2f} "
        f"target_negative_mass={opts.target_negative_mass:.3f} "
        f"fm_warmup={opts.fm_warmup} fm_ramp={opts.fm_ramp} "
        f"fm_max_weight={opts.fm_max_weight} ode_steps={opts.ode_steps} "
        f"detVAE={opts.deterministic_vae} "
        f"staticScorer={opts.static_scorer} noFM={opts.disable_fm} "
        f"directPredictor={opts.ablate_direct_predictor} "
        f"plainEncoder={opts.ablate_plain_encoder} "
        f"hCondition={opts.fm_condition_hidden}\n"
    )
    analysis_requested = (
        args.analysis_all
        or args.analysis_paper_all
        or args.analyze_path_samples
        or args.analyze_latent
        or args.analyze_flow
        or args.analyze_ode_steps
        or args.analyze_candidate_budget
        or args.analyze_latent_states
        or args.analyze_flow_difficulty
        or args.analyze_robustness
        or args.analyze_efficiency
        or args.analyze_case_study
        or args.analyze_multi_answer_case
        or args.analyze_vae_sources
    )
    if analysis_requested:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"checkpoint not found for analysis: {checkpoint}"
            )
        saved = torch.load(checkpoint, map_location=f"cuda:{args.gpu}")
        model.model.load_state_dict(saved["model_state_dict"], strict=True)
        split = args.analysis_split
        blocks = []

        if args.analysis_all or args.analyze_path_samples:
            sample_counts = _parse_int_list(args.path_sample_values)
            rows = model.evaluate_path_sample_sweep(split, sample_counts)
            blocks.append(
                model.format_analysis_rows(
                    f"{split.upper()} PATH-SAMPLE SWEEP",
                    rows,
                    ["samples", "MRR", "H1", "H10", "time", "peak_gb"],
                )
            )

        if args.analysis_all or args.analyze_latent:
            result = model.evaluate_latent_condition_variants(split)
            rows = [
                {"variant": name, **result[name]}
                for name in ("FullZ", "ZeroZ", "ShuffledZ")
            ]
            blocks.append(
                model.format_analysis_rows(
                    f"{split.upper()} LATENT-CONDITION DIAGNOSTIC",
                    rows,
                    ["variant", "MRR", "H1", "H10"],
                )
            )

        if args.analysis_all or args.analyze_flow:
            tau_values = _parse_float_list(args.flow_tau_values)
            rows, elapsed = model.evaluate_flow_evolution(split, tau_values)
            blocks.append(
                model.format_analysis_rows(
                    f"{split.upper()} FLOW-EVOLUTION",
                    rows,
                    [
                        "tau",
                        "MRR",
                        "MR",
                        "H1",
                        "H3",
                        "H10",
                        "MoveFromX0",
                        "DistToX1",
                        "GoldProb",
                        "GoldMargin",
                        "GoldMarginCov",
                    ],
                )
                + f"\nTIME\t{elapsed:.6f}"
            )

        if args.analysis_all or args.analyze_ode_steps:
            step_counts = _parse_int_list(args.ode_step_values)
            rows = model.evaluate_ode_step_sweep(split, step_counts)
            blocks.append(
                model.format_analysis_rows(
                    f"{split.upper()} ODE-STEP SWEEP",
                    rows,
                    ["steps", "MRR", "H1", "H10", "time", "peak_gb"],
                )
            )

        if args.analysis_all or args.analyze_candidate_budget:
            budgets = _parse_float_list(args.candidate_topk_values)
            if any(value <= 0.0 for value in budgets):
                raise ValueError("candidate_topk_values must all be positive")
            rows = model.evaluate_candidate_budget_sweep(split, budgets)
            blocks.append(
                model.format_analysis_rows(
                    f"{split.upper()} CANDIDATE-BUDGET SWEEP",
                    rows,
                    [
                        "budget",
                        "candidates",
                        "PPRGraphAnsRec",
                        "MRR",
                        "H1",
                        "H10",
                        "time",
                        "peak_gb",
                    ],
                )
            )

        if args.analysis_paper_all or args.analyze_latent_states:
            rows, corr, elapsed = model.evaluate_latent_reasoning_states(
                split,
                sample_count=args.latent_sample_count,
                topk=args.latent_analysis_topk,
            )
            blocks.append(
                model.format_analysis_rows(
                    f"{split.upper()} LATENT-REASONING-STATE ANALYSIS",
                    rows,
                    [
                        "answers",
                        "queries",
                        "LatentD",
                        "PredJSD",
                        "AR10_K1",
                        "AR10_K",
                        "DeltaAR10",
                        "GoldCov10_K",
                    ],
                )
                + f"\nLatentD_PredJSD_Corr\t{corr:.6f}"
                + f"\nTIME\t{elapsed:.6f}"
            )

        if args.analysis_paper_all or args.analyze_flow_difficulty:
            tau_values = _parse_float_list(args.flow_tau_values)
            rows, elapsed = model.evaluate_flow_by_query_difficulty(
                split,
                tau_values,
                max_depth=args.difficulty_max_depth,
            )
            blocks.append(
                model.format_analysis_rows(
                    f"{split.upper()} FLOW-DYNAMICS BY QUERY DIFFICULTY",
                    rows,
                    [
                        "difficulty",
                        "queries",
                        "tau",
                        "MRR",
                        "H1",
                        "H10",
                        "GoldProb",
                        "MoveFromX0",
                        "DistToX1",
                    ],
                )
                + f"\nTIME\t{elapsed:.6f}"
            )

        if args.analysis_paper_all or args.analyze_robustness:
            drop_values = _parse_float_list(args.robustness_drop_values)
            if any(value < 0.0 or value >= 1.0 for value in drop_values):
                raise ValueError("robustness drop values must lie in [0, 1)")
            robustness_seeds = _parse_int_list(args.robustness_seeds)
            raw_rows, summary_rows = model.evaluate_incomplete_evidence_robustness(
                split,
                drop_values,
                robustness_seeds,
            )
            blocks.append(
                model.format_analysis_rows(
                    f"{split.upper()} INCOMPLETE-EVIDENCE ROBUSTNESS SUMMARY",
                    summary_rows,
                    [
                        "drop",
                        "MRR_mean",
                        "MRR_std",
                        "H10_mean",
                        "H10_std",
                        "PPRRecall_mean",
                        "RelMRRDrop",
                    ],
                )
                + "\n\n"
                + model.format_analysis_rows(
                    f"{split.upper()} INCOMPLETE-EVIDENCE ROBUSTNESS RAW",
                    raw_rows,
                    [
                        "drop",
                        "seed",
                        "retained_edges",
                        "MRR",
                        "H1",
                        "H10",
                        "PPRGraphAnsRec",
                        "time",
                    ],
                )
            )

        if args.analysis_paper_all or args.analyze_efficiency:
            step_counts = _parse_int_list(args.ode_step_values)
            rows = model.evaluate_efficiency_scalability(split, step_counts)
            blocks.append(
                model.format_analysis_rows(
                    f"{split.upper()} EFFICIENCY-AND-SCALABILITY",
                    rows,
                    [
                        "steps",
                        "NFE",
                        "MRR",
                        "H1",
                        "H10",
                        "EndpointError",
                        "fm_ms_per_query",
                        "ms_per_query",
                        "queries_per_sec",
                        "peak_gb",
                        "params_m",
                    ],
                )
            )


        if args.analyze_vae_sources:
            rows, elapsed = model.evaluate_vae_source_interventions(split)
            blocks.append(
                model.format_analysis_rows(
                    f"{split.upper()} VAE-SOURCE INTERVENTION",
                    rows,
                    ["variant", "MRR", "H1", "H10", "DeltaMRR"],
                )
                + f"\nTIME\t{elapsed:.6f}"
            )

        if args.analyze_multi_answer_case:
            tau_values = _parse_float_list(args.flow_tau_values)
            case = model.evaluate_multi_answer_flow_case(
                split,
                tau_values,
                topk=args.multi_case_topk,
                case_index=args.multi_case_query_index,
            )
            blocks.append(
                model.format_multi_answer_flow_case(
                    f"{split.upper()} MULTI-ANSWER FLOW CASE", case
                )
            )

        if args.analysis_paper_all or args.analyze_case_study:
            tau_values = _parse_float_list(args.flow_tau_values)
            case_indices = (
                _parse_int_list(args.case_query_indices)
                if args.case_query_indices
                else None
            )
            cases = model.evaluate_case_study(
                split,
                tau_values,
                sample_count=args.case_sample_count,
                topk=args.case_topk,
                case_indices=case_indices,
            )
            blocks.append(
                model.format_case_study(
                    f"{split.upper()} REPRESENTATIVE CASE STUDY", cases
                )
            )

        analysis_text = "\n\n".join(blocks) + "\n"
        paper_analysis_requested = (
            args.analysis_paper_all
            or args.analyze_latent_states
            or args.analyze_flow_difficulty
            or args.analyze_robustness
            or args.analyze_efficiency
            or args.analyze_case_study
            or args.analyze_multi_answer_case
            or args.analyze_vae_sources
        )
        default_analysis_name = (
            f"{run_tag}_paper_analysis.txt"
            if paper_analysis_requested
            else f"{run_tag}_analysis.txt"
        )
        analysis_file = args.analysis_file or os.path.join(
            project_root, "results", name, default_analysis_name
        )
        os.makedirs(os.path.dirname(os.path.abspath(analysis_file)), exist_ok=True)
        with open(analysis_file, "w", encoding="utf-8") as handle:
            handle.write(args.data_path + "\n")
            handle.write(config)
            handle.write(analysis_text)
        return

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
        with open(perf_file, "a", encoding="utf-8") as handle:
            handle.write(epoch_line + "\n")
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
    final = f"[FINAL BEST EPOCH {best_epoch}] {best_valid}\t{test_str}\n"
    with open(perf_file, "a", encoding="utf-8") as handle:
        handle.write(final)


if __name__ == "__main__":
    main()
