import subprocess

import numpy as np
from scipy.stats import binom


def cal_ranks(scores, labels, filters):
    
    ranks = []
    labels = labels.astype(bool)
    filters = filters.astype(bool)
    for row in range(scores.shape[0]):
        for gold in np.flatnonzero(labels[row]):
            valid = ~filters[row]
            valid[gold] = True
            valid_scores = scores[row, valid]
            gold_score = scores[row, gold]
            higher = np.sum(valid_scores > gold_score)
            ties = np.sum(valid_scores == gold_score) - 1
            ranks.append(1.0 + higher + 0.5 * max(float(ties), 0.0))
    return ranks


def cal_performance(ranks, masks):
    
    if len(ranks) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ranks = np.asarray(ranks, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.float64)
    if masks.size != ranks.size:
        raise ValueError(f"mask count {masks.size} != rank count {ranks.size}")

    mrr = float(np.mean(1.0 / ranks))
    mean_rank = float(np.mean(ranks))
    h1 = float(np.mean(ranks <= 1))
    h3 = float(np.mean(ranks <= 3))
    h10 = float(np.mean(ranks <= 10))
    false_positive_rate = np.clip(
        (ranks - 1.0) / np.maximum(masks, 1.0), 0.0, 1.0
    )
    h10_50 = float(np.mean(binom.cdf(9, 50, false_positive_rate)))
    return mrr, mean_rank, h1, h3, h10, h10_50


def select_gpu():
    
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        memory = [
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise RuntimeError("unable to select a CUDA GPU with nvidia-smi") from error
    if not memory:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return int(np.argmin(memory))
