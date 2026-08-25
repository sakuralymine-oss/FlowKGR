

import math
from typing import Dict

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from models import FMGNNReasoner
from utils import cal_performance, cal_ranks


class BaseModel:
    def __init__(self, args, loader):
        self.model = FMGNNReasoner(args, loader).to(args.device)
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_batch = args.n_batch
        self.optimizer = Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.lamb
        )
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.params = args
        self.loss_weights = self._build_loss_weights(args)

    @staticmethod
    def _build_loss_weights(args) -> Dict[str, float]:
        return {
            "rec": args.rec_weight,
            "con": args.con_weight,
            "vae_rec": args.vae_rec_weight,
            "kl": args.kl_weight,
            "fm": args.fm_weight,
            "temperature": args.con_temperature,
            "num_negatives": args.con_negatives,
        }

    @staticmethod
    def _schedule_progress(epoch: int, warmup: int, ramp: int) -> float:
        if epoch <= warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        return min(max((epoch - warmup) / float(ramp), 0.0), 1.0)

    def _set_epoch_loss_weights(self, epoch: int):
        progress = self._schedule_progress(
            epoch, self.params.fm_warmup, self.params.fm_ramp
        )
        self.loss_weights["fm"] = (
            float(self.params.fm_weight)
            * progress
            * float(self.params.fm_max_weight)
        )

    def train_batch(self, epoch: int) -> float:
        self._set_epoch_loss_weights(epoch)
        n_batch = math.ceil(self.loader.n_train / self.n_batch)
        total_loss = 0.0
        self.model.train()

        for batch_id in range(n_batch):
            start = batch_id * self.n_batch
            end = min(self.loader.n_train, start + self.n_batch)
            subs, rels, labels = self.loader.get_batch(
                np.arange(start, end), data="train"
            )
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.model.compute_losses(
                subs, rels, labels, mode="train", weights=self.loss_weights
            )
            loss = losses["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at batch {batch_id}: {loss.item()}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.params.grad_clip
            )
            self.optimizer.step()
            total_loss += loss.detach().item()

        self.loader.shuffle_train()
        self.scheduler.step()
        return total_loss / max(n_batch, 1)

    def evaluate_split(self, data: str):
        if data not in {"valid", "test"}:
            raise ValueError(f"unknown evaluation split: {data}")
        n_data = self.loader.n_valid if data == "valid" else self.loader.n_test
        filters_dict = (
            self.loader.val_filters if data == "valid" else self.loader.tst_filters
        )
        n_batch = math.ceil(n_data / self.n_batch)
        ranking = []
        masks = []

        self.model.eval()
        with torch.inference_mode():
            for batch_id in range(n_batch):
                start = batch_id * self.n_batch
                end = min(n_data, start + self.n_batch)
                subs, rels, labels = self.loader.get_batch(
                    np.arange(start, end), data=data
                )
                scores = self.model(
                    subs,
                    rels,
                    mode="eval",
                    sample_z=False,
                    return_aux=False,
                ).detach().cpu().numpy()

                filters = np.zeros_like(labels, dtype=bool)
                for row, (sub, rel) in enumerate(zip(subs, rels)):
                    known = filters_dict.get((int(sub), int(rel)), [])
                    filters[row, np.asarray(known, dtype=np.int64)] = True
                    masks.extend(
                        [max(self.n_ent - len(known), 1)]
                        * int(labels[row].sum())
                    )
                ranking.extend(cal_ranks(scores, labels, filters))

        return cal_performance(np.asarray(ranking), masks)

    @staticmethod
    def format_metrics(prefix: str, metrics):
        mrr, mr, h1, h3, h10, _ = metrics
        return (
            f"[{prefix}] MRR:{mrr:.4f} MR:{mr:.1f} "
            f"H@1:{h1:.4f} H@3:{h3:.4f} H@10:{h10:.4f}"
        )
