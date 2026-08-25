

import math
from contextlib import contextmanager
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
        self.ema_enabled = bool(getattr(args, "ema", False))
        self.ema_decay = float(getattr(args, "ema_decay", 0.999))
        self.ema_updates = 0
        self.ema_state = {}
        if self.ema_enabled:
            self._reset_ema_state()

    def _reset_ema_state(self, source_state=None):
        source_state = self.model.state_dict() if source_state is None else source_state
        self.ema_state = {
            name: value.detach().clone() for name, value in source_state.items()
        }

    @torch.no_grad()
    def _update_ema(self):
        if not self.ema_enabled:
            return
        self.ema_updates += 1
        decay = min(
            self.ema_decay,
            (1.0 + float(self.ema_updates)) / (10.0 + float(self.ema_updates)),
        )
        for name, value in self.model.state_dict().items():
            shadow = self.ema_state[name]
            if torch.is_floating_point(value) or torch.is_complex(value):
                shadow.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                shadow.copy_(value.detach())

    def checkpoint_weights(self):
        if not self.ema_enabled:
            return {"model_state_dict": self.model.state_dict()}
        return {
            "model_state_dict": self.ema_state,
            "raw_model_state_dict": self.model.state_dict(),
            "ema_updates": self.ema_updates,
            "ema_decay": self.ema_decay,
        }

    def load_checkpoint_weights(self, saved, resume: bool = False):
        if resume and "raw_model_state_dict" in saved:
            live_state = saved["raw_model_state_dict"]
        else:
            live_state = saved["model_state_dict"]
        self.model.load_state_dict(live_state, strict=True)
        if self.ema_enabled:
            self._reset_ema_state(saved["model_state_dict"])
            self.ema_updates = int(saved.get("ema_updates", 0))

    @contextmanager
    def _evaluation_context(self, data: str):
        raw_state = None
        if self.ema_enabled:
            raw_state = {
                name: value.detach().clone()
                for name, value in self.model.state_dict().items()
            }
            self.model.load_state_dict(self.ema_state, strict=True)

        device_index = (
            self.params.device.index if self.params.device.type == "cuda" else None
        )
        devices = [] if device_index is None else [device_index]
        try:
            with torch.random.fork_rng(devices=devices):
                eval_seed = int(getattr(self.params, "seed", 1234)) + (
                    100_003 if data == "valid" else 200_003
                )
                torch.manual_seed(eval_seed)
                if device_index is not None:
                    torch.cuda.manual_seed(eval_seed)
                yield
        finally:
            if raw_state is not None:
                self.model.load_state_dict(raw_state, strict=True)

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
            "hard_negative_ratio": 0.0,
            "top10": 0.0,
            "top10_margin": args.top10_margin,
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
        total_epochs = max(int(self.params.epochs), 1)
        warmup = float(self.params.con_hard_warmup_fraction) * total_epochs
        ramp = float(self.params.con_hard_ramp_fraction) * total_epochs
        final_ratio = float(self.params.con_hard_final_ratio)
        if epoch <= warmup:
            hard_ratio = 0.0
        elif ramp <= 0.0:
            hard_ratio = final_ratio
        else:
            hard_ratio = final_ratio * min(
                max((epoch - warmup) / ramp, 0.0), 1.0
            )
        self.loss_weights["hard_negative_ratio"] = hard_ratio

        top10_warmup = float(self.params.top10_warmup_fraction) * total_epochs
        top10_ramp = float(self.params.top10_ramp_fraction) * total_epochs
        if epoch <= top10_warmup:
            top10_progress = 0.0
        elif top10_ramp <= 0.0:
            top10_progress = 1.0
        else:
            top10_progress = min(
                max((epoch - top10_warmup) / top10_ramp, 0.0), 1.0
            )
        self.loss_weights["top10"] = (
            float(self.params.top10_loss_weight) * top10_progress
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
            self._update_ema()
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
        n_ent = (
            int(self.loader.entity_count(data))
            if hasattr(self.loader, "entity_count")
            else int(self.n_ent)
        )
        n_batch = math.ceil(n_data / self.n_batch)
        ranking = []
        masks = []

        with self._evaluation_context(data):
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
                        mode=data,
                        sample_z=False,
                        return_aux=False,
                    ).detach().cpu().numpy()

                    filters = np.zeros_like(labels, dtype=bool)
                    for row, (sub, rel) in enumerate(zip(subs, rels)):
                        known = filters_dict.get((int(sub), int(rel)), [])
                        filters[row, np.asarray(known, dtype=np.int64)] = True
                        masks.extend(
                            [max(n_ent - len(known), 1)]
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
