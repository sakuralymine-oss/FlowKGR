

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


@dataclass
class CandidateState:
    scores: torch.Tensor
    nodes: torch.Tensor
    hidden: torch.Tensor
    candidate_mask: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
    recon_hidden: torch.Tensor
    corruption_mask: torch.Tensor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
        half = self.dim // 2
        frequency = torch.exp(
            torch.arange(half, device=tau.device, dtype=tau.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        embedding = torch.cat(
            [
                torch.sin(tau * frequency.unsqueeze(0)),
                torch.cos(tau * frequency.unsqueeze(0)),
            ],
            dim=-1,
        )
        if embedding.size(-1) < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.size(-1)))
        return self.proj(embedding)


class BellmanFordGNNLayer(nn.Module):
    

    def __init__(self, dim: int, attn_dim: int, n_rel: int, act):
        super().__init__()
        self.act = act
        self.rela_embed = nn.Embedding(2 * n_rel + 1, dim)
        self.Ws_attn = nn.Linear(dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1, bias=False)
        self.aggregate = nn.Sequential(
            nn.Linear(dim * 12, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.update = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        q_rel: torch.Tensor,
        hidden: torch.Tensor,
        edges: torch.Tensor,
        boundary: torch.Tensor,
        active: torch.Tensor,
        n_node: int,
    ):
        src = edges[:, 4]
        rel = edges[:, 2]
        dst = edges[:, 5]
        row = edges[:, 0]
        active_edge = active[src]

        if torch.any(active_edge):
            src = src[active_edge]
            rel = rel[active_edge]
            dst = dst[active_edge]
            row = row[active_edge]
            h_src = hidden[src]
            r_edge = self.rela_embed(rel)
            r_query = self.rela_embed(q_rel)[row]
            attention_hidden = F.relu(
                self.Ws_attn(h_src)
                + self.Wr_attn(r_edge)
                + self.Wqr_attn(r_query)
            )
            alpha = torch.sigmoid(self.w_alpha(attention_hidden))
            message = h_src * r_edge * alpha

            degree = scatter(
                torch.ones_like(dst, dtype=message.dtype),
                dst,
                dim=0,
                dim_size=n_node,
                reduce="sum",
            )
            count = degree.clamp_min(1.0).unsqueeze(-1)
            mean = scatter(message, dst, dim=0, dim_size=n_node, reduce="sum") / count
            maximum = scatter(message, dst, dim=0, dim_size=n_node, reduce="max")
            minimum = scatter(message, dst, dim=0, dim_size=n_node, reduce="min")
            second = scatter(
                message.square(), dst, dim=0, dim_size=n_node, reduce="sum"
            ) / count
            std = torch.sqrt((second - mean.square()).clamp_min(1e-12))
            statistics = torch.cat([mean, maximum, minimum, std], dim=-1)
            log_degree = torch.log1p(degree).clamp_min(math.log(2.0)).unsqueeze(-1)
            pna = torch.cat(
                [statistics, statistics * log_degree, statistics / log_degree],
                dim=-1,
            )
            incoming = degree > 0
            aggregated = self.aggregate(pna) * incoming.to(message.dtype).unsqueeze(-1)
        else:
            incoming = torch.zeros(n_node, dtype=torch.bool, device=hidden.device)
            aggregated = torch.zeros_like(hidden)

        delta = self.update(torch.cat([hidden, aggregated, boundary], dim=-1))
        next_active = active | incoming | boundary.ne(0).any(dim=-1)
        updated = self.out_norm(hidden + delta + boundary)
        updated = self.act(updated) * next_active.to(updated.dtype).unsqueeze(-1)
        return updated, next_active


class CandidateDenoisingVAE(nn.Module):
    

    def __init__(
        self,
        dim: int,
        feature_mask_rate: float,
    ):
        super().__init__()
        self.dim = dim
        self.feature_mask_rate = float(feature_mask_rate)
        self.encoder = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def encode(
        self,
        hidden: torch.Tensor,
        query_ctx: torch.Tensor,
        entity_state: torch.Tensor,
        corrupt: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if corrupt and self.feature_mask_rate > 0.0:
            corruption_mask = torch.rand_like(hidden).lt(self.feature_mask_rate)
            corrupted_hidden = hidden.masked_fill(corruption_mask, 0.0)
        else:
            corruption_mask = torch.zeros_like(hidden, dtype=torch.bool)
            corrupted_hidden = hidden
        encoded = self.encoder(
            torch.cat([corrupted_hidden, query_ctx, entity_state], dim=-1)
        )
        mu, logvar = encoded.chunk(2, dim=-1)
        return mu, logvar.clamp(min=-10.0, max=5.0), corruption_mask

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor, sample: bool
    ) -> torch.Tensor:
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class ConditionalScoreVectorField(nn.Module):
    

    def __init__(self, dim: int):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(dim)
        self.score_embedding = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.context_score = nn.Linear(dim, 1, bias=False)
        self.velocity = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.velocity[-1].weight)
        nn.init.zeros_(self.velocity[-1].bias)

    def forward(
        self,
        score_state: torch.Tensor,
        latent_condition: torch.Tensor,
        tau: torch.Tensor,
        row: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if score_state.dim() != 2 or score_state.size(-1) != 1:
            raise ValueError("score_state must have shape [num_candidates, 1]")
        token = self.token_projection(
            torch.cat(
                [
                    latent_condition,
                    self.score_embedding(score_state),
                    self.time_embedding(tau),
                ],
                dim=-1,
            )
        )
        attention_logits = self.context_score(token).squeeze(-1)
        row_max = scatter(
            attention_logits, row, dim=0, dim_size=batch_size, reduce="max"
        )[row]
        attention = torch.exp(attention_logits - row_max)
        denominator = scatter(
            attention, row, dim=0, dim_size=batch_size, reduce="sum"
        )[row].clamp_min(1e-12)
        attention = attention / denominator
        context = scatter(
            attention.unsqueeze(-1) * token,
            row,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )[row]
        return self.velocity(torch.cat([token, context, token * context], dim=-1))


class FMGNNReasoner(nn.Module):
    

    def __init__(self, params, loader):
        super().__init__()
        self.n_layer = int(params.n_layer)
        self.hidden_dim = int(params.hidden_dim)
        self.attn_dim = int(params.attn_dim)
        self.n_rel = int(params.n_rel)
        self.loader = loader
        self.topk = float(params.topk)
        self.unreached_score = min(float(getattr(params, "unreached_score", -1e9)), -1e9)
        self.ode_steps = int(getattr(params, "ode_steps", 4))
        self.eval_path_samples = int(getattr(params, "eval_path_samples", 1))
        self.fm_source_power = float(getattr(params, "fm_source_power", 2.0))
        self.fm_source_fraction = float(getattr(params, "fm_source_fraction", 0.50))
        self.target_negative_mass = float(getattr(params, "target_negative_mass", 0.05))

        activations = {
            "relu": nn.ReLU(),
            "tanh": torch.tanh,
            "idd": lambda value: value,
        }
        activation = activations[params.act]
        self.dropout = nn.Dropout(params.dropout)
        self.rela_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)
        self.layers = nn.ModuleList(
            [
                BellmanFordGNNLayer(
                    self.hidden_dim, self.attn_dim, self.n_rel, activation
                )
                for _ in range(self.n_layer)
            ]
        )
        for layer in self.layers:
            layer.rela_embed = self.rela_embed

        self.query_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.entity_state = nn.Embedding(self.loader.n_ent, self.hidden_dim)
        nn.init.xavier_uniform_(self.entity_state.weight)
        self.candidate_vae = CandidateDenoisingVAE(
            self.hidden_dim,
            feature_mask_rate=float(getattr(params, "feature_mask_rate", 0.30)),
        )
        self.vector_field = ConditionalScoreVectorField(self.hidden_dim)

    def _dense_from_node_logits(
        self,
        nodes: torch.Tensor,
        node_logits: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        scores = torch.full(
            (batch_size, self.loader.n_ent),
            self.unreached_score,
            dtype=node_logits.dtype,
            device=node_logits.device,
        )
        scores[nodes[:, 0], nodes[:, 1]] = node_logits
        return scores

    def _initial_hidden(self, q_rel: torch.Tensor, batch_size: int) -> torch.Tensor:
        del batch_size
        return torch.tanh(self.rela_embed(q_rel))

    def _build_query_context(
        self,
        q_rel: torch.Tensor,
        nodes: torch.Tensor,
        hidden: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        pooled = scatter(
            hidden, nodes[:, 0], dim=0, dim_size=batch_size, reduce="mean"
        )
        return self.query_projection(
            torch.cat([pooled, self.rela_embed(q_rel)], dim=-1)
        )

    def encode_candidates(self, subs, rels, mode: str = "train"):
        batch_size = len(subs)
        device = self.rela_embed.weight.device
        q_sub = torch.as_tensor(subs, dtype=torch.long, device=device)
        q_rel = torch.as_tensor(rels, dtype=torch.long, device=device)

        nodes = self.loader.get_ppr_subgraph(
            q_sub.detach().cpu().numpy(), self.topk, mode
        )
        ppr_mask = torch.zeros(
            (batch_size, self.loader.n_ent), dtype=torch.bool, device=device
        )
        ppr_mask[nodes[:, 0], nodes[:, 1]] = True
        edges = self.loader.get_subgraph_edges(
            nodes.detach().cpu().numpy(), batch_size, mode
        )
        boundary = torch.zeros((nodes.size(0), self.hidden_dim), device=device)
        head_mask = nodes[:, 1] == q_sub[nodes[:, 0]]
        if torch.any(head_mask):
            head_state = self._initial_hidden(q_rel, batch_size)
            boundary[head_mask] = head_state[nodes[head_mask, 0]]
        hidden = boundary.clone()
        active = head_mask.clone()
        for layer in self.layers:
            hidden, active = layer(
                q_rel, hidden, edges, boundary, active, nodes.size(0)
            )
            hidden = self.dropout(hidden)

        query_ctx = self._build_query_context(q_rel, nodes, hidden, batch_size)
        return q_rel, nodes, hidden, query_ctx, ppr_mask

    def _integrate_scores(
        self,
        latent_condition: torch.Tensor,
        row: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        state = torch.zeros(
            (latent_condition.size(0), 1),
            dtype=latent_condition.dtype,
            device=latent_condition.device,
        )
        dt = 1.0 / float(self.ode_steps)
        for step in range(self.ode_steps):
            tau_start = torch.full(
                (state.size(0),),
                step * dt,
                dtype=state.dtype,
                device=state.device,
            )
            tau_mid = tau_start + 0.5 * dt
            velocity_start = self.vector_field(
                state, latent_condition, tau_start, row, batch_size
            )
            midpoint = state + 0.5 * dt * velocity_start
            velocity_mid = self.vector_field(
                midpoint, latent_condition, tau_mid, row, batch_size
            )
            state = state + dt * velocity_mid
        return state







    def _node_scores(
        self,
        nodes: torch.Tensor,
        score_state: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        dense_logits = self._dense_from_node_logits(
            nodes, score_state.squeeze(-1), batch_size
        )
        return F.log_softmax(dense_logits, dim=1)

    def forward(
        self,
        subs,
        rels,
        mode: str = "train",
        sample_z: bool = False,
        return_aux: bool = False,
        path_samples: Optional[int] = None,
    ):
        batch_size = len(subs)
        (
            q_rel,
            nodes,
            hidden,
            query_ctx,
            candidate_mask,
        ) = self.encode_candidates(subs, rels, mode)
        del q_rel
        row = nodes[:, 0]
        entity_state = self.entity_state(nodes[:, 1])
        mu, logvar, corruption_mask = self.candidate_vae.encode(
            hidden, query_ctx[row], entity_state, corrupt=self.training
        )

        if path_samples is None:
            path_samples = 1 if self.training else self.eval_path_samples
        if path_samples < 1:
            raise ValueError("path_samples must be positive")

        final_score_paths = []
        latent_paths = []
        recon_hidden = None
        for path_index in range(path_samples):
            should_sample = self.training or sample_z or path_index > 0
            z = self.candidate_vae.reparameterize(mu, logvar, should_sample)
            if recon_hidden is None:
                recon_hidden = self.candidate_vae.decode(z)
            x1 = self._integrate_scores(z, row, batch_size)
            final_score_paths.append(self._node_scores(nodes, x1, batch_size))
            latent_paths.append(z)

        scores = torch.logsumexp(
            torch.stack(final_score_paths, dim=0), dim=0
        ) - math.log(float(path_samples))
        z_mean = torch.stack(latent_paths, dim=0).mean(dim=0)
        if not return_aux:
            return scores
        return CandidateState(
            scores=scores,
            nodes=nodes,
            hidden=hidden,
            candidate_mask=candidate_mask,
            mu=mu,
            logvar=logvar,
            z=z_mean,
            recon_hidden=recon_hidden,
            corruption_mask=corruption_mask,
        )

    @staticmethod
    def ranking_loss(
        scores: torch.Tensor,
        labels: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        reached_labels = labels * candidate_mask.to(labels.dtype)
        valid = reached_labels.sum(dim=1) > 0
        if not torch.any(valid):
            return scores.sum() * 0.0
        candidate_scores = scores.masked_fill(~candidate_mask, float("-inf"))
        log_prob = F.log_softmax(candidate_scores, dim=1)
        positive_count = reached_labels.sum(dim=1).clamp_min(1.0)
        row_loss = -(
            reached_labels * log_prob.masked_fill(~candidate_mask, 0.0)
        ).sum(dim=1)
        return (row_loss[valid] / positive_count[valid]).mean()

    @staticmethod
    def contrastive_loss(
        scores: torch.Tensor,
        labels: torch.Tensor,
        candidate_mask: torch.Tensor,
        temperature: float,
        num_negatives: int,
    ) -> torch.Tensor:
        positive = (labels > 0) & candidate_mask
        negative = (labels <= 0) & candidate_mask
        valid = positive.any(dim=1) & negative.any(dim=1)
        if not torch.any(valid):
            return scores.sum() * 0.0
        row_scores = scores[valid]
        row_positive = positive[valid]
        row_negative = negative[valid]
        positive_logit = torch.logsumexp(
            row_scores.masked_fill(~row_positive, float("-inf")) / temperature,
            dim=1,
        )

        
        
        
        
        
        hard_scores = row_scores.detach().masked_fill(
            ~row_negative, float("-inf")
        )
        count = min(int(num_negatives), row_scores.size(1))
        negative_index = torch.topk(hard_scores, k=count, dim=1).indices
        selected_is_negative = row_negative.gather(1, negative_index)
        negative_logit = (
            row_scores.gather(1, negative_index) / temperature
        ).masked_fill(~selected_is_negative, float("-inf"))

        logits = torch.cat([positive_logit.unsqueeze(1), negative_logit], dim=1)
        return F.cross_entropy(
            logits,
            torch.zeros(logits.size(0), dtype=torch.long, device=scores.device),
        )

    @staticmethod
    def vae_reconstruction_loss(state: CandidateState) -> torch.Tensor:
        target = state.hidden.detach()
        squared_error = (state.recon_hidden - target).square()
        if torch.any(state.corruption_mask):
            mask = state.corruption_mask.to(squared_error.dtype)
            return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)
        return squared_error.mean()

    @staticmethod
    def kl_loss(state: CandidateState) -> torch.Tensor:
        node_kl = -0.5 * torch.sum(
            1.0 + state.logvar - state.mu.square() - state.logvar.exp(), dim=-1
        )
        row = state.nodes[:, 0]
        batch_size = state.scores.size(0)
        row_sum = scatter(node_kl, row, dim=0, dim_size=batch_size, reduce="sum")
        row_count = scatter(
            torch.ones_like(node_kl),
            row,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        ).clamp_min(1.0)
        return (row_sum / row_count).mean()

    def _answer_target_logits(
        self, state: CandidateState, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes = state.nodes
        row = nodes[:, 0]
        entity = nodes[:, 1]
        batch_size = labels.size(0)
        positive = labels[row, entity] > 0
        negative = ~positive
        dtype = state.z.dtype
        positive_count = scatter(
            positive.to(dtype), row, dim=0, dim_size=batch_size, reduce="sum"
        )
        negative_count = scatter(
            negative.to(dtype), row, dim=0, dim_size=batch_size, reduce="sum"
        )
        valid_rows = (positive_count > 0) & (negative_count > 0)

        eps = self.target_negative_mass
        pos_prob = (1.0 - eps) / positive_count[row].clamp_min(1.0)
        neg_prob = eps / negative_count[row].clamp_min(1.0)
        probability = torch.where(positive, pos_prob, neg_prob).clamp_min(1e-12)
        target = probability.log().unsqueeze(-1)

        row_sum = scatter(
            target.squeeze(-1), row, dim=0, dim_size=batch_size, reduce="sum"
        )
        row_count = scatter(
            torch.ones_like(target.squeeze(-1)),
            row,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        ).clamp_min(1.0)
        target = target - (row_sum / row_count)[row].unsqueeze(-1)
        target = target.clamp(min=-12.0, max=12.0)
        return target, positive, negative, valid_rows

    def score_fm_loss(
        self, state: CandidateState, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row = state.nodes[:, 0]
        batch_size = labels.size(0)
        target, positive, negative, valid_rows = self._answer_target_logits(
            state, labels
        )
        if not torch.any(valid_rows):
            zero = state.scores.sum() * 0.0
            return zero, zero

        positive_count = scatter(
            positive.to(target.dtype),
            row,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )
        negative_count = scatter(
            negative.to(target.dtype),
            row,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )
        positive_weight = 0.5 / positive_count[row].clamp_min(1.0)
        negative_weight = 0.5 / negative_count[row].clamp_min(1.0)
        node_weight = torch.where(positive, positive_weight, negative_weight)
        node_weight = node_weight * valid_rows[row].to(node_weight.dtype)

        uniform = torch.rand(batch_size, device=state.z.device)
        tau_row = uniform.pow(self.fm_source_power)
        if self.fm_source_fraction > 0.0:
            source_rows = torch.rand(batch_size, device=state.z.device).lt(
                self.fm_source_fraction
            )
            tau_row = torch.where(source_rows, torch.zeros_like(tau_row), tau_row)
        tau = tau_row[row]
        path_state = tau.unsqueeze(-1) * target
        predicted_velocity = self.vector_field(
            path_state, state.z, tau, row, batch_size
        )
        target_velocity = target
        node_loss = (predicted_velocity - target_velocity).square().squeeze(-1)

        row_loss = scatter(
            node_weight * node_loss,
            row,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )
        fm_loss = row_loss[valid_rows].mean()

        source_tau = torch.zeros_like(tau)
        source_state = torch.zeros_like(target)
        source_prediction = self.vector_field(
            source_state, state.z, source_tau, row, batch_size
        )
        source_node_loss = (
            source_prediction - target_velocity
        ).square().squeeze(-1)
        source_row_loss = scatter(
            node_weight * source_node_loss,
            row,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )
        source_loss = source_row_loss[valid_rows].mean()
        return fm_loss, source_loss

    def compute_losses(
        self,
        subs,
        rels,
        labels,
        mode: str,
        weights: Dict[str, float],
    ):
        labels = torch.as_tensor(
            labels, dtype=torch.float32, device=self.rela_embed.weight.device
        )
        state = self.forward(
            subs,
            rels,
            mode=mode,
            sample_z=True,
            return_aux=True,
            path_samples=1,
        )
        rank_loss = self.ranking_loss(state.scores, labels, state.candidate_mask)
        contrastive = self.contrastive_loss(
            state.scores,
            labels,
            state.candidate_mask,
            weights.get("temperature", 1.0),
            int(weights.get("num_negatives", 64)),
        )
        vae_rec = self.vae_reconstruction_loss(state)
        kl = self.kl_loss(state)
        fm, _ = self.score_fm_loss(state, labels)
        total = (
            weights.get("rec", 1.0) * rank_loss
            + weights.get("con", 1.0) * contrastive
            + weights.get("vae_rec", 1.0) * vae_rec
            + weights.get("kl", 1e-4) * kl
            + weights.get("fm", 1.0) * fm
        )
        return {"loss": total}
