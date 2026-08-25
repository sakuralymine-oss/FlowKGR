

import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix


class DataLoader:
    

    def __init__(
        self,
        task_dir: str,
        n_batch: int = 32,
        fact_ratio: float = 0.9,
        remove_1hop_edges: bool = False,
        ppr_alpha: float = 0.15,
        ppr_iterations: int = 20,
    ):
        self.task_dir = task_dir
        self.n_batch = n_batch
        self.fact_ratio = float(fact_ratio)
        self.remove_1hop_edges = bool(remove_1hop_edges)
        self.ppr_alpha = float(ppr_alpha)
        self.ppr_iterations = int(ppr_iterations)
        if not 0.0 < self.ppr_alpha < 1.0:
            raise ValueError("ppr_alpha must be between zero and one")
        if self.ppr_iterations < 1:
            raise ValueError("ppr_iterations must be positive")
        self._ppr_cache_train = {}
        self._ppr_cache_eval = {}
        if not 0.0 < self.fact_ratio < 1.0:
            raise ValueError("fact_ratio must be between zero and one")
        self.entity2id = self._read_id_file(os.path.join(task_dir, "entities.txt"))
        self.relation2id = self._read_id_file(os.path.join(task_dir, "relations.txt"))
        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)
        self.id2entity = {idx: name for name, idx in self.entity2id.items()}
        self.id2relation = {idx: name for name, idx in self.relation2id.items()}

        
        
        source_facts = self.read_triples("facts.txt")
        source_train = self.read_triples("train.txt")
        self.valid_triples = self.read_triples("valid.txt")
        self.test_triples = self.read_triples("test.txt")

        
        
        self.observed_forward = [
            triple
            for triple in source_facts + source_train
            if triple[1] < self.n_rel
        ]
        if len(self.observed_forward) < 2:
            raise ValueError("facts.txt + train.txt must contain at least two forward triples")
        self.observed_triples = self.double_triples(self.observed_forward)
        self.eval_KG, self.eval_sub = self.load_graph(self.observed_triples)
        self.eval_transition = self._build_transition_matrix(self.eval_KG)

        
        self._resplit_fact_train()

        self.valid_q, self.valid_a = self.load_query(self.valid_triples)
        self.test_q, self.test_a = self.load_query(self.test_triples)

        all_known = self.observed_triples + self.valid_triples + self.test_triples
        filters = self.get_filter(all_known)
        self.val_filters = filters
        self.tst_filters = filters

        self.n_valid = len(self.valid_q)
        self.n_test = len(self.test_q)

    @staticmethod
    def _read_id_file(path: str) -> Dict[str, int]:
        mapping = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                mapping[parts[0]] = int(parts[1]) if len(parts) > 1 else len(mapping)
        ids = sorted(mapping.values())
        if ids != list(range(len(ids))):
            raise ValueError(f"IDs in {path} must be contiguous from zero")
        return mapping

    def read_triples(self, filename: str) -> List[List[int]]:
        
        triples = []
        path = os.path.join(self.task_dir, filename)
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 3:
                    raise ValueError(f"invalid triple in {path}: {line.rstrip()}")
                head, relation, tail = parts
                h = self.entity2id[head]
                r = self.relation2id[relation]
                t = self.entity2id[tail]
                triples.append([h, r, t])
                triples.append([t, r + self.n_rel, h])
        return triples

    def double_triples(self, triples: List[List[int]]) -> List[List[int]]:
        
        doubled = []
        for head, relation, tail in triples:
            doubled.append([head, relation, tail])
            doubled.append([tail, relation + self.n_rel, head])
        return doubled

    def _resplit_fact_train(self):
        
        order = np.random.permutation(len(self.observed_forward))
        split = int(len(order) * self.fact_ratio)
        split = min(max(split, 1), len(order) - 1)
        fact_forward = [self.observed_forward[index] for index in order[:split]]
        train_forward = [self.observed_forward[index] for index in order[split:]]
        self.fact_triples = self.double_triples(fact_forward)
        self.train_triples = self.double_triples(train_forward)
        if self.remove_1hop_edges:
            blocked_pairs = {
                (int(head), int(tail))
                for head, _, tail in self.train_triples
            }
            self.fact_triples = [
                triple
                for triple in self.fact_triples
                if (int(triple[0]), int(triple[2])) not in blocked_pairs
            ]
        self.train_KG, self.train_sub = self.load_graph(self.fact_triples)
        self.train_transition = self._build_transition_matrix(self.train_KG)
        self._ppr_cache_train = {}
        self.train_q, self.train_a = self.load_query(self.train_triples)
        self.n_train = len(self.train_q)

    def load_graph(self, triples: List[List[int]]) -> Tuple[np.ndarray, csr_matrix]:
        
        kg = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
        self_loop_rel = 2 * self.n_rel
        identity = np.column_stack(
            (
                np.arange(self.n_ent, dtype=np.int64),
                np.full(self.n_ent, self_loop_rel, dtype=np.int64),
                np.arange(self.n_ent, dtype=np.int64),
            )
        )
        kg = np.concatenate((kg, identity), axis=0)
        edge_count = kg.shape[0]
        subject_matrix = csr_matrix(
            (np.ones(edge_count), (np.arange(edge_count), kg[:, 0])),
            shape=(edge_count, self.n_ent),
        )
        return kg, subject_matrix

    def _build_transition_matrix(self, kg: np.ndarray) -> csr_matrix:
        
        non_self_loop = kg[:, 1] != (2 * self.n_rel)
        heads = kg[non_self_loop, 0]
        tails = kg[non_self_loop, 2]
        adjacency = csr_matrix(
            (np.ones(len(heads), dtype=np.float32), (heads, tails)),
            shape=(self.n_ent, self.n_ent),
            dtype=np.float32,
        )
        adjacency.data[:] = 1.0
        row_sum = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        inv_row_sum = np.zeros_like(row_sum, dtype=np.float32)
        np.divide(1.0, row_sum, out=inv_row_sum, where=row_sum > 0)
        return adjacency.multiply(inv_row_sum[:, None]).tocsr()

    def _resolve_subgraph_size(self, topk: float) -> int:
        
        topk = float(topk)
        if topk <= 0:
            raise ValueError("one-shot PPR topk must be positive")
        size = int(np.ceil(topk * self.n_ent)) if topk < 1.0 else int(topk)
        return int(min(max(size, 1), self.n_ent))

    def _ppr_scores(self, root: int, mode: str) -> np.ndarray:
        
        if mode == "train":
            transition = self.train_transition
            cache = self._ppr_cache_train
        elif mode in {"valid", "test", "eval"}:
            transition = self.eval_transition
            cache = self._ppr_cache_eval
        else:
            raise ValueError(f"unknown mode: {mode}")

        root = int(root)
        if root in cache:
            return cache[root]

        teleport = np.zeros(self.n_ent, dtype=np.float32)
        teleport[root] = 1.0
        scores = teleport.copy()
        for _ in range(self.ppr_iterations):
            scores = (
                self.ppr_alpha * teleport
                + (1.0 - self.ppr_alpha)
                * np.asarray(scores @ transition).reshape(-1)
            ).astype(np.float32, copy=False)
        scores[root] = max(float(scores[root]), 1.0)
        cache[root] = scores
        return scores

    def get_ppr_subgraph(
        self,
        subs: np.ndarray,
        topk: float,
        mode: str = "train",
    ) -> torch.Tensor:
        
        k = self._resolve_subgraph_size(topk)
        rows = []
        for row, root in enumerate(np.asarray(subs, dtype=np.int64)):
            scores = self._ppr_scores(int(root), mode)
            if k >= self.n_ent:
                entities = np.arange(self.n_ent, dtype=np.int64)
            else:
                entities = np.argpartition(-scores, kth=k - 1)[:k].astype(
                    np.int64, copy=False
                )
                if int(root) not in set(entities.tolist()):
                    entities[-1] = int(root)
                order = np.argsort(-scores[entities], kind="mergesort")
                entities = entities[order]
            rows.append(
                np.column_stack(
                    (
                        np.full(len(entities), row, dtype=np.int64),
                        entities.astype(np.int64, copy=False),
                    )
                )
            )
        nodes = (
            np.concatenate(rows, axis=0)
            if rows
            else np.zeros((0, 2), dtype=np.int64)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.as_tensor(nodes, dtype=torch.long, device=device)

    def get_subgraph_edges(
        self,
        nodes: np.ndarray,
        batch_size: int,
        mode: str = "train",
    ) -> torch.Tensor:
        
        if mode == "train":
            kg, subject_matrix = self.train_KG, self.train_sub
        elif mode in {"valid", "test", "eval"}:
            kg, subject_matrix = self.eval_KG, self.eval_sub
        else:
            raise ValueError(f"unknown mode: {mode}")

        nodes = np.asarray(nodes, dtype=np.int64)
        node_lookup = np.full((batch_size, self.n_ent), -1, dtype=np.int64)
        node_lookup[nodes[:, 0], nodes[:, 1]] = np.arange(
            nodes.shape[0], dtype=np.int64
        )

        node_1hot = csr_matrix(
            (np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])),
            shape=(self.n_ent, batch_size),
        )
        edge_1hot = subject_matrix.dot(node_1hot)
        edge_ids, active_node_ids = np.nonzero(edge_1hot)
        selected = np.concatenate(
            (active_node_ids[:, None], kg[edge_ids]), axis=1
        ).astype(np.int64, copy=False)

        source_index = node_lookup[selected[:, 0], selected[:, 1]]
        target_index = node_lookup[selected[:, 0], selected[:, 3]]
        keep = (source_index >= 0) & (target_index >= 0)
        selected = selected[keep]
        source_index = source_index[keep]
        target_index = target_index[keep]

        selected = np.concatenate(
            (selected, source_index[:, None], target_index[:, None]), axis=1
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.as_tensor(selected, dtype=torch.long, device=device)

    @staticmethod
    def load_query(
        triples: List[List[int]],
    ) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        
        grouped = defaultdict(set)
        for head, relation, tail in triples:
            grouped[(head, relation)].add(tail)
        queries = list(grouped.keys())
        answers = [np.asarray(sorted(grouped[q]), dtype=np.int64) for q in queries]
        return queries, answers

    def get_batch(self, batch_idx: np.ndarray, data: str = "train"):
        
        if data == "train":
            queries, answers = self.train_q, self.train_a
        elif data == "valid":
            queries, answers = self.valid_q, self.valid_a
        elif data == "test":
            queries, answers = self.test_q, self.test_a
        else:
            raise ValueError(f"unknown split: {data}")

        selected = [queries[int(index)] for index in batch_idx]
        subs = np.asarray([query[0] for query in selected], dtype=np.int64)
        rels = np.asarray([query[1] for query in selected], dtype=np.int64)
        labels = np.zeros((len(batch_idx), self.n_ent), dtype=np.float32)
        for row, index in enumerate(batch_idx):
            labels[row, answers[int(index)]] = 1.0
        return subs, rels, labels

    def shuffle_train(self):
        
        self._resplit_fact_train()


    def structural_shortest_distance(
        self, head: int, targets, max_depth: int = 8
    ) -> int:
        
        head = int(head)
        target_set = {int(value) for value in np.asarray(targets).reshape(-1)}
        if head in target_set:
            return 0
        if not hasattr(self, "_analysis_observed_adj"):
            adjacency = [[] for _ in range(self.n_ent)]
            for src, _, dst in self.observed_forward:
                src = int(src)
                dst = int(dst)
                adjacency[src].append(dst)
                adjacency[dst].append(src)
            self._analysis_observed_adj = adjacency
        adjacency = self._analysis_observed_adj
        frontier = {head}
        visited = {head}
        for depth in range(1, int(max_depth) + 1):
            next_frontier = set()
            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor in visited:
                        continue
                    if neighbor in target_set:
                        return depth
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier
        return -1

    @contextmanager
    def temporary_eval_edge_dropout(self, drop_rate: float, seed: int):
        
        drop_rate = float(drop_rate)
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError('drop_rate must be in [0, 1)')

        old_eval_KG = self.eval_KG
        old_eval_sub = self.eval_sub
        old_eval_transition = self.eval_transition
        old_cache = self._ppr_cache_eval
        try:
            if drop_rate <= 0.0:
                
                
                self._ppr_cache_eval = {}
                yield len(self.observed_forward)
                return

            rng = np.random.default_rng(int(seed))
            keep = rng.random(len(self.observed_forward)) >= drop_rate
            if not np.any(keep):
                
                keep[rng.integers(0, len(keep))] = True
            retained_forward = [
                triple for flag, triple in zip(keep.tolist(), self.observed_forward)
                if flag
            ]
            retained_triples = self.double_triples(retained_forward)
            self.eval_KG, self.eval_sub = self.load_graph(retained_triples)
            self.eval_transition = self._build_transition_matrix(self.eval_KG)
            self._ppr_cache_eval = {}
            yield len(retained_forward)
        finally:
            self.eval_KG = old_eval_KG
            self.eval_sub = old_eval_sub
            self.eval_transition = old_eval_transition
            self._ppr_cache_eval = old_cache

    @staticmethod
    def get_filter(triples: List[List[int]]):
        filters = defaultdict(set)
        for head, relation, tail in triples:
            filters[(head, relation)].add(tail)
        return {key: sorted(values) for key, values in filters.items()}
