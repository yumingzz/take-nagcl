"""
Centrality and community feature loader for TAKE.
"""
from typing import Dict, Tuple
import os
import numpy as np
import pandas as pd
import torch


class CentralityCommunityLoader:
    """
    Load DGCN3 centrality predictions and compute 6-dim structural features.
    """

    def __init__(
        self,
        dgcn_predictions_dir: str,
        edge_lists_dir: str,
        node_mapping_csv: str,
        alpha: float = 1.5,
        num_slices: int = 10,
        feature_set: str = "all",
        window_size: int = 2
    ):
        self.dgcn_predictions_dir = dgcn_predictions_dir
        self.edge_lists_dir = edge_lists_dir
        self.alpha = alpha
        self.num_slices = num_slices
        self.feature_set = feature_set
        self.window_size = window_size

        self.node_df = pd.read_csv(node_mapping_csv)
        self.centrality_dict: Dict[int, float] = {}
        self.community_dict: Dict[int, int] = {}
        self.feature_dict: Dict[int, np.ndarray] = {}
        self.turn_id_dict: Dict[int, int] = {}

        self._load_all_features()

    def _build_turn_id_dict(self) -> Dict[int, int]:
        turn_id_dict: Dict[int, int] = {}
        if self.node_df.empty:
            return turn_id_dict
        if "node_id" not in self.node_df.columns or "turn_id" not in self.node_df.columns:
            return turn_id_dict
        for _, row in self.node_df[["node_id", "turn_id"]].iterrows():
            try:
                nid = int(row["node_id"])
                tid = int(row["turn_id"])
                turn_id_dict[nid] = tid
            except Exception:
                continue
        return turn_id_dict

    def _load_centrality_for_slice(self, slice_id: int) -> Dict[int, float]:
        centrality = {}
        pred_file = os.path.join(
            self.dgcn_predictions_dir,
            f"alpha_{self.alpha}",
            f"tiage_{slice_id}.csv"
        )
        if os.path.exists(pred_file):
            df = pd.read_csv(pred_file, header=None, names=['node_id', 'centrality'])
            for _, row in df.iterrows():
                centrality[int(row['node_id'])] = float(row['centrality'])
        return centrality

    def _load_community_for_slice(self, slice_id: int) -> Dict[int, int]:
        edge_file = os.path.join(self.edge_lists_dir, f"tiage_{slice_id}.txt")
        if not os.path.exists(edge_file):
            return {}

        import networkx as nx
        import community as community_louvain

        graph = nx.Graph()
        with open(edge_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    graph.add_edge(u, v)

        partition = community_louvain.best_partition(graph)
        return partition

    def _rank_pct(self, values: np.ndarray) -> np.ndarray:
        if values.size <= 1:
            return np.zeros_like(values, dtype=np.float32)
        order = np.argsort(values, kind='mergesort')
        ranks = np.empty_like(order)
        ranks[order] = np.arange(values.size)
        return (ranks / float(values.size - 1)).astype(np.float32)

    def _filter_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_set == "none":
            return np.zeros_like(features, dtype=np.float32)
        if self.feature_set == "imp_pct":
            filtered = np.zeros_like(features, dtype=np.float32)
            filtered[1] = features[1]
            return filtered
        return features

    def _build_feature_dict(self) -> Dict[int, np.ndarray]:
        feature_dict: Dict[int, np.ndarray] = {}
        if self.node_df.empty:
            return feature_dict

        grouped = self.node_df.groupby('dialog_id', sort=False)
        for _, group in grouped:
            group_sorted = group.sort_values('turn_id')
            node_ids = group_sorted['node_id'].tolist()
            imp_raw = np.array(
                [self.centrality_dict.get(nid, 0.0) for nid in node_ids],
                dtype=np.float32
            )
            imp_pct = self._rank_pct(imp_raw)

            imp_delta_prev = np.zeros_like(imp_pct)
            if imp_pct.size > 1:
                imp_delta_prev[1:] = imp_pct[1:] - imp_pct[:-1]

            imp_delta_next = np.zeros_like(imp_pct)
            if imp_pct.size > 1:
                imp_delta_next[:-1] = imp_pct[1:] - imp_pct[:-1]

            imp_z_local = np.zeros_like(imp_raw)
            imp_minus_window_mean = np.zeros_like(imp_raw)
            for idx in range(imp_raw.size):
                start = max(0, idx - self.window_size)
                end = min(imp_raw.size, idx + self.window_size + 1)
                window_vals = imp_raw[start:end]
                mean = float(window_vals.mean()) if window_vals.size > 0 else 0.0
                std = float(window_vals.std()) if window_vals.size > 0 else 0.0
                imp_z_local[idx] = (imp_raw[idx] - mean) / (std + 1e-6)
                imp_minus_window_mean[idx] = imp_raw[idx] - mean

            for idx, nid in enumerate(node_ids):
                features = np.array(
                    [
                        imp_raw[idx],
                        imp_pct[idx],
                        imp_delta_prev[idx],
                        imp_delta_next[idx],
                        imp_z_local[idx],
                        imp_minus_window_mean[idx]
                    ],
                    dtype=np.float32
                )
                feature_dict[int(nid)] = self._filter_features(features)

        return feature_dict

    def _load_all_features(self) -> None:
        for slice_id in range(self.num_slices):
            self.centrality_dict.update(self._load_centrality_for_slice(slice_id))
            self.community_dict.update(self._load_community_for_slice(slice_id))
        self.feature_dict = self._build_feature_dict()
        self.turn_id_dict = self._build_turn_id_dict()

    def get_num_communities(self) -> int:
        if not self.community_dict:
            return 0
        return max(self.community_dict.values()) + 1

    def get_features_for_node(self, node_id: int) -> Tuple[np.ndarray, int]:
        features = self.feature_dict.get(node_id, np.zeros(6, dtype=np.float32))
        community = self.community_dict.get(node_id, 0)
        return features, community

    def get_batch_features(self, node_ids: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = node_ids.size(0)
        features = torch.zeros(batch_size, 6, device=device, dtype=torch.float32)
        communities = torch.zeros(batch_size, dtype=torch.long, device=device)
        # 使用 .item() 逐个获取避免 .tolist() 的 NumPy 依赖
        for i in range(batch_size):
            nid = int(node_ids[i].item())
            feat, comm = self.get_features_for_node(nid)
            features[i] = torch.tensor(feat, device=device, dtype=torch.float32)
            communities[i] = comm
        return features, communities

    def get_imp_raw(self, node_id: int) -> float:
        return float(self.centrality_dict.get(node_id, 0.0))

    def get_turn_id(self, node_id: int) -> int:
        return int(self.turn_id_dict.get(node_id, -1))
