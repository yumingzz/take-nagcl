"""
Neural network modules for encoding centrality and community features.
"""
import torch
import torch.nn as nn


class CentralityCommunityEncoder(nn.Module):
    """
    Encode 6-dim centrality features plus community id into a dense vector.
    """

    def __init__(
        self,
        num_communities: int,
        community_embed_dim: int = 64,
        centrality_hidden_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_communities = num_communities
        self.output_dim = output_dim

        self.community_embedding = nn.Embedding(
            num_embeddings=num_communities + 1,
            embedding_dim=community_embed_dim
        )

        self.centrality_mlp = nn.Sequential(
            nn.Linear(6, centrality_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(centrality_hidden_dim, centrality_hidden_dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(community_embed_dim + centrality_hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, centrality_features: torch.Tensor, community_ids: torch.Tensor) -> torch.Tensor:
        community_ids = community_ids.clamp(0, self.num_communities)
        comm_embed = self.community_embedding(community_ids)
        cent_embed = self.centrality_mlp(centrality_features)
        combined = torch.cat([comm_embed, cent_embed], dim=-1)
        fused = self.fusion(combined)
        fused = self.layer_norm(fused)
        return fused
