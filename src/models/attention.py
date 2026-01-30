import torch
import torch.nn as nn

# =============================================================================
# CROSS-ATTENTION MODULES
# =============================================================================
class AspectCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = query.unsqueeze(1)
        kv = key_value.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        refined = self.norm(q + self.dropout(attn_out))
        return refined.squeeze(1)


class AspectCrossAttentionModule(nn.Module):
    def __init__(self, aspect_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.theme_cross = AspectCrossAttention(aspect_dim, num_heads, dropout)
        self.action_cross = AspectCrossAttention(aspect_dim, num_heads, dropout)
        self.outcome_cross = AspectCrossAttention(aspect_dim, num_heads, dropout)
    
    def forward(
        self,
        anchor_aspects: dict[str, torch.Tensor],
        other_aspects: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            'theme': self.theme_cross(anchor_aspects['theme'], other_aspects['theme']),
            'action': self.action_cross(anchor_aspects['action'], other_aspects['action']),
            'outcome': self.outcome_cross(anchor_aspects['outcome'], other_aspects['outcome'])
        }
