import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# LOSSES
# =============================================================================
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=-1)
        return F.relu(neg_sim - pos_sim + self.margin).mean()


class AspectMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred: dict, target: dict) -> torch.Tensor:
        loss_theme = self.mse(pred['theme'], target['theme'])
        loss_action = self.mse(pred['action'], target['action'])
        loss_outcome = self.mse(pred['outcome'], target['outcome'])
        return (loss_theme + loss_action + loss_outcome) / 3

class AspectCosineLoss(nn.Module):
    def forward(self, pred: dict, target: dict) -> torch.Tensor:
        loss_theme = 1 - F.cosine_similarity(pred['theme'], target['theme'], dim=-1).mean()
        loss_action = 1 - F.cosine_similarity(pred['action'], target['action'], dim=-1).mean()
        loss_outcome = 1 - F.cosine_similarity(pred['outcome'], target['outcome'], dim=-1).mean()
        return (loss_theme + loss_action + loss_outcome) / 3


class AspectCrossLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor_pred_aspects: dict,
        cross_pos: dict,
        cross_neg: dict
    ) -> torch.Tensor:
        total = 0.0
        for aspect in ['theme', 'action', 'outcome']:
            anchor = anchor_pred_aspects[aspect]
            pos_sim = F.cosine_similarity(anchor, cross_pos[aspect], dim=-1)
            neg_sim = F.cosine_similarity(anchor, cross_neg[aspect], dim=-1)
            total += F.relu(neg_sim - pos_sim + self.margin).mean()
        return total / 3


class CombinedLoss(nn.Module):
    """Total = L_triplet + gamma * L_aspect + beta * L_cross"""
    
    def __init__(
        self,
        triplet_margin: float = 0.5,
        cross_margin: float = 0.3,
        gamma: float = 0.3,
        beta: float = 0.2,
        aspect_loss_type: str = 'mse'  # 'mse' or 'cosine'
    ):
        super().__init__()
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        if aspect_loss_type == 'mse':
            self.aspect_loss = AspectMSELoss()
        elif aspect_loss_type == 'cosine':
            self.aspect_loss = AspectCosineLoss()
        else:
            raise ValueError("aspect_loss_type must be 'mse' or 'cosine'")

        self.aspect_cross_loss = AspectCrossLoss(margin=cross_margin)
        self.gamma = gamma
        self.beta = beta
    
    def forward(self, outputs: dict) -> dict[str, torch.Tensor]:
        """
        Args:
            outputs: dict from model.forward()
        """
        # Triplet loss
        l_triplet = self.triplet_loss(
            outputs['anchor_emb'],
            outputs['positive_emb'],
            outputs['negative_emb']
        )
        
        # Aspect loss
        l_aspect = (
            self.aspect_loss(outputs['anchor_pred_aspects'], outputs['anchor_target_aspects']) +
            self.aspect_loss(outputs['positive_pred_aspects'], outputs['positive_target_aspects']) +
            self.aspect_loss(outputs['negative_pred_aspects'], outputs['negative_target_aspects'])
        ) / 3
        
        # Cross-attention loss
        l_cross = self.aspect_cross_loss(
            outputs['anchor_pred_aspects'],
            outputs['cross_pos'],
            outputs['cross_neg']
        )
        
        total = l_triplet + self.gamma * l_aspect + self.beta * l_cross
        
        return {
            'triplet_loss': l_triplet,
            'aspect_loss': l_aspect,
            'cross_loss': l_cross,
            'total_loss': total
        }
