import torch
import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from .attention import AspectCrossAttentionModule
from peft import LoraConfig, get_peft_model, TaskType

# DistilBERT
#target_modules=["q_lin", "v_lin"]  # Query and Value projections

# BERT / RoBERTa
#target_modules=["query", "value"]  # Can also add "key"

# MPNet (sentence-transformers/all-mpnet-base-v2)
#target_modules=["q", "v"]

# For more aggressive fine-tuning (better performance, more params)
target_modules=["query", "key", "value", "dense"]  # Add output projection


# =============================================================================
# MAIN ENCODER
# =============================================================================
class AspectSupervisedEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        projection_dim: int = 256,
        aspect_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
        use_lora: bool = False,  
        lora_r: int = 8,         #  LoRA rank
        lora_alpha: int = 16,    # LoRA scaling
        lora_dropout: float = 0.1,  # LoRA dropout
        target_modules: list[str] = ["q", "v"]  # For MPNet
    ):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.freeze_encoder = freeze_encoder
        self.use_lora = use_lora

        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,                    # Rank of low-rank matrices
                lora_alpha=lora_alpha,       # Scaling factor
                target_modules=target_modules,  # ! Depend on model architecture !! 
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            print(f"LoRA enabled: {self.encoder.print_trainable_parameters()}")
            self.freeze_encoder = False  # LoRA params are trainable
        
        # Freeze encoder if requested (and not using LoRA)
        elif freeze_encoder and not use_lora:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Main projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Aspect prediction heads
        self.theme_head = self._make_aspect_head(projection_dim, aspect_dim, dropout)
        self.action_head = self._make_aspect_head(projection_dim, aspect_dim, dropout)
        self.outcome_head = self._make_aspect_head(projection_dim, aspect_dim, dropout)
        
        # Aspect text encoder projection
        self.aspect_projection = nn.Sequential(
            nn.Linear(hidden_size, aspect_dim),
            nn.LayerNorm(aspect_dim)
        )
        
        # Cross-attention
        self.aspect_cross_attention = AspectCrossAttentionModule(aspect_dim, num_heads, dropout)
        
        self.projection_dim = projection_dim
        self.aspect_dim = aspect_dim
    
    def _make_aspect_head(self, in_dim: int, out_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    
    def _encode_raw(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder and not self.use_lora:
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._mean_pool(outputs.last_hidden_state, attention_mask)
    
    def _predict_aspects(self, story_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            'theme': self.theme_head(story_emb),
            'action': self.action_head(story_emb),
            'outcome': self.outcome_head(story_emb)
        }
    
    def _encode_aspect_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raw = self._encode_raw(input_ids, attention_mask)
        return self.aspect_projection(raw)
    
    def _process_single_story(
        self,
        story_ids: torch.Tensor,
        story_mask: torch.Tensor,
        theme_ids: torch.Tensor,
        theme_mask: torch.Tensor,
        action_ids: torch.Tensor,
        action_mask: torch.Tensor,
        outcome_ids: torch.Tensor,
        outcome_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Process one story: encode + predict aspects + target aspects."""
        # Story embedding (BERT + projection)
        story_emb = self.encode_story(story_ids, story_mask)
        
        # Predicted aspects
        pred_aspects = self._predict_aspects(story_emb)
        
        # Target aspects from text
        target_aspects = {
            'theme': self._encode_aspect_text(theme_ids, theme_mask),
            'action': self._encode_aspect_text(action_ids, action_mask),
            'outcome': self._encode_aspect_text(outcome_ids, outcome_mask)
        }
        
        return {
            'story_emb': story_emb,
            'pred_aspects': pred_aspects,
            'target_aspects': target_aspects
        }
    
    def tokenize(self, texts: list[str], device: torch.device, max_length: int = 512) -> dict:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt'
        )
        return {k: v.to(device) for k, v in encoded.items()}
    
    def encode_story(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """story text → embedding."""
        raw = self._encode_raw(input_ids, attention_mask)
        return self.projection(raw)
    
    def forward(self, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
        """
        TRAINING: Full forward pass for triplet with aspects.
        
        Args:
            batch: dict with keys for anchor/positive/negative stories and their aspects
            device: torch device
        
        Returns:
            dict with all embeddings, predictions, targets, and cross-attention outputs
        """
        # Tokenize all inputs
        def tok(texts):
            return self.tokenize(texts, device)
        
        # Actually, let me simplify the tokenization
        anchor_story_tok = tok(batch['anchor'])
        anchor_theme_tok = tok(batch['anchor_theme'])
        anchor_action_tok = tok(batch['anchor_action'])
        anchor_outcome_tok = tok(batch['anchor_outcome'])
        
        positive_story_tok = tok(batch['positive'])
        positive_theme_tok = tok(batch['positive_theme'])
        positive_action_tok = tok(batch['positive_action'])
        positive_outcome_tok = tok(batch['positive_outcome'])
        
        negative_story_tok = tok(batch['negative'])
        negative_theme_tok = tok(batch['negative_theme'])
        negative_action_tok = tok(batch['negative_action'])
        negative_outcome_tok = tok(batch['negative_outcome'])
        
        # Process each story
        anchor_out = self._process_single_story(
            anchor_story_tok['input_ids'], anchor_story_tok['attention_mask'],
            anchor_theme_tok['input_ids'], anchor_theme_tok['attention_mask'],
            anchor_action_tok['input_ids'], anchor_action_tok['attention_mask'],
            anchor_outcome_tok['input_ids'], anchor_outcome_tok['attention_mask']
        )
        
        positive_out = self._process_single_story(
            positive_story_tok['input_ids'], positive_story_tok['attention_mask'],
            positive_theme_tok['input_ids'], positive_theme_tok['attention_mask'],
            positive_action_tok['input_ids'], positive_action_tok['attention_mask'],
            positive_outcome_tok['input_ids'], positive_outcome_tok['attention_mask']
        )
        
        negative_out = self._process_single_story(
            negative_story_tok['input_ids'], negative_story_tok['attention_mask'],
            negative_theme_tok['input_ids'], negative_theme_tok['attention_mask'],
            negative_action_tok['input_ids'], negative_action_tok['attention_mask'],
            negative_outcome_tok['input_ids'], negative_outcome_tok['attention_mask']
        )
        
        # Cross-attention: anchor with positive/negative
        cross_pos = self.aspect_cross_attention(
            anchor_out['pred_aspects'], positive_out['pred_aspects']
        )
        cross_neg = self.aspect_cross_attention(
            anchor_out['pred_aspects'], negative_out['pred_aspects']
        )
        
        return {
            # Story embeddings
            'anchor_emb': anchor_out['story_emb'],
            'positive_emb': positive_out['story_emb'],
            'negative_emb': negative_out['story_emb'],
            # Predicted aspects
            'anchor_pred_aspects': anchor_out['pred_aspects'],
            'positive_pred_aspects': positive_out['pred_aspects'],
            'negative_pred_aspects': negative_out['pred_aspects'],
            # Target aspects
            'anchor_target_aspects': anchor_out['target_aspects'],
            'positive_target_aspects': positive_out['target_aspects'],
            'negative_target_aspects': negative_out['target_aspects'],
            # Cross-attention outputs
            'cross_pos': cross_pos,
            'cross_neg': cross_neg
        }
