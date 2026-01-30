import torch 
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from src.models.loss import CombinedLoss
from src.models.encoder import AspectSupervisedEncoder


# =============================================================================
# TRAINER 
# =============================================================================
class Trainer:
    def __init__(
        self,
        model: AspectSupervisedEncoder,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        triplet_margin: float = 0.5,
        cross_margin: float = 0.3,
        gamma: float = 0.3,
        beta: float = 0.2
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = CombinedLoss(triplet_margin, cross_margin, gamma, beta)
        
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    
    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        
        totals = {'triplet': 0, 'aspect': 0, 'cross': 0, 'total': 0}
        correct, total = 0, 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # forward pass
            outputs = self.model(batch, self.device)
            losses = self.criterion(outputs)
            
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # Accuracy
            with torch.no_grad():
                pos_sim = F.cosine_similarity(outputs['anchor_emb'], outputs['positive_emb'], dim=-1)
                neg_sim = F.cosine_similarity(outputs['anchor_emb'], outputs['negative_emb'], dim=-1)
                correct += (pos_sim > neg_sim).sum().item()
                total += outputs['anchor_emb'].size(0)
            
            totals['triplet'] += losses['triplet_loss'].item()
            totals['aspect'] += losses['aspect_loss'].item()
            totals['cross'] += losses['cross_loss'].item()
            totals['total'] += losses['total_loss'].item()
            
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        n = len(dataloader)
        return {
            'triplet_loss': totals['triplet'] / n,
            'aspect_loss': totals['aspect'] / n,
            'cross_loss': totals['cross'] / n,
            'total_loss': totals['total'] / n,
            'accuracy': correct / total
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Inference: only encode_story()."""
        self.model.eval()
        correct, total = 0, 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            anchor_tok = self.model.tokenize(batch['anchor'], self.device)
            positive_tok = self.model.tokenize(batch['positive'], self.device)
            negative_tok = self.model.tokenize(batch['negative'], self.device)
            
            anchor_emb = self.model.encode_story(anchor_tok['input_ids'], anchor_tok['attention_mask'])
            positive_emb = self.model.encode_story(positive_tok['input_ids'], positive_tok['attention_mask'])
            negative_emb = self.model.encode_story(negative_tok['input_ids'], negative_tok['attention_mask'])
            
            pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=-1)
            neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=-1)
            
            correct += (pos_sim > neg_sim).sum().item()
            total += anchor_emb.size(0)
        
        return correct / total

    @torch.no_grad()
    def evaluate2(self, dataloader: DataLoader, debug: bool = False) -> dict[str, float]:
        """Inference on story_emb concatenated with aspect embeddings.

        Returns:
            dict with accuracies for: story, theme, action, outcome, concatenated
        """
        self.model.eval()

        # Counters for different combinations
        correct_story = 0
        correct_theme = 0
        correct_action = 0
        correct_outcome = 0
        correct_final = 0
        total = 0

        def _cosine_triplet_sim(anchor, positive, negative):
            pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
            neg_sim = F.cosine_similarity(anchor, negative, dim=-1)
            return pos_sim, neg_sim

        for batch in tqdm(dataloader, desc="Evaluating"):
            anchor_tok = self.model.tokenize(batch['anchor'], self.device)
            positive_tok = self.model.tokenize(batch['positive'], self.device)
            negative_tok = self.model.tokenize(batch['negative'], self.device)

            anchor_emb = self.model.encode_story(anchor_tok['input_ids'], anchor_tok['attention_mask'])
            positive_emb = self.model.encode_story(positive_tok['input_ids'], positive_tok['attention_mask'])
            negative_emb = self.model.encode_story(negative_tok['input_ids'], negative_tok['attention_mask'])

            # Predict aspects
            anchor_aspects = self.model._predict_aspects(anchor_emb)
            positive_aspects = self.model._predict_aspects(positive_emb)
            negative_aspects = self.model._predict_aspects(negative_emb)

            # 1. Story embedding only
            pos_sim_story, neg_sim_story = _cosine_triplet_sim(anchor_emb, positive_emb, negative_emb)
            correct_story += (pos_sim_story > neg_sim_story).sum().item()

            # 2. Theme only
            pos_sim_theme, neg_sim_theme = _cosine_triplet_sim(
                anchor_aspects['theme'], positive_aspects['theme'], negative_aspects['theme']
            )
            correct_theme += (pos_sim_theme > neg_sim_theme).sum().item()

            # 3. Action only
            pos_sim_action, neg_sim_action = _cosine_triplet_sim(
                anchor_aspects['action'], positive_aspects['action'], negative_aspects['action']
            )
            correct_action += (pos_sim_action > neg_sim_action).sum().item()

            # 4. Outcome only
            pos_sim_outcome, neg_sim_outcome = _cosine_triplet_sim(
                anchor_aspects['outcome'], positive_aspects['outcome'], negative_aspects['outcome']
            )
            correct_outcome += (pos_sim_outcome > neg_sim_outcome).sum().item()

            # 5. Concatenated (story + all aspects)
            anchor_final = torch.cat([
                anchor_emb,
                anchor_aspects['theme'],
                anchor_aspects['action'],
                anchor_aspects['outcome']
            ], dim=-1)
            positive_final = torch.cat([
                positive_emb,
                positive_aspects['theme'],
                positive_aspects['action'],
                positive_aspects['outcome']
            ], dim=-1)
            negative_final = torch.cat([
                negative_emb,
                negative_aspects['theme'],
                negative_aspects['action'],
                negative_aspects['outcome']
            ], dim=-1)

            pos_sim_final, neg_sim_final = _cosine_triplet_sim(anchor_final, positive_final, negative_final)
            correct_final += (pos_sim_final > neg_sim_final).sum().item()

            total += anchor_emb.size(0)

        # Calculate accuracies
        accuracies = {
            'story': correct_story / total,
            'theme': correct_theme / total,
            'action': correct_action / total,
            'outcome': correct_outcome / total,
            'concatenated': correct_final / total
        }

        if debug:
            print("\n" + "="*60)
            print("DETAILED ACCURACY BREAKDOWN")
            print("="*60)
            print(f"Story Embedding Only:           {accuracies['story']:.4f} ({correct_story}/{total})")
            print(f"Theme Aspect Only:              {accuracies['theme']:.4f} ({correct_theme}/{total})")
            print(f"Action Aspect Only:             {accuracies['action']:.4f} ({correct_action}/{total})")
            print(f"Outcome Aspect Only:            {accuracies['outcome']:.4f} ({correct_outcome}/{total})")
            print(f"Concatenated (Story + Aspects): {accuracies['concatenated']:.4f} ({correct_final}/{total})")
            print("="*60 + "\n")

        return accuracies
    
    def fit(self, train_loader: DataLoader, dev_loader: DataLoader, epochs: int = 10, save_dir: str = "checkpoints"):
        """
        Train the model and save best checkpoints for each metric.

        Args:
            train_loader: Training data loader
            dev_loader: Development/validation data loader
            epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Track best accuracies for each metric
        best_accuracies = {
            'story': 0.0,
            'theme': 0.0,
            'action': 0.0,
            'outcome': 0.0,
            'concatenated': 0.0
        }

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print('='*60)

            # Training
            metrics = self.train_epoch(train_loader)
            print(f"Train - Triplet: {metrics['triplet_loss']:.4f}, "
                  f"Aspect: {metrics['aspect_loss']:.4f}, "
                  f"Cross: {metrics['cross_loss']:.4f}, "
                  f"Acc: {metrics['accuracy']:.4f}")

            # Evaluation
            dev_accuracies = self.evaluate2(dev_loader, debug=True)
            # Save best model for each metric
            saved_models = []
            for metric_name, current_acc in dev_accuracies.items():
                if current_acc > best_accuracies[metric_name]:
                    best_accuracies[metric_name] = current_acc
                    save_path = os.path.join(save_dir, f"best_model_{metric_name}.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'accuracy': current_acc,
                        'metric': metric_name,
                        'all_accuracies': dev_accuracies
                    }, save_path)
                    saved_models.append(metric_name)
                    print(f" New best {metric_name}! Saved to {save_path}")

            if not saved_models:
                print(f"  No improvements this epoch.")

        # Final summary
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE - Best Accuracies:")
        print('='*60)
        for metric_name, acc in best_accuracies.items():
            print(f"  {metric_name:15s}: {acc:.4f}")
        print('='*60)

        return best_accuracies
    
    def load_best_model(self, checkpoint_path: str) -> dict:
        """Load a saved checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            dict with checkpoint info (epoch, accuracy, etc.)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Metric: {checkpoint['metric']}")
        print(f"Accuracy: {checkpoint['accuracy']:.4f}")
        print(f"All accuracies at that epoch: {checkpoint['all_accuracies']}")

        return checkpoint