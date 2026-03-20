"""
Loss function for WS-TFA (Weakly Supervised Tiny Feature Aggregation).
Optimizes the network using only Image-Level Labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class WSTFALoss(nn.Module):
    """
    Weakly Supervised Loss for WS-TFA.
    Contains:
    1. MIL Classification Loss (BCE on image-level predictions)
    2. Alpha Regularization (preventing dynamic attention factors from degenerating)
    3. Box Regression Loss (via Pseudo-Label Mining)
    """

    def __init__(self, alpha_reg_weight: float = 0.01, box_loss_weight: float = 1.0, top_k_pseudo: int = 1):
        """
        Args:
            alpha_reg_weight (float): Weight for the alpha regularization term.
            box_loss_weight (float): Base weight for the bounding box regression loss.
            top_k_pseudo (int): Number of top scoring queries to use as pseudo ground truths.
        """
        super().__init__()
        self.alpha_reg_weight = alpha_reg_weight
        self.box_loss_weight = box_loss_weight
        self.top_k_pseudo = top_k_pseudo
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(
        self, 
        network_outputs: Dict[str, torch.Tensor], 
        image_labels: torch.Tensor,
        current_epoch: int = 0,
        warmup_epochs: int = 5
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the total loss.

        Args:
            network_outputs (Dict): Dictionary containing 'final_prob', 'alphas', and 'bboxes'.
                                    'final_prob' shape: [B, num_queries, num_classes]
                                    'bboxes' shape: [B, num_queries, 4]
                                    'alphas' dict: values shape [B, 1, 1, 1]
            image_labels (torch.Tensor): Ground truth image-level labels (multi-hot).
                                         Shape: [B, num_classes] (values 0 or 1).
            current_epoch (int): The current training epoch.
            warmup_epochs (int): Number of epochs before applying box regression loss.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: 
                - Total loss (scalar tensor)
                - Dictionary of individual loss components for logging
        """
        B, num_queries, num_classes = network_outputs['final_prob'].shape
        final_prob = network_outputs['final_prob'] # [B, num_queries, num_classes]
        bboxes = network_outputs['bboxes'] # [B, num_queries, 4]
        
        # ==========================================
        # 1. MIL Loss (Image-Level Classification)
        # ==========================================
        # 直接提取每张图片在每个类别上的最高置信度 (Max-Pooling) 
        image_level_probs = final_prob.max(dim=1)[0] 
        
        # 裁剪概率以防极端值导致 log(0) 崩溃 
        image_level_probs = torch.clamp(image_level_probs, min=1e-6, max=1.0 - 1e-6) 
        
        # 将概率安全地逆转为 Logits，完美对接 BCEWithLogitsLoss 
        image_level_logits = torch.logit(image_level_probs) 
        mil_loss = self.bce_loss(image_level_logits, image_labels.float())

        # ==========================================
        # 2. Alpha Regularization
        # ==========================================
        alpha_reg_loss = 0.0
        alphas = network_outputs['alphas']
        for name, alpha in alphas.items():
            # MSE between alpha and 0.5 to prevent extreme saturation
            alpha_reg_loss += F.mse_loss(alpha, torch.full_like(alpha, 0.5))
        if len(alphas) > 0:
            alpha_reg_loss = alpha_reg_loss / len(alphas)

        # ==========================================
        # 3. Pseudo-Label Mining & Box Regression Loss
        # ==========================================
        box_loss = torch.tensor(0.0, device=final_prob.device)
        valid_box_pairs = 0
        
        # Only compute Box Loss if past the warmup phase
        if current_epoch >= warmup_epochs:
            # We iterate over each image in the batch
            for b in range(B):
                # Find classes that actually exist in the ground truth for this image
                present_classes = torch.where(image_labels[b] == 1.0)[0]
                
                for cls_idx in present_classes:
                    # Get scores for all queries for this specific class
                    cls_scores = final_prob[b, :, cls_idx] # Shape: [num_queries]
                    
                    # Find the top K queries with the highest scores
                    # These are our "Pseudo GTs" for this class
                    topk_scores, topk_indices = torch.topk(cls_scores, k=self.top_k_pseudo)
                    
                    # Let's use the absolute highest scoring query's box as the Pseudo GT for this class
                    best_query_idx = topk_indices[0]
                    # Make sure the pseudo GT box is strictly detached to prevent gradients from flowing
                    # back through the pseudo GT target.
                    pseudo_gt_box = bboxes[b, best_query_idx].detach().clone() # [4]
                    
                    # We force the top K queries to regress towards the best pseudo GT box.
                    for k_idx in topk_indices:
                        if k_idx != best_query_idx:
                            pred_box = bboxes[b, k_idx] # [4]
                            box_loss = box_loss + self.l1_loss(pred_box, pseudo_gt_box)
                            valid_box_pairs += 1
                        
            if valid_box_pairs > 0:
                box_loss = box_loss / valid_box_pairs

        # Dynamic Box Loss Weight based on Epoch
        current_lambda_3 = self.box_loss_weight if current_epoch >= warmup_epochs else 0.0

        # ==========================================
        # 4. Total Loss
        # ==========================================
        total_loss = mil_loss + self.alpha_reg_weight * alpha_reg_loss + current_lambda_3 * box_loss

        loss_dict = {
            "loss_mil": mil_loss.detach(),
            "loss_alpha_reg": (self.alpha_reg_weight * alpha_reg_loss).detach() if isinstance(alpha_reg_loss, torch.Tensor) else 0.0,
            "loss_box": box_loss.detach() if isinstance(box_loss, torch.Tensor) else 0.0,
            "loss_total": total_loss.detach()
        }

        return total_loss, loss_dict


if __name__ == '__main__':
    # Simple syntax check
    loss_fn = WSTFALoss()
    print("WSTFALoss initialized successfully.")
