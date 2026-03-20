"""
Sparse Multiple Instance Learning (MIL) Head for WS-TFA.
Combines a Class-Agnostic DETR-like proposal generator with a Sparsemax-based
MIL classifier to suppress local dominance and filter noisy proposals.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

class PositionEmbeddingSine(nn.Module):
    """
    Standard 2D Sine Positional Encoding.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask (torch.Tensor): [B, H, W] boolean mask (False means valid spatial location).
        Returns:
            torch.Tensor: [B, C, H, W] positional embedding.
        """
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Interleave sin and cos
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class Sparsemax(torch.nn.Module):
    """
    Sparsemax activation function.
    Projects the input onto the probability simplex, resulting in sparse probabilities.
    Replaces Softmax to force low-confidence noisy proposals to exactly 0.
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max, _ = torch.max(x, dim=self.dim, keepdim=True)
        z = x - x_max
        zs, _ = torch.sort(z, dim=self.dim, descending=True)
        
        range_vec = torch.arange(1, zs.size(self.dim) + 1, dtype=x.dtype, device=x.device)
        shape = [1] * z.dim()
        shape[self.dim] = -1
        range_vec = range_vec.view(*shape)

        cumsum_zs = torch.cumsum(zs, dim=self.dim)
        bound = 1.0 + range_vec * zs
        
        # 🚨 修复核心：这才是绝对正确的数学条件 (1.0 > 0.0 永远成立，保证 k >= 1)
        is_valid = bound > cumsum_zs
        k = is_valid.sum(dim=self.dim, keepdim=True)
        
        k_idx = torch.clamp(k - 1, min=0)
        sum_k = torch.gather(cumsum_zs, self.dim, k_idx)
        tau = (sum_k - 1.0) / k.to(x.dtype)

        p = torch.clamp(z - tau, min=0.0)
        return p


class ClassAgnosticDETR(nn.Module):
    """
    Simplified Transformer Encoder-Decoder for Class-Agnostic Proposal Generation.
    Uses learnable Object Queries to generate bounding boxes and objectness scores.
    """

    def __init__(self, hidden_dim: int = 256, num_queries: int = 300, nheads: int = 8,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2, num_feature_levels: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )

        # Learnable Object Queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Scale-level embeddings to distinguish features from different FPN levels
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, hidden_dim))
        nn.init.normal_(self.level_embed)
        
        # 2D Positional Encoding
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)

        # Prediction Heads (Class Agnostic)
        # 1. Bounding Box Head (4 coordinates: cx, cy, w, h)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid() # Normalize coordinates to [0, 1]
        )
        
        # 2. Objectness Score Head (1 score per query)
        self.objectness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Probability score
        )

    def forward(self, features: torch.Tensor, spatial_shapes: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features (torch.Tensor): Flattened feature maps [B, Seq_Len, C]
            spatial_shapes (List[Tuple[int, int]]): Shapes of features from each level
            
        Returns:
            bboxes: [B, num_queries, 4]
            objectness_scores: [B, num_queries, 1]
            hs: Hidden states from decoder [B, num_queries, hidden_dim]
        """
        B = features.size(0)
        
        # Prepare position embeddings and scale embeddings
        pos_embeds = []
        for level, (H, W) in enumerate(spatial_shapes):
            # Create a dummy mask [B, H, W] with False (meaning valid)
            mask = torch.zeros((B, H, W), dtype=torch.bool, device=features.device)
            # Generate 2D sine positional embedding: [B, C, H, W]
            pos = self.pos_embed(mask)
            # Flatten to [B, H*W, C]
            pos = pos.flatten(2).permute(0, 2, 1)
            # Add scale-level embedding
            pos = pos + self.level_embed[level].view(1, 1, -1)
            pos_embeds.append(pos)
            
        # Concatenate positional embeddings
        pos_embeds = torch.cat(pos_embeds, dim=1) # [B, Seq_Len, C]
        
        # Add pos_embed to features before Transformer Encoder
        features_with_pos = features + pos_embeds
        
        # Prepare queries [B, num_queries, hidden_dim]
        query_embeds = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Transformer Forward Pass
        # Encoder takes features, Decoder takes queries
        memory = self.transformer.encoder(features_with_pos)
        # Use query_embeds as the target sequence for the decoder so that each query 
        # has a unique identity, preventing them from producing identical outputs.
        hs = self.transformer.decoder(query_embeds, memory)
        
        # Generate predictions
        bboxes = self.bbox_head(hs)
        objectness_scores = self.objectness_head(hs)

        return bboxes, objectness_scores, hs


class SparseMILHead(nn.Module):
    """
    Sparse MIL Detection Head.
    Combines Class-Agnostic DETR proposals with Sparsemax MIL classification.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 256, num_queries: int = 300):
        super().__init__()
        self.num_classes = num_classes
        
        # Class-Agnostic Proposal Generator
        self.proposal_generator = ClassAgnosticDETR(
            hidden_dim=hidden_dim, 
            num_queries=num_queries
        )
        
        # MIL Classification Branch
        # Linear layer mapping query hidden states to class logits
        self.mil_classifier = nn.Linear(hidden_dim, num_classes)
        
        # Sparsemax applied over the proposal dimension (dim=1)
        # This normalizes probabilities across all queries for each class,
        # forcing low-confidence queries to 0.
        self.sparsemax = Sparsemax(dim=1)

    def forward(self, features: torch.Tensor, spatial_shapes: List[Tuple[int, int]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features (torch.Tensor): Flattened multi-scale features [B, Seq_Len, C]
            spatial_shapes (List[Tuple[int, int]]): Shapes of features from each level
            
        Returns:
            Dict containing final probabilities, objectness, bboxes, and sparse mil probs.
        """
        # 1. Generate Class-Agnostic Proposals
        bboxes, objectness_scores, hs = self.proposal_generator(features, spatial_shapes)
        # hs shape: [B, 300, 256]
        # objectness_scores shape: [B, 300, 1]

        # 2. Sparsemax MIL Classifier
        mil_logits = self.mil_classifier(hs) # [B, 300, Num_Classes]
        # In testing phase, Sparsemax can be too aggressive if the model hasn't fully converged.
        # It forces almost everything to exact 0. We can fall back to softmax for testing 
        # or stick to sparsemax if fully trained. Let's see what mil_probs look like.
        mil_probs = self.sparsemax(mil_logits) # [B, 300, Num_Classes]

        # 3. Joint Probability Estimation
        # Final_Prob = MIL_Prob * Objectness_Scores
        # This suppresses proposals that have high MIL score (local discriminative part) 
        # but low objectness (not a complete object).
        # We also clamp to avoid exact 0s which might cause issues downstream.
        final_prob = (mil_probs * objectness_scores).clamp(min=1e-8, max=1.0)
        
        # If in eval mode and everything is clamped to 1e-8, it means Sparsemax killed everything
        # or Objectness is exactly 0. Let's return raw logits too for debugging if needed.
        return {
            "bboxes": bboxes,
            "objectness_scores": objectness_scores,
            "mil_probs": mil_probs,
            "mil_logits": mil_logits,
            "final_prob": final_prob
        }


if __name__ == '__main__':
    print("=== Testing Sparse MIL Detection Head ===")
    
    # Configuration
    B = 2
    Seq_Len = 400 # Mock sequence length of flattened FPN features
    C = 256
    Num_Classes = 20
    Num_Queries = 300
    
    # Dummy flattened features (e.g., from P5 or concatenated FPN levels)
    dummy_features = torch.randn(B, Seq_Len, C)
    spatial_shapes = [(10, 10), (10, 10), (10, 10), (10, 10)] # Mock spatial shapes
    print(f"Input Features Shape: {dummy_features.shape}")

    # Instantiate Head
    mil_head = SparseMILHead(num_classes=Num_Classes, hidden_dim=C, num_queries=Num_Queries)
    
    # Forward Pass
    outputs = mil_head(dummy_features, spatial_shapes)
    
    bboxes = outputs["bboxes"]
    obj_scores = outputs["objectness_scores"]
    mil_probs = outputs["mil_probs"]
    final_prob = outputs["final_prob"]
    
    print("\n--- Output Shapes ---")
    print(f"BBoxes: {bboxes.shape}")
    print(f"Objectness Scores: {obj_scores.shape}")
    print(f"MIL Probs (Sparse): {mil_probs.shape}")
    print(f"Final Joint Probs: {final_prob.shape}")
    
    # Validate Shapes
    assert bboxes.shape == (B, Num_Queries, 4)
    assert obj_scores.shape == (B, Num_Queries, 1)
    assert mil_probs.shape == (B, Num_Queries, Num_Classes)
    assert final_prob.shape == (B, Num_Queries, Num_Classes)
    
    # Validate Sparsemax Truncation Effect
    # Check how many probabilities are exactly zero
    zeros_count = (mil_probs == 0).sum().item()
    total_elements = mil_probs.numel()
    sparsity_ratio = zeros_count / total_elements
    print(f"\n--- Sparsemax Analysis ---")
    print(f"Total Elements in MIL Probs: {total_elements}")
    print(f"Number of Exact Zeros: {zeros_count}")
    print(f"Sparsity Ratio: {sparsity_ratio:.2%}")
    
    # Ensure Sparsemax sums to 1 over the query dimension (dim=1)
    # Check for batch 0, class 0
    sum_probs = mil_probs[0, :, 0].sum().item()
    print(f"Sum of MIL Probs for Batch 0, Class 0: {sum_probs:.4f}")
    assert abs(sum_probs - 1.0) < 1e-4, "Sparsemax probabilities must sum to 1 over dim=1"

    print("\n✅ Test Passed: Sparse MIL Head forward pass and Sparsemax functionality verified.")
