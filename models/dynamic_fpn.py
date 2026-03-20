"""
Dynamic Attention-based Feature Pyramid Network (FPN) for WS-TFA.
Replaces traditional 1:1 FPN addition with a dynamic attention mechanism
to adaptively fuse deep semantic features with shallow detailed features.
Also integrates the Feature Supplement Module (FSM) to refine P2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import sys
import os

# Ensure models module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsm import FeatureSupplementModule


class FusionFactorModule(nn.Module):
    """
    Computes a dynamic weight factor (alpha) for adaptive feature fusion.
    Concatenates deep upsampled features and shallow lateral features,
    applies a lightweight convolution, global average pooling, and sigmoid.
    """

    def __init__(self, in_channels: int) -> None:
        """
        Initializes the FusionFactorModule.

        Args:
            in_channels (int): The number of channels of a single feature map.
                               The concatenated input will have 2 * in_channels.
        """
        super().__init__()
        # Lightweight convolution to process concatenated features
        # We reduce channels for efficiency, e.g., to in_channels // 2 or a fixed small number
        inter_channels = max(in_channels // 2, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            # Output a single channel for the spatial alpha
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, deep_feat: torch.Tensor, shallow_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the fusion factor alpha.

        Args:
            deep_feat (torch.Tensor): Upsampled feature map from deeper layer.
            shallow_feat (torch.Tensor): Lateral feature map from shallower layer.

        Returns:
            torch.Tensor: Dynamic spatial weight factor alpha of shape (B, 1, H, W).
        """
        # Ensure spatial dimensions match (handle potential odd sizes)
        if deep_feat.shape[2:] != shallow_feat.shape[2:]:
            deep_feat = F.interpolate(deep_feat, size=shallow_feat.shape[2:], mode='nearest')

        # Concatenate along channel dimension
        concat_feat = torch.cat([deep_feat, shallow_feat], dim=1)

        # Compute alpha: (B, 2C, H, W) -> (B, 1, H, W) -> Sigmoid
        x = self.conv(concat_feat)
        alpha = self.sigmoid(x)

        return alpha


class DynamicAttentionFPN(nn.Module):
    """
    Dynamic Attention-based FPN.
    Builds P5 down to P2 using Dynamic Attention Fusion,
    and refines P2 into P2_prime using the FSM module and C1.
    """

    def __init__(
        self,
        in_channels_list: List[int] = [256, 512, 1024, 2048],
        out_channels: int = 256,
        c1_channels: int = 64
    ) -> None:
        """
        Initializes the DynamicAttentionFPN.

        Args:
            in_channels_list (List[int]): Channels of C2, C3, C4, C5.
            out_channels (int): Channels of the FPN output layers (P2-P5).
            c1_channels (int): Channels of C1, used for FSM.
        """
        super().__init__()
        
        # Lateral convolutions for C2, C3, C4, C5
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])

        # Output convolutions for P2, P3, P4, P5 (anti-aliasing after fusion)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

        # Fusion factor modules for P4, P3, P2 (3 fusions needed for 4 levels)
        self.fusion_modules = nn.ModuleList([
            FusionFactorModule(out_channels) for _ in range(len(in_channels_list) - 1)
        ])

        # Feature Supplement Module (FSM) to refine P2 -> P2_prime using C1
        self.fsm = FeatureSupplementModule(
            c1_channels=c1_channels,
            p2_channels=out_channels,
            dilation=2
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> dict:
        """
        Forward pass for the Dynamic Attention FPN.

        Args:
            features (Dict[str, torch.Tensor]): Dict containing 'C1' to 'C5' from backbone.

        Returns:
            dict: Dict containing 'P2', 'P2_prime', 'P3', 'P4', 'P5',
                  and a debug dict 'alphas' containing the learned weights.
        """
        # Extract features
        c1 = features['C1']
        c_laterals = [
            features['C2'],
            features['C3'],
            features['C4'],
            features['C5']
        ]

        # 1. Lateral Connections
        laterals = [
            lat_conv(c) for lat_conv, c in zip(self.lateral_convs, c_laterals)
        ]

        # 2. Top-down Pathway with Dynamic Attention Fusion
        # laterals: [P2_lat, P3_lat, P4_lat, P5_lat]
        # Build P5 first
        p_out = [laterals[-1]]
        
        # We will store the alphas for debugging/visualization
        alphas = {}

        # Iterate backwards from P4 down to P2
        for i in range(len(laterals) - 2, -1, -1):
            prev_p = p_out[-1] # The deeper P layer
            curr_lat = laterals[i] # The current lateral C layer
            
            # Upsample deeper layer
            prev_p_upsampled = F.interpolate(
                prev_p, size=curr_lat.shape[2:], mode='nearest'
            )

            # Compute dynamic weight alpha
            fusion_mod = self.fusion_modules[i]
            alpha = fusion_mod(prev_p_upsampled, curr_lat)
            
            # Store alpha for logging (layer index: P4 is idx 2, P3 is idx 1, P2 is idx 0)
            level_name = f"P{i+2}_alpha"
            alphas[level_name] = alpha

            # Dynamic Fusion: P_i = alpha * P_{i+1}_upsampled + (1 - alpha) * C_i_lat
            fused = alpha * prev_p_upsampled + (1 - alpha) * curr_lat
            
            p_out.append(fused)

        # Reverse p_out to align with [P2, P3, P4, P5]
        p_out = p_out[::-1]

        # 3. FPN Output Convolutions (Anti-aliasing)
        fpn_outs = [
            fpn_conv(p) for fpn_conv, p in zip(self.fpn_convs, p_out)
        ]
        
        p2, p3, p4, p5 = fpn_outs

        # 4. Refine P2 into P2_prime using FSM and C1
        p2_prime = self.fsm(c1, p2)

        return {
            "P2": p2,
            "P2_prime": p2_prime,
            "P3": p3,
            "P4": p4,
            "P5": p5,
            "alphas": alphas # Exposing alphas for inspection
        }

if __name__ == '__main__':
    # Simple test block to verify Dynamic FPN forward pass and alpha dimensions
    print("=== Testing Dynamic Attention FPN ===")
    
    from backbone import ResNet50Backbone

    # Instantiate backbone
    backbone = ResNet50Backbone(pretrained=False)
    
    # Instantiate Dynamic FPN
    # ResNet50 channels: C2=256, C3=512, C4=1024, C5=2048; C1=64
    dynamic_fpn = DynamicAttentionFPN(
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256,
        c1_channels=64
    )

    # Dummy input
    dummy_input = torch.randn(2, 3, 512, 512)
    print(f"Input Image Shape: {dummy_input.shape}")

    # Forward pass
    features_c = backbone(dummy_input)
    features_p = dynamic_fpn(features_c)

    print("\n--- Output Feature Shapes ---")
    print(f"P2_prime: {features_p['P2_prime'].shape}")
    print(f"P3: {features_p['P3'].shape}")
    print(f"P4: {features_p['P4'].shape}")
    print(f"P5: {features_p['P5'].shape}")

    print("\n--- Learned Dynamic Alphas ---")
    alphas = features_p['alphas']
    for level, alpha_tensor in alphas.items():
        print(f"{level} shape: {alpha_tensor.shape}, sample value (batch 0): {alpha_tensor[0, 0, 0, 0].item():.4f}")

    # Validation
    assert features_p['P2_prime'].shape == features_c['C2'].shape, "P2_prime spatial dimensions should match C2"
    assert features_p['P5'].shape == (2, 256, 16, 16), "P5 should be downsampled by 32 (512/32 = 16)"
    
    print("\n✅ Test Passed: Dynamic Attention FPN forward pass and alpha generation are correct.")
