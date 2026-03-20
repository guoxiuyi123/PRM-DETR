"""
Inference Pipeline and Post-processing for WS-TFA.
Handles forward pass, confidence filtering, coordinate conversion, and NMS.
"""

import torch
import torchvision
from typing import Dict, List, Tuple
import sys
import os

# Ensure models can be imported
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
from models.ws_tfa import WS_TFA_Net

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2).
    
    Args:
        boxes (torch.Tensor): Tensor of shape (..., 4) with [cx, cy, w, h] format.
        
    Returns:
        torch.Tensor: Tensor of shape (..., 4) with [x1, y1, x2, y2] format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)

def predict(
    model: torch.nn.Module, 
    image_tensor: torch.Tensor, 
    conf_threshold: float = 0.3,
    nms_iou_threshold: float = 0.4
) -> List[Dict[str, torch.Tensor]]:
    """
    Runs inference on a batch of images and applies post-processing.
    
    Args:
        model (nn.Module): The trained WS-TFA network.
        image_tensor (torch.Tensor): Input images of shape [B, 3, H, W].
        conf_threshold (float): Minimum confidence score to retain a bounding box.
        nms_iou_threshold (float): IoU threshold for Non-Maximum Suppression.
        
    Returns:
        List[Dict[str, torch.Tensor]]: A list of dictionaries (one for each image in batch).
            Each dict contains:
            - 'boxes': [N, 4] absolute coordinates [x1, y1, x2, y2]
            - 'scores': [N] confidence scores
            - 'labels': [N] class indices
    """
    model.eval()
    
    B, C, H, W = image_tensor.shape
    results = []
    
    with torch.no_grad():
        # 1. Forward Pass
        outputs = model(image_tensor)
        
        final_probs = outputs['final_prob'] # [B, 300, Num_Classes]
        pred_boxes = outputs['bboxes'] # [B, 300, 4] in [cx, cy, w, h] normalized [0, 1]
        
        # 2. Process each image in the batch
        for b in range(B):
            probs = final_probs[b] # [300, Num_Classes]
            boxes = pred_boxes[b] # [300, 4]
            
            # Note: Final probs are non-negative due to Sparsemax and Sigmoid.
            # But let's check max just in case
            
            # Find maximum probability and corresponding class for each query
            scores, labels = probs.max(dim=-1) # scores: [300], labels: [300]
            
            # 3. Confidence Filtering
            keep_mask = scores > conf_threshold
            
            # If we explicitly pass -1.0, we want everything to pass to test NMS
            if conf_threshold < 0:
                keep_mask = torch.ones_like(scores, dtype=torch.bool)
            
            filtered_boxes = boxes[keep_mask]
            filtered_scores = scores[keep_mask]
            filtered_labels = labels[keep_mask]
            
            # If no boxes survive the threshold
            if filtered_boxes.shape[0] == 0:
                results.append({
                    'boxes': torch.empty((0, 4), device=image_tensor.device),
                    'scores': torch.empty((0,), device=image_tensor.device),
                    'labels': torch.empty((0,), dtype=torch.int64, device=image_tensor.device)
                })
                continue
                
            # 4. Coordinate Conversion
            # Convert [cx, cy, w, h] normalized to [x1, y1, x2, y2] absolute
            xyxy_boxes = cxcywh_to_xyxy(filtered_boxes)
            
            # Scale up to image dimensions
            scale_tensor = torch.tensor([W, H, W, H], dtype=torch.float32, device=xyxy_boxes.device)
            abs_boxes = xyxy_boxes * scale_tensor
            
            # Ensure coordinates are within image boundaries
            abs_boxes[:, 0::2].clamp_(min=0, max=W)
            abs_boxes[:, 1::2].clamp_(min=0, max=H)
            
            # 5. Non-Maximum Suppression (NMS)
            # Since multiple classes might overlap, we use batched_nms which performs NMS independently per class
            keep_indices = torchvision.ops.batched_nms(
                boxes=abs_boxes,
                scores=filtered_scores,
                idxs=filtered_labels,
                iou_threshold=nms_iou_threshold
            )
            
            final_boxes = abs_boxes[keep_indices]
            final_scores = filtered_scores[keep_indices]
            final_labels = filtered_labels[keep_indices]
            
            results.append({
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels
            })
            
    return results

if __name__ == '__main__':
    print("=== Testing WS-TFA Inference Pipeline ===")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 20
    img_size = 800
    
    # 1. Initialize Model
    print("Loading Model...")
    model = WS_TFA_Net(num_classes=num_classes, pretrained_backbone=False).to(device)
    
    # 2. Generate Random Image Tensor [Batch=1, Channels=3, H=800, W=800]
    dummy_image = torch.randn(1, 3, img_size, img_size).to(device)
    print(f"Input Image Shape: {dummy_image.shape}")
    
    # For testing purposes, let's artificially boost some random query scores
    # so they survive the confidence threshold and we can test NMS.
    # We will inject this directly into the model's output temporarily or just use a low threshold.
    # Since model is random, max score might be very low. We'll use an extremely low threshold for the dummy test.
    test_threshold = -1.0 # Guarantee everything passes to test NMS
    
    
    # 3. Run Inference
    print(f"\nRunning Inference (Threshold = {test_threshold})...")
    predictions = predict(
        model=model, 
        image_tensor=dummy_image, 
        conf_threshold=test_threshold, 
        nms_iou_threshold=0.4
    )
    
    # 4. Print Results
    result = predictions[0] # Get first image in batch
    boxes = result['boxes']
    scores = result['scores']
    labels = result['labels']
    
    print(f"\n--- Final Detections (After NMS) ---")
    print(f"Total Objects Detected: {boxes.shape[0]}")
    
    if boxes.shape[0] > 0:
        # Print top 5 detections
        print("\nTop Detections:")
        # Sort by score just for printing
        sorted_scores, sorted_idx = torch.sort(scores, descending=True)
        for i in range(min(5, len(sorted_idx))):
            idx = sorted_idx[i]
            box = boxes[idx].cpu().numpy()
            score = scores[idx].item()
            label = labels[idx].item()
            
            print(f"  [{i+1}] Class: {label:2d} | Score: {score:.4f} | Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            
    print("\n✅ Inference Pipeline Test Passed!")