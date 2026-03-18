import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prm_detr import PRMDETR
from models.backbone import ResNetBackbone
from models.neck.hybrid_encoder_prm import HybridEncoderPRM
from models.head.prm_decoder import PRMDecoder

def parse_args():
    parser = argparse.ArgumentParser(description="PRM-DETR Inference")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--weights", default="checkpoints/prm_detr_latest.pth", help="Path to model weights")
    parser.add_argument("--output_dir", default="outputs/", help="Directory to save the result")
    parser.add_argument("--conf_thresh", type=float, default=0.3, help="Confidence threshold")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 构建模型
    backbone = ResNetBackbone()
    neck = HybridEncoderPRM(in_channels_list=[64, 128, 256, 512], hidden_dim=256)
    head = PRMDecoder(num_classes=10, hidden_dim=256, num_queries=200)
    model = PRMDETR(backbone, neck, head).to(device)
    
    # 加载权重
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        # 如果是包含了完整状态字典的 checkpoint，提取 model_state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Weights file {args.weights} not found. Using random weights.")
        
    model.eval()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取图像
    print(f"Processing image: {args.image_path}")
    image_np = cv2.imread(args.image_path)
    if image_np is None:
        raise ValueError(f"Could not read image from {args.image_path}")
        
    orig_h, orig_w = image_np.shape[:2]
    
    # 预处理：BGR转RGB，缩放至640x640，转Tensor
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize((640, 640))
    img_tensor = TF.to_tensor(pil_img).unsqueeze(0).to(device)  # [1, 3, 640, 640]
    
    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
        
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_points = outputs['pred_points'][0]  # [num_queries, 2]
    
    # 计算得分
    probs = pred_logits.sigmoid()
    scores, labels = probs.max(dim=-1)
    
    # 筛选
    keep = scores > args.conf_thresh
    final_scores = scores[keep]
    final_labels = labels[keep]
    final_points = pred_points[keep]
    
    # 可视化
    result_img = image_np.copy()
    count = 0
    
    for i in range(len(final_scores)):
        norm_x, norm_y = final_points[i].cpu().numpy()
        class_id = final_labels[i].item()
        
        # 还原坐标
        x = int(norm_x * orig_w)
        y = int(norm_y * orig_h)
        
        # 画红色圆点
        cv2.circle(result_img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
        
        # 画类别文本 (绿色，右上偏移)
        text = f"C{class_id}"
        cv2.putText(result_img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        count += 1
        
    # 保存结果
    base_name = os.path.basename(args.image_path)
    out_path = os.path.join(args.output_dir, f"result_{base_name}")
    cv2.imwrite(out_path, result_img)
    
    print(f"成功检测到 {count} 个目标，结果保存在: {out_path}")

if __name__ == "__main__":
    main()
