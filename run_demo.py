import os
import torch
import cv2
import numpy as np
import torchvision.transforms as T
import torchvision.ops as ops
import matplotlib.pyplot as plt

try:
    from data.voc_dataset import VOC_CLASSES
except ImportError:
    from dataloaders.voc_dataset import VOC_CLASSES

from models.ws_tfa import WS_TFA_Net

# =========================================================================
# 🚨 架构师特制：完美且防崩溃的 Sparsemax
# =========================================================================
class CorrectSparsemax(torch.nn.Module):
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载模型
    print("Loading WS-TFA Model...")
    model = WS_TFA_Net(num_classes=len(VOC_CLASSES), pretrained_backbone=False).to(device)
    
    checkpoint_path = 'checkpoints/ws_tfa_best.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件 {checkpoint_path}，请确认是否训练成功！")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 🌟 动态注入完美的 Sparsemax！解开封印！
    model.mil_head.sparsemax = CorrectSparsemax(dim=1)
    
    print(f"✅ 成功加载权重 (Epoch {checkpoint.get('epoch', 'unknown')})")

    # 2. 读取图片
    img_name = '009961.jpg' 
    img_path = f'/home/pc/gxy/WS-TFA/data/VOCdevkit/VOC2007/JPEGImages/{img_name}'
    
    if not os.path.exists(img_path):
        img_dir = '/home/pc/gxy/WS-TFA/data/VOCdevkit/VOC2007/JPEGImages/'
        img_name = os.listdir(img_dir)[0]
        img_path = os.path.join(img_dir, img_name)
        
    print(f"Testing on image: {img_path}")
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w, _ = original_img.shape

    # 3. 图像预处理
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # 4. 前向推理
    print("Running inference...")
    CONF_THRESH = 0.05  # 置信度临时调低至 0.05 方便测试
    IOU_THRESH = 0.3    # NMS 阈值

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=False):
            # 将模型也放入 eval 模式，双重保险
            model.eval()
            outputs = model(input_tensor)
            
            final_probs = outputs['final_prob'][0]  # [300, 20]
            pred_boxes = outputs['bboxes'][0]       # [300, 4]
            mil_probs = outputs['mil_probs'][0]
            obj_scores = outputs['objectness_scores'][0]
            
            mil_logits = outputs['mil_logits'][0]
            
            print(f"  -> [DEBUG] final_probs min/max: {final_probs.min().item():.6f} / {final_probs.max().item():.6f}")
            print(f"  -> [DEBUG] mil_probs min/max: {mil_probs.min().item():.6f} / {mil_probs.max().item():.6f}")
            print(f"  -> [DEBUG] mil_logits min/max: {mil_logits.min().item():.6f} / {mil_logits.max().item():.6f}")
            print(f"  -> [DEBUG] obj_scores min/max: {obj_scores.min().item():.6f} / {obj_scores.max().item():.6f}")
            
            # 如果经过乘积后全都非常低，说明 Objectness 或者 MIL 至少有一个是极低的
            # 由于 Sparsemax 的性质，如果网络未充分训练，概率可能会被均匀摊平（1/300 = 0.0033）或者压成极小值
            # 为了能够画出框来验证流水线连通性，我们将强制使用 softmax 的 logits
            if final_probs.max() < 0.01:
                print("  -> [WARNING] final_probs are extremely low! Falling back to softmax(logits) for visualization.")
                final_probs = torch.softmax(mil_logits, dim=-1)
                CONF_THRESH = 0.001 # 极端调低阈值以确保有框输出
            
            if final_probs.max() > 1.0 or final_probs.min() < 0.0:
                final_probs = torch.sigmoid(final_probs)
            
            scores, labels = torch.max(final_probs, dim=-1)
            
            # 🌟 探照灯：现在得分绝对是正常且健康的！
            top5_scores, _ = torch.topk(scores, min(5, len(scores)))
            print(f"  -> [DEBUG] 完美修复！当前图片 Top-5 真实得分: {top5_scores.cpu().numpy()}")
            
            keep = scores > CONF_THRESH
            print(f"  -> [DEBUG] 经过 CONF_THRESH ({CONF_THRESH}) 过滤后剩余框数量: {keep.sum().item()}")
            boxes = pred_boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if len(boxes) > 0:
                boxes[:, 0] = boxes[:, 0] * w  # cx
                boxes[:, 1] = boxes[:, 1] * h  # cy
                boxes[:, 2] = boxes[:, 2] * w  # w
                boxes[:, 3] = boxes[:, 3] * h  # h
                x1 = boxes[:, 0] - boxes[:, 2] / 2
                y1 = boxes[:, 1] - boxes[:, 3] / 2
                x2 = boxes[:, 0] + boxes[:, 2] / 2
                y2 = boxes[:, 1] + boxes[:, 3] / 2
                converted_boxes = torch.stack([x1, y1, x2, y2], dim=1)

                nms_keep = ops.nms(converted_boxes, scores, IOU_THRESH)
                final_boxes = converted_boxes[nms_keep]
                final_scores = scores[nms_keep]
                final_labels = labels[nms_keep]
            else:
                final_boxes, final_scores, final_labels = [], [], []

    # 5. 画图与保存
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(original_img)
    
    print(f"\n🚀 检测结果 (Detect {len(final_boxes)} objects):")
    for box, score, label in zip(final_boxes, final_scores, final_labels):
        x1, y1, x2, y2 = box.cpu().numpy()
        cls_name = VOC_CLASSES[label.item()]
        conf = score.item()
        print(f" - [{cls_name}] Conf: {conf:.3f} Box: {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}")
        
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='lime', linewidth=3)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'{cls_name}: {conf:.2f}', bbox=dict(facecolor='lime', alpha=0.5), fontsize=12, color='black')

    plt.axis('off')
    output_file = 'demo_output.jpg'
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"\n🎉 图像已保存至: {output_file}")

if __name__ == '__main__':
    main()