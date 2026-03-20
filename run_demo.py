import os
import torch
import cv2
import numpy as np
import torchvision.transforms as T
import torchvision.ops as ops
import matplotlib.pyplot as plt

# 导入类别和模型
try:
    from data.voc_dataset import VOC_CLASSES
except ImportError:
    from dataloaders.voc_dataset import VOC_CLASSES

from models.ws_tfa import WS_TFA_Net

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
    print(f"✅ 成功加载权重 (Epoch {checkpoint.get('epoch', 'unknown')})")

    # 2. 随机挑选一张 VOC 的真实测试图片
    img_name = '000005.jpg' 
    img_path = f'/home/pc/gxy/WS-TFA/data/VOCdevkit/VOC2007/JPEGImages/{img_name}'
    
    if not os.path.exists(img_path):
        print(f"找不到图片 {img_path}，自动从目录抓取第一张...")
        img_dir = '/home/pc/gxy/WS-TFA/data/VOCdevkit/VOC2007/JPEGImages/'
        img_name = os.listdir(img_dir)[0]
        img_path = os.path.join(img_dir, img_name)
        
    print(f"Testing on image: {img_path}")
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w, _ = original_img.shape

    # 3. 图像预处理 (和训练时保持一致)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # 4. 前向推理
    print("Running inference...")
    
    # 🚨 架构师特调阈值：弱监督的联合概率通常在 0.1 左右，所以必须设低！
    CONF_THRESH = 0.05  
    IOU_THRESH = 0.3    # NMS 阈值

    with torch.no_grad():
        # 强制退回 FP32 保证推理稳定
        with torch.amp.autocast('cuda', enabled=False):
            outputs = model(input_tensor)
            
            # 🚨 直接使用探测到的准确键名
            final_probs = outputs['final_prob'][0]  # [300, 20]
            pred_boxes = outputs['bboxes'][0]       # [300, 4]
            
            # 兼容模型输出 logits 需要经过 sigmoid 的情况
            if final_probs.max() > 1.0 or final_probs.min() < 0.0:
                final_probs = torch.sigmoid(final_probs)
            
            # 找到每个 Query 的最高得分类别
            scores, labels = torch.max(final_probs, dim=-1)
            
            # 🌟 探照灯：打印全图最高的 5 个得分
            top5_scores, _ = torch.topk(scores, min(5, len(scores)))
            print(f"  -> [DEBUG] 当前图片 Top-5 原始得分: {top5_scores.cpu().numpy()}")
            
            # 过滤掉低于阈值的背景框
            keep = scores > CONF_THRESH
            boxes = pred_boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # 坐标反归一化 (从 0~1 恢复到原图的像素坐标)
            if len(boxes) > 0:
                boxes[:, 0] = boxes[:, 0] * w  # cx
                boxes[:, 1] = boxes[:, 1] * h  # cy
                boxes[:, 2] = boxes[:, 2] * w  # w
                boxes[:, 3] = boxes[:, 3] * h  # h
                # [cx, cy, w, h] 转换为 [x1, y1, x2, y2]
                x1 = boxes[:, 0] - boxes[:, 2] / 2
                y1 = boxes[:, 1] - boxes[:, 3] / 2
                x2 = boxes[:, 0] + boxes[:, 2] / 2
                y2 = boxes[:, 1] + boxes[:, 3] / 2
                converted_boxes = torch.stack([x1, y1, x2, y2], dim=1)

                # NMS (非极大值抑制) 去除重叠框
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
        
        # 在画面上画框
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'{cls_name}: {conf:.2f}', bbox=dict(facecolor='red', alpha=0.5), fontsize=12, color='white')

    plt.axis('off')
    output_file = 'demo_output.jpg'
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"\n🎉 图像已保存至: {output_file}")

if __name__ == '__main__':
    main()