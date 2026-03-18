import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prm_detr import PRMDETR
from models.backbone import ResNetBackbone
from models.neck.hybrid_encoder_prm import HybridEncoderPRM
from models.head.prm_decoder import PRMDecoder
from engine.matcher import PointMatcher
from engine.criterion import PointCriterion
from utils.metrics import PointEvaluator
from dataset import build_dataloader

def main():
    # 1. 解析配置
    config_path = 'configs/dataset/visdrone_point.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Loaded config from {config_path}")
    
    # 手动注入 batch_size
    if 'batch_size' not in config:
        config['batch_size'] = 8
    
    # 2. 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. 构建 DataLoader
    print("Building DataLoaders...")
    train_loader = build_dataloader(config, is_train=True)
    val_loader = build_dataloader(config, is_train=False)
    print(f"Train loader size: {len(train_loader)}, Val loader size: {len(val_loader)}")

    # 4. 实例化模型组件
    backbone = ResNetBackbone()
    neck = HybridEncoderPRM(in_channels_list=[64, 128, 256, 512], hidden_dim=256)
    # 将 num_queries 提高到 200，以适应 VisDrone 中密集的小目标
    head = PRMDecoder(num_classes=config['num_classes'], hidden_dim=256, num_queries=200)
    model = PRMDETR(backbone, neck, head).to(device)

    # 5. 实例化损失函数
    matcher = PointMatcher()
    weight_dict = {"loss_ce": 1.0, "loss_point": 5.0}
    criterion = PointCriterion(num_classes=config['num_classes'], matcher=matcher, weight_dict=weight_dict).to(device)

    # 6. 设置优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
    torch.backends.cudnn.benchmark = True # 开启底层加速
    
    # 初始化验证器
    evaluator = PointEvaluator(distance_thresholds=[0.01, 0.05, 0.1])

    # 7. 运行训练循环
    num_epochs = 50
    log_file = open('train_log.txt', 'a')
    print(f"Starting training loop for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        # --- 训练阶段 ---
        model.train()
        criterion.train()
        total_train_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # 数据迁移到 device
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            # 🚀 1. 纯 FP32 前向传播 (原汁原味，绝不溢出)
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            
            # 🚀 2. 反向传播求导
            total_loss.backward()
            
            # 🚀 3. 🚨 必须保留！极其重要的 DETR 梯度裁剪防爆神器
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            # 🚀 4. 更新权重
            optimizer.step()
                       
            total_train_loss += total_loss.item()
            
            # 打印日志
            if batch_idx % 50 == 0:
                loss_ce = loss_dict['loss_ce'].item()
                loss_point = loss_dict['loss_point'].item()
                log_str = f"Epoch [{epoch}/{num_epochs}][{batch_idx}/{len(train_loader)}] | loss_ce: {loss_ce:.4f} | loss_point: {loss_point:.4f} | Total: {total_loss.item():.4f}"
                print(log_str)
                log_file.write(log_str + '\n')
                log_file.flush()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_log_str = f"Epoch {epoch} finished. Avg Train Loss: {avg_train_loss:.4f}"
        print(avg_log_str)
        log_file.write(avg_log_str + '\n')
        log_file.flush()
        
        # 学习率步进
        lr_scheduler.step()
        
        # --- 验证阶段 ---
        print("Running validation...")
        model.eval()
        evaluator.reset()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(device)
                outputs = model(images)
                pred_logits = outputs['pred_logits']
                pred_points = outputs['pred_points']
                
                bs = images.shape[0]
                for i in range(bs):
                    probs = pred_logits[i].sigmoid()
                    scores, _ = probs.max(dim=-1)
                    points = pred_points[i]
                    
                    # 确保真实标签也放在显卡上，否则后面计算距离会报错
                    gt_points = targets[i]['points'].to(device)
                    
                    # 过滤低分预测加速评估
                    keep = scores > 0.05
                    evaluator.update(
                        pred_points=points[keep],
                        pred_scores=scores[keep],
                        gt_points=gt_points
                    )
                    
        # 累积并打印指标
        aps = evaluator.accumulate()
        print("-" * 50)
        
        # 将验证结果写入日志
        log_file.write(f"Epoch {epoch} Validation Results:\n")
        for k, v in aps.items():
            log_file.write(f"  {k} : {v:.4f}\n")
        log_file.flush()
        
        # 保存权重
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }, 'checkpoints/prm_detr_latest.pth')
        print(f"🚀 Epoch {epoch} 权重已成功保存...")

    log_file.close()
    print("Training loop finished successfully!")

if __name__ == "__main__":
    main()