import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # 修复了 Warning
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm

from models.ws_tfa import WS_TFA_Net
from models.loss import WSTFALoss
from data.voc_dataset import WSOD_VOCDataset, get_wsod_transforms, VOC_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="Train WS-TFA on PASCAL VOC")
    parser.add_argument('--data_dir', type=str, default='/home/pc/gxy/WS-TFA/data', help='Path to VOC dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (adjust based on VRAM)')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Learning rate for backbone')
    parser.add_argument('--lr_transformer', type=float, default=1e-4, help='Learning rate for FPN and Transformer Head')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Epochs before enabling Box Loss')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"=== Starting WS-TFA Training on {args.device.upper()} ===")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'wstfa_voc_{timestamp}')
    writer = SummaryWriter(log_dir)
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Initializing Model...")
    model = WS_TFA_Net(
        num_classes=len(VOC_CLASSES), 
        pretrained_backbone=True,
        fpn_out_channels=256,
        num_queries=300
    ).to(args.device)
    
    criterion = WSTFALoss(alpha_reg_weight=0.01, box_loss_weight=1.0, top_k_pseudo=3).to(args.device)
    
    print("Preparing Dataloader...")
    train_transforms = get_wsod_transforms(is_train=True, target_size=800)
    
    # 🚨 核心修复：移除了危险的 try-except，并且强行指定 download=False 绕过死掉的官方服务器
    dataset = WSOD_VOCDataset(
        root=args.data_dir,
        year='2007',
        image_set='trainval',
        download=False, 
        transforms=train_transforms
    )
    
    dataset_len = len(dataset)
    print(f"\n✅ 成功加载了 {dataset_len} 张 PASCAL VOC 真实图片！\n")
    
    if dataset_len == 0:
        raise RuntimeError("Dataset is empty. 你的 trainval.txt 可能被破坏了！")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if args.device == 'cuda' else 0,
        pin_memory=(args.device == 'cuda'),
        drop_last=False # 保证不会除以 0
    )
    
    # Optimizer and Scheduler
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    transformer_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
            
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': transformer_params, 'lr': args.lr_transformer}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler('cuda') # 修复了 Warning
    
    best_mil_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_total_loss, epoch_mil_loss, epoch_box_loss = 0.0, 0.0, 0.0
        
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        if epoch < args.warmup_epochs:
            print("Status: WARM-UP (Box Loss is 0)")
        else:
            print("Status: FULL TRAINING (Box Loss Active)")
            
        pbar = tqdm(dataloader, desc="Training")
        
        for images, labels in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=(args.device == 'cuda')):
                outputs = model(images)
                total_loss, loss_dict = criterion(outputs, labels, current_epoch=epoch, warmup_epochs=args.warmup_epochs)
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_total_loss += total_loss.item()
            epoch_mil_loss += loss_dict['loss_mil'].item()
            epoch_box_loss += loss_dict.get('loss_box', torch.tensor(0.0)).item()
            
            pbar.set_postfix({'Tot': f"{total_loss.item():.3f}", 'MIL': f"{loss_dict['loss_mil'].item():.3f}"})
            
            writer.add_scalar('Step/Total_Loss', total_loss.item(), global_step)
            global_step += 1
            
        num_batches = len(dataloader)
        avg_total_loss = epoch_total_loss / num_batches
        avg_mil_loss = epoch_mil_loss / num_batches
        
        print(f"Epoch {epoch+1} Summary: Avg Total: {avg_total_loss:.4f} | Avg MIL: {avg_mil_loss:.4f}")
        scheduler.step()
        
        if avg_mil_loss < best_mil_loss:
            best_mil_loss = avg_mil_loss
            save_path = os.path.join('checkpoints', 'ws_tfa_best.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, save_path)
            print(f"⭐ New best MIL loss ({best_mil_loss:.4f})! Saved to {save_path}")
            
    writer.close()
    print("\n🎉 Training Complete!")

if __name__ == '__main__':
    main()