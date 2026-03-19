"""
Main Training Script for WS-TFA (Weakly Supervised Transformer-Fusion Attention).
Implements the training loop, mixed precision (AMP), and TensorBoard logging.
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm

from models.ws_tfa import WS_TFA_Net
from models.loss import WSTFALoss
from data.voc_dataset import WSOD_VOCDataset, get_wsod_transforms, VOC_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="Train WS-TFA on PASCAL VOC")
    parser.add_argument('--data_dir', type=str, default='./data/voc', help='Path to VOC dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (adjust based on VRAM)')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Learning rate for backbone')
    parser.add_argument('--lr_transformer', type=float, default=1e-4, help='Learning rate for FPN and Transformer Head')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Epochs before enabling Box Loss')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Initialization
    print(f"=== Starting WS-TFA Training on {args.device.upper()} ===")
    
    # TensorBoard Writer
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'wstfa_voc_{timestamp}')
    writer = SummaryWriter(log_dir)
    print(f"Logging to {log_dir}")
    
    # Create checkpoints dir
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize Model
    print("Initializing Model...")
    model = WS_TFA_Net(
        num_classes=len(VOC_CLASSES), 
        pretrained_backbone=True,
        fpn_out_channels=256,
        num_queries=300
    ).to(args.device)
    
    # Initialize Loss
    criterion = WSTFALoss(
        alpha_reg_weight=0.01, 
        box_loss_weight=1.0, 
        top_k_pseudo=3
    ).to(args.device)
    
    # Dataloader
    print("Preparing Dataloader...")
    train_transforms = get_wsod_transforms(is_train=True, target_size=800)
    
    # We will use the fallback logic if VOC download fails
    try:
        dataset = WSOD_VOCDataset(
            root=args.data_dir,
            year='2007',
            image_set='trainval',
            download=True,
            transforms=train_transforms
        )
    except Exception as e:
        print(f"Warning: Official VOC download failed. Using dummy data for pipeline verification.")
        
        # Create dummy directory structure
        img_dir = os.path.join(args.data_dir, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        ann_dir = os.path.join(args.data_dir, 'VOCdevkit', 'VOC2007', 'Annotations')
        set_dir = os.path.join(args.data_dir, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(set_dir, exist_ok=True)
        
        # Create a dummy image
        import numpy as np
        from PIL import Image
        img = Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8))
        img.save(os.path.join(img_dir, '000001.jpg'))
        img.save(os.path.join(img_dir, '000002.jpg'))
        
        # Create a dummy XML annotation
        xml_content = """<annotation>
            <folder>VOC2007</folder>
            <filename>000001.jpg</filename>
            <object>
                <name>dog</name>
                <bndbox><xmin>48</xmin><ymin>240</ymin><xmax>195</xmax><ymax>371</ymax></bndbox>
            </object>
        </annotation>"""
        with open(os.path.join(ann_dir, '000001.xml'), 'w') as f: f.write(xml_content)
        with open(os.path.join(ann_dir, '000002.xml'), 'w') as f: f.write(xml_content.replace('000001', '000002'))
            
        # Create trainval.txt
        with open(os.path.join(set_dir, 'trainval.txt'), 'w') as f:
            f.write('000001\n000002\n')
            
        dataset = WSOD_VOCDataset(
            root=args.data_dir,
            year='2007',
            image_set='trainval',
            download=False,
            transforms=train_transforms
        )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    
    # 2. Optimizer and Scheduler
    # Separate learning rates: smaller for pretrained backbone, larger for new transformer layers
    backbone_params = []
    transformer_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            transformer_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': transformer_params, 'lr': args.lr_transformer}
    ], weight_decay=1e-4)
    
    # Cosine Annealing Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 3. Mixed Precision (AMP)
    scaler = GradScaler()
    
    # 4. Training Loop
    best_mil_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_mil_loss = 0.0
        epoch_box_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        if epoch < args.warmup_epochs:
            print("Status: WARM-UP (Box Loss is 0)")
        else:
            print("Status: FULL TRAINING (Box Loss Active)")
            
        pbar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            optimizer.zero_grad()
            
            # Forward pass with Mixed Precision
            with autocast(enabled=(args.device == 'cuda')):
                outputs = model(images)
                
                # Compute Loss
                total_loss, loss_dict = criterion(
                    outputs, 
                    labels, 
                    current_epoch=epoch, 
                    warmup_epochs=args.warmup_epochs
                )
            
            # Backward pass with Gradient Scaling
            scaler.scale(total_loss).backward()
            
            # Gradient Clipping to prevent exploding gradients in Transformer
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            epoch_total_loss += total_loss.item()
            epoch_mil_loss += loss_dict['loss_mil'].item()
            epoch_box_loss += loss_dict['loss_box'].item()
            
            pbar.set_postfix({
                'Total': f"{total_loss.item():.3f}",
                'MIL': f"{loss_dict['loss_mil'].item():.3f}",
                'Box': f"{loss_dict['loss_box'].item():.3f}"
            })
            
            # TensorBoard step-level logging
            writer.add_scalar('Step/Total_Loss', total_loss.item(), global_step)
            writer.add_scalar('Step/MIL_Loss', loss_dict['loss_mil'].item(), global_step)
            writer.add_scalar('Step/Box_Loss', loss_dict['loss_box'].item(), global_step)
            writer.add_scalar('Step/Alpha_Loss', loss_dict['loss_alpha_reg'].item(), global_step)
            
            global_step += 1
            
        # Epoch-level processing
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_mil_loss = epoch_mil_loss / len(dataloader)
        avg_box_loss = epoch_box_loss / len(dataloader)
        
        # Log Learning Rates
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_transformer = optimizer.param_groups[1]['lr']
        writer.add_scalar('Epoch/LR_Backbone', current_lr_backbone, epoch)
        writer.add_scalar('Epoch/LR_Transformer', current_lr_transformer, epoch)
        
        writer.add_scalar('Epoch/Avg_Total_Loss', avg_total_loss, epoch)
        writer.add_scalar('Epoch/Avg_MIL_Loss', avg_mil_loss, epoch)
        writer.add_scalar('Epoch/Avg_Box_Loss', avg_box_loss, epoch)
        
        print(f"Epoch {epoch+1} Summary: Avg Total: {avg_total_loss:.4f} | Avg MIL: {avg_mil_loss:.4f} | Avg Box: {avg_box_loss:.4f}")
        
        # Step the scheduler
        scheduler.step()
        
        # 5. Model Saving
        if avg_mil_loss < best_mil_loss:
            best_mil_loss = avg_mil_loss
            save_path = os.path.join('checkpoints', 'ws_tfa_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mil_loss': best_mil_loss,
            }, save_path)
            print(f"⭐ New best MIL loss ({best_mil_loss:.4f})! Model saved to {save_path}")
            
    writer.close()
    print("\n🎉 Training Complete!")

if __name__ == '__main__':
    main()