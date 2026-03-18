import torch
from torch.utils.data import DataLoader
from .point_dataset import PointSupervisedDataset

def point_collate_fn(batch):
    """
    自定义的 collate_fn。
    因为每张图的目标数量不同，不能直接 stack targets。
    """
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # 将图像堆叠成 [B, C, H, W]
    images = torch.stack(images, dim=0)
    # targets 保持为 list of dicts
    return images, targets

def build_dataloader(config, is_train=True):
    if is_train:
        img_dir = config['train_img_dir']
        label_dir = config['train_label_dir']
        shuffle = True
    else:
        img_dir = config['val_img_dir']
        label_dir = config['val_label_dir']
        shuffle = False
        
    dataset = PointSupervisedDataset(img_dir, label_dir, is_train=is_train)
    
    batch_size = config.get('batch_size', 8)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=point_collate_fn,
        num_workers=4,      # 开启多进程加载
        pin_memory=True     # 加速传输到显卡
    )
    
    return dataloader