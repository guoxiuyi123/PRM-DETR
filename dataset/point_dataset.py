import os 
import cv2 
import torch 
import numpy as np 
from torch.utils.data import Dataset 
from PIL import Image 
import torchvision.transforms.functional as TF 
from .transforms import PointCopyPaste 
 
class PointSupervisedDataset(Dataset): 
    def __init__(self, img_dir, label_dir, is_train=False): 
        self.img_dir = img_dir 
        self.label_dir = label_dir 
        self.is_train = is_train 
        self.copy_paste = PointCopyPaste(prob=0.5) if is_train else None 
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))] 
 
    def __len__(self): 
        return len(self.img_names) 
 
    def __getitem__(self, idx): 
        img_name = self.img_names[idx] 
        img_path = os.path.join(self.img_dir, img_name) 
        
        # 1. 使用 cv2 读取以便处理绝对坐标 
        image_np = cv2.imread(img_path) 
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) 
        h, w, _ = image_np.shape 
 
        # 2. 读取标注并转换为绝对像素坐标 
        label_name = os.path.splitext(img_name)[0] + '.txt' 
        label_path = os.path.join(self.label_dir, label_name) 
        abs_points = [] 
        if os.path.exists(label_path): 
            with open(label_path, 'r') as f: 
                for line in f.readlines(): 
                    parts = line.strip().split() 
                    if len(parts) >= 3: 
                        class_id = int(parts[0]) 
                        # 将归一化坐标还原为绝对坐标 
                        px, py = float(parts[1]) * w, float(parts[2]) * h 
                        abs_points.append([px, py, class_id]) 
        abs_points = np.array(abs_points) if len(abs_points) > 0 else np.empty((0, 3)) 
 
        # 3. 训练阶段执行 CopyPaste 增强 
        if self.is_train and len(abs_points) > 0 and self.copy_paste is not None: 
            image_np, abs_points = self.copy_paste(image_np, abs_points) 
            h, w, _ = image_np.shape # 更新宽高 
 
        # 4. 转换为 PIL 并缩放到模型需要的 640x640 
        image = Image.fromarray(image_np).resize((640, 640)) 
        image = TF.to_tensor(image) 
 
        # 5. 重组为归一化目标格式 
        labels, points = [], [] 
        for pt in abs_points: 
            if 0 <= pt[0] <= w and 0 <= pt[1] <= h: 
                points.append([pt[0] / w, pt[1] / h]) 
                labels.append(int(pt[2])) 
 
        targets = {} 
        if len(labels) > 0: 
            targets['labels'] = torch.tensor(labels, dtype=torch.int64) 
            targets['points'] = torch.tensor(points, dtype=torch.float32) 
        else: 
            targets['labels'] = torch.zeros((0,), dtype=torch.int64) 
            targets['points'] = torch.zeros((0, 2), dtype=torch.float32) 
 
        return image, targets
