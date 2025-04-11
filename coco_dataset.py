import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from config import *
from pycocotools.coco import COCO

class COCODataset(Dataset):
    """
    COCO数据集加载器
    """
    def __init__(self, root_dir, ann_file, transform=None):
        """
        初始化COCO数据集
        
        参数:
            root_dir (str): 图像根目录
            ann_file (str): 标注文件路径
            transform (callable, optional): 图像转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(ann_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        # 获取所有类别
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.num_classes = len(self.categories)
        
        # 创建类别ID到索引的映射
        self.cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(self.categories)}
        
        # 如果没有提供转换，使用默认转换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(TRAINING_CONFIG['image_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=TRANSFORM_CONFIG['normalize_mean'], 
                                  std=TRANSFORM_CONFIG['normalize_std'])
            ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        返回:
            image (Tensor): 图像张量
            target (Tensor): 多标签目标张量
        """
        # 加载图像
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(self.root_dir, img_info['file_name'])).convert('RGB')
        
        # 获取图像的标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 创建多标签向量
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        for ann in anns:
            cat_idx = self.cat_id_to_idx[ann['category_id']]
            labels[cat_idx] = 1.0
        
        # 应用数据转换
        image = self.transform(image)
        
        # 打印调试信息
        if idx == 0:  # 只打印第一个样本的信息
            print(f"Image shape: {image.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Labels dtype: {labels.dtype}")
        
        # 确保图像和标签的维度匹配
        if image.size(0) != 3:  # 如果通道数不是3
            print(f"Warning: Image has {image.size(0)} channels, expected 3")
            image = image[:3]  # 只取前3个通道
        
        # 确保图像尺寸正确
        if image.size(1) != TRAINING_CONFIG['image_size'][0] or image.size(2) != TRAINING_CONFIG['image_size'][1]:
            print(f"Warning: Image size is {image.size(1)}x{image.size(2)}, expected {TRAINING_CONFIG['image_size']}")
            image = transforms.Resize(TRAINING_CONFIG['image_size'])(image)
        
        return image, labels

def get_coco_datasets():
    """
    获取COCO训练集和验证集
    
    返回:
        train_dataset: 训练集
        val_dataset: 验证集
    """
    # 创建训练集
    train_dataset = COCODataset(
        root_dir=COCO_TRAIN_DIR,
        ann_file=COCO_TRAIN_ANN
    )
    
    # 创建验证集
    val_dataset = COCODataset(
        root_dir=COCO_VAL_DIR,
        ann_file=COCO_VAL_ANN
    )
    
    return train_dataset, val_dataset 