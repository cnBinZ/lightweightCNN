import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
import time
from LightweightClassifierNet import LargeKernelClassifierNet
import matplotlib.pyplot as plt
from config import *  # 导入配置文件
from tqdm import tqdm
import numpy as np
from coco_dataset import get_coco_datasets


def train_model(model, train_loader, val_loader, device, num_epochs=TRAINING_CONFIG['num_epochs']):
    """
    训练模型
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        num_epochs: 训练轮数
    """
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 多标签分类使用BCEWithLogitsLoss
    optimizer = optim.SGD(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],
        momentum=TRAINING_CONFIG['momentum'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=TRAINING_CONFIG['scheduler_factor'],
        patience=TRAINING_CONFIG['scheduler_patience'],
        verbose=True
    )
    
    # 记录最佳验证损失
    best_val_loss = float('inf')
    
    # 记录训练和验证损失
    train_losses = []
    val_losses = []
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 使用tqdm显示进度条
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, targets in train_bar:
            # 将数据移到设备
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            # 对于多标签分类，我们需要分别计算每个类别的准确率
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predicted == targets).sum().item()
            train_total += targets.numel()
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': f'{train_loss / (train_bar.n + 1):.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, targets in val_bar:
                # 将数据移到设备
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 统计
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.numel()
                
                # 更新进度条
                val_bar.set_postfix({
                    'loss': f'{val_loss / (val_bar.n + 1):.4f}',
                    'acc': f'{100. * val_correct / val_total:.2f}%'
                })
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_SAVE_CONFIG['best_model_name']))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # 每隔一定轮数保存检查点
        if (epoch + 1) % MODEL_SAVE_CONFIG['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(MODEL_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
        
        # 打印训练和验证结果
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {100. * train_correct / train_total:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {100. * val_correct / val_total:.2f}%")
    
    # 保存训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'))
    plt.close()
    
    return model


def get_device():
    """根据配置选择合适的设备"""
    if DEVICE_CONFIG['use_mps'] and torch.backends.mps.is_available():
        return torch.device("mps")
    elif DEVICE_CONFIG['use_cuda'] and torch.cuda.is_available():
        return torch.device("cuda")
    elif DEVICE_CONFIG['use_cpu']:
        return torch.device("cpu")
    else:
        raise RuntimeError("No available device found")

def get_transforms():
    """根据配置创建数据转换"""
    train_transform = transforms.Compose([
        transforms.ColorJitter(*TRANSFORM_CONFIG['color_jitter']),
        transforms.RandomCrop(TRAINING_CONFIG['image_size'], 
                            padding=TRAINING_CONFIG['padding'], 
                            fill=TRAINING_CONFIG['fill_color']),
        transforms.RandomHorizontalFlip() if TRANSFORM_CONFIG['random_horizontal_flip'] else transforms.Lambda(lambda x: x),
        transforms.RandomVerticalFlip() if TRANSFORM_CONFIG['random_vertical_flip'] else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])
    
    val_transform = transforms.Compose([
        transforms.ColorJitter(*TRANSFORM_CONFIG['color_jitter']),
        transforms.RandomCrop(TRAINING_CONFIG['image_size'], 
                            padding=TRAINING_CONFIG['padding'], 
                            fill=TRAINING_CONFIG['fill_color']),
        transforms.ToTensor()
    ])
    
    return {'train': train_transform, 'val': val_transform}

def main():
    """
    主函数：加载数据、初始化模型并训练
    """
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    
    # 创建模型保存目录
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 下载COCO数据集（如果不存在）
    from config import download_coco_dataset
    download_coco_dataset()
    
    # 获取COCO数据集
    train_dataset, val_dataset = get_coco_datasets()
    
    # 创建数据加载器，减少工作进程数量
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=True,
        num_workers=0,  # 设置为0，使用主进程加载数据
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=False,
        num_workers=0,  # 设置为0，使用主进程加载数据
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    model = LargeKernelClassifierNet(new_resnet=True, num_classes=TRAINING_CONFIG['num_classes'])
    model = model.to(device)
    
        # 训练模型
    model = train_model(model, train_loader, val_loader, device)


if __name__ == "__main__":
    main() 