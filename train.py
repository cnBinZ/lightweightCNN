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


def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=25, save_dir=None):
    """
    训练模型的主函数
    参数:
        model: 待训练的模型
        device: 运行设备
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        save_dir: 模型保存目录
    """
    # 记录最佳模型
    best_acc = 0.0
    best_model_path = None
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 开始训练
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置为训练模式
                dataloader = train_loader
            else:
                model.eval()   # 设置为评估模式
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # 遍历数据
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 训练阶段进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # 计算epoch的损失和准确率
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            # 记录训练过程
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if save_dir:
                    best_model_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save(model.state_dict(), best_model_path)
                    print(f'Best model saved to {best_model_path}')
        
        print()
    
    # 训练结束后保存最后的模型
    if save_dir:
        final_model_path = os.path.join(save_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f'Final model saved to {final_model_path}')
    
    # 绘制训练过程
    plot_training_process(train_losses, train_accs, val_losses, val_accs, save_dir)
    
    return model, best_model_path


def plot_training_process(train_losses, train_accs, val_losses, val_accs, save_dir=None):
    """
    绘制训练过程的损失和准确率曲线
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图像
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    
    plt.show()


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
    主函数：设置训练参数并开始训练
    """
    # 设置环境变量
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 获取设备
    device = get_device()
    print(f'Using device: {device}')
    
    # 获取数据转换
    data_transforms = get_transforms()
    
    # 加载数据集
    image_datasets = {
        x: torchvision.datasets.ImageFolder(
            TRAIN_DATA_DIR if x == 'train' else VAL_DATA_DIR,
            data_transforms[x]
        ) for x in ['train', 'val']
    }
    
    # 创建数据加载器
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=TRAINING_CONFIG['batch_size'],
            shuffle=True,
            num_workers=TRAINING_CONFIG['num_workers']
        ) for x in ['train', 'val']
    }
    
    # 获取类别信息
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f'Classes: {class_names}')
    print(f'Number of classes: {num_classes}')
    
    # 保存类别索引
    class_indices = {str(i): name for i, name in enumerate(class_names)}
    with open(CLASS_INDICES_PATH, 'w') as f:
        import json
        json.dump(class_indices, f)
    
    # 创建模型
    model = LargeKernelClassifierNet(new_resnet=True)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    
    # 训练模型
    print("Starting training...")
    start_time = time.time()
    
    model, best_model_path = train_model(
        model=model,
        device=device,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        save_dir=MODEL_DIR
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best model saved to: {best_model_path}")


if __name__ == '__main__':
    main() 