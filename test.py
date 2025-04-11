import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision import models
import os
from LightweightClassifierNet import LargeKernelClassifierNet
from config import *  # 导入配置文件
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from coco_dataset import get_coco_datasets


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
    transform = transforms.Compose([
        transforms.ColorJitter(*TRANSFORM_CONFIG['color_jitter']),
        transforms.RandomCrop(TRAINING_CONFIG['image_size'], 
                            padding=TRAINING_CONFIG['padding'], 
                            fill=TRAINING_CONFIG['fill_color']),
        transforms.RandomHorizontalFlip() if TRANSFORM_CONFIG['random_horizontal_flip'] else transforms.Lambda(lambda x: x),
        transforms.RandomVerticalFlip() if TRANSFORM_CONFIG['random_vertical_flip'] else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])
    return transform


def test(model, device, test_loader, threshold=0.5):
    """
    测试模型性能
    
    参数:
        model: 待测试的模型
        device: 运行设备（CPU/GPU）
        test_loader: 测试数据加载器
        threshold: 分类阈值
    """
    model.eval()  # 设置为评估模式
    test_loss = 0
    correct = 0
    total = 0
    loss_fc = nn.BCEWithLogitsLoss()  # 多标签分类使用BCEWithLogitsLoss
    
    # 用于计算每个类别的准确率
    class_correct = [0] * TRAINING_CONFIG['num_classes']
    class_total = [0] * TRAINING_CONFIG['num_classes']
    
    with torch.no_grad():  # 不计算梯度
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)  # 数据转移到指定设备
            output = model(data)  # 前向传播
            test_loss += loss_fc(output, target).item()  # 计算损失
            
            # 对于多标签分类，我们需要分别计算每个类别的准确率
            predicted = (torch.sigmoid(output) > threshold).float()
            
            # 计算总体准确率
            correct += (predicted == target).sum().item()
            total += target.numel()
            
            # 计算每个类别的准确率
            for i in range(TRAINING_CONFIG['num_classes']):
                class_correct[i] += ((predicted[:, i] == 1) & (target[:, i] == 1)).sum().item()
                class_total[i] += (target[:, i] == 1).sum().item()
    
    # 计算平均损失和准确率
    test_loss /= len(test_loader)
    overall_accuracy = 100. * correct / total
    
    # 打印总体结果
    print(f'\nTest set: Average loss: {test_loss:.4f}, Overall Accuracy: {overall_accuracy:.2f}%')
    
    # 打印每个类别的准确率
    print("\nPer-class accuracy:")
    for i in range(TRAINING_CONFIG['num_classes']):
        if class_total[i] > 0:
            accuracy = 100. * class_correct[i] / class_total[i]
            print(f"{COCO_CLASSES[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"{COCO_CLASSES[i]}: N/A (no samples)")


def main():
    """
    主函数：加载数据、初始化模型并进行测试
    """
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    
    # 获取COCO数据集
    _, val_dataset = get_coco_datasets()
    
    # 创建数据加载器
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    # 测试自定义模型
    model_test = LargeKernelClassifierNet(new_resnet=True, num_classes=TRAINING_CONFIG['num_classes'])
    model_test.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, MODEL_SAVE_CONFIG['best_model_name']), 
        map_location=device
    ))
    model_test = model_test.to(device)
    print("\nTesting LightweightClassifierNet model:")
    test(model_test, device, val_loader)
    
    # 测试GoogLeNet模型
    model_googlenet = models.googlenet(num_classes=TRAINING_CONFIG['num_classes'], aux_logits=False, init_weights=False)
    model_googlenet.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "googlenet_model.pth"), 
        map_location=device
    ))
    model_googlenet = model_googlenet.to(device)
    print("\nTesting GoogLeNet model:")
    test(model_googlenet, device, val_loader)
    
    # 测试ResNet模型
    model_resnet = models.resnet18(pretrained=False)
    num_fc_in = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_fc_in, TRAINING_CONFIG['num_classes'])  # 修改最后的全连接层
    model_resnet.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "resnet_model.pth"), 
        map_location=device
    ))
    model_resnet = model_resnet.to(device)
    print("\nTesting ResNet model:")
    test(model_resnet, device, val_loader)


if __name__ == '__main__':
    main()