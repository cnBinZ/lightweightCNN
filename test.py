import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision import models
import os
from LightweightClassifierNet import LargeKernelClassifierNet
from config import *  # 导入配置文件


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


def test(model, device, test_loader):
    """
    测试模型性能
    参数:
        model: 待测试的模型
        device: 运行设备（CPU/GPU）
        test_loader: 测试数据加载器
    """
    model.eval()  # 设置为评估模式
    test_loss = 0
    correct = 0
    loss_fc = nn.CrossEntropyLoss()  # 交叉熵损失函数
    
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 数据转移到指定设备
            output = model(data)  # 前向传播
            test_loss += loss_fc(output, target)  # 计算损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计正确预测数

    # 计算平均损失和准确率
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    """
    主函数：加载数据、初始化模型并进行测试
    """
    # 获取设备
    device = get_device()
    print(f'Using device: {device}')

    # 获取数据转换
    data_transform = get_transforms()
    
    # 加载数据集
    train_dataset = torchvision.datasets.ImageFolder(
        root=TRAIN_DATA_DIR,
        transform=data_transform
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=VAL_DATA_DIR,
        transform=data_transform
    )

    # 创建数据加载器
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers']
    )

    # 获取类别名称
    class_names = train_dataset.classes
    print(f'Classes: {class_names}')

    # 测试自定义模型
    model_test = LargeKernelClassifierNet(new_resnet=True)
    model_test.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, MODEL_SAVE_CONFIG['best_model_name']), 
        map_location=device
    ))
    model_test = model_test.to(device)
    print("\nTesting LightweightClassifierNet model:")
    test(model_test, device, val_dataloader)

    # 测试GoogLeNet模型
    model_googlenet = models.GoogLeNet(num_classes=len(class_names), aux_logits=False, init_weights=False)
    model_googlenet.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "googlenet_model.pth"), 
        map_location=device
    ))
    model_googlenet = model_googlenet.to(device)
    print("\nTesting GoogLeNet model:")
    test(model_googlenet, device, val_dataloader)

    # 测试ResNet模型
    model_resnet = models.resnet18(pretrained=False)
    num_fc_in = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_fc_in, len(class_names))  # 修改最后的全连接层
    model_resnet.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "resnet_model.pth"), 
        map_location=device
    ))
    model_resnet = model_resnet.to(device)
    print("\nTesting ResNet model:")
    test(model_resnet, device, val_dataloader)


if __name__ == '__main__':
    main()