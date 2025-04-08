from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision import models
import os
from LightweightClassifierNet import *


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
    # 数据集路径
    data_dir = ''

    # 训练数据集
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=transforms.Compose([
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),  # 颜色抖动
            transforms.RandomCrop((128, 256), padding=(8, 8), fill=(255, 255, 255)),  # 随机裁剪
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomVerticalFlip(),  # 随机垂直翻转
            transforms.ToTensor()  # 转换为张量
        ])
    )
    
    # 验证数据集
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=transforms.Compose([
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),  # 颜色抖动
            transforms.RandomCrop((128, 256), padding=(8, 8), fill=(255, 255, 255)),  # 随机裁剪
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomVerticalFlip(),  # 随机垂直翻转
            transforms.ToTensor()  # 转换为张量
        ])
    )

    # 创建数据加载器
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=20, shuffle=4)

    # 获取类别名称
    class_names = train_dataset.classes
    print('class_names:{}'.format(class_names))

    # 设置运行设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:{}'.format(device.type))

    # 测试自定义模型
    model_test = LargeKernelClassifierNet(True)
    model_test.load_state_dict(torch.load("", map_location=device))
    model_test = model_test.to(device)
    test(model_test, device, val_dataloader)

    # 测试GoogLeNet模型
    model_googlenet = models.GoogLeNet(num_classes=2, aux_logits=False, init_weights=False)
    model_googlenet.load_state_dict(torch.load("", map_location=device))
    model_googlenet = model_googlenet.to(device)
    test(model_googlenet, device, val_dataloader)

    # 测试ResNet模型
    model_resnet = models.resnet18(pretrained=False)
    num_fc_in = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_fc_in, 2)  # 修改最后的全连接层
    model_resnet.load_state_dict(torch.load("", map_location=device))
    model_resnet = model_resnet.to(device)
    test(model_resnet, device, val_dataloader)


if __name__ == '__main__':
    main()