# LightweightCNN

这是一个基于PyTorch实现的轻量级卷积神经网络项目，主要用于图像分类任务。该项目实现了一个具有大卷积核的轻量级分类网络，并提供了训练、测试和预测功能。

## 项目结构

```
.
├── LightweightClassifierNet.py  # 主要的网络模型实现
├── TestNet.py                   # 测试网络实现
├── test.py                      # 模型测试脚本
├── predict.py                   # 图像预测脚本
└── class_indices.json          # 类别索引文件
```

## 主要特性

- 实现了轻量级的卷积神经网络架构
- 支持大卷积核操作
- 包含多种残差块设计
- 提供了完整的训练、测试和预测流程
- 支持CPU和GPU训练

## 网络架构

该网络包含以下主要组件：

1. `ConvBn2d`: 卷积+批归一化层
2. `ConvBnReLU2d`: 卷积+批归一化+ReLU激活层
3. `PoolResBlock`: 池化残差块
4. `DownResBlock`: 下采样残差块
5. `ClassifyTailBlock`: 分类尾部块
6. `LargeKernelClassifierNet`: 主网络架构

## 环境要求

- Python 3.x
- PyTorch
- torchvision
- PIL
- matplotlib

## 使用方法

### 1. 训练模型

使用 `test.py` 进行模型训练和测试：

```bash
python test.py
```

### 2. 预测图像

使用 `predict.py` 进行单张图像的预测：

```bash
python predict.py
```

## 数据预处理

模型使用的数据预处理包括：
- 颜色抖动 (ColorJitter)
- 随机裁剪 (RandomCrop)
- 随机水平翻转 (RandomHorizontalFlip)
- 随机垂直翻转 (RandomVerticalFlip)
- 转换为张量 (ToTensor)

## 模型特点

1. 轻量级设计：通过优化网络结构，减少参数量
2. 大卷积核：使用大尺寸卷积核提升特征提取能力
3. 残差连接：采用多种残差块设计，提升网络性能
4. 批归一化：使用批归一化加速训练并提高模型稳定性
5. Dropout：使用Dropout防止过拟合

## 注意事项

1. 确保数据集按照正确的目录结构组织
2. 训练前检查GPU可用性
3. 根据实际需求调整模型参数和训练参数
4. 注意数据预处理步骤的一致性

## 许可证

[待补充]

## 贡献指南

[待补充] 