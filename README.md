# 轻量级CNN模型训练与测试

本项目实现了一个轻量级的CNN模型，用于图像分类任务。项目包含模型定义、训练、测试和预测等完整功能。

## 项目结构

```
lightweightCNN/
├── config.py           # 配置文件
├── LightweightClassifierNet.py  # 模型定义
├── train.py           # 训练脚本
├── test.py           # 测试脚本
├── predict.py        # 预测脚本
├── models/           # 模型保存目录
└── data/             # 数据集目录
    ├── train/        # 训练数据
    └── val/          # 验证数据
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- torchvision
- Pillow
- matplotlib

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

项目使用 `config.py` 进行统一配置管理，主要包含以下配置项：

### 路径配置
```python
# 数据目录配置
TRAIN_DATA_DIR = "data/train"  # 训练数据目录
VAL_DATA_DIR = "data/val"      # 验证数据目录
MODEL_DIR = "models"           # 模型保存目录
CLASS_INDICES_PATH = "models/class_indices.json"  # 类别索引文件
```

### 设备配置
```python
DEVICE_CONFIG = {
    'use_mps': True,   # 是否使用 MPS (Apple Silicon GPU)
    'use_cuda': True,  # 是否使用 CUDA
    'use_cpu': True    # 是否使用 CPU
}
```

### 训练配置
```python
TRAINING_CONFIG = {
    'batch_size': 16,          # 批次大小
    'num_workers': 4,          # 数据加载线程数
    'learning_rate': 0.001,    # 学习率
    'num_epochs': 20,          # 训练轮数
    'image_size': (128, 256),  # 图像大小
    'padding': (8, 8),         # 填充大小
    'fill_color': (255, 255, 255)  # 填充颜色
}
```

### 数据转换配置
```python
TRANSFORM_CONFIG = {
    'color_jitter': (0.05, 0.05, 0.05, 0.05),  # 颜色抖动参数
    'random_horizontal_flip': True,  # 是否随机水平翻转
    'random_vertical_flip': True     # 是否随机垂直翻转
}
```

### 模型保存配置
```python
MODEL_SAVE_CONFIG = {
    'best_model_name': 'best_model.pth',  # 最佳模型文件名
    'checkpoint_interval': 5              # 检查点保存间隔
}
```

## 使用方法

### 训练模型

```bash
python train.py
```

### 测试模型

```bash
python test.py
```

### 预测单张图片

```bash
python predict.py path/to/image.jpg path/to/model.pth
```

## 模型说明

项目包含三种模型实现：

1. LightweightClassifierNet：自定义的轻量级CNN模型
2. GoogLeNet：使用torchvision实现的GoogLeNet模型
3. ResNet18：使用torchvision实现的ResNet18模型

## 注意事项

1. 确保数据集按照正确的目录结构组织
2. 根据实际硬件情况调整 `DEVICE_CONFIG` 配置
3. 可以根据需要调整 `TRAINING_CONFIG` 中的参数
4. 模型文件会自动保存在 `models` 目录下

## 许可证

MIT License 