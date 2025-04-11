# 轻量级CNN模型训练与测试

本项目实现了一个轻量级的CNN模型，用于图像分类任务。项目支持COCO数据集的多标签分类，并包含模型定义、训练、测试和预测等完整功能。

## 项目结构

```
lightweightCNN/
├── config.py           # 配置文件
├── LightweightClassifierNet.py  # 模型定义
├── coco_dataset.py    # COCO数据集加载器
├── train.py           # 训练脚本
├── test.py           # 测试脚本
├── predict.py        # 预测脚本
├── models/           # 模型保存目录
└── data/             # 数据集目录
    └── coco/         # COCO数据集目录
        ├── train2017/  # COCO训练集图像
        ├── val2017/    # COCO验证集图像
        └── annotations/ # COCO标注文件
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- torchvision
- Pillow
- matplotlib
- tqdm

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集准备

本项目使用COCO数据集进行训练和测试。请按照以下步骤准备数据集：

1. 下载COCO数据集：
   - 训练集：http://images.cocodataset.org/zips/train2017.zip
   - 验证集：http://images.cocodataset.org/zips/val2017.zip
   - 标注文件：http://images.cocodataset.org/annotations/annotations_trainval2017.zip

2. 解压数据集到项目目录：
   ```
   data/
   └── coco/
       ├── train2017/  # 训练集图像
       ├── val2017/    # 验证集图像
       └── annotations/ # 标注文件
           ├── instances_train2017.json
           └── instances_val2017.json
   ```

## 配置说明

项目使用 `config.py` 进行统一配置管理，主要包含以下配置项：

### 路径配置
```python
# 数据目录配置
COCO_DIR = "data/coco"  # COCO数据集目录
COCO_TRAIN_DIR = os.path.join(COCO_DIR, "train2017")  # COCO训练集目录
COCO_VAL_DIR = os.path.join(COCO_DIR, "val2017")  # COCO验证集目录
COCO_ANNOTATIONS_DIR = os.path.join(COCO_DIR, "annotations")  # COCO标注目录
COCO_TRAIN_ANN = os.path.join(COCO_ANNOTATIONS_DIR, "instances_train2017.json")  # COCO训练集标注
COCO_VAL_ANN = os.path.join(COCO_ANNOTATIONS_DIR, "instances_val2017.json")  # COCO验证集标注

MODEL_DIR = "models"  # 模型保存目录
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
    'image_size': (224, 224),  # 图像大小
    'num_classes': 80,         # COCO数据集类别数
    'weight_decay': 1e-4,      # 权重衰减
    'momentum': 0.9,           # 动量
    'scheduler_factor': 0.1,   # 学习率调度器因子
    'scheduler_patience': 3    # 学习率调度器耐心值
}
```

### 数据转换配置
```python
TRANSFORM_CONFIG = {
    'color_jitter': (0.05, 0.05, 0.05, 0.05),  # 颜色抖动参数
    'random_horizontal_flip': True,  # 是否随机水平翻转
    'normalize_mean': [0.485, 0.456, 0.406],  # 标准化均值 (ImageNet)
    'normalize_std': [0.229, 0.224, 0.225]    # 标准化标准差 (ImageNet)
}
```

### 模型保存配置
```python
MODEL_SAVE_CONFIG = {
    'best_model_name': 'best_model.pth',  # 最佳模型文件名
    'checkpoint_interval': 5,              # 检查点保存间隔
    'save_dir': MODEL_DIR                 # 保存目录
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

1. LightweightClassifierNet：自定义的轻量级CNN模型，支持多标签分类
2. GoogLeNet：使用torchvision实现的GoogLeNet模型
3. ResNet18：使用torchvision实现的ResNet18模型

## 多标签分类

本项目支持COCO数据集的多标签分类，每个图像可能包含多个对象类别。预测结果将返回所有置信度高于阈值的类别。

## 注意事项

1. 确保COCO数据集按照正确的目录结构组织
2. 根据实际硬件情况调整 `DEVICE_CONFIG` 配置
3. 可以根据需要调整 `TRAINING_CONFIG` 中的参数
4. 模型文件会自动保存在 `models` 目录下
5. 对于多标签分类，预测时使用sigmoid激活函数和阈值进行类别判断

## 许可证

MIT License 