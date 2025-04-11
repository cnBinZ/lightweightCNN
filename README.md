# 轻量级CNN模型训练与测试

本项目实现了一个轻量级的CNN模型，用于图像分类任务。项目支持COCO数据集的多标签分类，并包含模型定义、训练、测试和预测等完整功能。

## 项目结构

```
lightweightCNN/          # 项目根目录
├── config.py           # 配置文件
├── LightweightClassifierNet.py  # 模型定义
├── coco_dataset.py    # COCO数据集加载器
├── train.py           # 训练脚本
├── test.py           # 测试脚本
└── predict.py        # 预测脚本

../data/               # 数据目录（位于项目根目录的上级目录）
└── coco/
    ├── train2017/    # 训练集图像
    ├── val2017/      # 验证集图像
    └── annotations/   # 标注文件
        ├── instances_train2017.json
        └── instances_val2017.json
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

本项目使用COCO数据集进行训练和测试。
程序会自动下载COCO数据集，下载目录如下：
   ```
   ../data/              # 数据目录（位于项目根目录的上级目录）
   └── coco/
       ├── train2017/    # 训练集图像
       ├── val2017/      # 验证集图像
       └── annotations/   # 标注文件
           ├── instances_train2017.json
           └── instances_val2017.json
   ```

## 配置说明

项目使用 `config.py` 进行统一配置管理，包含以下主要配置：

- 数据路径配置：数据集和模型保存路径
- 设备配置：支持 CPU、CUDA 和 MPS (Apple Silicon GPU)
- 训练参数：批次大小、学习率、训练轮数等
- 数据转换：图像预处理参数
- 模型保存：检查点保存策略

详细配置请参考 `config.py` 文件。

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

项目实现了轻量级CNN模型（LightweightClassifierNet），支持多标签分类任务。模型特点：

- 轻量级设计，适合在资源受限环境运行
- 支持多标签分类
- 使用现代CNN架构设计
- 支持批量训练和推理

## 多标签分类

本项目支持COCO数据集的多标签分类，每个图像可能包含多个对象类别。预测结果将返回所有置信度高于阈值的类别。

## 注意事项

1. 确保在项目根目录的上级目录中创建data目录，并按照正确的目录结构组织COCO数据集
2. 根据实际硬件情况调整设备配置
3. 模型文件会自动保存在 `models` 目录下
4. 对于多标签分类，预测时使用sigmoid激活函数和阈值进行类别判断

## 许可证

MIT License 