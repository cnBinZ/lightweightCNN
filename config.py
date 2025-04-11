import os
import json
import torch
import requests
import zipfile
from tqdm import tqdm

# 获取项目根目录（上一级目录）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据相关路径
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
VAL_DATA_DIR = os.path.join(DATA_DIR, 'val')

# COCO数据集URL
COCO_URLS = {
    'train': 'http://images.cocodataset.org/zips/train2017.zip',
    'val': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

def download_file(url, filename):
    """
    下载文件并显示进度条
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def download_coco_dataset():
    """
    下载并解压COCO数据集
    """
    # 检查数据集是否已经存在
    if (os.path.exists(COCO_TRAIN_DIR) and 
        os.path.exists(COCO_VAL_DIR) and 
        os.path.exists(COCO_TRAIN_ANN) and 
        os.path.exists(COCO_VAL_ANN)):
        print("COCO dataset already exists, skipping download.")
        return

    # 创建必要的目录
    os.makedirs(COCO_DIR, exist_ok=True)
    os.makedirs(COCO_ANNOTATIONS_DIR, exist_ok=True)

    # 下载并解压数据集
    for name, url in COCO_URLS.items():
        zip_path = os.path.join(COCO_DIR, f'{name}.zip')
        
        # 如果文件不存在，则下载
        if not os.path.exists(zip_path):
            print(f"Downloading {name} dataset...")
            download_file(url, zip_path)
        
        # 解压文件
        print(f"Extracting {name} dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(COCO_DIR)
        
        # 删除zip文件
        os.remove(zip_path)
        print(f"{name} dataset ready!")

# 数据目录配置
COCO_DIR = os.path.join(ROOT_DIR, "data/coco")  # COCO数据集目录
COCO_TRAIN_DIR = os.path.join(COCO_DIR, "train2017")  # COCO训练集目录
COCO_VAL_DIR = os.path.join(COCO_DIR, "val2017")  # COCO验证集目录
COCO_ANNOTATIONS_DIR = os.path.join(COCO_DIR, "annotations")  # COCO标注目录
COCO_TRAIN_ANN = os.path.join(COCO_ANNOTATIONS_DIR, "instances_train2017.json")  # COCO训练集标注
COCO_VAL_ANN = os.path.join(COCO_ANNOTATIONS_DIR, "instances_val2017.json")  # COCO验证集标注

# 模型相关路径
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

# 训练相关配置
TRAINING_CONFIG = {
    'batch_size': 64,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'num_workers': 4,
    'image_size': (224, 224),
    'padding': (8, 8),
    'fill_color': (255, 255, 255),
    'num_classes': 80,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'scheduler_factor': 0.1,
    'scheduler_patience': 3
}

# 数据增强配置
TRANSFORM_CONFIG = {
    'color_jitter': (0.05, 0.05, 0.05, 0.05),
    'random_horizontal_flip': True,
    'random_vertical_flip': False,
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225]
}

# 类别索引配置
CLASS_INDICES = {
    "0": "class1",
    "1": "class2",
    "2": "class3"
    # 可以根据实际类别添加更多
}

# 确保必要的目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(VAL_DATA_DIR, exist_ok=True)

# 保存类别索引到文件
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'class_indices.json')
with open(CLASS_INDICES_PATH, 'w') as f:
    json.dump(CLASS_INDICES, f, indent=4)

# 设备配置
DEVICE_CONFIG = {
    'use_mps': True,  # 是否使用MPS（M芯片GPU）
    'use_cuda': True,  # 是否使用CUDA
    'use_cpu': True   # 是否使用CPU
}

# 模型保存配置
MODEL_SAVE_CONFIG = {
    'best_model_name': 'best_model.pth',
    'checkpoint_interval': 5,
    'save_dir': MODEL_DIR,
}

# 日志配置
LOG_CONFIG = {
    'log_interval': 100,  # 每隔多少个batch打印一次日志
    'save_training_curves': True  # 是否保存训练曲线图
}

# COCO类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 创建类别索引字典
COCO_CLASS_INDICES = {str(i): name for i, name in enumerate(COCO_CLASSES)} 