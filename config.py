import os
import json

# 获取项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据相关路径
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
VAL_DATA_DIR = os.path.join(DATA_DIR, 'val')

# 模型相关路径
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

# 训练相关配置
TRAINING_CONFIG = {
    'batch_size': 16,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'num_workers': 4,
    'image_size': (128, 256),
    'padding': (8, 8),
    'fill_color': (255, 255, 255)
}

# 数据增强配置
TRANSFORM_CONFIG = {
    'color_jitter': (0.05, 0.05, 0.05, 0.05),
    'random_horizontal_flip': True,
    'random_vertical_flip': True
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
    'final_model_name': 'final_model.pth',
    'save_interval': 5  # 每隔多少个epoch保存一次
}

# 日志配置
LOG_CONFIG = {
    'log_interval': 100,  # 每隔多少个batch打印一次日志
    'save_training_curves': True  # 是否保存训练曲线图
} 