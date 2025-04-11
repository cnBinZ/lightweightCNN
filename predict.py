import torch
import torchvision.transforms as transforms
from PIL import Image
import json
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
        transforms.Resize((TRAINING_CONFIG['image_size'][0] + 32, TRAINING_CONFIG['image_size'][1] + 32)),
        transforms.CenterCrop(TRAINING_CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORM_CONFIG['normalize_mean'], 
                           std=TRANSFORM_CONFIG['normalize_std'])
    ])
    return transform

def load_model(model_path):
    """加载模型"""
    # 创建模型实例
    model = LargeKernelClassifierNet(new_resnet=True, num_classes=TRAINING_CONFIG['num_classes'])
    
    # 加载模型权重
    device = get_device()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path):
    """预处理图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 应用转换
    transform = get_transforms()
    image_tensor = transform(image)
    
    return image_tensor.unsqueeze(0)  # 添加批次维度

def predict(model, image_tensor, threshold=0.5):
    """
    进行预测
    
    参数:
        model: 模型
        image_tensor: 图像张量
        threshold: 分类阈值
    
    返回:
        predicted_classes: 预测的类别列表
        confidences: 对应的置信度列表
    """
    device = get_device()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs)  # 使用sigmoid激活函数进行多标签分类
    
    # 获取预测的类别和置信度
    predicted_classes = []
    confidences = []
    
    for i, prob in enumerate(probabilities[0]):
        if prob > threshold:
            predicted_classes.append(COCO_CLASSES[i])
            confidences.append(prob.item())
    
    # 按置信度排序
    sorted_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    predicted_classes = [predicted_classes[i] for i in sorted_indices]
    confidences = [confidences[i] for i in sorted_indices]
    
    return predicted_classes, confidences

def main(image_path, model_path):
    """主函数"""
    # 加载模型
    model = load_model(model_path)
    
    # 预处理图像
    image_tensor = preprocess_image(image_path)
    
    # 进行预测
    predicted_classes, confidences = predict(model, image_tensor)
    
    # 打印结果
    print(f"Image: {os.path.basename(image_path)}")
    print("Predicted classes:")
    for cls, conf in zip(predicted_classes, confidences):
        print(f"  - {cls}: {conf:.2%}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict.py <image_path> <model_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    main(image_path, model_path)
