import torch
import torchvision.transforms as transforms
from PIL import Image
import json
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

def load_model(model_path):
    """加载模型和类别索引"""
    # 加载类别索引
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    # 创建模型实例
    model = LargeKernelClassifierNet(new_resnet=True)
    
    # 加载模型权重
    device = get_device()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, class_indices

def preprocess_image(image_path):
    """预处理图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 应用转换
    transform = transforms.Compose([
        transforms.ColorJitter(*TRANSFORM_CONFIG['color_jitter']),
        transforms.RandomCrop(TRAINING_CONFIG['image_size'], 
                            padding=TRAINING_CONFIG['padding'], 
                            fill=TRAINING_CONFIG['fill_color']),
        transforms.ToTensor()
    ])
    
    # 转换图像
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # 添加批次维度

def predict(model, image_tensor, class_indices):
    """进行预测"""
    device = get_device()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # 获取预测的类别名称
    predicted_class_name = class_indices[str(predicted_class)]
    
    return predicted_class_name, confidence

def main(image_path, model_path):
    """主函数"""
    # 加载模型和类别索引
    model, class_indices = load_model(model_path)
    
    # 预处理图像
    image_tensor = preprocess_image(image_path)
    
    # 进行预测
    predicted_class, confidence = predict(model, image_tensor, class_indices)
    
    # 打印结果
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict.py <image_path> <model_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    main(image_path, model_path)
