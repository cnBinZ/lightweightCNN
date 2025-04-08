import os
import json
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from LightweightClassifierNet import *


def main():
    """
    主函数：加载模型并进行单张图片的预测
    """
    # 设置环境变量，避免某些系统上的库冲突
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # 设置运行设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义数据预处理流程
    data_transform = transforms.Compose([
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),  # 颜色抖动
        transforms.RandomCrop((128, 256), padding=(8, 8), fill=(255, 255, 255)),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.ToTensor()  # 转换为张量
    ])

    # 加载待预测图片
    img_path = ""
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)  # 显示原始图片
    
    # 预处理图片
    img = data_transform(img)  # 应用数据转换
    img = torch.unsqueeze(img, dim=0)  # 添加batch维度

    # 加载类别索引文件
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 创建模型实例
    model = LargeKernelClassifierNet(True).to(device)

    # 加载模型权重
    weights_path = ""
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    missing_keys, unexpected_keys = model.load_state_dict(
        torch.load(weights_path, map_location=device),
        strict=False
    )

    # 进行预测
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        # 获取模型输出
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)  # 计算softmax概率
        predict_cla = torch.argmax(predict).numpy()  # 获取预测类别

    # 输出预测结果
    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_cla)],
        predict[predict_cla].numpy()
    )
    plt.title(print_res)  # 在图片上显示预测结果
    print(print_res)  # 打印预测结果
    plt.show()  # 显示图片


if __name__ == '__main__':
    main()
