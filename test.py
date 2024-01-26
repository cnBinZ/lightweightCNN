from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision import models
import os
from LargeKernelClassifierNet import *



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fc = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fc(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    data_dir = 'E:\\data\\areca_texture'

    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
                                                 transform=transforms.Compose(
                                                     [
                                                         transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                                         transforms.RandomCrop((128, 256), padding=(8, 8),
                                                                               fill=(255, 255, 255)),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.RandomVerticalFlip(),
                                                         transforms.ToTensor()
                                                     ]))
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'test'),
                                               transform=transforms.Compose(
                                                     [
                                                         transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                                         transforms.RandomCrop((128, 256), padding=(8, 8),
                                                                               fill=(255, 255, 255)),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.RandomVerticalFlip(),
                                                         transforms.ToTensor()
                                                     ]))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=20, shuffle=4)

    # 类别名称
    class_names = train_dataset.classes
    print('class_names:{}'.format(class_names))

    # 训练设备  CPU/GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:{}'.format(device.type))


    model_test = LargeKernelClassifierNet(True)
    model_test.load_state_dict(torch.load("C:\\Users\\jidan\\Desktop\\model\\TestNet_model.pth", map_location=device))
    model_test = model_test.to(device)
    test(model_test, device, val_dataloader)

    model_googlenet = models.GoogLeNet(num_classes=2, aux_logits=False, init_weights=False)
    model_googlenet.load_state_dict(torch.load("C:\\Users\\jidan\\Desktop\\model\\GoogLeNet_model.pth", map_location=device))
    model_googlenet = model_googlenet.to(device)

    test(model_googlenet, device, val_dataloader)

    model_resnet = models.resnet18(pretrained=False)
    num_fc_in = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_fc_in, 2)
    model_resnet.load_state_dict(torch.load("C:\\Users\\jidan\\Desktop\\model\\resnet_model.pth", map_location=device))
    model_resnet = model_resnet.to(device)

    test(model_resnet, device, val_dataloader)

if __name__ == '__main__':
    main()