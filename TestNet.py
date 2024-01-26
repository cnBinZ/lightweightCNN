import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms

import os
import matplotlib.pyplot as plt
from LargeKernelClassifierNet import *
import pandas as pd

data_dir = 'D:\\实习\\data\\areca_texture'

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

train_dataloader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=4)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=100, shuffle=4)

# 类别名称
class_names = train_dataset.classes
print('class_names:{}'.format(class_names))

# 训练设备  CPU/GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('trian_device:{}'.format(device.type))

# -------------------------模型选择，优化方法， 学习率策略----------------------
Loss_list = []
Accuracy_list = []

model = LargeKernelClassifierNet(True)

# 模型迁移到CPU/GPU
model = model.to(device)
#print(model)
# 定义损失函数
loss_fc = nn.CrossEntropyLoss()

# 选择优化方法
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)

# 学习率调整策略
exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.65)  # step_size

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5.0e-4, nesterov=True)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.75)

# -------------0---训练过程-----------------
num_epochs = 100
for epoch in range(num_epochs):

    running_loss = 0.0

    for i, sample_batch in enumerate(train_dataloader):
        inputs = sample_batch[0]
        labels = sample_batch[1]

        model.train()

        # GPU/CPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # foward
        outputs = model(inputs)

        # loss
        loss = loss_fc(outputs, labels)
        #loss =nn.functional.nll_loss(outputs, labels)
        # loss求导，反向
        loss.backward()

        # 优化
        optimizer.step()

        #
        running_loss += loss.item()

        # 測試
        if i % 200 == 29:
            correct = 0
            total = 0
            model.eval()
            for images_test, labels_test in val_dataloader:
                images_test = images_test.to(device)
                labels_test = labels_test.to(device)

                outputs_test = model(images_test)
                _, prediction = torch.max(outputs_test, 1)
                correct += (torch.sum((prediction == labels_test))).item()
               # print(prediction, labels_test, correct)
                total += labels_test.size(0)
            Loss_list.append(running_loss / num_epochs)
            Accuracy_list.append(correct / total)
            print('[{}, {}] running_loss = {:.5f} accurcay = {:.5f}'.format(epoch + 1, i + 1, running_loss / num_epochs,
                                                                        correct / total))
            running_loss = 0.0

    exp_lr_scheduler.step()
os.environ['KMP_DUPLICATE_LIB_OK']='True'
print('training finish !')
# x1 = range(0, 30)
# x2 = range(0, 30)
# y1 = Accuracy_list
# y2 = Loss_list
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-')
# plt.title('Test accuracy vs. epoches')
# plt.ylabel('Test accuracy')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '.-')
# plt.xlabel('Test loss vs. epoches')
# plt.ylabel('Test loss')
# plt.show()
#
# csvList = []
# csvList.append(Loss_list)
# csvList.append(Accuracy_list)
# csv = pd.DataFrame(csvList, columns=list((range(1, 31))),  index=list(('Loss', 'acc'))).T
# # csv.to_csv('C:\\Users\\jidan\\Desktop\\model\\data\\TestNet.csv')
# #
# # torch.save(model.state_dict(), 'C:\\Users\\jidan\\Desktop\\model\\TestNet_model.pth')
