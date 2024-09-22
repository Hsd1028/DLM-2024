import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    # 是否使用GPU 自动选择
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
                                     transforms.RandomResizedCrop(224),  # 随机裁剪图片
                                     transforms.RandomHorizontalFlip(),  # 以一定的概率对图像进行水平翻转  同一事物的 不同角度的 观察
                                     transforms.ToTensor(),  # 将图像映射成张量 并计算 归一化（0-255 映射 到 0-1 ）
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # 对图像进行标准化处理
                                    ]),
        "val": transforms.Compose([
                                   transforms.Resize((224, 224)),  # 调整图像的大小
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
                      }

    # 获取数据的路径位置
    data_root = r"E:\DATA"  # get data root path
    image_path = os.path.join(data_root, "LM_2024", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # 添加训练集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # 训练集的个数
    train_num = len(train_dataset)

    # 添加标签集
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    # 标签集的个数
    val_num = len(validate_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32  # 训练批次

    # 加载 训练集
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)
    # 加载 验证集
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=True,
                                                  num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.__next__()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)  # 加载训练模型

    net.to(device)  # 添加到指定的设备上
    loss_function = nn.CrossEntropyLoss()  # 定义损失函数
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)  # 定义 优化器

    epochs = 10
    save_path = r'E:\DATA\LM_2024\trains_result\AlexNet.pth' # 保存权重的一个路径
    best_acc = 0.0  # 最佳 准确率
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()  # 管理 Dropout 启用
        running_loss = 0.0  # 统计在训练过程中的 平均损失
        train_bar = tqdm(train_loader, file=sys.stdout)
        # 遍历数据集
        for step, data in enumerate(train_bar):
            images, labels = data  # 获取 图片 和它的标签
            optimizer.zero_grad()  # 清空之前的梯度信息
            outputs = net(images.to(device))  # 进行训练
            loss = loss_function(outputs, labels.to(device))  # 计算 预测值与真实值的差距
            loss.backward()  # 将损失 反向传播到前面的节点中
            optimizer.step()  # 更新每一个节点的参数

            # print statistics
            running_loss += loss.item()  # 获得 损失的数据

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()  # 管理 Dropout 关闭
        acc = 0.0  # accumulate accurate number / epoch
        #  进行验证
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data  # 获取 验证的图片 和它的标签
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # 求得输出的最大值 作为输出值
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # 将预测值与真实标签进行对比

        val_accurate = acc / val_num  # 计算测试集的准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        # 是否更新 准确率
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()