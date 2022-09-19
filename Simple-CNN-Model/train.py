# 使用简单CNN进行Fashion Mnist的分类
import argparse
import matplotlib.pyplot as plt
import os

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CNN

def main(args):
    """
    Train主函数
    """
    # * 打印参数 * #
    print("Args:",args)

    # * 读取数据集 * #
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./dataset/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    ) # 下载Train数据集

    test_dataset = torchvision.datasets.FashionMNIST(
        root="./dataset/",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    ) # 下载Test数据集

    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True) # 使用Dataloader读取数据，划分batch及做shuffle处理
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False) # Test不需要shuffle

    # * 定义模型 * #
    device = torch.device(args.device) # 使用显卡还是cpu
    model = CNN().to(device) # 模型实例
    criterion = nn.CrossEntropyLoss().to(device) # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # 优化器使用Adam,传入需要训练的参数及学习率

    # * 训练模型 * #
    losses = [] # 用于打印训练完成的Loss趋势图
    for epoch in range(args.epochs): # 每个epoch
        losses_iter = []
        for iter, (images, labels) in enumerate(train_dataloader): # 最大iter数 = 样本总数/batchsize，向上取整
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # 梯度清零

            y_hat = model(images) # 模型输出
            loss = criterion(y_hat, labels) # 计算loss
            loss.backward() # 反向传播
            optimizer.step() # 执行优化器

            losses_iter.append(loss.cpu().data.item())

            if (iter+1) % 100 == 0: # 每100个iter打印一次结果
                print(f"Epoch: [{epoch+1}/{args.epochs}] [{iter+1}/{len(train_dataloader)}] "
                      f"Loss: {loss.data.item()}")
        losses.append(sum(losses_iter)/len(losses_iter))

    # * 评估模型 * #
    # 测试模型在测试集上的准确率
    model.eval() # 推理模式
    correct, total = 0,0
    for images, labels in test_dataloader:
        images = images.to(device)
        y_hat = model(images).cpu()

        _, pred = torch.max(y_hat.data, 1) # 索引每行的最大值，pred为最大值所对应的索引
        total += labels.size(0) # 总数合计
        correct += (pred == labels).sum().item() # 统计每个batch对应元素相等的个数

    print(f"Test Accuracy: ",correct/total)

    # * 绘制训练图像 * #
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.show()

    # * 保存模型 * #
    save_dir = "./save/"
    os.makedirs(save_dir,exist_ok=True) # 如果文件夹不存在，则创建文件夹
    torch.save(model, save_dir + f"model_e{args.epochs}.pth")

if __name__ == "__main__":
    # * 命令项选项与参数解析,可在此处定义模型参数 * #
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=200) # 划分batchsize
    parser.add_argument('--epochs', type=int, default=10) # 训练最大epoch
    parser.add_argument('--lr', type=float, default=1e-3) # 学习率
    parser.add_argument('--device', type=str, default="cpu")  # 值为cpu或cuda，表明用哪种设备跑

    # * 主函数 * #
    main(parser.parse_args())