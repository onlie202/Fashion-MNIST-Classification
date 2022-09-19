# 加载模型并测试输出
import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T

def main(args):
    # * 打印参数 * #
    print("Args:", args)

    # * 加载测试集 * #
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./dataset/",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )  # 下载Test数据集
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    classes_chn = ["T恤/上衣", "裤子", "套头衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "踝靴"]

    # * 加载保存的模型 * #
    model = torch.load(args.pth_file)
    model.to(args.device)
    model.eval()

    # * 读取图片并识别结果和可视化 * #
    img_show = T.ToPILImage()
    for index, (images, labels) in enumerate(test_dataloader):
        if index == 3: # 只看3张图
            break
        images = images.to(args.device)
        y_hat = model(images).cpu()

        _, pred = torch.max(y_hat.data, 1)  # 索引每行的最大值，pred为最大值所对应的索引
        print(f"Predict: [{classes[pred]}] [{classes_chn[pred]}]")
        print(f"Truth: [{classes[labels]}] [{classes_chn[labels]}]\n")
        img_show(images.squeeze(0)).show() # 可视化图片

if __name__ == "__main__":
    # * 命令项选项与参数解析,可在此处定义模型参数 * #
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_file', type=str, default="./save/model_e10.pth")  # 保存的模型文件
    parser.add_argument('--device', type=str, default="cpu")  # 值为cpu或cuda，表明用哪种设备跑

    main(parser.parse_args())