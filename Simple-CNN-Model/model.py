# CNN模型定义
# 卷积操作输出计算式:H_Out/W_Out = (H/W - kernel_size + 2*padding)/stride + 1

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        """
        模型各个模块的定义
        """
        super(CNN, self).__init__()
        # Conv2d -> BN -> ReLU -> MaxPool
        # Input:1*28*28, Output:16*14*14 (Channel, Height, Width)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Conv2d -> BN -> ReLU
        # Input:16*14*14, Output:32*12*12
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Conv2d -> BN -> ReLU -> MaxPool
        # Input: 32*12*12, Output: 64*5*5
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # FC Layer -> 10 Classes
        self.fc = nn.Linear(64 * 5 * 5, 10)

    def forward(self, x):
        """
        定义前向传播
        """
        x = self.pool1(self.layer1(x))
        x = self.layer2(x)
        x = self.pool2(self.layer3(x))
        y = self.fc(torch.flatten(x, 1)) # x从第二维开始平坦化，因为x的实际形状为(Batchsize, Channel, Height, Weight)

        return y
