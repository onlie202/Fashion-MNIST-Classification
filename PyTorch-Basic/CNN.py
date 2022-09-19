# Tutorials Web:
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

# 构建一个简单神经网络

# 卷积层输出计算式：
# N=(W−F+2P)/S+1
# N：输出大小
# W：输入大小
# F：卷积核大小
# P：填充值大小
# S：步长大小

import torch
import torch.nn as nn

class CNNNet(nn.Module):
    def __init__(self):
        # nn.Model的子类
        super(CNNNet, self).__init__()
        # Convolution Layers
        # nn.Conv2d(in_channels,out_channels,kernel)
        # Conv2d Out: (batch_size,depth,rows,cols)
        # Default: padding = 0, stride = 1
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        # Full Connected Layers
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        # Max Pooling (2,2)
        # 最大池化默认步长和窗口一致,n = (28-2)/2 + 1 =14
        # 1*32*32->[conv1]=6*28*28->[max_pool]=6*14*14
        x = torch.max_pool2d(torch.relu(self.conv1(x)),2)
        # Max Pooling (2,2)
        # 6*14*14->[conv2]=16*10*10->[max_pool]=16*5*5
        x = torch.max_pool2d(torch.relu(self.conv2(x)),2)
        # 全连接层的输入即为池化层的输出
        # 平铺，建立全连接层
        x = torch.flatten(x,1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Softmax将张量的每个元素缩放到（0,1）区间且和为1,dim=1表示按行计算
        x = torch.softmax(self.fc3(x),dim=1)
        # x = self.fc3(x)

        return x

def CNN_train(dataset, label, net:CNNNet):
    # 训练程序
    optimizer = torch.optim.SGD(net.parameters(),lr=1e-2)
    epochs = 100
    for i in range(epochs):
        # 手动将梯度缓冲区设为0
        optimizer.zero_grad()
        output = net(dataset)
        # 定义LOSS
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {i + 1}, Loss: {loss}")

if __name__ == "__main__":
    net0 = CNNNet()
    # 模拟输入一张图片(Dataset)
    input_sample = torch.randn(10, 1, 32, 32)
    input_label = torch.softmax(torch.rand(10,10),dim=1)
    # input_label = input_label.view(1,-1)
    CNN_train(input_sample,input_label,net0)

