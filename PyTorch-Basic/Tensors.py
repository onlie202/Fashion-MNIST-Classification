import torch

# * 构造Tensor * #
t0 = torch.Tensor(5, 3)  # 构造一个未初始化的5*3的矩阵
t1 = torch.rand(5, 3)  # 构造一个随机初始化的矩阵
print(t0.size(), t0.shape)
print(type(t0.size()), type(t0.shape)) # torch.Size本质上为Tuple

# * 运算示例（加法） * #
# 其他运算：https://pytorch-cn.readthedocs.io/zh/latest/package_references/Tensor/
t2 = torch.rand(5, 3)
add0 = t1 + t2 # 加法1
add1 = torch.add(t1, t2) # 加法2
print(add0 == add1)

# * 切片操作 * #
s0 = t1[:,1] # 取第2列，非复制操作
print(f"{t1}\n{s0}\n{s0.shape}")
s0[0] = -1
print(f"{t1}\n{s0}")

# * Tensor 与 numpy array 互转 * #
# Tensor和numpy array相互转换后仍共享存储地址
import numpy as np

n0 = t1.numpy() # Tensor转numpy array
print(type(n0))
n1 = np.ones((5,3))
t3 = torch.from_numpy(n1) # numpy array 转 Tensor
print(type(t3))

# * GPU/CPU设备切换 * #
try:
    t1_gpu = t1.cuda()
    t1_cpu = t1_gpu.cpu()
except Exception as e:
    raise RuntimeError(e)
print(t1_gpu, "\n", t1_cpu)



