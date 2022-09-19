import torch

w = torch.tensor([[1., 2.],
                  [3., 4.]], requires_grad=True)  # 由于需要计算梯度，所以requires_grad设置为True

x = torch.tensor([[1., 5.],
                  [1., 5.]], requires_grad=True)

a = w + x
b = w + 1

y = torch.sum(a * b)
print(y)

y.backward() # 对y进行反向传播
print(w.grad) # 输出w的梯度
print(x.grad) # 输出x的梯度

