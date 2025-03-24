---
title: "🧠 线性神经网络"
outline: deep

tags: "deepLearning"
updateTime: "2024-11-26 22:52"
---
![云边小卖部](/public/window.jpg)
>有人哭，有人笑，有人输，有人老。

# 线性网络



## 前言


### 数据操作实现及预处理

***

```py
//导入numpy库
import numpy as np
//创建数组
np_array=np.array([1,3,4])

//第二种方法，用pytorch库
import torch
//创建4x5全一矩阵
torch_tensor=torch.ones(4,5)
```

#### 一个模型的通常步骤

- 导入相关库
- 初始化变量
- 引入数据集
- 定义softmax或者其他概率处理函数
- 定义网络，计算y=xw+b实现前向传播
- 定义损失函数
- 定义参数更新的优化器
- 训练模型
- 预测模型

在 PyTorch 中，`torch` 和 `tensor` 都有许多常用的方法。以下是一些常用的操作和方法：

### 1. `torch` 常用方法：
`torch` 提供了创建张量和处理张量的许多方法。常见的有：

- **创建张量：**
  
  ```python
  torch.tensor([1, 2, 3])  # 从 Python 列表创建张量
  torch.ones(size)         # 创建全 1 的张量
  torch.zeros(size)        # 创建全 0 的张量
  torch.eye(n)             # 创建nxn单位矩阵
  
  torch.arange(start, end, step)  # 创建范围内等差数列张量,该范围不包含end，如果省略start默认从0开始到end-1
  例如：torch.arange(1,3)结果：[1,2]
  
  torch.matmul(张量a，张量b) #mat->matrix mul->multiplication 矩阵乘法
  torch.mul(a,b)#普通的乘法
  
  torch.normal(mean,std,size=(num,num))#生成随机数，mean是均值，std是标准差，size是张量形状
  torch.linspace(start, end, steps)  # 创建范围内线性均分的张量
  ```
  
- **随机数生成：**
  ```python
  torch.rand(size)         # 生成 [0, 1) 区间内均匀分布的随机张量
  torch.randn(size)        # 生成标准正态分布的随机张量
  torch.randint(low, high, size)  # 生成整数随机张量
  ```

- **张量初始化：**
  ```python
  torch.empty(size)        # 创建一个未初始化的张量
  torch.full(size, value)  # 创建一个指定值的张量
  ```

- **张量操作：**
  ```python
  torch.cat(tensors, dim=0)# 拼接多个张量
  torch.dot(a,b) #计算两个一维向量的内积 
  dim=0行改变 dim=1列数改变
  ```

### 2. `tensor` 对象常用方法：
`tensor` 对象是 PyTorch 的核心数据结构，常用方法有：

- **形状操作：**
  ```python
  tensor.reshape(num,num)#重塑为num x num张量，设置为-1时，表示自动计算维度
  #例如：y.reshape(-1,3)，若y有9个元素，则-1代表9/3=3
  tensor.shape             # 查看张量的形状
  tensor.size()            # 返回张量的大小
  tensor.view(shape)       # 改变张量的形状（等效 reshape）
  ```
  
- **张量数学操作：**
  
  ```python
  tensor.add(other)        # 张量相加
  tensor.sub(other)        # 张量相减
  tensor.mul(other)        # 元素级相乘
  tensor.div(other)        # 元素级相除
  tensor.pow(exponent)     # 每个元素求幂
  tensor.sqrt()            # 张量开平方
  tensor.sum(dim)          # 指定维度求和
  tensor.mean(dim)         # 指定维度求均值
  tensor.max(dim)          # 获取最大值
  tensor.min(dim)          # 获取最小值
  tensor.clamp(min, max)   # 将张量的值限制在指定范围内
  ```
  
- **索引与切片：**
  ```python
  tensor[index]            # 通过索引访问张量
  tensor[:, i]             # 选取指定维度的切片
  tensor.index_select(dim, indices)  # 按索引选取张量的特定元素
  ```

- **与 NumPy 互操作：**
  ```python
  tensor.numpy()           # 将 PyTorch 张量转为 NumPy 数组
  torch.from_numpy(numpy_array)  # 将 NumPy 数组转为 PyTorch 张量
  ```

- **GPU 操作：**
  ```python
  tensor.cuda()            # 将张量移动到 GPU
  tensor.cpu()             # 将张量移动到 CPU
  ```

- **梯度相关：**
  ```python
  tensor.requires_grad_()  # 设置张量需要计算梯度
  tensor.grad              # 获取张量的梯度
  tensor.backward()        # 反向传播计算梯度
  ```

### 示例代码：
```python
import torch

# 创建一个 3x3 张量
tensor = torch.rand(3, 3)

# 张量的数学操作
result = tensor.add(2)   # 每个元素加 2
result = tensor.sum(dim=1)  # 对每一行求和

# 改变形状
reshaped = tensor.view(1, 9)

# GPU 操作
if torch.cuda.is_available():
    tensor = tensor.cuda()

print(tensor)
```



## 神经网络



### 线性神经网络

***

#### 线性层入门(穷举法)

- 结构y=wx，利用穷举法找到最优参数w
- 绘制w-loss图像(后续都是用epoch-loss)
- 寻找loss最低点对应的w

##### 代码演示

```py
# 线性网络基础入门  利用穷举法寻找最适超参数w  这里忽略了b，y=wx
import numpy as np
# 绘图工具
import  matplotlib.pyplot as plt
from torch.nn.functional import mse_loss

# 数据预处理
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
# w初始化为0
w=0
# 前向传播和损失计算（穷举不需要更新参数和梯度下降）
def forward(x):
    return x*w
# mse损失，即均方平均损失
def Loss(x,y):
    y_hat=forward(x)
    return  (y_hat-y)*(y_hat-y)

# 直接开始穷举
# 初始化两个列表装w和对应的损失值，为后续绘图使用
w_list=[]
l_list=[]
# w从0-4挨个穷举
for w in np.arange(0.0,4.1,0.5):
    print('w=',w)
    l_sum = 0
    for (x,y) in zip(x_data,y_data):
        y_hat=forward(x)
        loss=Loss(x,y)

        # 计算总损失
        l_sum+=loss
        print('\t',x,y,y_hat,loss)
    # 添加到列表
    w_list.append(w)
    # 均方损失
    l_list.append(l_sum/3)
    print('mse=',l_sum/3)

# 绘制图像
plt.plot(w_list,l_list)
# 设置横纵坐标
plt.ylabel('loss')
plt.xlabel('w')
# 展示
plt.show()
```

#### 线性层入门（梯度下降GD）

梯度下降更新公式：w=w - 学习率*偏导

##### 代码演示

```py
"""
使用梯度下降寻找最优超参数w
"""
import numpy as np
import matplotlib.pyplot as plt

# 数据预处理
x_data=[1,2,3]
y_data=[2,4,6]

w=1
epoch_list=[]
loss_list=[]
def forward(x):
    return x*w
# 定义损失和gd
def Loss(x,y):
    y_hat=forward(x)
    return (y_hat-y)*(y_hat-y)
# grad=2(x*w-y)*w
def grad_decent(x,y):
    return  2*(x*w-y)*w
for epoch in range(100):
    epoch_list.append(epoch)
    l_sum=0
    grad_sum=0
    for(x,y) in zip(x_data,y_data):
        # 单个损失
        loss=Loss(x,y)
        # 单个梯度
        grad=grad_decent(x,y)
        grad_sum+=grad
        l_sum+=loss
    w-=0.1*grad_sum/3
    loss_list.append(l_sum / 3)
    print('epoch=',epoch,'w=',w,'loss=',l_sum/3)
print('x=4,predicted=',forward(4))
# 绘图
plt.plot(epoch_list,loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```



#### 线性层（随机梯度下降SGD）

- 使用随机梯度下降，不在是使用所有损失的和l_sum的均值作为损失，而是将每一个loss作为损失并且求导
- SGD随机梯度下降 与梯度下降GD相比
- 优点：结果更精确，性能更好
- 缺点：所需时间更长
  - 既要性能又要时间效率，折中，使用miniBatch，将每个epoch分为小批量使用sgd(后续使用)

##### 代码演示

```py
"""
使用随机梯度下降，不在是使用所有损失的和l_sum的均值作为损失，而是将每一个loss作为损失并且求导
"""
# SGD随机梯度下降 与梯度下降GD相比
# 优点：结果更精确，性能更好
# 缺点：所需时间更长
# 既要性能又要时间效率，折中，使用miniBatch，将每个epoch分为小批量使用sgd

import matplotlib.pyplot as plt

# 数据预处理
x_data=[1,2,3]
y_data=[2,4,6]

w=1

def forward(x):
    return x*w
# 定义损失和gd
def Loss(x,y):
    y_hat=forward(x)
    return (y_hat-y)*(y_hat-y)
# grad=2(x*w-y)*w
def grad_decent(x,y):
    return  2*(x*w-y)*w
for epoch in range(100):

    grad_sum=0
    for(x,y) in zip(x_data,y_data):
        # 单个损失
        loss=Loss(x,y)
        # 单个梯度
        grad=grad_decent(x,y)
        # 不再根据损失的和作为梯度更新，而是每一个都会更新
        w -= 0.1 * grad
        print('epoch=', epoch, 'w=', w, 'loss=', loss)


print('x=4,predicted=',forward(4))
# 绘图
# plt.plot(epoch_list,loss_list)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
```





#### 线性层（pytorch实现）

***

之前我们的梯度都是根据具体的模型得出来的计算公式，但是在现实的模型里面，且不说能不能得出，其复杂程度难以想象
因此用tensor张量的形式，将计算图保存下来，每次只需要backward就可以将每一个节点的梯度值计算出来并且保存

##### 代码演示

```py
# 引入pytorch张量，实现反向传播
"""
之前我们的梯度都是根据具体的模型得出来的计算公式，但是在现实的模型里面，且不说能不能得出，其复杂程度难以想象
因此用tensor张量的形式，将计算图保存下来，每次只需要backward就可以将每一个节点的梯度值计算出来并且保存
"""
import torch

# 数据预处理 虽然这里定义的时候是列表形式，但是跟w相乘后会自动转为张量，所以也可以定义为张量 以下两种等效
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# x_data=[1,2,3]
# y_data=[2,4,6]
# 注意这里的true首字母要大写，不然识别不了
# require_grad表示是否需要求梯度
w=torch.Tensor([1.0])
w.requires_grad=True

def forward(x):
    return x*w
def Loss(x,y):
    y_hat=forward(x)
    return (y_hat-y)**2

for epoch in range(100):
    for (x,y) in zip(x_data,y_data):
        # 前向传播得到loss
        loss=Loss(x,y)
        # 反向传播将计算图中每个节点的梯度都保存下来
        loss.backward()
        # sgd优化
        """
        不能直接w-=0.01*w.grad
        因为只要w为张量，那么根据w引出来的计算图中所有变量都是张量，张量中不仅包含了data值，还保存了相应的梯度值
        所以要用.data来拿到值的对象，同时里面的grad也是tensor，所以也要用data，这样才不会影响到计算图
        若是一个标量形式的tensor，那么.item就可以直接拿到标量值而非tensor对象
        """
        w.data-=0.1*w.grad.data
        # 将w的梯度值清零，不然每次反向传播都会叠加
        w.grad.data.zero_()
# 同样这里不能使用forward（4）来得到输出，不然会输出一个tensor对象
# 因为w是tensor，x*w会自动将x也变成tensor
print("x=4, y=",forward(4).item())



```

##### 代码演示2

```py
"""
使用pytorch的nn.module构造模型实现线性回归（nn=net nural神经网络，只要是神经网络的模型实现都需要继承自nn）

"""
import torch
import torch.nn as nn



# 数据预处理
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# 定义网络模型
class Linear(nn.Module):
    # __init__是py的构造器，必须要有
    def __init__(self):
        # 这就是调用父类的构造器，所有的模型照着写就行了，不用深究
        super(Linear,self).__init__()
        # torch.nn.linear有三个参数，分别是输入维度，输出维度，是否有偏置量b(默认为true)，返回的是一个对象
        self.linear=torch.nn.Linear(1,1,bias=True)
    # forward是module强制子类实现的，而且必须就叫这名字，返回的是输出结果
    def forward(self,x):
        # linear实现y=wx+b并返回结果
        y_pred=self.linear(x)
        return y_pred

# 实例化model
model=Linear()
# 定义损失和优化器

# 只有一个参数，决定求和后要不要求均值（并不影响）
criterion = torch.nn.MSELoss(reduction='mean')
# model.parameters()能获得所有有梯度的参数，将这些参数都更新权重，lr是学习率（步长），还能设置权重衰退等参数
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
# 训练
for epoch in range(10000):
    # 获得输出
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)

    # 梯度清零后反向传播更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 每100个打印一次训练信息
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")
# 打印更新结果
print(f"w={model.linear.weight.item()},b={model.linear.bias.item()}")

# 测试
x_test = torch.tensor([[4.0]])
print(f"y_test={model(x_test).item()}")
```



#### 线性层（二分类）

##### 代码实现

```py
"""
由线性回归->二分类（逻辑斯蒂回归）
二分类重要的是求y与y_hat分布的差异同时输出值在0-1区间
因为不是值得差异，而是分布的差异，所以用到交叉熵损失（二维）。

与线性回归一共有三处差异，其他完全相同
需要引入functional，用sigmoid将输出值固定在0-1
损失使用交叉熵bce而非mse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import  matplotlib.pyplot as plt

# 数据预处理
x_data = torch.tensor([[1.0], [2.0], [3.0]])
# 不是预测，而是分类
y_data = torch.tensor([[0], [0], [1.0]])

# 定义网络模型
class Linear(nn.Module):
    # __init__是py的构造器，必须要有
    def __init__(self):
        # 这就是调用父类的构造器，所有的模型照着写就行了，不用深究
        super(Linear,self).__init__()
        # torch.nn.linear有三个参数，分别是输入维度，输出维度，是否有偏置量b(默认为true)，返回的是一个对象
        self.linear=torch.nn.Linear(1,1,bias=True)
    # forward是module强制子类实现的，而且必须就叫这名字，返回的是输出结果
    def forward(self,x):
        # linear实现y=wx+b并返回结果
        y_pred=F.sigmoid(self.linear(x)) #第二处差异
        return y_pred

# 实例化model
model=Linear()
# 定义损失和优化器

# 只有一个参数，决定求和后要不要求均值（并不影响）
criterion = torch.nn.BCELoss(reduction='mean')#第三处差异
# model.parameters()能获得所有有梯度的参数，将这些参数都更新权重，lr是学习率（步长），还能设置权重衰退等参数
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
# 训练
for epoch in range(10000):
    # 获得输出
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)

    # 梯度清零后反向传播更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 每100个打印一次训练信息
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")
# 打印更新结果
print(f"w={model.linear.weight.item()},b={model.linear.bias.item()}")

# 测试
x_test = torch.tensor([[4.0]])
print(f"y_test={model(x_test).item()}")


# 用plt展示图像 曲线非常近似sigmoid
x=np.linspace(0,10,200)
x_t=torch.Tensor(x).view(200,1)
y_t=model(x_t)
y=y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('probability of pass')
plt.grid()
plt.show()
```



#### 线性层（多分类问题）

***

##### 代码实现

```py
import torch
# 引入数据集和预处理
from torchvision import datasets, transforms
# 引入 DataLoader 用于加载数据集
from torch.utils.data import DataLoader
# 引入优化器
import torch.optim as optim
# 引入激活函数和交叉熵损失函数
import torch.nn.functional as F
# 一次训练处理64个样本
batch_size = 64

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# 加载手写数字数据集
train_dataset = datasets.MNIST(root='../dataset/minist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 加载测试数据集
test_dataset = datasets.MNIST(root='../dataset/minist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义神经网络
class Net(torch.nn.Module):
    def __init__(self):
        # 调用父类初始化函数
        super(Net, self).__init__()
        # 定义网络结构 
        self.fc1 = torch.nn.Linear(784, 512)  # 输入层到第一隐藏层
        self.fc2 = torch.nn.Linear(512, 256)  # 第一隐藏层到第二隐藏层
        self.fc3 = torch.nn.Linear(256, 128)  # 第二隐藏层到第三隐藏层
        self.fc4 = torch.nn.Linear(128, 64)   # 第三隐藏层到第四隐藏层
        self.fc5 = torch.nn.Linear(64, 10)    # 第四隐藏层到输出层

    def forward(self, x):
        x = x.view(-1, 784)  # 将输入展平成 (batch_size, 784)
        x = F.relu(self.fc1(x))  # ReLU 激活函数
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # 输出层不使用激活函数（交叉熵损失函数内部会应用 softmax）
        return x

# 实例化模型、定义损失函数和优化器
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 训练过程
def train(epoch):
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()
        if batch_idx % 300 == 299:    # 每300个mini-batch打印一次损失
            print(f'[{epoch + 1}, {batch_idx + 1}] loss: {running_loss / 300:.3f}')
            running_loss = 0.0

# 测试过程
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试时不需要计算梯度
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 获取最大概率的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')

# 训练和测试模型
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

```

#####  当前理解

***

- 先调用相关包，以及进行数据集的一些预处理
- 继承torch.nn.module搭建模型
- 实例化，定义好损失和优化函数
- 进行训练：前向，反向，更新
- 测试模型





### 卷积神经网络

***

