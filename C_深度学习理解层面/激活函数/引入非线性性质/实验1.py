import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 数据库
red_points = np.array([[-1, -1], [-0.5, 1], [0, -1], [0.5, 1]])
blue_points = np.array([[1, 1], [0, 1], [-0.5, -1], [0.5, -1]])

# 画图 描点
plt.scatter(red_points[:, 0], red_points[:, 1], color='red')
plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Dataset')
plt.legend()
plt.show()

# 构建神经网络模型  输入层 —— 隐藏层 —— 输出层   两个线性层和两个激活函数
model = nn.Sequential(
    nn.Linear(2, 10),  # 输入层到隐藏层
    nn.ReLU(),  # 使用ReLU激活函数
    nn.Linear(10, 1),  # 隐藏层到输出层
    nn.Sigmoid()  # 使用Sigmoid激活函数
)

# 准备数据
X = torch.tensor(np.concatenate([red_points, blue_points]), dtype=torch.float32)
y = torch.tensor([0] * len(red_points) + [1] * len(blue_points), dtype=torch.float32)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器

# 训练模型  训练1000轮
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()   # 梯度清零
    outputs = model(X).squeeze()  # 模型训练
    loss = criterion(outputs, y)  # 计算了模型的预测值outputs与真实标签y之间的损失
    loss.backward()  # 计算损失相对于模型参数的梯度
    optimizer.step()  # 调整权重


# 绘制决策边界
def plot_decision_boundary(model, X):
    # 计算网格的范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 生成网格点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    # 获取模型对网格点的预测
    Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).detach().numpy()
    # 绘制等高线图
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdBu)
    # 绘制数据点
    plt.scatter(red_points[:, 0], red_points[:, 1], color='red', label='Red Points')
    plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue', label='Blue Points')
    # 设置坐标轴和标题
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.legend()
    plt.show()


plot_decision_boundary(model, X)
