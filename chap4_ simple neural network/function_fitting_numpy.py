import numpy as np
import matplotlib.pyplot as plt

# 1. 定义目标函数
def target_function(x):
    return np.sin(2 * np.pi * x) + 0.5 * x

# 2. 生成数据
np.random.seed(42)
x_train = np.linspace(-1, 1, 100).reshape(-1, 1)
y_train = target_function(x_train)

x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
y_test = target_function(x_test)

# 标准化输入数据
x_mean, x_std = np.mean(x_train), np.std(x_train)
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

# 3. 定义激活函数（ReLU）
def relu(x):
    return np.maximum(0, x)

# 4. 初始化神经网络参数（He 初始化）
input_size = 1
hidden_size = 50  # 增加神经元数量
output_size = 1
learning_rate = 0.005  # 初始学习率
epochs = 5000  # 增加训练轮数
batch_size = 20  # Mini-batch 大小

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
b2 = np.zeros((1, output_size))

# 5. 训练神经网络（使用 Mini-batch SGD）
n_samples = x_train.shape[0]
for epoch in range(epochs):
    # 随机打乱数据
    indices = np.random.permutation(n_samples)
    x_train, y_train = x_train[indices], y_train[indices]
    
    for i in range(0, n_samples, batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # 前向传播
        hidden_layer = relu(np.dot(x_batch, W1) + b1)
        output_layer = np.dot(hidden_layer, W2) + b2
        
        # 计算损失 (均方误差)
        loss = np.mean((output_layer - y_batch) ** 2)
        
        # 反向传播
        d_output = 2 * (output_layer - y_batch) / batch_size
        dW2 = np.dot(hidden_layer.T, d_output)
        db2 = np.sum(d_output, axis=0, keepdims=True)
        
        d_hidden = np.dot(d_output, W2.T) * (hidden_layer > 0)
        dW1 = np.dot(x_batch.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        # 更新参数
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
    
    # 逐步降低学习率
    if epoch % 1000 == 0:
        learning_rate *= 0.9
    
    # 打印损失
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# 6. 预测
test_hidden = relu(np.dot(x_test, W1) + b1)
y_pred = np.dot(test_hidden, W2) + b2

# 7. 结果可视化
plt.figure(figsize=(8, 5))
plt.scatter(x_train * x_std + x_mean, y_train, label="Ground Truth", color='b', s=10)
plt.plot(x_test * x_std + x_mean, y_pred, label="NN Prediction", color='r', linewidth=2)
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Function Approximation using Numpy ReLU Network with He Init & Mini-batch")
plt.show()
