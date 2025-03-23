import numpy as np
import tensorflow as tf
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

# 3. 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)  # 输出层，无激活函数
])

# 4. 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

# 5. 训练模型
history = model.fit(x_train, y_train, epochs=500, verbose=0)

# 6. 预测
y_pred = model.predict(x_test)

# 7. 结果可视化
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, label="Ground Truth", color='b', s=10)
plt.plot(x_test, y_pred, label="NN Prediction", color='r', linewidth=2)
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Function Approximation using ReLU Network")
plt.show()
