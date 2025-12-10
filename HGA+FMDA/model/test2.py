import numpy as np
import matplotlib.pyplot as plt

# 生成x值，例如从-2到2的范围
x_values = np.linspace(-1, 1, 100)

# 计算以e为底的指数值
# y_values = 1 / (1 + np.exp(-(np.abs(x_values))))
# y_values = 1 - np.exp(-(x_values) * 3)
# y_values = 1 - np.exp(-(x_values) * 3)
# y_values = np.exp(-(np.abs(x_values)) * 2)
y_values = - np.exp(-2*(x_values))
# y_values = y_values / y_values.max()
# y_values = 2/ (1+np.exp(-2*(x_values) * 0.5)) - 1
# y_values = 1 / np.log(np.abs(x_values) + 1)
# y_values = -np.log(1 + np.exp(-2*(x_values)))
print(y_values.max())
print(y_values.min())
# 绘制曲线
plt.plot(x_values, y_values, label='e^x')

# 添加标签和标题
plt.xlabel('x')
plt.ylabel('e^x')
plt.title('Exponential Curve with base e')

# 添加图例
plt.legend()

# 显示图形
plt.show()