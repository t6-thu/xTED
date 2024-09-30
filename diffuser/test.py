import matplotlib.pyplot as plt
import numpy as np
from noise import OUNoise
# 生成一组示例数据，这里以正态分布为例
OUGen = OUNoise(shape=(1000,1000), theta = 0.15, mu=0, sigma=0.01)
GaussianNoise = np.random.randn(1000)
data = OUGen.gen_noise()[:, 0]

# 绘制直方图
plt.plot(range(1000), data)
plt.show()
plt.savefig('test.png')
plt.close()

# # 绘制核密度估计图
# plt.figure(figsize=(10, 5))
# plt.hist(data, bins=30, density=True, alpha=0.5, color='blue', edgecolor='black')
# plt.title('Histogram and Kernel Density Estimation of Data Distribution')
# plt.xlabel('Value')
# plt.ylabel('Density')

# # 绘制核密度估计曲线
# density, bins, _ = plt.hist(data, bins=30, density=True, alpha=0)
# plt.plot(bins, 1/(np.std(data) * np.sqrt(2 * np.pi)) * np.exp( - (bins - np.mean(data))**2 / (2 * np.std(data)**2) ), color='red', linewidth=2)
# plt.legend(['Kernel Density Estimation', 'Histogram'])
# plt.grid(True)
# plt.show()
# plt.savefig('kernel.png')
# plt.close()
