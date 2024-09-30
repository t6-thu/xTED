
import numpy as np
from itertools import combinations
from scipy.stats import ks_2samp

def mean_ks_statistic(samples):
    """
    计算Mean Kolmogorov-Smirnov statistic

    参数：
    samples: 一个包含多个样本的列表，每个样本应该是一个一维NumPy数组

    返回值：
    mean_ks: Mean Kolmogorov-Smirnov statistic
    """
    num_samples = len(samples)
    ks_values = []

    # 对每一对样本计算KS统计量
    for sample1, sample2 in combinations(samples, 2):
        ks_statistic, _ = ks_2samp(sample1, sample2)
        ks_values.append(ks_statistic)

    # 计算平均KS统计量
    mean_ks = np.mean(ks_values)

    return mean_ks

# # 示例用法
sample1 = np.random.normal(loc=0, scale=1, size=100)
sample2 = np.random.normal(loc=0, scale=1, size=100)
sample3 = np.random.normal(loc=1, scale=1, size=100)

mean_ks = mean_ks_statistic([sample1, sample3])
print("Mean Kolmogorov-Smirnov statistic:", mean_ks)
