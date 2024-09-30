from tkinter import Grid
from turtle import color
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np

alg_names = ['target', 'target + source', 'target + unedit source']

fig = plt.figure(figsize=[7, 7])

# sim  real
y_t = [27.44]
y_ts = [36.84]
y_tus = [30.16]

err_t = [1.99]
err_ts = [1.68]
err_tus = [3.58]

xticks_label = ['halfcheetah-mr']
x = np.arange(1)

# alg_names = ['H2O+(IQL)']

err_attr = {"elinewidth": 1, "ecolor": "black", "capsize": 3}
bar_width = 0.05

color_list = ['rebeccapurple', (236/255, 174/255, 65/255), (49/255, 207/255, 162/255), (218/255, 107/255, 148/255), (63/255, 115/255, 191/255)]

plt.bar(x - bar_width, y_t, yerr=err_t, color=color_list[1], error_kw=err_attr, width=bar_width)
plt.bar(x, y_ts, yerr=err_ts, color=color_list[3], error_kw=err_attr, width=bar_width)
plt.bar(x + bar_width, y_tus, yerr=err_tus, color=color_list[4], error_kw=err_attr, width=bar_width)
plt.legend(alg_names, fontsize=20, loc="lower right")
plt.ylabel("Average Return", fontsize=20)
plt.xticks(x, labels=xticks_label, fontsize=20)

plt.yticks(fontsize=20)

plt.subplots_adjust(top=0.99)

plt.savefig('figures/halfcheetah-mr.pdf')
plt.show()
