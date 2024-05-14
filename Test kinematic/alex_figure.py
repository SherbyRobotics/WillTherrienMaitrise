#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

l_xmaxx = 0.475 / 100.0

x_axis = ['Baseline', 'Normalized', 'PCA (n=3)' , 'PCA (n=2)' ,'Augmented', 'Pi','Pi+fillers','Augmented Pi']

Xmaxx_mae = np.array([
3.24,
4.25,
5.25,
52.64,
2.75,
2.04,
1.33,
1.38,
]) * l_xmaxx

limo_mae = np.array([
46.99,
46.88,
46.07,
59.72,
22.83,
7.61,
8.97,
6.04,
]) * l_xmaxx

racecar_mae = np.array([
30.99,
30.85,
32.14,
52.58,
13.79,
2.78,
4.19,
2.23,
]) * l_xmaxx

merged_mae = np.array([
10.41,
10.24,
17.41,
53.83,
3.83,
2.45,
2.15,
1.78,
]) * l_xmaxx

# algo_names = ['Base', 'Normalized', 'PCA(n=3)' , 'PCA(n=2)' ,'Augmented']







fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Mean Absolute Error for $Y$ predictions [m] ')
ax1.plot(x_axis, merged_mae ,  '--b', marker="x",label="Merged")
ax1.plot(x_axis, racecar_mae ,  '--r', marker="o",label="Small")
ax1.plot(x_axis, limo_mae ,  '--g', marker="*",label="Long")
ax1.plot(x_axis, Xmaxx_mae ,  '--k', marker="+",label="Big")
plt.legend(loc='upper right')
plt.xticks(rotation = 45)
plt.grid()
plt.show()
fig.savefig('alex_figure2.png', dpi=300, bbox_inches='tight')