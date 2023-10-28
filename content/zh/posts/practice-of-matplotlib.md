---
title: "Matplotlib 绘图例子"
date: 2023-10-19T05:26:51Z
draft: false
description: 包括概率分布函数，激活函数，SST/SSR/SSE，2D vs. 3D 等。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Matplotlib
categories:
- Notes
libraries:
- mathjax
---




## 附：理解画布

`figure` 可以理解为整个画布，`ax/axes` 可以理解为具体的绘图区域（包括标题、坐标轴、刻度等）。
`plt` 直接绘制在整个画布上，`ax/axes` 绘制在自己的一亩三分地中，并且有的复杂功能 `plt` 上没有。因此稍微复杂的图都**不建议**直接使用 `plt`.

### 画一个（推荐）

画复杂单图推荐，可扩展性较强。

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.patches import PathPatch

x = np.linspace(0, 2 * np.pi, 50)
y = np.sin(x)

# 以下两种方式二选一
# 方式一
fig, ax = plt.subplots()

# # 方式二
# fig = plt.figure()
# ax = fig.add_subplot()

ax.plot(x, y-1)
ax.plot(x, y)
ax.plot(x, y+1)

# 添加辅助线
vertices = [(0, -2), (0, 2), (1.5, 2), (1.5, -2), (0, -2)]
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
path = Path(vertices, codes)
pathpatch = PathPatch(path, facecolor='none', edgecolor='green', ls='--', lw=0.7)
ax.add_patch(pathpatch)

# 填充颜色
ax.fill_between(x=x, y1=y+1, y2=y, color='lightskyblue', alpha=0.75)

plt.savefig('single-ax.svg')
plt.show()
```

<img src='https://user-images.githubusercontent.com/46241961/278818513-50f9de61-e2a3-4fb4-9a12-a8aaacad7ac0.svg' alt='single-ax'>

### 画一组（推荐）

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

x = np.linspace(0, 2 * np.pi, 50)
y = np.sin(x)

# 配置画布
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharey=True)
ax1, ax2, ax3 = axes

# 大标题
fig.suptitle('This is Figure', fontweight='bold')

# Axes
# Tip: 可以在以下三行中的任意一行后，测试直接 plt 的效果
ax1.plot(x, y-1, color='#4EACC5', label=r'$y=\sin(x)-1$')
ax2.plot(x, y, color='#FF9C34', label=r'$y=\sin(x)$')
ax3.plot(x, y+1, color='#4E9A06', label=r'$y=\sin(x)+1$')

# Axes 标题
ax1.set_title('This is Axes1')
ax2.set_title('This is Axes2')
ax3.set_title('This is Axes3')

# Axes 图例
ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best', frameon=False)

# Axes 轴标签和刻度
ax1.set_xlabel('xlabel')
ax1.set_ylabel('ylabel')
ax2.set_xticks([i for i in range(0, 7, 2)])
ax3.set_xticks(())

plt.savefig('multiple-axes.svg')
plt.show()
```

<img src='https://user-images.githubusercontent.com/46241961/278814329-f3333ef6-be84-4ea2-9564-6d24001929d8.svg' alt='multiple-axes'>