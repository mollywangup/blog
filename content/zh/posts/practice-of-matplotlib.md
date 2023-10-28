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


## 小功能

### 

## 附：理解画布

`figure` 可以理解为整个画布，`ax/axes` 可以理解为具体的绘图区域（包括坐标轴、刻度、标题等）。
`plt` 直接绘制在整个画布上，`ax/axes` 绘制在自己的一亩三分地之中。因此稍微复杂的图都**不建议**直接使用 `plt`.

### 画一个

画复杂单图推荐。

```python
import numpy as np
import matplotlib.pyplot as plt

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

plt.show()
```

### 画一组

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 50)
y = np.sin(x)

# 配置画布：1x3
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