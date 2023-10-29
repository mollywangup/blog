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

### 颜色填充

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots()

ax.plot(x, y, color='green')
ax.fill(x, y, facecolor='green', edgecolor=None, alpha=0.25)
ax.fill_between(x=x, y1=y+1, y2=y+2, color='lightskyblue', alpha=0.75)

plt.savefig('fill-and-fill_between.svg')
plt.show()
```

<img src='https://user-images.githubusercontent.com/46241961/278825203-d5b282fb-9d70-480b-9ad1-9461e28dbd4f.svg' alt='fill-and-fill_between'>

<!-- ###  -->

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

<!-- 
参考
https://liyangbit.com/pythonvisualization/matplotlib-top-50-visualizations/
https://www.huaxiaozhuan.com/%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80/chapters/2_probability.html

激活函数那个：https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/normalization/basic_normalization.html#id2
熵那个：https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC1%E6%AD%A5%20-%20%E5%9F%BA%E6%9C%AC%E7%9F%A5%E8%AF%86/03.2-%E4%BA%A4%E5%8F%89%E7%86%B5%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0/

mathjax符号那个：https://www.oscaner.com/skill/others/mathjax-symbol.html


一般：
向量范数那个：https://sunocean.life/blog/blog/2020/08/31/deep-learning-math-norm#%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%8E%E8%8C%83%E6%95%B0 -->


