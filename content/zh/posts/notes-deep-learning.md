---
title: "学习笔记：吴恩达深度学习"
date: 2023-10-21T12:36:53Z
draft: false
description: 神经网络。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Deep Learning
- TensorFlow
categories:
- Notes
libraries:
- mathjax
---

挖坑中...

<!-- 神经网络（Neural Network），解决**分类+回归**问题。 -->

## 激活函数

为了在神经网络中将线性输出转化为非线性输出。

结论：
1. 隐藏层建议使用 ReLU；
2. 输出层根据预测值选择：
   - 二分类问题：Sigmoid
   - 多分类问题：Softmax
   - 预测值非负问题：ReLU
   - 预测值可正可负可零问题：Linear（即不使用激活函数）

<!-- ### Sign

符号函数，公式如下：

$$
sgn \space x = 
\begin{cases}
-1, & \text{if $x < 0$} \\\\
0, & \text{if $x = 0$} \\\\
1, & \text{if $x \ge 0$}
\end{cases}
$$ -->

### Linear

线性激活函数，也称作 no activation，本质上相当于没有使用激活函数。

$$ 
f(x) = x 
$$

$$ 
f'(x) = 1
$$

<img src='https://user-images.githubusercontent.com/46241961/277111054-df2a4eb0-f099-40bc-a3b6-61b44ae7ea58.svg' alt='ActivationFunction_Linear' width=80%>

### Sigmoid

也称作 Logistic function，适用二分类问题。

$$ 
f(x) = \frac{1}{1+e^{-x}} \in (0,1) 
$$

$$
f'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} \in (0,0.25)
$$

<img src='https://user-images.githubusercontent.com/46241961/276302604-44080d48-59a9-4ce1-ab2a-83e058ac00af.svg' alt='ActivationFunction_Sigmoid' width=80%>

### tanh

$$
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1, 1)
$$

<img src='https://user-images.githubusercontent.com/46241961/276302656-2e0dfbdc-a990-4486-bec9-502441bfe07a.svg' alt='ActivationFunction_tanh' width=80%>

### ReLU

$$
f(x) =
\begin{cases}
x & \text{if $x \geq 0$} \\\\
0 & \text{if $x < 0$}
\end{cases} \space\space\space \text{or} \space\space\space
f(x) = \max(0, x)
$$

$$
f'(x) =
\begin{cases}
1 & \text{if $x \geq 0$} \\\\
0 & \text{if $x < 0$}
\end{cases}
$$

<img src='https://user-images.githubusercontent.com/46241961/276302720-d6f6ffe9-6a1c-45a3-9bbc-1fe15938f289.svg' alt='ActivationFunction_ReLU' width=80%>

### Softmax

适用多分类问题。

$$
p(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{k}e^{x_j}}, 0 < p(x_i) < 1, \sum_{i} p(x_i)= 1
$$
