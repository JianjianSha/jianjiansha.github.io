---
title: PyTorch.optim.SGD
p: pytorch/optim_SGD
date: 2020-01-02 16:25:32
tags: PyTorch
mathjax: true
---

# 1. SGD
<!-- more -->

## 1.1 weight decay
为了过拟合，通常在损失函数中增加正则项，记原来损失（MSE 或者 CE 等）为 $L_0$，那么添加正则项后的损失为，

$$L=L_0+\frac 1 2 \lambda \cdot \|\mathbf \theta\|_2^2$$
如图 1 所示，
![](/images/pytorch/overfitting.png) <center>图 1 过拟合图示（来源《Deep Learning with PyTorch》）</center>
下半部分展示了过拟合的图，这种曲线的多项式比上图曲线的多项式，其系数 $\mathbf \theta$ 的绝对值更大（曲线的部分段变化更快，说明斜率绝对值更大，也就是 $|\mathbf \theta_i|$ 更大），所以增加正则项作为惩罚，其中 $\lambda$ 作为平衡因子，也称 `weight decay`。于是，求导时，损失对每个权重参数的梯度多了一项
$$\frac {\partial L}{\partial \mathbf \theta_i}=\frac {\partial L_0}{\partial \mathbf \theta_i}+\lambda \cdot \mathbf \theta_i$$

## 1.2 momentum
使用 SGD 训练时，有时候下降比较慢，甚至会陷入导局部最小值中，如图 2 所示，引入 momentum 可以加快收敛速度，我们知道 SGD 的参数更新公式为
$$\theta_{t+1} = \theta_t -\epsilon \cdot d\theta_t$$
而使用 momentum 的更新公式为
$$\begin{aligned} v_{t+1} & = \mu \cdot v_t + d\theta_t
\\\\ \theta_{t+1} &= \theta_t - \epsilon \cdot v_{t+1}=\theta_t-\epsilon \cdot \mu \cdot v_t - \epsilon \cdot d\theta_t \end{aligned} \qquad(1)$$
其中 $\theta_0$ 为初始权值参数值，$v_0=0$，$\epsilon$ 为学习率，$\mu$ 为 momentum 系数。从上两式中可见，如果当前 velocity（v 值） 与梯度方向一致，那么将会加快权值参数的变化量。在局部最小值附近（见图 2），由于 velocity 累计了之前的梯度，所以有望冲出局部最小值区域。
![](/images/pytorch/momentum.png) <center>图 2</center>
有的地方写成如下形式：
$$\begin{aligned}v_{t+1}&=\mu \cdot v_t - d\theta_t
\\\\\theta_{t+1}&=\theta_t+\epsilon \cdot v_{t+1}=\theta_t +\epsilon \cdot \mu \cdot v_t-\epsilon \cdot d\theta_t \end{aligned}\qquad(2)$$
实际上，当 $v_0=0$ 时，这两组更新公式本质相同。

caffe 框架以及在 Sutskever. [1] 中，更新公式为：
$$\begin{aligned}v_{t+1}&=\mu \cdot v_t + \epsilon \cdot d\theta_t
\\\\ \theta_{t+1}&=\theta_t - v_{t+1}=\theta_t-\mu \cdot v_t-\epsilon \cdot d\theta_t\end{aligned} \qquad(3)$$

或者 
$$\begin{aligned}v_{t+1}&=\mu \cdot v_t - \epsilon \cdot d\theta_t
\\\\ \theta_{t+1}&=\theta_t + v_{t+1}=\theta_t+\mu \cdot v_t-\epsilon \cdot d\theta_t\end{aligned} \qquad(3')$$

假设学习率 $\epsilon$ 保持不变，那么在 $v_0=0$ 时，(3) 式中的 $v_{t+1}$ 是 (1) 式中的 $\epsilon$ 倍（在其他变量均相同的情况下），
$$\begin{aligned}v_1^{(3)}&=\epsilon \cdot d\theta_0 = \epsilon \cdot v_1^{(1)}
\\\\ v_2^{(3)}& = \mu \cdot v_1^{(3)}+\epsilon \cdot d\theta_1=\epsilon \cdot [\mu \cdot v_1^{(1)}+d\theta_1]=\epsilon \cdot v_2^{(1)}
\\\\ &\cdots \end{aligned}$$

所以 (1) 式中更新 $\theta_{t+1}$ 时 $v_{t+1}$ 前面添加了系数 $\epsilon$ 后，(1) 和 (3) 也是等价的，但是前提条件是: 1. $v_0=0$；2. 学习率 $\epsilon$ 保持不变。

随着训练 epoch 增加，学习率可能会衰减，例如衰减为 10%，那么 (1) 和 (3) 会出现不同，我们继续写下迭代计算过程以探究为何会发生不同。记 在 t+1 时刻发生 $\epsilon$ 衰减，衰减前后分别为 $\epsilon_1, \ \epsilon_2$，且 $\epsilon_2 = 0.1 \epsilon_1$，
$$\begin{aligned}v_t^{(3)} &= \epsilon_1 \cdot v_t^{(1)}
\\\\ \theta_{t+1}^{(3)}&=\theta_t- \mu \cdot v_t^{(3)} - \epsilon_2 \cdot d\theta_t=\theta_t- \epsilon_1 \cdot \mu \cdot v_t^{(1)} - \epsilon_2 \cdot d\theta_t
\\\\ \theta_{t+1}^{(1)}&=\theta_t-\epsilon_2\cdot \mu \cdot v_t^{(1)} - \epsilon_2 \cdot d\theta_t=\theta_t-0.1\epsilon_1\cdot \mu \cdot v_t^{(1)} - \epsilon_2 \cdot d\theta_t\end{aligned}$$

显然，(1) 式的更新方式在学习率衰减时能够有更加明显的体现，参数更新量明显变小，而 (3) 式此时的 velocity 相对较大，参数更新的量没有明显变小。当然随着训练迭代的推进，(3) 式的参数更新量也会逐渐变小，这是因为 velocity 不断更新后，逐渐被较新的 $\epsilon_2 \cdot d\theta$ 主导，而先前累积的 $v_t^{(3)}$ 占比会越来越小，
$$\begin{aligned} v_{t+n}&=\mu \cdot v_{t+n-1}+\epsilon_2 d\theta_{t+n-1}\\\\ &=\mu^2 \cdot v_{t+n-2}+\mu \cdot \epsilon_2 \cdot d\theta_{t+n-2}+\epsilon_2 \cdot d\theta_{t+n-1}
\\\\&=\cdots
\\\\&=\mu^n \cdot v_t + \mu^{n-1} \cdot \epsilon_2 \cdot d\theta_{t}+\mu^{n-2} \cdot \epsilon_2 \cdot  d\theta_{t+1} + \cdots + \mu^0 \cdot \epsilon_2 \cdot d\theta_{t+n-1}\end{aligned}$$
由于 $\mu <1$，当 $n$ 较大时，上式第一项即 $v_t$ 对 velocity 贡献可以忽略。

### 1.2.1 dampening

阅读 PyTorch 中这部分的[源码](https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py#L71)，发现还使用了一个参数 `dampening`，这个参数指使用 momentum 更新时，对当前梯度的抑制，即 (1) 式中 velocity 更新变为
$$v_{t+1} = \mu \cdot v_t + \text{dampening} \cdot d\theta_t$$
其实很多地方写成 $v_{t+1} = \mu \cdot v_t + (1-\mu) \cdot d\theta_t$，由于 $0 \le \mu < 1$，这表示 $v_{t+1}$ 在 $\min (v_t, d\theta_t)$ 与 $\max (v_t, d\theta_t)$ 之间。不过，实际计算中，`dampening` 常使用默认值 `0`。

## 1.3 Nesterov
在 momentum 一节使用式 (1) (3) 进行介绍，其实是为了与 PyTorch [源码](https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py#L71) 或者 [文档](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) 对应，但是很多论文或者博客中更常用式 (3') 的形式（将学习率放入 $v_{t+1}$ 的计算式中），
$$\theta_{t+1}=\theta_t+v_{t+1}$$
这里，$v_{t+1}$ 的定义有很多种，例如经典 momentum，NAG (Nesterov Accelerated Gradient) 等。

### 1.3.1 经典 momentum（CE）

我们重写一遍经典 momentum 的参数更新公式（式 2），
$$\begin{aligned}v_{t+1}&=\mu \cdot v_t - \epsilon \cdot \nabla f(\theta_t)
\\\\ \theta_{t+1}&=\theta_t + \mu \cdot v_t - \epsilon \cdot \nabla f(\theta_t)\end{aligned}$$

### 1.3.2 NAG
NAG 一次迭代过程分为两步：
1. 梯度下降步骤
   
   $$\theta_{t+1} = y_t - \epsilon_t \cdot \nabla f(y_t) \qquad(4)$$
2. momentum
   
   $$ y_{t+1}=\theta_{t+1} + \mu_{t+1} \cdot (\theta_{t+1}-\theta_t) \qquad(5)$$
   
初始时令 $y_0=\theta_0$，即这两个变量起点相同。

NAG 中我们对 $y$ 做梯度下降，得到的值为 $\theta$ 的新值，而非 $y$ 新值，$y$ 的新值是在 $\theta$ 的基础之上再增加 $\mu_{t+1} \cdot (\theta_{t+1}-\theta_t)$ 这么多的更新量。如图 3，
![](/images/pytorch/NAS_0.png) <center>图 3. NAG 过程示意图</center>
如果 $\mu \equiv 0$，NAG 就是普通的梯度下降 SD。


注意，这里将 $\mu, \ \epsilon$ 系数带上时刻下标。下面我们推导 velocity 的迭代计算式。


### 1.3.3 Sutskever Nesterov Momentum
NAG 中参数 $\theta$ 的更新在梯度下降之后，在 momentum 之前。现在我们根据 NAG 推导出 velocity 项。首先要说明的是，需要将 NAG 迭代过程的两个步骤顺序对换，即 `momentum-GD-momentum-GD-...` 的顺序。

已知，
$$y_t=\theta_t+\mu_t \cdot(\theta_t-\theta_{t-1})$$

写成以下形式，
$$y_t=\theta_t + \mu_t \cdot v_t$$

可根据 (4) 式消去 $y_t$，需要注意的是，(4) 式表示 t 时刻迭代过程中的梯度下降步骤，到 Sutskever Nesterov Momentum 中则为 t-1 时刻迭代中的 momentum 步骤，即 $\theta_{t+1} = y_t - \epsilon_t \cdot \nabla f(y_t)$，与上式联合可消去 $y_t$， 得
$$\theta_{t+1} = \theta_t+\mu_t \cdot v_t-\epsilon_t \cdot \nabla f(\theta_t + \mu_t \cdot v_t) \qquad(6)$$

于是，

$$v_{t+1} = \mu_t \cdot v_t - \epsilon_t \cdot \nabla f(\theta_t+\mu_t \cdot v_t)  \qquad(7) $$

图 4 是经典 momentum 与 NAG 方法的图示比较。
![](/images/pytorch/NAG.png) <center>图 4 </center>


### 1.3.4 Bengio Nesterov Momentum
NAG 中我们的模型参数是 $\theta$，但是其更新不是对自身做梯度下降，而是对 $y$ 做梯度下降，进一步地，$y$ 的更新则又反过来依赖于 $\theta$ 的 momentum。

定义一个新变量，表示经过 momentum 更新后的 $\theta$ 值，或者更准确地讲，是 momentum 更新后的模型参数的值。

$$\Theta_{t-1}=\theta_{t-1} + \mu_{t-1} \cdot v_{t-1}$$
这里可能感觉有点绕，一会 $\theta$，一会 $\Theta$，到底哪个是表示模型参数。我是这么理解的，初始时模型参数为 $\theta_0$，此后更新迭代过程中，$\Theta$ 才表示模型参数，$\theta$ 只作为中间变量。


根据 velocity 的定义 (7) 式，有
$$v_t=\mu_{t-1} \cdot v_{t-1} - \epsilon_{t-1} \cdot \nabla f(\Theta_{t-1})$$

$v_t$ 依然是 $\theta$ “中间”变量的更新量。

根据 $\Theta$ 定义，
$$\Theta_{t+1}-\mu_{t+1} \cdot v_{t+1}= \theta_{t+1}
\\\\ \Theta_t-\mu_t \cdot v_t= \theta_t$$
以及 (6) 式，有
$$\Theta_{t+1}-\mu_{t+1} \cdot v_{t+1}=\Theta_t-\mu_t \cdot v_t+\mu_t \cdot v_t - \epsilon_t \cdot \nabla f(\Theta_t)$$
化简得，
$$\Theta_{t+1}=\Theta_t+\mu_{t+1} \cdot v_{t+1}-\epsilon_t \cdot \nabla f(\Theta_t)$$

继续代入 (7) 式，有
$$\Theta_{t+1}=\Theta_t+\mu_{t+1} \cdot[\mu_t \cdot v_t - \epsilon_t \cdot \nabla f(\Theta_t)]-\epsilon_t \cdot \nabla f(\Theta_t)$$

展开得，
$$\Theta_{t+1}=\Theta_t+\mu_{t+1} \cdot \mu_t \cdot v_t-\mu_{t+1} \cdot \epsilon_t \cdot \nabla f(\Theta_t)-\epsilon_t \cdot \nabla f(\Theta_t) \qquad(8)$$

写成 $\Theta_{t+1}=\Theta_t + V_{t+1}$ 的形式，于是
$$V_{t+1}=\mu_{t+1} \cdot \mu_t \cdot v_t-\mu_{t+1} \cdot \epsilon_t \cdot \nabla f(\Theta_t)-\epsilon_t \cdot \nabla f(\Theta_t)$$
 就是 $\Theta$ 的更新量，等价于 `(3')` 式中的 $v_{t+1}$，对应到 (1) 式中的 $v_{t+1}$ 的形式，去掉 $\epsilon$，以及 $-$ 变成 $+$，易得， 
$$\begin{aligned} V_{t+1}&=\mu_{t+1} \cdot \mu_t \cdot v_t+\mu_{t+1} \cdot \nabla f(\Theta_t)+ \nabla f(\Theta_t) 
\\\\ &=\mu_{t+1} \cdot [\mu_t \cdot v_t+ \nabla f(\Theta_t)] + \nabla f(\Theta_t)  \end{aligned} \qquad(9)$$
此时 $\Theta$ 的更新为
$$\Theta_{t+1}=\Theta_t - \epsilon_t \cdot V_{t+1} \qquad(10)$$

__(9) 和 (10) 式就对应 PyTorch 源码中 `SGD.step` 在 `nesterov=True` 时的计算过程。__


# 参考：

[1] On the importance of initialization and momentum in deep learning. Ilya Sutskever

[2] [Nesterov Accelerated Gradient and Momentum](https://jlmelville.github.io/mize/mesterov.html)

# 更多阅读
[1] [ORF523: Nesterov's Accelerated Gradient Descent](https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/)