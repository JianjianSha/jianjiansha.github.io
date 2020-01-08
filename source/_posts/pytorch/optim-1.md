---
title: PyTorch.optim
p: pytorch/optim-1
date: 2020-01-06 14:38:40
tags:
---

# 1. Adagrad

## 1.1 原理
所有的参数形成一个参数向量，对每个参数使用不同的学习率。例如在时间步 `t`，第 `i` 个参数 $\theta_i$ 的梯度为 $g_{t,i}$，
$$g_{t,i} = \nabla_{\theta}J(\theta_{t,i})$$
SGD 的更新方式为，
$$\theta_{t+1,i}=\theta_{t,i}-\eta \cdot g_{t,i}$$
其中学习率 $\eta$ 恒定。

Adagrad 对每个参数在不同时间步调整学习率，参数更新为
$$\theta_{t+1,i}=\theta_{t,i}-\frac {\eta} {\sqrt{G_{t,ii}+\epsilon}} \cdot g_{t,i} \qquad(1)$$

其中 $G_t \in \mathbb R^{d \times d}$ 是一个对角矩阵，对角线上每个元素 $G_{t,ii}$ 是参数 $\theta_i$ 从时间步 `0` 到时间步 `t` 的梯度的平方和，
$$G_{t,ii}=\sum_{\tau=0}^t g_{\tau,i}^2 \qquad(2)$$

$\epsilon$ 为平滑因子，用于避免分母为 0，一般取值 `1e-8`。

将 (1) 式向量化，
$$\theta_{t+1}=\theta_t - \frac \eta {\sqrt {G_t+\epsilon}} \odot g_t \qquad(3)$$

其中 $\odot$ 表示矩阵与向量相乘。通常，$\eta=0.01$。

Adagrad 的优点是不需要手动调整学习率，缺点是随着迭代次数的增加，分母逐渐增大，导致最后变得非常小，学习过程非常缓慢甚至停止。

关于 Adagrad 调整学习率的理论分析可参考论文 [1]。

## 1.2 PyTorch 实现
PyTorch 的 Adagrad 实现中除了学习率 `lr` 和平滑因子 `eps`，还是增加了几个参数：
1. 学习率衰减因子 `lr_decay`
2. 权重衰减因子 `weight_decay`
3. 累加初始值 `G`，这是 (2) 式中累加的一个初始值

参数更新步骤如下：

设置累加初值
$$G_0=[G,...,G]$$
其中 $G_0$ 是一个向量（对角矩阵的对角线元素），与参数数量相同。

在时间步 `t`，

1. 增加权重衰减项（正则项）的梯度
   
   $$g_t := g_t + \lambda_{\theta} \cdot \theta_t$$

2. 学习率衰减为 
   
   $$\eta := \frac {\eta} {1+ t \cdot \lambda_{\eta}}  $$

3. 累加梯度平方
   
   $$G_{t+1} = G_t+ g_t \cdot g_t$$

4. 更新参数
   
   $$\theta_{t+1} = \theta_t - \frac \eta {\sqrt{G_t} + \epsilon}\cdot g_t$$

以上，向量的计算全部按元素进行（标量则在需要的时候广播为向量）。（不同的参数具有不同的调整后的学习率）

# 2. Adadelta
## 2.1 原理
Adadelta 是在 Adagrad 的基础上对学习率一味单调递减进行修改，不再对之前所有时间步的梯度做平方和，而是限制一个最近时间步的窗口，窗口大小为 `w`。

然而，由于存储 `w` 个梯度平方值效率较低，所以改为使用梯度的衰减均值，如下
$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1- \gamma)g_t^2$$
它的平方根就变成了 RMS（均方根，区别是每个元素的权重由 `1/n` 变成依次递增的值），
$$\text{RMS}[g]_t = \sqrt{E[g^2]_t + \epsilon}$$
这样，越早期时间步的梯度平方，其权重越低，贡献也小，越近期的梯度平方，贡献越大。$\gamma$ 可取 `0.9`。

于是，参数更新为，
$$\Delta \theta_t = -\frac \eta {\text{RMS}[g]_t}\cdot g_t \qquad(4)
\\\\\theta_{t+1}=\theta_t + \Delta \theta_t$$

更进一步地，更新量 $\Delta \theta$ 与 $\theta$ 在单位空间上不匹配，这在 SGD，momentum 以及 Adagrad 中也存在同样的问题，即
$$\Delta x 单位 \propto g 单位 \propto \frac {\partial f} {\partial x} \propto  \frac 1 {x 单位}$$
上式最后一步中假定了目标函数 `f` 是无单位的。这个单位空间不匹配如何理解呢？假设 `x` 表示距离，例如 米 $m$，损失函数 `f` 无量纲，根据上式，发现 `x` 的更新量的单位为 $m^{-1}$，显然这是不匹配的。为了实现匹配的目的，首先类似 $g^2$ 的衰减均值，定义更新量的衰减均值，
$$E[\Delta \theta^2]_t = \gamma \cdot E[\Delta \theta^2]_{t-1} + (1-\gamma)\Delta \theta_t^2$$

均方根为，
$$\text{RMS}[\Delta\theta]_t=\sqrt {E[\Delta \theta^2]_t+\epsilon}$$
残念，由于 $\Delta \theta_t$ 未知，所以上式也未知，所以近似使用 $\text{RMS}[\Delta\theta]_{t-1}$ 来代替，然后这个值就作为 (4) 式中的 $\eta$。

于是最终参数更新方式为，
$$\Delta \theta_t = -\frac {\text{RMS}[\Delta\theta]_{t-1}} {\text{RMS}[g]_t}\cdot g_t \qquad(5)
\\\\\theta_{t+1}=\theta_t + \Delta \theta_t$$
注意到 `RMS` 中有平方根计算，所以，$\text{RMS}[\Delta\theta]_{t-1}$ 与 $\theta$ 量纲匹配，而 $\text{RMS}[g]_t$ 与 $g$ 量纲匹配，所以 (5) 式中 $\Delta \theta$ 与 $\theta$ 量纲匹配。

## 2.2 PyTorch 实现
PyTorch 的 Adadelta 实现使用 (5) 式，非常简单，不再啰嗦。

# 3. RMSprop
RMSprop 就是 Adadelta 中 (4) 式，通常 $\gamma=0.9$，$\eta=0.001$。简单，我们直接看 PyTorch 实现部分。

## 3.1 PyTorch 实现
RMSprop 的 `step` 方法部分代码如下，
```python
# 对每个参数
square_avg = state['square_avg']    # 参数对应的梯度平方的衰减均值（也称 moving average）
alpha = group['alpha']              # 对应上文公式中的 gamma

if group['weight_decay'] != 0:
    grad = grad.add(group['weight_decay'], p.data)  # 添加正则项的梯度

square_avg.mul_(alpha).addcmul_(1-alpha, grad, grad)    # 计算 E[g^2]

if group['centered']:               # 使用 centered 版本的 RMSprop
    grad_avg = state['grad_avg']    # 获取 梯度衰减平均
    grad_avg.mul_(alpha).add_(1-alpha, grad)    # 更新 梯度衰减平均
    # 先归一化，然后计算 RMS[g]
    avg = square_avg.addcmul_(-1, grad_avg, grad_avg).sqrt_().add_(group['eps'])
else:
    # 直接计算 RMS[g]
    avg = square_avg.sqrt_().add_(group['eps'])

if group['momentum'] > 0:       # 使用动量
    buf = state['momentum_buffer']  # 获取动量缓存
    buf.mul_(group'momentum').addcdiv_(grad, avg)   # 更新 velocity，与 (6) 式一致
    p.data.add_(-group['lr'], buf)
else:
    p.data.addcdiv_(-group['lr'], grad, avg)
```
上面代码中，如果不使用 `centered` 和 `momentum`，那么代码逻辑与 (4) 式完全一致，所以我们只看 `centered` 和 `momentum` 是如何进行的。
### centered
对 梯度 $g$ 归一化，然后计算 平方 的衰减均值，如下
$$E\{[g-E(g)]^2\}=E(g^2)-[E(g)]^2$$
其中 $E(\cdot)$ 计算衰减均值。于是，
$$RMS[g]=\sqrt{E\{[g-E(g)]^2\}}+\epsilon=\sqrt{E(g^2)-[E(g)]^2}+\epsilon$$

### momentum
我们回顾一下普通的 SGD 参数更新方式：
$$\theta_{t+1}=\theta_t - \eta \cdot \nabla f(\theta_t)$$
然后带有 momentum 的 SGD 参数更新方式：
$$v_{t+1}=\mu \cdot v_t + \nabla f(\theta_t)
\\\\\theta_{t+1}=\theta_t - \eta \cdot v_{t+1}$$

根据 (4) 式，现在已知 RMSprop 的参数更新方式为，
$$\theta_{t+1}=\theta_t -\frac \eta {\text{RMS}[g]_t}\cdot g_t$$
类比 SGD，可知带有 momentum 的 RMSprop 参数更新方式为，
$$v_{t+1}=\mu \cdot v_t + \frac {g_t} {\text{RMS}[g]_t} \qquad(6)
\\\\ \theta_{t+1}=\theta_t - \eta \cdot v_{t+1}$$

# 4. Rprop
## 4.1 原理
Rprop 表示 resilient propagation。

在 SGD 中，参数更新方向为负梯度方向，更新步长为梯度乘以一个系数（学习率），但是让更新步长直接与梯度成正比不一定是好选择，例如（来源 [2]）
![]()<center>图 1. 三个函数在相同的地方有最小值，但是 `f'(x)` 不同</center>
上图中，三个函数的最小值均在相同地方，所以各自更新步长可以差不多，但是如果使用 学习率乘以梯度 作为步长，显然三者的更新步长将会相差几个数量级，更糟的是，可能还会出现 梯度消失 和 梯度爆炸。

Rprop 仅利用梯度的（负）方向，参数更新如下，
$$\theta_{t+1} = \theta_t + \Delta \theta_t=\theta_t - \Delta_t \cdot \text{sign}[\nabla f(\theta_t)] \qquad(7)$$
其中 $\Delta_t$ 表示时间步 `t` 处的更新步长，并且不同参数的更新步长也不同，例如第 `i` 个参数在时间步 `t` 的更新步长为 $\Delta_{t,i}$。

在每个时间步，计算各参数的梯度以及更新步长。根据当前时间步的梯度与上一时间步的梯度的符号是否一致，来调整更新步长，思路如下：
- 如果符号一致，那么应该增大更新步长，以更快的到达最小值处
- 如果符号相反，这表示刚好跨过最小值处，那么应该减小更新步长，以避免再次跨过最小值处
  
更新步长调整方案如下，
$$\Delta_t=\begin{cases}\min(\Delta_{t-1} \cdot \eta^+, \ \Delta_{max}) & \nabla f(\theta_t) \cdot \nabla f(\theta_{t-1}) > 0 \\\\ \max(\Delta_{t-1} \cdot \eta^-, \Delta_{min}) & \nabla f(\theta_t) \cdot \nabla f(\theta_{t-1}) < 0 \\\\ \Delta_{t-1} & \text{otherwise} \end{cases}$$

其中 $\eta^+ > 1 > \eta^->0$，$\eta^+, \ \eta^-$ 分别用于增大步长和减小步长，并使用 $\Delta_{min}, \ \Delta_{max}$ 来限制步长范围。通常，$\Delta_{min}$ 过小 或者 $\Delta_{max}$ 过大 都不是问题，因为实际的更新步长可以快速调整到合适值。$\alpha$ 通常取 `1.2`，$\beta$ 取 `0.5`。$\Delta_0$ 为初始更新步长，作为超参数，事先给定，在 PyTorch 实现中为 `0.01`。

在论文 [3] 中，作者具体讨论了四种参数更新方式，`Rprop+`，`Rprop-`，`iRprop+`，`iRprop-`，上述的参数更新方式对应 `Rprop-`，其余三种方法可阅读 [3]，这里不再一一具体介绍。[3] 的实验结果表明，`iRprop-` 的更新方式综合最优，PyTorch 的实现正是采用了 `iRprop-`。

# 参考
[1] Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. John Duchi.

[2] [RProp](https://florian.github.io/rprop/) 

[3] Improving the Rprop Learning Algorithm. Christian Igel.