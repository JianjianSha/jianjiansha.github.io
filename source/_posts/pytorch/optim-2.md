---
title: PyTorch.optim
p: pytorch/optim-2
date: 2020-01-08 18:19:54
tags: 
    - PyTorch
    - DL
mathjax: true
---

# 1. Adam
Adam 表示 Adaptive Moment Estimation。
<!-- more -->
## 1.1 原理
梯度和梯度平方的衰减如下，
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
\\\\ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad(1)$$
其中 $\beta_1 < 1, \ \beta_2 < 1$，$m_t$ 和 $v_t$ 分别是梯度 $g$ 的一阶矩和二阶矩的样本估计（$g$ 看作随机变量）。由于 $m$ 和 $v$ 均初始化为 0，即 $m_0=0, \ v_0 = 0$，所以这两个样本估计均是有偏估计，且偏向 0，尤其在刚开始的时间步（t 较小）和衰减率较小时（$1-\beta$ 较小，$\beta$ 接近 1）。

令 $E(g)=\mu$，$g_1, g_2, ...$ 来自于 $g$ 且独立同分布，那么
$$E(m_t)=E\left(\sum_{\tau=1}^t \beta_1^{t-\tau} (1-\beta_1) g_{\tau}\right)=(1-\beta_1)\sum_{\tau=1}^t \beta_1^{t-\tau}E(g_{\tau})=\mu (1-\beta_1)\sum_{\tau=1}^t \beta_1^{t-\tau}=\mu(1-\beta_1^t)$$ 
可见，当 t 较小且 $\beta_1 \rightarrow 1$，$E(m_t) \rightarrow 0$

为了抵消这些偏向，取以下计算进行校正，
$$\hat m_t=\frac {m_t} {1-\beta_1^t}
\\\\ \hat v_t = \frac {v_t} {1-\beta_2^t}$$

其中 上标 `t` 表示指数，即 `t` 个 $\beta$ 相乘。 通过上面的分析，可知，除以 $1-\beta^t$ 后，$E(\hat m_t)=\mu$，为无偏估计。

然后类似 Adadelta 和 RMSprop 中那样，更新公式为，
$$\theta_{t+1}=\theta_t - \frac {\eta} {\sqrt{\hat v_t}+\epsilon} \hat m_t \qquad(2)$$
其中 $\eta$ 为初始学习率，是一个初始时给定的超参数。

## 1.2 AMSGrad 变体
修改 $v$ 的计算式如下，
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\\\\ v_{t,m} = \max\left(v_{t-1,m}, \ \beta_2 v_{t-1} + (1-\beta_2) g_t^2\right)$$
其中 $v_{0,m}=0$。

然后 $v$ 的无偏估计改为，
$$\hat v_{t,m}=\frac {v_{t,m}} {1-\beta_2^t}$$
参数更新公式调整为，
$$\theta_{t+1}=\theta_t - \frac {\eta} {\sqrt{\hat v_{t,m}}+\epsilon} \hat m_t \qquad(3)$$
其中 $\hat m_t$ 的计算部分与前面的保持一致。

AMSGrad 比 Adam 降低了学习率。

# 2. Adamax
在 Adam 的基础上将 (1) 式泛化，不局限于 $l_2$ 范数，如下
$$v_t = \beta_2^p v_{t-1} + (1-\beta_2^p)|g_t|^p$$
其中注意 $p$ 为指数。

将上式中的 $v_{t-1}$ 展开，
$$\begin{aligned} v_t  &= (1-\beta_2^p)|g_t|^p + \beta_2^p[(1-\beta_2^p)|g_{t-1}|^p+\beta_2^p v_{t-2}]
\\\\ & = (1-\beta_2^p)\sum_{i=1}^t \beta_2^{p(t-i)} |g_i|^p
\end{aligned}$$

令 $p \rightarrow \infin$，并定义 $u_t = \lim_{p \rightarrow \infin}(v_t)^{1/p}$，结合上式有，
$$\begin{aligned} u_t  = \lim_{p \rightarrow \infin}(v_t)^{1/p} &= \lim_{p \rightarrow \infin}\left((1-\beta_2^p)\sum_{i=1}^t \beta_2^{p(t-i)} |g_i|^p\right)^{1/p}
\\\\ &= \lim_{p \rightarrow \infin} (1-\beta_2^p)^{1/p} \left(\sum_{i=1}^t \beta_2^{p(t-i)} |g_i|^p\right)^{1/p}
\\\\ &= \lim_{p \rightarrow \infin} \left(\sum_{i=1}^t \beta_2^{p(t-i)} |g_i|^p\right)^{1/p}
\\\\ &=\max (\beta_2^{t-1}|g_1|,\beta_2^{t-2}|g_2|,...,\beta_2^{0}|g_t|)\end{aligned}$$

于是可得以下迭代公式，
$$u_t = \max(\beta_2 u_{t-1}, \ |g_t|)$$
其中初始值 $u_0=0$。

用 $u_t$ 替换 Adam 中的 $\sqrt{\hat v_t}+\epsilon$，于是 更新公式为，
$$\theta_{t+1} = \theta_t - \frac \eta {u_t} \hat m_t \qquad(4)$$
其中 $\hat m_t$ 的计算方式与 Adam 中一致。

# 3. AdamW
Adam 中，梯度中事先包含了正则惩罚项，即
$$g := g+\lambda \theta$$
然后再计算梯度的一阶矩和二阶矩的无偏估计。现在考虑将权重衰减项从梯度 $g$ 中解耦出来，直接附加到参数衰减 $\theta$ 上，调整 (2) 式得到 AdamW 的参数更新公式，
$$\theta_{t+1}=\theta_t - \lambda \eta \theta_t - \frac {\eta} {\sqrt{\hat v_t}+\epsilon} \hat m_t$$

# 4. Nadam
回顾一下 momentum 版本的 SGD 更新方式，
$$v_{t+1} = \gamma v_t + \eta g_t
\\\\ \theta_{t+1}=\theta_t - v_{t+1}$$
然后 NAG 的更新方式，先从当前参数处更新 momentum 的量到达一个新的位置 （(5) 式），然后从新位置处进行梯度下降，作为本次更新后的参数（(6, 7) 式），数学描述如下，
$$y_t = \theta_t + \mu v_t  \qquad(5)
\\\\ g_t = \nabla f(y_t)    \qquad(6)
\\\\ \theta_{t+1}=y_t - \gamma g_t \qquad(7)$$

联合上面三式可知，
$$v_{t+1}=\theta_{t+1}-\theta_t=\mu v_t - \gamma g_t$$
初始时，$t=0, \ v_0=0 \Rightarrow y_0=\theta_0$。

根据 [PyTorch.optim.SGD](2020/01/02/pytorch/optim_SGD) 中的公式 (8)、(9)、(10)，易知 NAG 等价于以下更新过程，
$$\begin{cases}g_t = \nabla f(\theta_t)
\\\\ v_{t+1} = \gamma v_t + \eta g_t
\\\\ v_{t+1}' = \gamma v_{t+1} + \eta g_t
\\\\ \theta_{t+1} = \theta_t - v_{t+1}'\end{cases} \qquad(8)$$
可见，做了两次的 momentum 更新，相比普通的 momentum 的 SGD，增加了一次 look ahead 的 momentum。注意，$v_{t+1}'$ 与 $v_{t+2}$ 是不一样的。

接着再回顾 Adam 中的参数更新，根据 (2) 式，得
$$\theta_{t+1}=\theta_t - \frac {\eta} {\sqrt{\hat v_t}+\epsilon} \frac {m_t} {1-\beta_1^t}\qquad(9)$$
其中 $m_t$ 包含了一次 momentum 更新，
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
增加一次 momentum 更新，
$$m_t'=\beta_1 m_t + (1-\beta_1) g_t$$
代入 (9) 式，于是参数更新变为，
$$\begin{aligned}\theta_{t+1}&=\theta_t - \frac \eta {\sqrt {\hat v_t} + \epsilon}\frac {m_t'} {1-\beta_1^t}\\\\&=\theta_t - \frac \eta {\sqrt {\hat v_t} + \epsilon}\frac {\beta_1 m_t + (1-\beta_1) g_t} {1-\beta_1^t}
\\\\&=\theta_t - \frac \eta {\sqrt {\hat v_t} + \epsilon}\left(\beta_1 \hat m_t+\frac {1-\beta_1}{1-\beta_1^t} g_t \right)\end{aligned} \qquad(10)$$

(10) 式就是 Nadam 的参数更新公式。

也可以按如下过程理解，
$$\hat m_t = \frac {m_t} {1-\beta_1^t}=\frac {\beta_1 m_{t-1} + (1-\beta_1) g_t} {1-\beta_1^t}=\frac {\beta_1 \hat m_{t-1}(1-\beta_1^{t-1}) + (1-\beta_1) g_t} {1-\beta_1^t}=\beta_1 \hat m_{t-1}+\frac {1-\beta_1}{1-\beta_1^t} g_t$$
其中最后一步用了近似处理。事实上 (10) 式第一步中，将 $m_t$ 替换为 $m_t'$ 时，分母也应该替换为 $1-\beta_1^{t+1}$，因为 $m_t'$ 真正的无偏估计就应该要除以 $1-\beta_1^{t+1}$，但是我们都忽略这个微小的差别。

根据上式，可得，
$$\hat m_t'=\beta_1 \hat m_t + \frac {1-\beta_1}{1-\beta_1^t} g_t$$
代入 (2) 得 Nesterov momentum 加成的 Adam 变体的 更新公式，与 (10) 式相同。 