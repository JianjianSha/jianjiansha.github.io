---
title: Denoising Diffusion Implicit Models
date: 2022-07-21 09:26:16
tags: diffusion model
mathjax: true
summary: DDIM-非马尔可夫转移过程
---

论文：[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

扩散模型 DDPM 可以生成高质量的图像，但是缺点是需要的迭代次数非常多，生成图像的时间较长。作者提出 DDIM 模型，提高扩散模型的生成效率。

# 1. 背景

简单回顾一下 DDPM 。

前向过程是一个马尔科夫链，给定样本 $x_0$ 的分布 $q(x_0)$，那么加噪后的隐变量联合分布为

$$q(x_{1:T}|x_0)=\prod_{t=1}^T q(x_t|x_{t-1})$$

其中转移过程为

$$q(x_t|x_{t-1})=\mathcal N(\sqrt {1-\beta_t}x_{t-1}, \beta_t I)$$

令 $\alpha_t=1-\beta_t$，$\overline \alpha_t = \prod_{s=1}^t \alpha_s$，经过推导得

$$q(x_t|x_0)=\int q(x_{1:t}|x_0) dx_{1:(t-1)}=\mathcal N(\sqrt {\overline \alpha_t} x_0, (1-\overline \alpha_t)I)$$

显然 $\overline \alpha_t$ 是递减的，所以当 $T$ 足够大，$q(x_T|x_0) \rightarrow \mathcal N(0, I)$，这表明加噪最后可以得到近似各向同性的高斯噪声。

DDPM 则是从一个标准高斯分布 $x_T$ 出发，逐步 denoising，最后得到 $x_0$ 的分布，

$$p_{\theta}(x_0)=\int p_{\theta}(x_{0:T}) dx_{1:T}$$

其中

$$p_{\theta}(x_{0:T})=p_{\theta}(x_T) \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)$$

优化目标函数为

$$\max_{\theta} \mathbb E_{q(x_0)}[\log p_{\theta}(x_0)] \le \max_{\theta} \mathbb E_{q(x_{0:T})} [\log p_{\theta}(x_{0:T})-\log q(x_{1:T}|x_0)]$$

<details>
<summary>推导上述不等式的过程</summary>

为了简洁，省略 $p$ 的下标 $\theta$，

$$\begin{aligned} \mathbb E_{q(x_0)}[\log p(x_0)] &= \mathbb E_{q(x_{0:T})}[\log p(x_0)]
\\\\ &= \mathbb E_{q(x_{0:T})}[\log p(x_0) + \log p(x_{1:T}|x_0) - \log p(x_{1:T}|x_0)]
\\\\ &= \mathbb E_{q(x_{0:T})}[\log p(x_{0:T}) - \log q(x_{1:T}|x_0) + \log q(x_{1:T}|x_0) - \log p(x_{1:T}|x_0)]
\\\\ &= \mathbb E_{q(x_{0:T})}[\log p(x_{0:T}) - \log q(x_{1:T}|x_0)] + D_{KL}(q(x_{1:T}|x_0) || p(x_{1:T}|x_0))
\\\\ & \le \mathbb E_{q(x_{0:T})}[\log p(x_{0:T}) - \log q(x_{1:T}|x_0)]
\end{aligned}$$

最后一步推导是因为 KL 散度 $D_{kl}(P||Q) \ge 0$，且 $P=Q$ 时等号成立。

</details>

根据推导（推导过程见 [DDPM](2022/06/27/diffusion_model/ddpm) 一文），上式等价于对下式求最小，

$$L_{\gamma}(\epsilon_{\theta})=\sum_{t=1}^T \gamma_t \mathbb E_{x_0 \sim q(x_0), \epsilon_t \sim \mathcal N(0,I)} [\|\epsilon_{\theta}^{(t)}(x_t) -\epsilon_t\|_2^2]$$

其中 $\gamma_t$ 是乘积因子，DDPM 中有 $\gamma_t=1$。

# 2. 非马尔可夫过程的变分推断

## 2.1 非马尔可夫前向过程

考虑如下形式的隐变量联合分布

$$q_{\sigma}(x_{1:T}|x_0)=q_{\sigma}(x_T|x_0) \prod_{t=2}^T q_{\sigma}(x_{t-1}|x_t, x_0)$$

上式使用 $q_ {\sigma} (x_ {t-1}|x_t, x_0)$ 代替马尔可夫过程 $q_(x_t|x_{t-1})$，故上式不是一个马尔可夫过程。上式中令 $q_{\sigma}(x_T|x_0)=\mathcal N(\sqrt {\overline \alpha_T} x_0, (1-\overline \alpha_T) I)$ ，这样做使得最终变量的分布与 DDPM 中相同。

对 $t>1$ 有

$$q_{\sigma}(x_{t-1}|x_t,x_0)=\mathcal N \left (\sqrt {\overline \alpha_{t-1}} x_0 + \sqrt {1-\overline \alpha_{t-1} -\sigma_t^2} \cdot \frac {x_t-\sqrt {\overline \alpha_t} x_0}{\sqrt {1-\overline \alpha_t}} , \sigma_t^2 I\right) \tag{1}$$

$\sigma_t$ 用于控制前向过程中的随机性强度，当 $\sigma_t \rightarrow 0$ 时，(1) 式高斯分布在期望处变得很陡，变成一个脉冲，那么只要观察到 $x_0, \ x_t$，那么 $x_{t-1}$ 就被固定（在期望处）。$\sigma_t$ 值的选择在下文算法一栏中有介绍。

选择 (1) 式是为了确保满足 $q_{\sigma}(x_t|x_0)=\mathcal N(\sqrt {\overline \alpha_t} x_0, (1-\overline \alpha_t)I), \ t=1,\ldots, T$，

证明：

根据归纳法证明。假设有

$$q_{\sigma}(x_t|x_0)=\mathcal N(\sqrt {\overline \alpha_t}x_0, (1-\overline \alpha_t)I) \tag{2}$$

初始条件为 $t=T$ 时，上式成立，那么如果能证明如下 (3) 式成立，那么结论得证。

$$q_{\sigma}(x_{t-1}|x_0)=\mathcal N(\sqrt {\overline \alpha_{t-1}}x_0, (1-\overline \alpha_{t-1})I) \tag{3}$$


首先，计算边缘分布

$$q_{\sigma}(x_{t-1}|x_0)=\int_{x_t} q_{\sigma}(x_t|x_0) q_{\sigma}(x_{t-1}|x_t, x_0) dx_t$$

积分项中条件分布为 (1) 式，

以及我们的假设条件

$$q_{\sigma}(x_ {t-1}|x_0)=\mathcal N(\sqrt {\overline \alpha_t}x_0, (1-\overline \alpha_t)I)$$

将上式和 (1) 式代入边缘分布，可计算出结果。

对于两个随机向量 $\mathbf x_1, \mathbf x_2$，记联合概率密度分布的参数如下，

$$\mu = (\mu_1, \mu_2)$$

$$\Sigma=\begin{bmatrix} \sigma_1^2 & \rho \sigma_1 \sigma_2 \\\\ \rho \sigma_1 \sigma_2 & \sigma_2^2 \end{bmatrix}$$

那么条件分布为（这部分知识可参考相关教材），

$$p(x_1|x_2) = \mathcal N(\mu_1 + \rho \sigma_1 / \sigma_2 (x_2 - \mu_2), (1 - \rho^2) \sigma_1 ^2)$$

$$p(x_2|x_1) = \mathcal N(\mu_2 + \rho \sigma_2 / \sigma_1(x_1 - \mu_1), (1 - \rho^2) \sigma_2^2)$$

根据条件分布公式，令 $x_1, \ x_2$ 分别表示 $x_{t-1}, \ x_t$，可得

$$\mathbb E[x_{t-1}|x_0]=\mathbb E[x_{t-1}|x_t,x_0]\vert _ {x_1=\sqrt{\overline \alpha _ t}x_0} =\sqrt {\overline \alpha_{t-1}} x_0 + \sqrt {1-\overline \alpha_{t-1} -\sigma_t^2} \cdot \frac {\sqrt {\overline \alpha_t}x_0-\sqrt {\overline \alpha_t} x_0}{\sqrt {1-\overline \alpha_t}}=\sqrt {\overline \alpha_{t-1}} x_0$$


比较条件分布的期望表达式以及 (1) 式的期望表达式，可知 

$$\rho^2 \mathbb V_{t-1}/\mathbb V_t = (1-\overline \alpha_{t-1} -\sigma_t^2)/(1-\overline \alpha_t)$$

代入 $\mathbb V_t = (1-\overline \alpha_t)$ 到上式中，化简后可得

$$\mathbb V_{t-1}= \sigma_t^ 2/(1-\rho^2)= \sigma_t^2 / (1-\frac {1-\overline \alpha_{t-1} -\sigma_t^2}{\mathbb V_{t-1}})$$

解上式得，

$$\text{Cov} = \mathbb V_{t-1}I=(1-\overline \alpha_{t-1})I$$

故 (3) 式得证，于是满足 (1) 式的分布可以确保（3）式成立。证毕。

结论：**DDIM与DDPM 在每个 time step 上得到的加噪数据的分布其实是相同的，只是每个 time step 转移的过程不同，DDPM 是马尔可夫过程，DDIM 是非马尔可夫过程。**


根据贝叶斯定理，可以得到前向过程为

$$q_{\sigma}(x_t|x_{t-1}, x_0)= \frac {q_{\sigma}(x_{t-1}|x_t,x_0) q_{\sigma}(x_t|x_0)}{q_{\sigma}(x_{t-1}|x_0)}$$


于是前向过程不再是马尔科夫过程，因为 $\mathbf x_t$ 同时依赖于 $x_{t-1}, x_0$，如图 1，

![](/images/diffusion_model/ddim_1.png)
<center>图 1. 左：DDPM；右：非马尔可夫过程</center>

## 2.2 生成过程

定义一个可训练的生成过程 $p_{\theta}(x_{0:T})$，其中 $p_{\theta}^{(t)}(x_{t-1}|x_t)$ 利用 $q_{\sigma}(x_{t-1}|x_t, x_0)$。

训练阶段，给定一个样本 $x_0 \sim q(x_0)$，以及一个随机变量 $\epsilon_t \sim \mathcal N(0,I)$，根据 (2) 式可以得到

$$x_t=\sqrt {\overline \alpha_t} x_0 +\sqrt {1-\overline \alpha_t} \epsilon_t$$

那么模型输出 $\epsilon_{\theta}^{(t)}(x_t)$ 用于预测 $\epsilon_t$，损失函数为 $L_{\gamma}(\epsilon_{\theta})$ 。

生成过程（反向过程）中，给定一个带噪观测数据 $x_t$，根据 (2) 式进行变换得到 $x_0$ 的预测，为

$$f_{\theta}^{(t)}(x_t) = (x_t - \sqrt {1-\overline \alpha_t} \cdot \epsilon_{\theta}^{(t)}(x_t))/\sqrt {\overline \alpha_t} \tag{4}$$

得到 $x_0$ 的预测后，根据 (1) 式，可以得到 $x_{t-1}$，需要注意 $t=1$ 是特殊情况不使用 (1) 式，反向过程总结如下，

$$p_{\theta}^{(t)}(x_{t-1}|x_t) = \begin{cases} \mathcal N(f_{\theta}^{(1)}(x_1), \sigma_1^2 I) & t=1 \\\\ q_{\sigma}(x_{t-1}|x_t, f_{\theta}^{(t)}(x_t)) & \text{o.w.}\end{cases} \tag{5}$$

上式第二种 case 就是 (1) 式，只是其中可观测数据 $x_0$ 使用 $f_{\theta}^{(t)}(x_t)$ 代替。
上式第一种 case 就是以 $f_{\theta}^{(1)}(x_1)$ 为中心的高斯分布，毕竟 $f_{\theta}^{(1)}(x_1)$ 就是基于 $x_1$ 对 $x_0$ 的预测，方差 $\sigma_1^2I$ 与 (1) 式保持一致，如果仍用第二种形式，那就变成了 $q_{\sigma} (x _ 0|x _1, f _ {\theta}^{(1)}(x_1))$，这个表达式本身是基于 $x_1$ 预测 $x_0$ 的分布，而其用到的条件变量 $f _ {\theta}^{(1)}(x _1)$ 也是基于 $x_1$ 预测 $x_0$ 的值，引起冲突。

第 `1` 节内容也提到过，优化目标函数为最小化下式：

$$J_{\sigma}(\epsilon_{\theta})=\mathbb E_{q_{\sigma}(x_{0:T})}[\log q_{\sigma}(x_{1:T}|x_0) - \log p_{\theta}(x_{0:T})]
\\\\ =\mathbb E_{q_{\sigma}(x_{0:T})}[\log q_{\sigma}(x_T|x_0)+\sum_{t=2}^T \log q_{\sigma}(x_{t-1}|x_t,x_0)-\sum_{t=1}^T \log p_{\theta}^{(t)}(x_{t-1}|x_t)-\log p_{\theta}(x_T)]$$

这与 DDPM 中相同，因为目标函数均为 $\mathbb E_{q(x_0)} [-\log p_{\theta}(x_0)]$，DDIM 与 DDPM 的区别在于状态转移过程不同（是否是马尔可夫过程），而推导 $\mathbb E_{q(x_0)} [-\log p_{\theta}(x_0)]$ 过程中不涉及到具体的状态转移过程。

仿照 DDPM，上式目标函数也可以分为 $L_T$、$L_{1:T-1}$ 和 $L_0$，其中 $L_T$ 包含了 $q_{\sigma}(x_T|x_0)$ 和 $p(x_T)=\mathcal N(x_T;\mathbf 0, I)$，这两个分布均与模型参数 $\theta$ 无关，可看作常量。

### 2.2.1 $L_{1:T-1}$

$L_{t-1}$ 的表达式为

$$L_{t-1}=\mathbb E_q \left[-\log \frac {p_{\theta}(x_{t-1}|x _ t)}{q_{\sigma}(x _ {t-1}|x _ t,x _ 0)} \right]$$

再次强调这些表达式的形式均与 DDPM 中相同，区别是两者的 $q(x_{t-1}|x _ t, x _ 0)$ 的表达式不同。经过与 DDPM 中相同的推导，

$$L_{t-1} = \mathbb E_{q(x_0, x _ t)} \left [ \frac 1 {2 \mathbb V_t} ||\tilde \mu_t(x_t, x_0) - \mu_{\theta}(x_t, t)||^2 \right] + C$$

根据 (1) 式，可知

$$\tilde \mu_t(x_t, x_0) = \sqrt {\overline \alpha_{t-1}} x_0 + \sqrt {1-\overline \alpha_{t-1} -\sigma_t^2} \cdot \frac {x_t-\sqrt {\overline \alpha_t} x_0}{\sqrt {1-\overline \alpha_t}}$$

$\mu_{\theta}(x_t, t)$ 是反向过程 $p_{\theta}(x_{t-1}|x_t)$ 分布的期望，我们应该尽量让 $p_{\theta}(x_{t-1}|x_t)$ 逼近 $q_{\sigma}(x_{t-1}|x_t, x _ 0)$，这样才能使得 $L_{t-1}$ 尽可能小。再根据重参数技巧，

$$\mu_{\theta}(x_t, t) = \sqrt {\overline \alpha_{t-1}} x_0 + \sqrt {1-\overline \alpha_{t-1} -\sigma_t^2} \cdot \frac {x_t-\sqrt {\overline \alpha_t} f_{\theta}^{(t)}(x_t)}{\sqrt {1-\overline \alpha_t}}$$

于是 

$$\tilde \mu_t(x_t, x_0) - \mu_{\theta}(x_t, t) = \frac { \sqrt {1-\overline \alpha_{t-1} -\sigma_t^2}}{\sqrt {1-\overline \alpha_t}} \cdot \sqrt { \overline \alpha_t} \cdot ( f_{\theta}^{(t)}(x_t)-x_0)$$

根据 (4) 式，可知 $\sqrt {\overline \alpha_t} \cdot ( f_{\theta}^{(t)}(x_t)-x_0)= \sqrt {1-\overline \alpha _ t} (\epsilon_{\theta}^{(t)}- \epsilon_t)$，于是上式转化为，

$$\tilde \mu_t(x_t, x_0) - \mu_{\theta}(x_t, t) =\sqrt {1-\overline \alpha_{t-1} -\sigma_t^2}  (\epsilon_{\theta}^{(t)}- \epsilon_t)$$

故仍然可以等价于求下式最小值，

$$L_{simple}^{(t-1)}(\theta) = \gamma \mathbb E_{t,x_0,\epsilon, \sigma^2} \left[||\epsilon_{\theta}^{(t)}- \epsilon_t||^2 \right]$$

上式中，$\gamma$ 是权重因子。

从上式看出对于不同的 $\sigma$ 值，实际上都会训练出相同的模型，即，$\theta$ 参数相同，而得到的目标函数的差值是与 $\theta$ 无关的值，实际上对于某确定的 $\alpha$ 值，$J_{\sigma}$ 均等价于 $J_{\alpha}$ ，所谓等价是指模型参数 $\theta$ 保持不变 。

**定理 1** 对于任意 $\sigma > 0$，存在 $\alpha \in \mathbb R_{\ge 0}^T$ 以及 $C \in \mathbb R$，使得 $J_{\sigma} = J_{\alpha} + C$

### 2.2.2 模型训练

训练过程与 DDPM 的训练相同。

# 3. 通用生成过程的采样

## 3.1 DDIM

根据 (1) (4) (5) 式，可知从 $x_t$ 生成 $x_{t-1}$ 为

$$x_{t-1}=\sqrt {\overline \alpha_{t-1}} \left(\frac {x_t-\sqrt {1-\overline \alpha_t} \epsilon_{\theta}^{(t)}(x_t)}{\sqrt {\overline \alpha_t}}\right)+\sqrt {1-\overline \alpha_{t-1} - \sigma_t^2}\cdot \epsilon_{\theta}^{(t)}(x_t) + \sigma_t \epsilon_t' \tag{6}$$

其中 $\epsilon_t' \sim \mathcal N(0, 1)$ 是反向过程中的随机噪声。注意使用上标 `'` 标注用于区别前向过程中的噪声 $\epsilon_t$，$\epsilon_t$ 表示 $x_t=\sqrt {\overline \alpha_t} x_0 +\sqrt {1-\overline \alpha_t} \epsilon_t$，而 $\epsilon_{\theta}^{(t)}(x_t)$ 就是模拟逼近 $\epsilon_t$ 。

注意 $t=1$ 时应该使用 (5) 式的第一个 case，得到

$$x_0=\frac {x_1 - \sqrt {1-\overline \alpha_1} \cdot \epsilon_{\theta}^{(1)}(x_1)} {\sqrt {\overline \alpha_1}} + \sigma_1 \epsilon_1'$$

不同的 $\sigma$ 值会导致生成结果不同，当然模型输出 $\epsilon_{\theta}$ 是相同的，所以不需要重新训练模型。一种特殊情况是，当 

$$\sigma_t = \sqrt {\frac {1-\overline \alpha_{t-1}}{1-\overline \alpha_t}} \sqrt {1-\frac {\overline \alpha_t}{\overline \alpha_{t-1}}} \tag{7}$$

时，前向过程变成马尔可夫过程，生成过程则变成 DDPM。

证明如下：

根据贝叶斯定理，可以得到前向过程为

$$q_{\sigma}(x_t|x_{t-1}, x_0)= \frac {q_{\sigma}(x_{t-1}|x_t,x_0) q_{\sigma}(x_t|x_0)}{q_{\sigma}(x_{t-1}|x_0)} \tag{8}$$

将 (1) (2) (3) 式代入 (8) 式，

$$q_{\sigma}(x_t|x_{t-1}, x_0) \propto \exp \{-\frac 1 {2\sigma_t^2} \|x_{t-1} - (\sqrt {\overline \alpha_{t-1}} x_0 + \sqrt {1-\overline \alpha_{t-1} -\sigma_t^2} \cdot \frac {x_t-\sqrt {\overline \alpha_t} x_0}{\sqrt {1-\overline \alpha_t}})\|^2
\\\\ - \frac 1 {2(1-\overline \alpha_t)}\|x_t-\sqrt {\overline \alpha_t}x_0\|^2 + \frac 1 {2(1-\overline \alpha_{t-1})}\|x_{t-1} - \sqrt {\overline \alpha_{t-1}}x_0\|^2\} \tag{9}$$


根据第 `1` 节内容，有 $\alpha_t=1-\beta_t$，$\overline \alpha_t = \prod_{s=1}^t \alpha_s$，再根据 (7) 式有

$$\sigma_t^2 = \frac {1-\overline \alpha_{t-1}}{1-\overline \alpha_t}(1-\alpha_t)=\frac {1-\overline \alpha_{t-1}}{1-\overline \alpha_t}\beta_t$$

$$\sqrt {1-\overline \alpha_{t-1} -\sigma_t^2}/\sqrt {1-\overline \alpha_t}=\frac {1-\overline \alpha_{t-1}}{1-\overline \alpha_t} \sqrt {\alpha_t}$$

于是 

$$\sqrt {\overline \alpha_{t-1}} - \frac {\sqrt {1-\overline \alpha_{t-1} -\sigma_t^2}}{\sqrt {1-\overline \alpha_t}} \sqrt {\overline \alpha_t}=\frac {\sqrt {\overline \alpha_{t-1}}}{1-\overline \alpha_t} \beta_t$$

考虑 (9) 式中 $x_0^2$ 的因子，统一忽略常数因子 $1/2$ ，那么有

$$-\frac 1 {\sigma_t^2} \frac {\overline \alpha_{t-1}}{(1-\overline \alpha_t)^2} \beta_t^2-\frac {\overline \alpha_t}{1-\overline \alpha_t}+ \frac {\overline \alpha_{t-1}}{1-\overline \alpha_{t-1}}=0$$

即，没有 $x_0^2$ 项，同样的可知没有 $x_0$ 项，所以 $q_{\sigma}(x_t|x_{t-1}, x_0)$ 展开式中不含 $x_0$ 和 $x_0^2$，这表明此时 $x_t$ 仅依赖于 $x_{t-1}$ ，退化为 马尔可夫过程 / DDPM 。

## 3.2 加速生成过程

生成过程通常被认为是对反向过程的近似，故前向过程有 $T$ steps，那么生成过程也是 $T$ steps。

降噪目标函数 $L_{\mathbf 1}$ （即损失项权重因子 $\gamma= \mathbf 1$）在 $q_{\sigma}(x_t|x_0)$ 固定（固定为 (2) 式）的情况下不依赖于具体的前向过程，故考虑 steps 小于 $T$ 的前向过程。

加速情况下，推断过程（即前向过程）为

$$q_{\sigma,\tau}(x_{1:T}|x_0)=q_{\sigma,\tau}(x_{\tau_S}|x_0) \prod_{i=1}^S q_{\sigma,\tau}(x_{\tau_{i-1}}|x_{\tau_i}, x_0) \prod_{t \in \overline \tau} q_{\sigma,\tau}(x_t|x_0) \tag{10}$$

其中 $\tau$ 是长度为 $S$ 的 $[1,\ldots, T]$ 的递增子序列，且 $\tau_S=T$ 。$\overline \tau$ 是补集。

(10) 式的乘积项为

$$q_{\sigma,\tau}(x_t|x_0)=\mathcal N(\sqrt {\overline \alpha_t} x_0, (1-\overline \alpha_t) I), \quad \forall t \in \overline \tau \cup \{T\} \tag{11}$$

$$q_{\sigma,\tau}(x_{\tau_{i-1}}|x_{\tau_i},x_0)=\mathcal N\left(\sqrt {\overline \alpha_{\tau_{i-1}}} x_0 + \sqrt {1-\overline \alpha_{\tau_{i-1}} - \sigma_{\tau_i}^2} \frac {x_{\tau_i}-\sqrt {\overline \alpha_{\tau_i}}x_0}{\sqrt {1-\overline \alpha_{\tau_i}}} , \sigma_{\tau_i}^2I\right), \quad \forall i \in [S] \tag{12}$$

设计为 (12) 式，这样就保证了

$$q_{\sigma,\tau}(x_{\tau_i}|x_0) = \mathcal N(\sqrt {\overline \alpha_{\tau_i}}x_0, (1-\overline \alpha_{\tau_i})I), \quad \forall i \in [S] \tag{13}$$

(11) (13) 两式就保证了对所有的 $t \in [T]$，边缘分布均满足 (2) 式。

---
**算法 1：** 加速生成过程

输入：前向过程的 step 总数 $N$，$\beta_i, \ i=1,\ldots,N$，特别地令 $\beta_0=0$
    &emsp; 子序列长度 $S$

步骤：

根据规则（如 linear，等间隔）生成子序列 $\tau_1,\ldots, \tau_S$，确保 $\tau_S=N$

采样 $x_N \sim \mathcal N(\mathbf 0, I)$

**for** $t=S,\ldots, 1$

&emsp; $x_0'=(x_{\tau_t} - \sqrt {1-\overline \alpha_{\tau_t}} \cdot \epsilon_{\theta}^{(\tau_t)}(x_{\tau_t}))/\sqrt {\overline \alpha_{\tau_t}}$

&emsp; $\mu_{\theta}(\tau_{t-1})=\begin{cases}\frac {\sqrt {\overline \alpha_{\tau_{t-1}}}(1-\overline \alpha_{\tau_t}/\overline \alpha_{\tau_{t-1}})}{1-\overline \alpha_{\tau_t}} x_0'+\frac {\sqrt {\overline \alpha_{\tau_t}/ \overline \alpha_{\tau_{t-1}}}(1-\overline \alpha_{t-1})}{1-\overline \alpha_{\tau_t}}  x_{\tau_t} & DDPM \\\\ \sqrt {\overline \alpha_{\tau_{t-1}}} x_0' + \sqrt {1-\overline \alpha_{\tau_{t-1}} - \sigma_{\tau_t}^2} \frac {x_{\tau_t}-\sqrt {\overline \alpha_{\tau_t}}x_0'}{\sqrt {1-\overline \alpha_{\tau_t}}} & DDIM\end{cases}$

&emsp; **if** $t>1$

&emsp; &emsp; 采样 $z \sim \mathcal N(\mathbf 0, I)$

&emsp; &emsp; $\sigma_{\tau_t}=\begin{cases} \sqrt {1-\overline \alpha_{\tau_t} / \overline \alpha_{\tau_{t-1}}} & DDPM（具体看下方解释） \\\\ \eta\sqrt {\frac {1-\overline \alpha_{\tau_{t-1}}}{1-\overline \alpha_{\tau_t}}}\cdot \sqrt {1-\frac {\overline \alpha_{\tau_t}}{\overline \alpha_{\tau_{t-1}}}} & DDIM（具体看下方解释） \end{cases}$

&emsp; &emsp; $x_{\tau_{t-1}}=\mu_{\theta}(\tau_{t-1})+ \sigma_{\tau_t} z$

&emsp; **else**

&emsp; &emsp; $x_{\tau_{t-1}}=\mu_{\theta}(\tau_{t-1})$

---

<font color="red">注意：</font> 

1. 以上算法中，不能直接使用 $\alpha_{\tau_t}$ 和 $\beta_{\tau_t}$，而是要根据 $\alpha_t = \overline \alpha_t / \overline \alpha_{t-1}$
使用 $\overline \alpha_{\tau_t} / \overline \alpha_{\tau_{t-1}}$ 来代替 $\alpha_{\tau_t}$，同样地使用 $1-\overline \alpha_{\tau_t} / \overline \alpha_{\tau_{t-1}}$ 代替 $\beta_{\tau_t}$。需要搞清楚的是，通常

    $$\overline \alpha_{\tau_t} / \overline \alpha_{\tau_{t-1}} \ne \alpha_{\tau_t}=\overline \alpha_{\tau_t} / \overline \alpha_{\tau_t -1}$$

    这是因为 $\tau_{t-1} \ne \tau_t -1$ 。

2. DDPM 中 $\mu_{\theta}(\tau_{t-1})$ 其实就是 [DDPM](2022/06/27/diffusion_model/ddpm) 一文中的 (12) 式，但是要将 $\alpha_{\tau_t}$ 替换为 $\overline \alpha_{\tau_t} / \overline \alpha_{\tau_{t-1}}$，将 $\beta_{\tau_t}$ 替换为 $1-\overline \alpha_{\tau_t} / \overline \alpha_{\tau_{t-1}}$ 。

2. $\sigma_{\tau_t}(\eta)=\eta\sqrt {\frac {1-\overline \alpha_{\tau_{t-1}}}{1-\overline \alpha_{\tau_t}}}\cdot \sqrt {1-\frac {\overline \alpha_{\tau_t}}{\overline \alpha_{\tau_{t-1}}}}$

    当 $\eta=1$ 时，DDIM 转变为 DDPM，当 $\eta=0$ 时转变为  deterministic DDIM。作者在 cifar10 数据集上还使用了 $\hat \sigma_{\tau_t}=\sqrt {1-\frac {\overline \alpha_{\tau_t}}{\overline \alpha_{\tau_{t-1}}}} > \sigma_{\tau_t}(\eta=1)$

3. 上面算法中，DDPM 的反向过程中的噪声方差 $\sigma_{\tau_t}^2=1-\overline \alpha_{\tau_t} / \overline \alpha_{\tau_{t-1}}$，对应于 $\sigma_t^2=\beta_t$，实际上 DDPM 还有另一个方差选择 $\sigma_t^2=\tilde \beta_t$，对应于反向过程中噪声方差的上下限，如果选择下限，那么 $\sigma_{\tau_t}=\sqrt {\frac {1-\overline \alpha_{\tau_{t-1}}}{1-\overline \alpha_{\tau_t}}}\cdot \sqrt {1-\frac {\overline \alpha_{\tau_t}}{\overline \alpha_{\tau_{t-1}}}}$

生成过程定义为

$$p_{\theta}(x_{0:T})=\underbrace {p_ {\theta}(x_T) \prod_{i=1}^S p_{\theta}^{(\tau_i)}(x_{\tau_{i-1}}|x_{\tau_i})}_ {\text{use to produce samples}} \times \prod_{t \in \overline \tau} p_ {\theta}^{(t)}(x_0|x_t) \tag{14}$$

其中

$$p_{\theta}^{(\tau_i)}(x_{\tau_{i-1}}|x_{\tau_i})=q_{\sigma,\tau}(x_{\tau_{i-1}}|x_{\tau_i}, f_{\theta}^{(\tau_i)}(x_{\tau_i})) \quad  i \in [S], i > 1$$

$$p_{\theta}^{(t)}(x_0|x_t)=\mathcal N(f_{\theta}^{(t)}(x_t), \sigma_t^2I) \quad \text{o.w.}$$

训练时还跟之前一样训练，生成样本时，则使用 $\text{reversed}(\tau)$ 生成，如图 2

![](/images/diffusion_model/ddim_2.png)
<center>图 2. 加速生成样本的示意图</center>

# 4. 实验

每个数据集均使用相同的训练模型，$T=1000$，训练目标为 $L_{\mathbf 1}$。生成样本使用子序列 $\tau$，方差 $\sigma$ 从随机 DDPM 和确定 $DDIM$ 中插值获得。$\sigma$ 形式如下

$$\sigma_{\tau_i}(\eta)=\eta \sqrt {(1-\overline \alpha_{\tau_{i-1}})/(1-\overline \alpha_{\tau_i})} \sqrt {1-\overline \alpha_{\tau_i}/\overline \alpha_{\tau_{i-1}}} \tag{15}$$

其中 $\eta \in \mathbb R_{\ge 0}$ 用于控制。当 $\eta=1$ 时，如 (7) 式所示，模型退化为 DDPM，如果 $\eta=0$，那么模型为确定 DDIM（随机性消失）。

作者也考虑了 $\sigma > \sigma(\eta=1)$ 情况，方差取值为 

$$\hat \sigma_{\tau_i} = \sqrt {1-\overline \alpha_{\tau_i}/\overline \alpha_{\tau_{i-1}}} \tag{16}$$

由于 $1-\overline \alpha_{\tau_{i-1}} < 1- \overline \alpha_{\tau_i}$ ，故 $\hat \sigma_{\tau_i} > \sigma_{\tau_i}(\eta=1)$ 。


# 5. ODE

根据 (6) 式，当噪声标准差 $\sigma_t=0$ 时，可以改写为

$$\frac {x_{t-1}}{\sqrt {\overline \alpha_{t-1}}}=\frac {x_t} {\sqrt {\overline \alpha_t}}+ \left(\sqrt {\frac {1-\overline \alpha_{t-1}}{\overline \alpha_{t-1}}}- \sqrt {\frac {1-\overline \alpha_t}{\overline \alpha_t}}\right) \epsilon_{\theta}(x_t,t) \tag{17}$$

当 $T$ 足够大，将 (17) 式看作某个常微分方程的欧拉积分形式，令 $\overline x = x/\sqrt {\overline \alpha}$，且令 

$$\sigma(t)=\sqrt {\frac {1-\overline \alpha_t}{\overline \alpha_t}} \tag{18}$$

注意，这里的 $\sigma(t)$ 与前面 (1) 式中标准差 $\sigma_t$ 是两个不同的概念，本节 $\sigma(t)$ 仅仅是一个符号，由 (18) 式定义。

于是 (17) 式变为

$$\overline x(t-\Delta t)=\overline x(t) + (\sigma(t-\Delta t)-\sigma(t)) \cdot \epsilon_{\theta}\left(\frac {\overline x(t)}{\sqrt {\sigma^2(t)+1}}, t\right)$$

令 $\Delta t \rightarrow 0$ ，于是上式变为

$$\frac {d\overline x(t)}{dt}=\frac {d \sigma(t)}{dt} \cdot \epsilon_{\theta} \left(\frac {\overline x(t)}{\sqrt {\sigma^2(t)+1}}, t \right) \tag{19}$$

给定 $\beta_t, t=1,\ldots, T$，就可以得到 $\{\alpha_t\}_{t=1}^T$ 和 $\{\overline \alpha_t\}_{t=1}^T$ 。

例如 $T=1000$，$\beta_1=10^{-4}$ 线性增加到 $\beta_T=0.02$ 为例，那么 

$$\beta_t=\frac {t-1} {T-1} (\beta_T-\beta_1) + \beta_1 \approx \frac {t\beta_T}{T}\tag{20}$$

上式对 $t \in [0, T]$ 范围内的所有实数均有效。根据已知条件 $x(T) \sim \mathcal N(\mathbf 0,I)$ 以及根据 (20) 式可以计算出 $\overline \alpha(t), \ \sigma(t)$ 的值， 计算 $\overline x(0)$，从而得到 $x(0)$ 。解 (19) 式就是解常微分方程。

计算 $\overline \alpha(t)$ 过程如下：

$$\log \overline \alpha(t)=\sum_{i=1}^t \log \alpha_i=\sum_{i=1}^t \log (1-\beta_i)\approx \sum_{i=1}^t \log (1-\frac {0.02i}T) 
\\ \approx -\sum_{i=1}^t \frac {0.02i}T=-\frac {0.01t(t+1)}T \approx - \frac {10t ^ 2}{T^ 2}$$

于是

$$\overline \alpha(t)=e^ {-10t ^ 2/T^2}$$

再根据 (18) 式得到

$$\sigma(t) = e ^ {5 t^2 / T ^ 2} \sqrt {1-e^{-10t ^ 2/T ^ 2}}$$


