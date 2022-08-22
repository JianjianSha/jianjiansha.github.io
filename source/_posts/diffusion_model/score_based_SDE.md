---
title: Score-Based Generative Modeling Through SDE
date: 2022-07-26 14:13:15
tags: diffusion model
mathjax: true
---

论文：[Score-based generative modeling through stochastic differential equations](https://arxiv.org/abs/2011.13456)

代码：[yang-song/score-sde](https://github.com/yang-song/score_sde)


# 1. 简介

两类概率生成模型。

## 1.1 SMLD

score matching + Langevin dynamics （SMLD）

score matching，模型 $s_{\theta}(\mathbf x)$ 模拟 $\nabla_{\mathbf x} \log p_{data}(\mathbf x)$，由于 $p_{data}(\mathbf x)$ 往往不可知，故采用 denoising score matching 方法，即，对数据做一个非常小的噪声扰动，

$$p_{\sigma}(\tilde {\mathbf x}|\mathbf x)=\mathcal N(\tilde {\mathbf x};\mathbf x,\sigma^2I) \tag{1}$$

根据上下文，可以知道 $I$ 为单位矩阵，下同。可以得到边缘分布

$$p_{\sigma}(\tilde {\mathbf x})=\int p_{data}(\mathbf x) p_{\sigma}(\tilde {\mathbf x}|\mathbf x) d\mathbf x \tag{2}$$

考虑一系列的扰动噪声幅度 $\sigma_{min}=\sigma_1 < \ldots < \sigma_N = \sigma_{max}$ 。$\sigma_{min}$ 必须足够小使得 $p_{\sigma_{min}}(\mathbf x) \approx p_{data}(\mathbf x)$ 。根据 [NCSN](/2022/07/22/diffusion_model/NCSN) 分析，由于数据位于低维度 manifolds，且多 mode 之间是数据低密度区域，会引起很多问题，所以**先选择最大的噪声扰动，然后逐步降低噪声幅度**。

（SMLD 论文中使用的网络框架作者称为 NCSN。）

训练的目标函数为

$$\theta^{\star}=\arg \min_{\theta} \sum_{i=1}^N \sigma_i^2 \mathbb E_{p_{data}(\mathbf x)}\mathbb E_{p_{\sigma_i}(\tilde {\mathbf x}|\mathbf x)}[\|\mathbf s_{\theta} - \nabla_{\tilde {\mathbf x}} \log p_{\sigma_i}(\tilde {\mathbf x}|\mathbf x)\|_2^2] \tag{3}$$

给定样本数据 $\mathbf x$，条件概率密度 $p_{\sigma_i}(\tilde {\mathbf x}|\mathbf x)$ 是我们预先设计好的 (设计扰动噪声方差 $\sigma_i$ 即可)，故条件概率密度已知，以此来监督训练模型参数 $\theta$ 。

假设数据足够多，且模型空间组大大，那么最优参数模型 $\mathbf s_{\theta{\star}}(\mathbf x,\sigma)$ 几乎匹配所有的 $\nabla_{\mathbf x} \log p_{\sigma}(\mathbf x), \ i=1,\ldots, N$ 。

使用 Langevin dynamics 采样

$$\mathbf x_i^m = \mathbf x_i^{m-1}+ \epsilon_i \mathbf s_{\theta{\star}}(\mathbf x_i^{m-1}, \sigma_i) + \sqrt {2\epsilon_i} \mathbf z_i^m, \ m=1,\ldots, M \tag{4}$$

其中 $\epsilon_i$ 是 step size，$M$ 是 step number，$\mathbf z_i \sim \mathcal N(0,I)$。依据 (4) 式的采样过程为，重复 $i=N,N-1,\ldots, 1$，对每个 $i$ 值，设置 $\mathbf x_i^0=\mathbf x_{i+1}^M$，其中初始值 $\mathbf x_N^0 \sim \mathcal N(\mathbf x|\mathbf 0, \sigma_{max}^2 I)$。

## 1.2 DDPM

denoising diffusion probabilistic models，先前向过程对数据逐步加噪，然后反向过程中逐步降噪。即数据分布为 $\mathbf x_0 \sim p_{data}(\mathbf x)$，设置前向过程 

$$p(\mathbf x_i|\mathbf x_{i-1})=\mathcal N(\mathbf x_i; \sqrt {1-\beta_i} \mathbf x_{i-1},\beta_i I)\tag{5}$$

$\beta_i$ 是噪声方差，且 $0<\beta_1,\beta_2,\ldots, \beta_N < 1 $，由此可推导出

$$p_{\alpha_i}(\mathbf x_i|\mathbf x_0)=\mathcal N(\mathbf x_i; \sqrt {\alpha_i} \mathbf x_0, (1-\alpha_i)I) \tag{6}$$

其中 $\alpha_i = \prod_{j=1}^i (1-\beta_j)$，与原文中的 $\overline \alpha_i$ 相同，这里为了表达简单，省掉了顶部横线。

与 SMLD 相似，采用 denoising score matching，那么目标函数为

$$\theta^{\star}=\arg\min_{\theta} \sum_{i=1}^N (1-\alpha_i)\mathbb E_{p_{data}(\mathbf x)} \mathbb E_{p_{\alpha_i}(\tilde {\mathbf x}|\mathbf x)} [\|\mathbf s_{\theta}(\tilde {\mathbf x},i)-\nabla_{\tilde {\mathbf x}} \log p_{\alpha_i}(\tilde {\mathbf x}|\mathbf x)\|_2^2] \tag{7}$$

(7) 式中取 $\mathbf x=\mathbf x_0$。

根据 [NCSN](/2022/07/22/diffusion_model/NCSN) 分析，权重因子 $\lambda(\alpha_i) \propto (1-\alpha_i)$ 。

根据 [DDPM](/2022/06/27/diffusion_model/ddpm) 中 (12) 式，这里再写出来如下，

$$\tilde {\mu}_i(\mathbf x_i, \mathbf x_0)=\frac {\sqrt {\alpha_{i-1}}\beta_i}{1- \alpha_i}\mathbf x_0+\frac {\sqrt {\alpha_i}(1-\overline \alpha_{i-1})}{1- \alpha_i} \mathbf x_i \tag{8}$$

根据 (7) 式，训练完毕后应有

$$\mathbf s_{\theta}(\mathbf x_i,i)=\nabla_{\mathbf x} \log p_{\alpha_i}(\mathbf x|\mathbf x_0)|_{\mathbf x=\mathbf x_i}=\frac {\sqrt {\alpha_i} \mathbf x_0 -\mathbf x_i}{1-\alpha_i} \tag{9}$$

(9) 式代入 (8) 式得 

$$\mu_{\theta}(\mathbf x_i)=\frac 1 {\sqrt {1-\beta_i}} (\mathbf x_i+\beta_i \mathbf s_{\theta}(\mathbf x_i, i)) \tag{10}$$

于是反向过程的转换为

$$p_{\theta}(\mathbf x_{i-1}|\mathbf x_i)=\mathcal N(\mathbf x_{i-1}; \frac 1 {\sqrt {1-\beta_i}} (\mathbf x_i+\beta_i \mathbf s_{\theta}(\mathbf x_i, i)), \beta_iI) \tag{11}$$

其中方差直接使用 $\beta_i$ 。

解出 (7) 式得到最优解 $\mathbf s_{\theta^{\star}}(\mathbf x,i)$，然后生成过程为从 $\mathbf x_N \sim \mathcal N(\mathbf 0, I)$ 开始，依据 (11) 式进行采样，

$$\mathbf x_{i-1}=\frac 1 {\sqrt {1-\beta_i}} (\mathbf x_i+\beta_i \mathbf s_{\theta^{\star}}(\mathbf x_i, i)) +\sqrt {\beta_i}\mathbf z_i, \quad i=N,N-1,\ldots, 1 \tag{12}$$

# 2. SDE 生成模型

思想：将上述按步加噪泛化到无限步加噪。为此，根据随机微分方程建立数据转换过程。

## 2.1 SDE 扰动

构造一个扩散过程 $\{\mathbf x(t)\}_{t=0}^T$，时刻 $t$ 是一个连续型变量，范围是 $t\in [0,T]$，数据分布记为 $\mathbf x(0) \sim p_0$，扩散最后的分布为 $\mathbf x(T) \sim p_T$。扩散过程使用 $It\hat o$ SDE 刻画，

$$d\mathbf x = \mathbf f(\mathbf x, t) dt + g(t) d\mathbf w \tag{13}$$

其中 $d\mathbf w =W_{t+\Delta t}-W_t \sim \mathcal N(0, \Delta t)$ 是标准布朗运动。关于 SDE 更多的知识可参考 [这里](/2022/07/25/math/SDE) 。

记 $\mathbf x(t)$ 的分布为 $p_t(\mathbf x)$，从 $\mathbf x(s)$ 到 $\mathbf x(t)$ 的转移记为 $p_{st}(\mathbf x(t)|\mathbf x(s))$，其中 $0 \le s < t \le T$ 。

## 2.2 反转 SDE

根据 $\mathbf x(T) \sim p_T$ 以及反向过程，可以采样得到来自 $p_0$ 的样本。根据 Brian D O Anderson. Reverse-time diffusion equation models. 1982. 的论文可知，反向 SDE 为

$$d\mathbf x=[\mathbf f(\mathbf x,t)-g(t)^2 \nabla_{\mathbf x} \log p_t(\mathbf x)] dt + g(t) d\overline {\mathbf w} \tag{14}$$

## 2.3 SDE 的得分估计

为了计算 (14) 式，我们回顾一下，

目标函数为

$$\theta^{\star}=\arg \min_{\theta} \mathbb E_t \{\lambda(t) \mathbb E_{p_0} \mathbb E_{p_{0t}}[\|\mathbf s_{\theta}(\mathbf x(t), t)-\nabla_{\mathbf x(t)} \log p_{0t}(\mathbf x(t)|\mathbf x(0))\|_2^2]\}\tag{15}$$

这里，$\lambda: [0,T] \rightarrow \mathbb R_{>0}$ 是一个权重函数，$t$ 是连续型在 $[0,T]$ 上均匀采样。当数据量足够以及模型容量足够大，那么可以求得 (15) 式的最优解记为 $\mathbf s_{\theta^{\star}}(\mathbf x,t) \approx \nabla_{\mathbf x} \log p_t$，对几乎所有的 $\mathbf x$ 和 $t$ 均成立。


## 2.4 例子

前面分析的通过 (15) 式训练模型，但是其中 $p_{0t}$ 是什么形式，以及通过 (14) 式进行采样，(14) 式中的 $\mathbf f(\mathbf x,t)$ 和 $g(t)$ 又分别是什么。 这一节通过具体的例子进行讲解。

### 2.4.1 VE
variance exploding （SMLD）

马尔可夫链转移关系为

$$\mathbf x_i = \mathbf x_{i-1} + \sqrt {\sigma_i^2 - \sigma_{i-1}^2}\mathbf z_{i-1}, \quad i=1,\ldots,N \tag{16}$$

其中 $\mathbf z_{i-1} \sim \mathcal N(\mathbf 0, I)$，$\mathbf x_0 \sim p_{data}$。$\sigma_{min} = \sigma_1 < \sigma_2 < \cdots < \sigma_N=\sigma_{max}$ 。从 (16) 式可计算 $\mathbb E[\mathbf x_i]=\mathbf x_0$，$\mathbb V[\mathbf x_i] = \sigma_i I$ （将 $\mathbf x_0$ 看作常量），于是

$$\mathbf x_i |\mathbf x_0 \sim \mathcal N(\mathbf x_0, (\sigma_i^2-\sigma_0^2)I) \tag{17}$$

当 $N \rightarrow \infty$ ，马尔科夫链 $\{\mathbf x_i\}_{i=1}^N$ 变成连续型随机过程 $\{\mathbf x(t)\}_{t=0}^1$，将原来的 $i$ 变为 $t=i/N$ 即可，$\sigma_i$ 和 $\mathbf z_i$ 类似地变为 $\sigma(t)$ 和 $\mathbf z(t)$，那么 (16) 式为

$$\mathbf x(t+\Delta t)=\mathbf x(t) + \sqrt {\sigma^2(t+\Delta t)-\sigma^2(t)}\mathbf z(t) = \mathbf x(t) + \sqrt {\frac {\Delta[\sigma^2(t)]}{\Delta t}\Delta t} \cdot \mathbf z(t)$$

由于 $\mathbf z(t) \sim \mathcal N(\mathbf 0,I)$，那么 $\sqrt {\Delta t} \mathbf z(t) \sim \mathcal N(0, \Delta t\cdot I)$，即布朗运动的增量随机变量 $d\mathbf w$，结合上式可得

$$d\mathbf x = \sqrt {\frac {d[\sigma^2(t)]}{dt}}d\mathbf w \tag{18}$$

对比 (13) 式，可知

$$\mathbf f(\mathbf x,t)=\mathbf 0, \quad g(t)=\sqrt {\frac {d[\sigma^2(t)]}{dt}} \tag{19}$$

于是我们知道了 $p_{0i}(\mathbf x_i|\mathbf x_0)$ 为 (17) 式，$\mathbf f(\mathbf x, t)$ 和 $g(t)$ 为 (19) 式，可以据此来训练模型并采样。
### 2.4.2 VP

variance preserving（DDPM）

用在 DDPM 中，离散马尔科夫链为

$$\mathbf x_i =\sqrt {1-\beta_i} \mathbf x_{i-1} + \sqrt {\beta_i} \mathbf z_{i-1} , \quad i=1,\ldots,N \tag{20}$$

其中 $\mathbf z_{i-1} \sim \mathcal N(\mathbf 0, I)$ 。$0< \beta_i < 1, \ \forall i=1,\ldots,N$。

令 $\overline \beta_i = N \beta_i$，那么 (20) 式变为

$$\mathbf x_i =\sqrt {1-\frac {\overline \beta_i}{N}} \mathbf x_{i-1} + \sqrt {\frac {\beta_i} N} \mathbf z_{i-1}$$

当 $N \rightarrow \infty$ 时，令 $\beta(\frac i N)=\overline \beta_i$，注意这里的 $\beta(\frac i N)$ 与原来的 $\beta_i$ 不相等，关系为 $\beta(\frac i N)=N \beta_i$，根据 $\Delta t=\frac 1 N$，那么 $\beta_i=\beta(\frac i N) / N=\beta(\frac i N) \Delta t$。令 $\mathbf x(\frac i N)=\mathbf x_i$，$\mathbf z(\frac i N)=\mathbf z_i$，这两个变量的变换与 VE 中相同。

于是 (20) 式变为


$$\begin{aligned}\mathbf x(t+\Delta t)&=\sqrt {1-\beta(t+\Delta t) \Delta t} \cdot \mathbf x(t)+\sqrt {\beta (t+\Delta t)\Delta t} \cdot \mathbf z(t)
\\ & \approx \mathbf x(t) - \frac 1 2 \beta(t+\Delta t)\Delta t \ \mathbf x(t) + \sqrt {\beta(t+\Delta t)\Delta t} \ \mathbf z(t)
\\& \approx \mathbf x(t) - \frac 1 2 \beta(t)\Delta t \ \mathbf x(t) + \sqrt {\beta(t)\Delta t} \ \mathbf z(t)
\end{aligned} \tag{21}$$

上式推导对 $\sqrt {1-\beta(t+\Delta t)\Delta t}$ 进行了泰勒一阶展开，根据上式推导结果可知

$$d\mathbf x = -\frac  1 2 \beta(t) \mathbf x \ dt + \sqrt {\beta(t)} \ d\mathbf w \tag{22}$$

对比 (13) 式可知

$$\mathbf f(\mathbf x, t)=-\frac 1 2 \beta(t) \mathbf x, \quad g(t) =\sqrt {\beta(t)} \tag{23}$$

其中 $\beta(t)=N \beta_{t\cdot N}$ 。

接下来计算 $p(\mathbf x_i|\mathbf x_0)$。在原 DDPM 论文中，$q(\mathbf x_i |\mathbf x_0)$ 由 (6) 式给出。然而这里将 $\beta_i$ 修改为 $\beta(t)$，其含义已经发生变化，所以现在是依据 (22) 式而非 (20) 式来描述马尔可夫链转移，故 $q(\mathbf x_i |\mathbf x_0)$ 不能使用 (6) 式，我们重新推导。

对 (21) 式两边取期望，

$$\mathbf e(t+\Delta t)=\mathbf e(t)  - \frac 1 2 \beta(t) \Delta t  \mathbf e(t)+\mathbf 0 \tag{24}$$

变换得

$$d \mathbf e(t) = -\frac 1 2 \beta(t) \mathbf e(t) \ dt \tag{25}$$

对上式两边积分得，

$$\mathbf e(t)=C e^{-\frac 1 2 \int_0^t \beta(s)ds} \tag{26}$$

根据初值条件 $\mathbf e(0)=\mathbf e_0$，可得 $C=\mathbf e_0$，于是

$$\mathbf e(t)=\mathbf e_0  e^{-\frac 1 2 \int_0^t \beta(t)ds} \tag{27}$$

对 (21) 式两边取协方差，

$$\Sigma(t+\Delta t)=(1-\beta(t+\Delta t) \Delta t) \Sigma(t)+\beta(t+\Delta t)\Delta t I
\\ \approx (1-\beta(t) dt)\Sigma(t) + \beta(t) dt I \tag{28}$$

故

$$d \Sigma(t)= \beta(t) (I - \Sigma(t)) dt \tag{29}$$

积分得

$$\Sigma(t)=Ce^{\int_0^t -\beta(s) ds} + I \tag{30}$$

根据初值条件 $\Sigma(t)=\Sigma_0$，知 $C=\Sigma_0 -I$，于是

$$\Sigma(t)=(\Sigma_0 -I)e^{\int_0^t -\beta(s) ds} + I \tag{31}$$

于是条件分布为

$$\mathbf x_t| \mathbf x_0 \sim  \mathcal N\left(\mathbf e_0  e^{-\frac 1 2 \int_0^t \beta(t)ds}, (\Sigma_0 -I)e^{\int_0^t -\beta(s) ds} + I\right) \tag{32}$$

如果 $\mathbf x_0$ 已知，那么 $\mathbf e_0 =\mathbf x_0$，且 $\Sigma_0=\mathbf 0$，(32) 式变为

$$\mathbf x_t| \mathbf x_0 \sim  \mathcal N\left(\mathbf x_0  e^{-\frac 1 2 \int_0^t \beta(t)ds},  -Ie^{\int_0^t -\beta(s) ds} + I\right) \tag{33}$$

### 2.4.3 sub-VP

受 VP SDE 启发，作者提出了一个新的 SDE，称为 sub-VP SDE，如下

$$d\mathbf x = -\frac 1 2 \beta(t) \mathbf x \ dt + \sqrt {\beta(t)(1-e^{-2\int_0^t \beta(s)ds})} d\mathbf w \tag{34}$$

与 DDPM VP 一样，$\mathbf x(t)$ 的期望满足 (25) 式，于是期望为 (27) 式。

协方差根据 (28) (29) 式进行调整，得

$$d\Sigma(t) =\beta(t) (I - I e^{-2\int_0^t \beta(s)ds} - \Sigma(t)) dt$$

积分得 

$$\Sigma(t)=Ce^{\int_0^t -\beta(s) ds} + I + I e^{-2\int_0^t \beta(s)ds}$$

根据初值条件 $\Sigma(t)=\Sigma_0$，于是 $\Sigma_0=C+I+I$，于是

$$\Sigma(t)=(\Sigma_0-2I)e^{\int_0^t -\beta(s) ds} + I + I e^{-2\int_0^t \beta(s)ds}\tag{35}$$

**性质：**

1. 当 $\Sigma_{VP}(0)=\Sigma_{sub-VP}(0)$ 时，

    $$\Sigma_{VP}(t)-\Sigma_{sub-VP}(t)=I(e^{\int_0^t -\beta(s) ds}-e^{-2\int_0^t \beta(s)ds}) \succ 0 \cdot I$$

    这表明，使用 sub-VP SDE，$\mathbf x_t |\mathbf x_0$ 分布的方差更小。

当 $\mathbf x_0$ 已知时，

$$\mathbf x_t|\mathbf x_0 \sim \mathcal N \left(\mathbf x_0  e^{-\frac 1 2 \int_0^t \beta(t)ds},  (1-e^{\int_0^t -\beta(s) ds})^2 I\right) \tag{36}$$

# 3. 解反向 SDE

根据设计好的 $\mathbf f(\mathbf x,t)$ 和 $g(t)$ 可以计算出 $\mathbf x_t|\mathbf x_0$ 的分布，然后可以训练 score-based 模型 $\mathbf s_{\theta}$ ，根据前向 SDE 可以得到 reverse-time SDE（根据论文 Anderson 1982），然后使用数值解法以生成来自 $p_0$ 分布的样本。
数值解法包括 Euler-Maruyama 和 随机 Runge-Kutta 方法等。

## 3.1 反向扩散采样

给定前向 SDE

$$d\mathbf x = \mathbf f(\mathbf x, t) dt + \mathbf G(t) d\mathbf w \tag{37}$$

其中 $\mathbf x \in \mathbb R^d, \ \mathbf G(t) \in \mathbb R^{d \times d}$ 。


离散化 (37) 式，将 $t, \ dt$ 从 (37) 式中剥离

$$\mathbf x_{i+1} = \mathbf x_i + \mathbf f_i(\mathbf x_i) + \mathbf G_i \mathbf z_i, \quad i=0,1,\ldots,N-1 \tag{38}$$

其中 $\mathbf z_i \sim \mathcal N(\mathbf 0, I)$ 。

根据论文 Anderson 1982，reverse-time SDE 为

$$d\mathbf x = [\mathbf f(\mathbf x, t)-\mathbf G(t)\mathbf G(t)^{\top} \nabla_{\mathbf x} \log p_t(\mathbf x)] dt + \mathbf G(t) d\overline {\mathbf w} \tag{39}$$

注意 (39) 式中 $\overline {\mathbf w}$ 表示从时间 $T$ 开始到时间 $0$ 结束，即 $\mathbf w_{s-t} - \mathbf w_s \sim \mathcal N(\mathbf 0, t I)$ 。(39) 式离散化为

$$\mathbf x_{i+1}-\mathbf x_i=\mathbf f_{i+1}(\mathbf x_{i+1})-\mathbf G_{i+1}\mathbf G_{i+1}^{\top} \mathbf s_{\theta}(\mathbf x_{i+1},i+1) - \mathbf G_{i+1}\mathbf z_{i+1}$$

变换得

$$\mathbf x_i=\mathbf x_{i+1}-\mathbf f_{i+1}(\mathbf x_{i+1})+\mathbf G_{i+1}\mathbf G_{i+1}^{\top} \mathbf s_{\theta}(\mathbf x_{i+1},i+1) + \mathbf G_{i+1}\mathbf z_{i+1}, \quad i=0,\ldots, N-1 \tag{40}$$

(40) 式就是离散化的采样规则。注意这里使用 $\mathbf f_{i+1}(\mathbf x_{i+1})$ 而不使用 $\mathbf f_{i}(\mathbf x_{i+1})$ 的写法是为了下标统一，故 (40) 式中的 $\mathbf f_1,\ldots, \mathbf f_N$ 就是 (38) 式中的 $\mathbf f_0, \ldots, \mathbf f_{N-1}$，对 $\mathbf G$ 的下标也是如此。

### 3.1.1 reverse-time VE SDE 采样

将 (40) 式应用于 (16) 式，得到 reverse-time VE SDE 采样器。将 (16) 和 (38) 式下标 $i$ 的取值范围统一为 $i=0,\ldots, N-1$，对比 (16) 和 (38) 式可知 (40) 式中的的相关函数为

$$\mathbf f_{i+1}(\mathbf x_{i+1})=\mathbf 0, \quad \mathbf G_{i+1}=I\sqrt {\sigma_{i+1}^2 - \sigma_i^2}$$



于是采样算法流程如下方算法 1 的蓝色部分。

---
**算法 1**： PC 采样（VE SDE）
$\mathbf x_N \sim \mathcal N(\mathbf 0, \sigma_{max}^2I)$
**for** $i=N-1,\ldots, 0$ **do**
<font color='blue'>
&emsp; $\mathbf x_i' \leftarrow \mathbf x_{i+1} + \sqrt {\sigma_{i+1}^2 - \sigma_i^2} \mathbf s_{\theta^{\star}}(\mathbf x_{i+1}, \sigma_{i+1})$
&emsp; $\mathbf z \sim \mathcal N(\mathbf 0, I)$
&emsp; $\mathbf x_i \leftarrow \mathbf x_i' + \sqrt {\sigma_{i+1}^2 - \sigma_i^2} \mathbf z$
</font>
<font color='orange'>
&emsp; <font style='font-weight:bold'>for</font> $j=1,\ldots, M$ <font style='font-weight:bold'>do</font>
&emsp; &emsp; $\mathbf z \sim \mathcal N(\mathbf 0, I)$
&emsp; &emsp; $\mathbf x_i \leftarrow  \mathbf x_i + \epsilon_i \mathbf s_{\theta^{\star}}(\mathbf x_i, \sigma_i) + \sqrt {2 \epsilon_i} \mathbf z$
</font>
**return** $\mathbf x_0$

---


### 3.1.2 reverse-time VP SDE 采样

将 (40) 式应用于 (20) 式，得到 reverse-time VE SDE 采样器。对比 (20) 和 (38) 式可知 (40) 式中的的相关函数为

$$\mathbf f_{i+1}(\mathbf x_{i+1})=(\sqrt{1-\beta_{i+1}}-1)\mathbf x_{i+1}, \quad \mathbf G_{i+1}=I\sqrt {\beta_{i+1}}$$

于是采样算法流程如下方算法 2 的蓝色部分。

---
**算法 2**： PC 采样（VP SDE）
$\mathbf x_N \sim \mathcal N(\mathbf 0, \sigma_{max}^2I)$
**for** $i=N-1,\ldots, 0$ **do**
<font color='blue'>
&emsp; $\mathbf x_i' \leftarrow (2-\sqrt{1-\beta_{i+1}})\mathbf x_{i+1} + \beta_{i+1} \mathbf s_{\theta^{\star}}(\mathbf x_{i+1}, i+1)$
&emsp; $\mathbf z \sim \mathcal N(\mathbf 0, I)$
&emsp; $\mathbf x_i \leftarrow \mathbf x_i' + \sqrt {\beta_{i+1}} \mathbf z$
</font>
<font color='orange'>
&emsp; <font style='font-weight:bold'>for</font> $j=1,\ldots, M$ <font style='font-weight:bold'>do</font>
&emsp; &emsp; $\mathbf z \sim \mathcal N(\mathbf 0, I)$
&emsp; &emsp; $\mathbf x_i \leftarrow  \mathbf x_i + \epsilon_i \mathbf s_{\theta^{\star}}(\mathbf x_i, i) + \sqrt {2 \epsilon_i} \mathbf z$
</font>
**return** $\mathbf x_0$

---

作者称类似 算法 1 和 2 这样的基于 (40) 式的采样为 _reverse diffusion samplers_ 。



DDPM 中 ancestral sampling 为 (12) 式，下面再次给出了，省去前面翻看的麻烦

$$\mathbf x_{i-1}=\frac 1 {\sqrt {1-\beta_i}} (\mathbf x_i+\beta_i \mathbf s_{\theta}(\mathbf x_i, i)) +\sqrt {\beta_i}\mathbf z_i, \quad i=N,N-1,\ldots, 1 \tag{12}$$

根据泰勒展开，

$$(1-\beta_i)^{-\frac 1 2}=1+\frac 1 2 \beta_i + \frac 3 8 \beta_i^2 + \cdots$$

当 $\beta_i \rightarrow 0$ 时，取一阶近似，代入 (12) 式

$$\mathbf x_{i-1}=(1+\frac 1 2 \beta_i)\mathbf x_i + (\beta_i + \frac 1 2 \beta_i^2)\mathbf s_{\theta}(\mathbf x_i, i) + \sqrt {\beta_i} \mathbf z_i \tag{41-1}$$

另一方面，我们将 (23) 式代入 (40) 式，注意这里要将 $\beta(t)$ 变为 $\beta_i$，

$$\mathbf x_{i-1}=\mathbf x_i + \frac 1 2 \beta_i \mathbf x_i +  \beta_i\mathbf s_{\theta}(\mathbf x_i, i) + \sqrt{\beta_i} \mathbf z_i \tag{41-2}$$

忽略 (41-1) 式中的二阶项 $\beta_i^2$，可以发现 (41-1) 和 (42-2) 等价，即 **reverse-time SDE 出发推导出来的采样规则与离散情况的反向采样规则等价**。


## 3.2 Predictor-Corrector 采样器

作者提出，可以使用 score-based MCMC 方法，例如 Langevin MCMC，从分布 $p_t$ 中直接采样，从而纠正 SDE 数值求解的结果。

具体而言，每个 step $i$，SDE 数值求解会给出一个样本估计，此过程充当 “predictor”，如算法 1 和 2 中蓝色部分的 $\mathbf x_i$，然后，score-based MCMC 方法纠正这个样本估计，如算法 1 和 2 中橙色部分（使用 Langevin dynamics 方法），此过程充当 “corrector”。

算法 1 和 2 中的 PC 采样就是指 Predictor-Corrector 采样。

## 3.3 概率流

对 scored-based 模型，求解 reverse-time SDE 有另一种数值法。对于扩散，存在一种确定性过程，在其转移路径上边缘分布与 SDE 的情况相同，这种确定性过程满足某个 ODE，

$$d\mathbf x=[\mathbf f(\mathbf x, t)-\frac 1 2 g(t)^2 \nabla_{\mathbf x} \log p_t(\mathbf x)]dt \tag{42}$$

详情参见附录 D，也就是将 (A5) 式中的 $\mathbf G(\mathbf x,t)$ 替换为 $g(t)$ 就得到 (42) 式。这样，一旦得分函数已知，(42) 式描述的过程就被确定。称 (42) 式这个 ODE 为 概率流 ODE （因为没有 $d\mathbf w$ 这一项，所以 (42) 式本质就是一个常微分方程）。当得分函数由时间相关的score-based model（这个 model 通常是神经网络） 近似时，就是一个 neural ODE 。


# 4. 总结

设计一个前向 SDE，包括 $\mathbf f(\mathbf x, t)$ 和 $\mathbf G(\mathbf x, t)$ ，从而可以计算出 reverse-time SDE 以及 $p(\mathbf x|\mathbf x_0)$，根据 score matching，可训练出得分函数 $\mathbf s_{\theta}(\mathbf x, \sigma)$ （这里模型 $\mathbf s_{\theta}$ 的输入是训练数据 $\mathbf x_0$ 以及对训练数据做扰动的噪声方差 $\sigma$）。然后使用 PC 采样生成来自 $p_0$ 的样本。

通过 SMLD 和 DDPM，验证了 SDE 求解问题的可行性，并与传统的离散化迭代过程（扩散过程）等价。所以我们可以直接从 SDE 出发，求解生成模型。


# APPENDIX

## A

考虑如下前向 SDE，

$$d\mathbf x = \mathbf f(\mathbf x, t) dt + \mathbf G(\mathbf x, t) d\mathbf w \tag{A1}$$

注意 (37) 式是上式的一个特例：$\mathbf G$ 函数中没有 $\mathbf x$ 这个自变量。(A1) 式对应的 reverse-time SDE 为

$$d\mathbf x = \{\mathbf f(\mathbf x, t)-\nabla_{\mathbf x} [\mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top}] - \mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top} \nabla_{\mathbf x} \log p_t(\mathbf x)\}dt + \mathbf G(\mathbf x, t)d\overline {\mathbf w} \tag{A2}$$

其中 $\mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top}$ 表示两个矩阵相乘。对于矩阵 $\mathbf F(\mathbf x)\stackrel{\Delta}=[\mathbf f^1(\mathbf x), \cdots, \mathbf f^d(\mathbf x)]^{\top}$，定义

$$\nabla F(\mathbf x)\stackrel{\Delta}=[\nabla \mathbf f^1(\mathbf x), \cdots, \nabla \mathbf f^d(\mathbf x)]^{\top}$$

上式针对 $\mathbf x$ 求梯度。

记 reverse-time SDE 为

$$d\mathbf x = \overline {\mathbf f}(\mathbf x, t)dt + \overline {\mathbf G}(\mathbf x, t) d \overline {\mathbf w} \tag{A3}$$

那么给定初值条件 $\mathbf x_T$ 时，可得 SDE 的解为

$$\mathbf x_T - \mathbf x_t=\int_t^{T} \overline {\mathbf f}(\mathbf x, s)ds + \int_t^T \overline {\mathbf G}(\mathbf x, s) d\overline {\mathbf w}(s) \tag{A5}$$


## D

### D.1 概率流 ODE 推导

考虑 (A1) 式 SDE，推导出概率流 ODE 如下（推导过程略，见原论文附录），

$$d\mathbf x = \tilde {\mathbf f}(\mathbf x, t) dt + \tilde {\mathbf G}(\mathbf x, t) d\mathbf w$$

其中 $\tilde {\mathbf G}(\mathbf x, t) := \mathbf 0$ 且

$$\tilde {\mathbf f}:=\mathbf f(\mathbf x, t) - \frac 1 2 \nabla [\mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top}] - \frac 1 2 \mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top}\nabla_{\mathbf x} \log p_t(\mathbf x) \tag{A5}$$


