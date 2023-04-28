---
title: guided-diffusion
date: 2022-09-30 14:02:23
tags: diffusion model
mathjax: true
img:
summury: 支持基于标签的图像生成
---

论文：[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2015.05233)
源码：[guided-diffusion](https://github.com/openai/guided-diffusion)


# 1. 框架改进

## 1.1 AdaGN

adaptive group normalization（AdaGN）


$$ AdaGN(h,y)=y_s \cdot GN(h)+y_b \tag{1}$$

其中 $y=[y_s,y_b]$ 是 timestep 和 class 的 embedding 经过一个线性变换而来，这里用到了图像的分类信息 $y_b$。

# 2. 分类器指导

可以使用一个分类器的梯度作为一个预训练的扩散模型的条件，从而指导扩散模型根据条件生成图像，具体而言，我们训练一个分类器 $p_{\phi}(y|x_t,t)$ 然后使用梯度 $\nabla_{x_t} \log p_{\phi}(y|x_t, t)$ 来指导采样过程，最终生成的图像趋于标签 $y$ 。

## 2.1 条件逆噪声过程

DDPM 是无条件的扩散模型，反向过程为 $p_{\theta}(x_t,|x_{t+1})$，增加一个条件——标签 $y$ 后，采样过程可表示为

$$p_{\theta, \phi}(x_t|x_{t+1},y) = Z p_{\theta}(x_t|x_{t+1}) p_{\phi}(y|x_t) \tag{2}$$ 

证明过程参考论文附录 H，这里略。

回顾 DDPM 的反向过程可知，

$$p_{\theta}(x_t|x_{t+1})=\mathcal N(\mu, \Sigma) \tag{3}$$

取对数

$$\log p_{\theta}(x_t|x_{t+1})=-\frac 1 2 (x_t -\mu)^{\top}\Sigma (x_t-\mu) + C \tag{4}$$

对于分类器预测概率对数，可以假设其具有比 $\Sigma^{-1}$ 小的曲率（即 $\log p(y|x_t)$ 曲线变化更缓慢，或者说方差较大），这个假设是合理的，因为 $\|\Sigma\| \rightarrow 0$，于是使用 Taylor 展开，

$$\begin{aligned}\log p_{\phi}(y|x_t) & \approx \log p_{\phi}(y|x_t)| _{x_t=\mu} + (x_t-\mu)\nabla_{x_t} \log p_{\phi}(y|x_t)| _{x_t=\mu}
\\ &=(x_t-\mu) g + C_1
\end{aligned} \tag{5}$$


结合上面两式，可知

$$\begin{aligned} \log (p_{\theta}(x_t|x_{t+1})p_{\phi}(y|x_t)) & \approx -\frac 1 2 (x_t-\mu)^{\top} \Sigma^{-1}(x_t-\mu) + (x_t-\mu) g + C_2 
\\\\ &= -\frac 1 2 (x_t -\mu -\Sigma g)^{\top} \Sigma^{-1} (x_t -\mu -\Sigma g) + C_3
\\\\ &= \log p(z) + C_4 
\end{aligned} \tag{6}$$

其中 $z \sim \mathcal N(\mu + \Sigma g , \Sigma)$ 。

根据 DDPM 中分析，

$$\mu = \frac 1 {\sqrt {\alpha_t}} \left(x_t - \frac {\beta_t}{\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(x_t)\right) \tag{7}$$

且 $\Sigma$ 我们选取固定的 $\beta_t$，并且 (6) 式 中对 $\Sigma g$ 项增加一个因子 $s=\frac 1 {\sqrt {\alpha_t}} > 1$，那么

$$\begin{aligned}\mu+ s\Sigma g &= \frac 1 {\sqrt {\alpha_t}} \left(x_t - \frac {\beta_t}{\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(x_t)\right) + \frac 1 {\sqrt {\alpha_t}}\beta_t \cdot g
\\\\ &=\frac 1 {\sqrt {\alpha_t}} \left(x_t -\frac {\beta_t}{\sqrt {1-\overline \alpha_t}}[\epsilon_{\theta}(x_t) - \sqrt {1-\overline \alpha_t} \cdot g]\right)
\end{aligned}$$

令

$$\hat \epsilon(x_t) := \epsilon_{\theta}(x_t) - \sqrt {1-\overline \alpha_t}  \nabla_{x_t} \log p_{\phi}(y|x_t) \tag{8}$$

于是，引入分类器后，扩散模型反向过程的改变为

$$\epsilon_{\theta}(x_t) \rightarrow \hat \epsilon(x_t) \tag{9}$$

当然分类器梯度因子 $s$ 也可以选择其他 $>1$ 的值，可以理解为 $\Sigma_{\theta}$ 选择不同于 $\beta_t$ 的某个值。

## 2.2 条件 DDIM

DDIM 与 DDPM 有所不同，DDIM 前向过程不是马尔可夫过程，但是 DDIM 前向过程依然满足

$$q(x_t|x_0)=\mathcal N(\sqrt {\overline \alpha_t} x_0,  {1-\overline \alpha_t}I) \tag{10}$$

存在关系

$$x_t = \sqrt{\overline \alpha_t} x_0 + \sqrt {1-\overline \alpha_t} \cdot \epsilon \tag{11}$$

模型预测 $\epsilon$ 为 $\epsilon_{\theta}(x_t)$ ，反向过程中 $p_{\theta}(x_t)$ 应尽量逼近 $q(x_t|x_0)$，于是

$$p_{\theta}(x_t) = \mathcal N(\sqrt {\overline \alpha_t} x_0',  {1-\overline \alpha_t} I) \tag{12}$$

其中 $x_0'$ 是模型在 $t$ timestep 对 $x_0$ 的一次预测，根据 (11) 式有 

$$x_0'(x_t) =\frac 1 {\sqrt {\overline \alpha_t}} [x_t - \sqrt {1-\overline \alpha_t} \cdot \epsilon_{\theta}(x_t)] \tag{13}$$

建立 score matching 与扩散模型之间的联系如下，

$$\nabla_{x_t}  \log p_{\theta}(x_t) = - \frac {x_t - \sqrt {\overline \alpha_t} x_0'}{1-\overline \alpha_t} = -\frac 1 {\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(x_t) \tag{14}$$

当考虑了分类器的梯度时，得分函数应该逼近 $\nabla_{x_t} \log (p_{\theta}(x_t) p_{\phi}(y|x_t))$ ，即 target 中要包含 $y$，这样生成的图像其标签才会趋于 $y$，而

$$\begin{aligned} \nabla_{x_t} \log (p_{\theta}(x_t) p_{\phi}(y|x_t)) &= \nabla_{x_t} \log p_{\theta}(x_t) + \nabla_{x_t} \log p_{\phi}(y|x_t) 

\\\\ &=-\frac 1 {\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(x_t)+\nabla_{x_t} \log p_{\phi}(y|x_t)
\end{aligned}$$

作 (8) 式变换得到 $\hat \epsilon(x_t)$ ，于是 

$$\nabla_{x_t} \log (p_{\theta}(x_t) p_{\phi}(y|x_t))=-\frac 1 {\sqrt {1-\overline \alpha_t}} \hat \epsilon(x_t) \tag{15}$$

(12) 式为无条件扩散模型的得分函数，(15) 式为有条件扩散模型的得分函数，同样是对应 (9) 式的变换。


---
**算法 1**：分类器指导的扩散模型（DDPM）采样过程，扩散模型描述为 $(\mu_{\theta}(x_t), \Sigma_{\theta}(x_t))$，分类器记为 $p_{\phi}(y|x_t)$

**输入**：分类标签 $y$，梯度因子 $s$

采样 $x_T \leftarrow \mathcal N(0,I)$

**for** $t=T,\ldots, 1$ **do**

&emsp; $\mu, \Sigma \leftarrow \mu_{\theta}(x_t), \Sigma_{\theta}(x_t)$

&emsp; 采样 $x_{t-1} \leftarrow \mathcal N(\mu+s\Sigma \nabla_{x_t} \log p_{\phi}(y|x_t), \Sigma)$

**end for**

**return** $x_0$

---
**算法 2**：分类器指导的 DDIM 采样过程，扩散模型为 $\epsilon_{\theta}(x_t)$，分类器为 $p_{\phi}(y|x_t)$

**输入** 分类标签 $y$，梯度因子 $s$

采样 $x_T \leftarrow \mathcal N(0, I)$

**for** $t=T,\ldots, 1$ **do**

&emsp; $\hat \epsilon \leftarrow \epsilon_{\theta}(x_t) - \sqrt {1-\overline \alpha_t} \nabla_{x_t} \log p_{\phi} (y|x_t)$

&emsp; $x_{t-1} \leftarrow \sqrt {\overline \alpha_{t-1}}\left(\frac {x_t - \sqrt {1-\overline \alpha_t} \hat \epsilon} {\sqrt {\overline \alpha_t}} \right) + \sqrt {1-\overline \alpha_{t-1}} \hat \epsilon$

**end for**

**return** $x_0$

---

# 3. 训练

## 3.1 模型说明

扩散模型框架采样 UNet，分类器使用 UNet 的下采样部分（即，Encoder 部分），使用 ImageNet 数据集，在输出 $8 \times 8$ spatial size 的 layer 之后使用 attention pool，从而将 spatial size 变成 $1 \times 1$ 。


分类器使用部分加噪的图片进行训练，这里加噪过程由对应的扩散模型进行加噪，加噪后的图片进行随机 crop 然后作为分类器的训练输入，以降低过拟合。

## 3.2 分类器梯度因子

如果分类器梯度因子 $s=1$，作者实验发现分类器将大约 50% 的最终采样结果预测为想要的标签，但是这些采样结果从视觉上看与这些标签并不匹配，将分类器梯度因子放大，可以修复这个问题。


注意到 

$$s \cdot  \nabla_{x} \log p(y|x) = \nabla_x \log \frac 1 Z p(y|x)^s$$

其中 $Z$ 是常量。当 $s$ 越大，分布变得越尖锐，这意味着，使用大的梯度因子会更集中在分类器预测概率的 mode 上，这有利于生成更准确（指分类准确）的样本（但是多样性更低）。





