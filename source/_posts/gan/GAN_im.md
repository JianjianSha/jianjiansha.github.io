---
title: GAN 的几种改进
date: 2019-07-25 20:24:41
tags: GAN
mathjax: true
---

# 1. WGAN

论文 [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

这篇论文主要研究并解决 GAN 训练不稳定的问题，以及 GAN 在生成图片质量和多样性两方面无法兼顾的问题。WGAN 从真实数据分布与模型分布之间的距离着手，提出使用 Earth-Mover (EM) 距离，

$$W(\mathbb P_r, \mathbb P_g)=\inf_{\gamma \in \prod(\mathbb P_r, \mathbb P_g)} \mathbb E_{(x,y)\sim \gamma} [\|x-y\|] \tag{1}$$

其中 $\mathbb P_r, \mathbb P_g$ 分别表示真实概率分布和模型分布，$\prod(\mathbb P_r, \mathbb P_g)$ 表示边缘分布满足 $\mathbb P_r, \mathbb P_g$ 的所有联合分布的集合，$(x,y)\sim \gamma$ 表示从一个联合分布中采样得到一个真实数据和一个模型生成数据。

但是 (1) 式难以直接计算。根据  Kantorovich-Rubinstein duality，可以转换为计算

$$W(\mathbb P_r, \mathbb P_{\theta})=\sup_{\|f\|_L \le 1} \mathbb E_{x \sim \mathbb P_r}[f(x)] - \mathbb E_{x\sim \mathbb P_{\theta}}[f(x)] \tag{2}$$

(2) 式求上确界是对所有的 1-Lipschitz 函数 $f:\mathcal X \rightarrow \mathbb R$ 。

对于判别器使用函数 $f_w$ 表示，模型参数的取值空间为 $w \in \mathcal W$ ，那么优化目标变为

$$\max_{w\in \mathcal W} \mathbb E_{x \sim \mathbb P_r}[f_w(x)] - \mathbb E_{z \sim p(z)}[f_w(g_{\theta}(z))] \tag{3}$$

为了使 $f_w(x)$ 满足 1-Lipschitz，一种简单的方法是将参数 $w$ clamp到 $[-0.01,0.01]$ 之间，算法如图 1 所示，

![](/images/gan/wgan_1.png)

# 3. SNGAN

论文：[ Spectral normalization for generative adversarial networks](https://arxiv.org/abs/1802.05957)

作者提出一种新型的权重归一化方法，名为 “光谱归一化” spectral normalization ，解决判别器训练过程中的不稳定现象。

## 3.1 方法介绍

考虑判别器网络，由下式表示

$$f(x,\theta)=W^{L+1} a_L(W^L(a_{L-1}(W^{L-1}(\cdots a_1(W^1 x))))) \tag{1}$$

其中 $W^l$ 是权重参数，$a_l$ 表示非线性激活函数，为了简单，忽略 bias。

$f(x, \theta)$ 是最后一层特征，判别器的最终输出为

$$D(x,\theta)= \mathcal A(f(x,\theta)) \tag{2}$$

其中 $\mathcal A$ 是激活函数，例如 sigmoid（表示输出为概率），这取决于我们使用何种测度距离。GAN 的目标函数为

$$\min_G \max_D V(G,D)=\Bbb E_{x \sim p_{data}(x)}[\log D(x)] + \Bbb E_{z \sim p_z(z)}[\log(1-D(G(z)))] \tag{3}$$

固定生成器 G，那么根据 (3) 式，最优判别器为

$$D_G^{\star}(x)=\frac {q_{data}(x)}{q_{data}(x)+p_G(x)} \tag{4}$$

machine learning community 指出，判别器的函数空间的选择至关重要，对 GAN 的影响很大。Lipschitz 连续性在确保统计量边界上起到重要作用。对于 (3) 式，可写为

$$D_G^{\star}(x)=\text{sigmoid} (f^{\star}(x)), \ f^{\star}(x)=\log q_{data}(x)-\log p_G(x) \tag{4}$$

导数为

$$\nabla_x f^{\star}(x) = \frac 1 {q_{data}(x)} \nabla_x q_{data}(x) - \frac 1 {p_G(x)}(x) \tag{5}$$

(5) 式这个导数可能会无边界甚至无法计算。这就促使我们考虑对 $f(x)$ 施加正则。

一种方法是控制判别器函数的 Lipschitz 常数。满足如下性质的任意连续函数 $f(x)$  称为K-Lipschitz:

$$||f(x_2)-f(x_1)|| \le K ||x_2-x_1|| \tag{6}$$

作者也采用这种方法，即判别器函数 $f(x)$ 满足 K-Lipschitz，

$$\arg \max_{||f||_{Lip} \le K} \ V(G, D) \tag{7}$$

### 3.1.1 光谱归一化

限制判别器网络中每层 $g: h_{in} \rightarrow h_{out}$ 的光谱范数，从而控制判别器函数 $f$ 的 Lipschitz 常数。

根据定义，Lipschitz 范数 $\|g\|_{Lip} = \sup_{h} \sigma(\nabla g(h))$，其中 $\sigma(A)$ 是矩阵 A 的光谱范数，

$$\sigma(A):=\max_{h:h\ne 0} \ \frac {\|A h\|_2}{\|h\|_2}=\max_{\|h\|_2 \le 1} \|Ah\|_2 \tag{8}$$

上式等于 $A$ 的最大奇异值（奇异值分解中的 $\Sigma$ 对角线最大元素值）。于是，对于线性层 $g(h)=Wh$，有

$$\|g\|_{Lip}=\sup_h \sigma(\nabla g(h))=\sup_h \sigma(W)=\sigma(W) \tag{9}$$

如果激活函数满足 $\|a_l\|_{Lip}=1$，那么根据不等式 $\|g_1 \circ g_2\|_{Lip} \le \|g_1\|_{Lip} \cdot \|g_2\|_{Lip}$ 可知

$$\|f\|_{Lip} \le \|g_{L+1}\|_{Lip} \cdot \|a_L\|_{Lip} \cdot \|g_L\|_{Lip} \cdots \|a_1\|_{Lip} \cdot \|g_1\|_{Lip} = \prod_{l=1}^{L+1} \sigma(W^l) \tag{10}$$

ReLU 和 Leaky ReLU 均满足  $\|a_l\|_{Lip}=1$。

光谱归一化指对权重矩阵 W 的光谱进行归一化使得 $\sigma(W)=1$，

$$\overline W_{SN}(W):=W/\sigma(W) \tag{11}$$

使用 (11) 式对权重矩阵光谱归一化，使得 $\|f\|_{Lip} \le 1$ （上边界为 1）。

本文的光谱归一化与 [spectral norm regularization](https://arxiv.org/pdf/1705.10941.pdf) 不同，后者是通过增加显示的正则项到目标函数中，从而对光谱范数进行惩罚，修改后的目标函数如下

$$\min_{\theta} \frac 1 K \sum_{i=1}^K L(f_{\theta}(x_i),y_i) + \frac {\lambda} 2 \sum_{l=1}^L \sigma(W^l)^2$$

其中 $K$ 为训练集大小，$\lambda > 0$ 是平衡因子。

### 3.1.2 光谱范数近似求解

前面已经提到，光谱范数 $\sigma(W)$ 是 W 的最大奇异值。如果使用奇异值分解来求 $\sigma(W)$ ，计算效率并不高。作者使用 power iteration 方法求解，计算量不大。

对于权重 $W$，随机初始化一个向量 $\tilde u$，$\tilde u$ 满足各向同性分布， 例如在球面上的均匀分布，那么当 $W$ 的主要奇异值没有重根时，且 $\tilde u$ 与第一个左奇异向量不正交（这个假设是有效的，因为正交的概率为 0），那么应用 power 方法可得到第一个左奇异向量和右奇异向量，如下

$$\tilde v \leftarrow W^{\top} \tilde u /\|W^{\top} \tilde u\|_2, \quad \tilde u \leftarrow W \tilde / \|W\tilde \|_2 \tag{12}$$

上式中 $\tilde v$ 为首个右奇异向量，$\tilde u$ 为首个左奇异向量。于是近似的光谱范数为 

$$\sigma(W) \approx \tilde u^{\top} W \tilde v \tag{13}$$

算法如图 2，

![](/images/gan/sngan_1.png)

<center>图 2.</center>

由于每一轮权重矩阵 $W$ 变化非常小，所以其最大的奇异值的变化也非常小。每一轮更新时，$\tilde u$ 使用相同的值，即 $\tilde u$ 初始化的值。

### 3.1.3 梯度分析

根据 (11) 式可得，

$$\frac {\partial \overline W_{SN}(W)}{\partial W_{ij}}=\frac {W'\sigma(W)- W \sigma'(W)}{\sigma^2(W)}=\frac 1 {\sigma(W)} E_{ij}-\frac 1 {\sigma^2(W)} \sigma'(W) W$$

注意，上式中 $W'$ 表示矩阵对标量求导，易知 $W'=E_{ij}$，其中 $E_{ij}$ 表示与 $W$ 相同 shape 的矩阵，且仅有 $(i,j)$ 处的元素为 1，其余元素均为 0。

$\sigma'(W)$ 表示标量对标量求导，所以上式推导的最后将 $W$ 至于最后，$\frac 1 {\sigma^2(W)} \sigma'(W)$ 表示标量，移到前面，作为矩阵 $W$ 的系数。根据 (13) 式，$\sigma(W)$ 展开后是若干项之和，其中包含 $W_{ij}$ 的项为 $\tilde u_i \tilde v_j=[\tilde u \tilde v^{\top}]_{ij}$，其中 $\tilde u, \tilde v$ 为 $W$ 首个左奇异向量和右奇异向量，使用 $u_1, v_1$ 表示 $W$ 首个左奇异向量和右奇异向量（以免引起混淆），那么上式梯度推导为

$$\frac {\partial \overline W_{SN}(W)}{\partial W_{ij}}=\frac 1 {\sigma(W)}E_{ij} - \frac {[u_1 v_1^{\top}]_{ij}}{\sigma^2(W)} W = \frac 1 {\sigma(W)}(E_{ij}-[u_1 v_1^{\top}]_{ij}\overline W_{SN}) \tag{14}$$

记 $\overline W_{SN}$ 所表示的 layer 的输入为 $h$，输出为 $h^{l+1}=\overline W_{SN}\cdot h$，目标函数为 $V(G,D)$，那么梯度为

$$\begin{aligned}\frac {\partial V(G,D)}{\partial W}&=\frac {\partial V(G,D)}{\partial h^{l+1}} \frac {\partial h^{l+1}}{\partial \overline W_{SN}}\frac {\partial \overline W_{SN}}{\partial W}
\\&=\frac {\partial V(G,D)}{\partial h^{l+1}}h^{\top}\frac {\partial \overline W_{SN}}{\partial W}
\\&=\frac 1 {\sigma(W)}(\delta h^{\top} - \lambda  u_1 v_1^{\top})
\end{aligned} \tag{15}$$


其中 $\delta = \frac {\partial V(G,D)}{\partial h^{l+1}}$ 表示列向量。$\lambda = \delta^{\top} \overline W_{SN} h$ 为一标量。

(15) 式的推导中（下面省去了 SN 下标），

$h_i^{l+1}=\overline W_{i,:} h$，故 $\frac {\partial h_i^{l+1}}{\partial \overline W_{ij}} = h_j$，于是 $\frac {\partial V(G,D)}{\partial \overline W_{ij}} = \delta_i \cdot h_j$，即

$$\frac {\partial V(G,D)}{\partial \overline W}=\delta h^{\top}$$

根据 (14) 式结论，导数的第一项 $E_{ij}$，这表示只有 $\frac {\partial \overline W_{ij}}{\partial W_{ij}}=1$ 有有效梯度，其他梯度均为 0，据此可得 $\frac {\partial V(G,D)}{\partial W_{ij}}=(\delta h^{\top})_{ij} \cdot \frac 1 {\sigma(W)}$，即 $\frac {\partial V(G,D)}{\partial W}=\frac 1 {\sigma(W)} \delta h^{\top}$。根据 (14) 式导数的第二项可知，

$$\frac {\partial V(G,D)}{\partial W_{ij}}=\sum_{m,n} \frac {\partial V(G,D)}{\partial \overline W_{mn}} \frac {\partial \overline W_{mn}}{\partial W_{ij}}=\sum_{m,n} (\delta h^{\top})_{mn} (u_1v_1^{\top})_{ij} \overline W_{mn}= \delta^{\top} \overline W h (u_1v_1^{\top})_{ij}$$

于是 

$$\frac {\partial V(G,D)}{\partial W}=\delta^{\top} \overline W h (u_1v_1^{\top})$$

根据 (15) 式，当 $\delta h^{\top} = k u_1 v_1^{\top}$ 时，$\frac {\partial V}{\partial W}=0$ 。

## 3.2 网络框架

|$z \in \mathbb R^{128} \sim \mathcal N(0, I)$, Mg=4|out channel|out size|
|--|--|--|
|dense $\rightarrow M_g \times M_g \times 512 \rightarrow$ reshape|512|4x4x512|
|$4 \times 4$, stride=2, padding=1, deconv. BN ReLU|256|8x8x256|
|$4 \times 4$, stride=2, padding=1, deconv. BN ReLU|128|16x16x128|
|$4 \times 4$, stride=2, padding=1, deconv. BN ReLU|64|32x32x64|
|$3 \times 3$, stride=1, padding=1, conv. | 3 | 32x32x3|

表 1. 生成器网络框架。SVHN 和 CIFAR-10 数据集使用 $M_g=4$；STL-10 数据集使用 $M_g=6$，表中数据使用 $M_g=4$ 计算得到。

|RGB image $x \in \mathbb R^{M \times M \times 3}$, M=32|output channel|out size|
|--|--|--|
|3x3, stride=1, padding=1, conv 1ReLU|64|32x32x64|
|4x4, stride=2, padding=1, conv 1ReLU|64|16x16x64|
|3x3, stride=1, padding=1, conv 1ReLU|128|16x16x128|
|4x4, stride=2, padding=1, conv 1ReLU|128|8x8x128|
|3x3, stride=1, padding=1, conv 1ReLU|256|8x8x256|
|4x4, stride=2, padding=1, conv 1ReLU|256|4x4x64|
|3x3, stride=1, padding=1, conv 1ReLU|512|4x4x512|
|dense $\rightarrow 1$||1|

表 2. 判别器网络框架。SVHN 和 CIFAR-10 使用 $M=32$，STL-10 使用 $M=48$ 。



# 4. SA-GAN

论文：[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)

将 self-attention 应用到 GAN 中。传统的基于 conv 的 GAN，仅利用了特征面商的局部信息。SA-GAN 则通过 attention 模块利用特征面上所有的信息以生成图像细节。

