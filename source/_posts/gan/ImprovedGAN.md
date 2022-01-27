---
title: ImprovedGAN
date: 2019-08-01 15:48:46
tags: GAN
mathjax: true
---
标题 [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)

源码 [improved_gan](https://github.com/openai/improved_gan)
<!-- more -->
# 简介
GAN 是基于博弈论学习生成模型的一类方法总称。GAN 目的是训练一个生成网络其生成样本的分布可以拟合真实数据分布。虽然 [DCGAN](2019/07/23/GAN) 在 GAN 中引入 conv+BN+ReLU 在一定程度上改善了生成器，但是我们认为 GAN 这个零和博弈问题具有高维参数且非凸，需要达到 Nash 均衡才是最佳解，而传统的基于目标函数梯度下降方法目的并非用于寻找 Nash 均衡。本文提出了以下改进方法：
1. 特征匹配
2. 小批量特征
3. 虚拟批归一化

# GAN 训练收敛
训练 GAN 意味着寻找二人非合作博弈中的 Nash 均衡，每个玩家希望最小化自己的损失函数，即生成器损失 $J^{(G)}(\mathbf {\theta}^{(D)}, \mathbf {\theta}^{G})$ 和判别器损失 $J^{(D)}(\mathbf {\theta}^{(D)}, \mathbf {\theta}^{G})$，Nash 均衡指 $J^{(D)}$ 关于 $\theta^{(D)}$ 最小，同时 $J^{(G)}$ 关于 $\theta^{(G)}$ 最小。寻找 Nash 均衡点是一个比较困难的问题，虽然某些特殊情况下有算法可以解决，但是由于这里损失函数非凸且参数维度很高，那些方法均不适用。以上 Nash 均衡的说明让我们从直觉上认为需要同时最小化 G 和 D 的损失。但是很不幸，更新 $\theta^{(D)}$ 降低 $J^{(D)}$ 却增大 $J^{(G)}$，更新 $\theta^{(G)}$ 以降低 $J^{(G)}$ 但是会增大 $J^{(D)}$。这就是导致梯度下降法难以收敛（往往是在一个轨道上一直反复，而不会到达最佳点）。例如一个玩家根据 x 来最小化 xy，另一个玩家根据 y 来最小化 -xy，梯度下降法更新会使得 x 和 y 值构成的点在一个椭圆上往复，而不会收敛到 x=y=0。本文介绍以下方法启发式的促使更新达到收敛。

## 特征匹配
特征匹配使用新的生成器损失函数以解决 GAN 训练不稳定问题。新的目标函数不是直接最大化 D 的输出（G 原本的目标是让 D 对生成样本有越大越好的输出），而是让 G 生成的样本能够匹配真实数据的统计量，这是一种更直接的思想。具体而言，训练 G 以匹配特征的期望值，这个特征来自于 D 的网络中间层。令 $\mathbf {f(x)}$ 表示 D 网络中间层的激活响应，即前面所指的特征，那么 G 的新目标函数为
$$\|\Bbb E_{\mathbf x \sim p_{data}} \mathbf {f(x)}-\Bbb E_{\mathbf z \sim p_{\mathbf z}}\mathbf f(G(\mathbf z))\|_2^2$$
G 的训练目标就是最小化这个目标损失。

## 小批量判别
GAN 训练失败的原因之一是生成器训练时总是会陷入一组参数无法逃脱，我这里称其为“陷入点”，当临近“陷入点”时，生成器的输出点总是很相似，而这些相似的点会让判别器总是指向一个差不多的方向，由于判别器 __独立处理__ 每个样本，这些样本对应的梯度相互之间无法合作，缺乏一种反馈机制去通知生成器让其输出相互之间尽可能不相似，生成器所有的输出都向同一个点竞争，这个点是为了让判别器判别为真实的数据，所以结果就是生成器陷入一组模型参数无法自拔，陷入之后，判别器通过学习又能够将这个点判别为来自生成器，但是梯度 __无法区分__ 各个不同的输出，于是判别器的梯度会一直在空间中将生成器产生的这个“陷入点”推来推去，导致算法无法收敛。一种显然的解决办法是让判别器不独立处理每个样本，而是一次能看到多个样本的合并，这就是小批量判别方法。

现在我们的实验建模瞄准于区分生成器的各个相互靠得很近得样本。小批量中样本之间接近程度按如下方法计算：  
令 $\mathbf {f(x_i)} \in \Bbb R^A$ 表示输入 $\mathbf x_i$ 对应的特征向量，这个特征由 D 网络中间层产生，然后将特征向量乘以一个张量 $T \in \Bbb R^{A \times B \times C}$，结果是一个矩阵 $M_i \in \Bbb R^{B \times C}$，对于输入样本编号 $i \in \{1,...,n\}$，得到对应的矩阵 $\{M_i |i=1,...,n\}$，计算两两矩阵的各行向量之间的 L1 距离，然后应用负指数函数，
$$c_b(\mathbf x_i, \mathbf x_j)=\exp(-\|M_{i,b}-M_{j,b}\| _ {L_1}) \in \Bbb R, \quad i,j \in \{1,...,n\}, \quad b \in \{1,...,B\}$$

其中下标 b 表示 row index。如图 1，minibatch layer 中样本 $\mathbf x_i$ 对应的输出定义为，
$$\begin{aligned} &o(\mathbf x_i) _ b = \sum_{j=1}^n c _ b(\mathbf x_i, \mathbf x_j) \in \Bbb R
\\\\ &o(\mathbf x_i)=\left[o(\mathbf x_i) _ 1,...o(\mathbf x_i) _ B \right] \in \Bbb R^B
\\\\ &o(\mathbf X) \in \Bbb R^{n \times B} \end{aligned}$$

然后，将 minibatch layer 的输出 $o(\mathbf x_i)$ 与 minibatch layer 的输入 $\mathbf {f(x_i)}$ concatenate 起来，作为 D 的下一 layer 的输入。对生成样本和训练数据分别计算 minibatch layer 特征。
![](/images/ImprovedGAN_fig1.png)

## 历史平均
修改每个玩家（G 和 D）的损失使得包含 $\|\mathbf \theta -\frac 1 t \sum_{i=1}^t \theta[i]\|^2$，其中 $\theta[i]$ 是历史时期 i 的参数值。

## 单边标注平滑
Label 平滑，就是将分类器的 target 值由 0 和 1 替换为一个平滑的值如 0.9 或 0.1。我们将正例 target 替换为 $\alpha$，负例 target 替换为 $\beta$，那么最佳判别器变为
$$D(\mathbf x)=\frac {\alpha p_{data}(\mathbf x) + \beta p_{model}(\mathbf x)}{p_{data}(\mathbf x)+p_{model}(\mathbf x)}$$

- 当 $p_{data}(\mathbf x) \gg p_{model}(\mathbf x)$ 时，$D(\mathbf x) \rightarrow \alpha$
- 当 $p_{data}(\mathbf x) \ll p_{model}(\mathbf x)$ 时，$D(\mathbf x) \rightarrow \beta$

当然我们也可以按 [GAN](2019/7/23/GAN) 中那样推导 $D^{\ast}$，推导过程这里略过，只是此时目标变为
$$\min_G \max_D V(D,G)=\Bbb E_{x \sim p_{data}(x)}[\log (D(x)-\beta)] + \Bbb E_{z \sim p_z(z)}[\log(\alpha-D(G(z)))] \quad (1)$$

这里约定正例 target 大于负例 target，即 $\alpha > \beta$，由 (1) 式，可知 D 输出范围为 $\beta < D(x) < \alpha$。

由于分子中出现 $p_{model}$，那么当 $p_{data} \rightarrow 0$，且 $p_{model}$ 足够大时，来自 $p_{model}$ 的错误样本将得不到促使向真实数据靠近的激励，所以只对正例 label 平滑处理为 $\alpha$，负例 label 依然为 0。

## 虚拟批归一化
DCGAN 中使用了批归一化 BN 使得网络优化更加有效，但是也会带来问题，比如一个输入 $\mathbf x$，其对应的输出高度依赖于同一 minibatch 中的其他输入 $\mathbf x'$。为了避免这个问题，本文使用了虚拟批归一化 VBN，每个样本输入 $\mathbf x$ 的归一化过程基于 reference batch 中样本的统计量以及 $\mathbf x$ 自身，reference batch 是在训练初期选定并固定不变，reference batch 使用统计量进行归一化。由于 VBN 计算强度较高，故只在 G 网络中使用。

# 图像质量评估
GAN 的性能评估最直接的方法就是人类观察员判断，缺点是难以公平公正。本文提出了一个自动评估方法：应用 Inception 模型到每个生成样本上，以获得条件 label 分布 $p(y|\mathbf x)$，那些包含有意义目标的图像的条件 label 分布 $p(y|\mathbf x)$ 应该具有较低的信息熵，也就是说，具有较低的不确定性，这意味着，对于给定的输入 $\mathbf x$（包含有意义目标的图像），Inception 模型每次输出值 y （比如图像分类 c）比较稳定变化很小。但是我们又希望生成模型能够生成各种不同的图像，即对于不同的噪声输入 z，G 能够生成各种不同的图像，分别以这些不同的图像作为输入， Inception 模型的输出也尽可能不同（不确定性较大），这说明 $\int p(y|\mathbf x=G(z)) dz$ 应该具有较大的信息熵。结合以上这两点要求，性能指标为这两个分布 KL 散度的期望，
$$\exp [\Bbb E_{\mathbf x} \mathbf {KL}(p(y|\mathbf x)\|p(y)) ]$$

应用指数函数仅仅是为了便于比较值的大小。

# 半监督学习
考虑一个标准分类器，输入为 $\mathbf x$，共有 K 种类别，输出为长度 K 的向量 $[l_1,...,l_K]$，表示每个类别的得分，通过 softmax 得到对应的概率：
$$p_{model}(y=j|\mathbf x)=\frac {\exp l_j} {\sum_{k=1}^K \exp l_k}$$

在监督学习中，此模型的训练是最小化交叉熵（或最大化 log 似然函数）。

增加来自生成器 G 的样本到数据集中，可以实现标准分类器的半监督学习，G 生成样本标记类别 y=K+1，分类器的输出维度改为 K+1，利用 $p_{model}(y=K+1|\mathbf x)$ 判断输入 $\mathbf x$ 是生成样本的概率，与 GAN 中的 $1-D(\mathbf x)$ 是对应的。也可以使用未标注数据进行学习，对于来自 K 个类别的真实数据，需要最大化 $\log p_{model}(y \in \{1,...,K\}|\mathbf x)$（log 似然函数），假设数据集中一半是真实数据，一半是生成数据，那么分类器训练的损失函数为，
$$\begin{aligned} &L=-\Bbb E_{\mathbf x,y \sim p_{data}(\mathbf x,y)}[\log p_{model}(y|\mathbf x)] - \Bbb E_{\mathbf x \sim G} [\log p_{model}(y=K+1|\mathbf x)]=L_{supervised}+L_{unsupervised}
\\\\ &L_{supervised}=-\Bbb E_{\mathbf x,y \sim p_{data}(\mathbf x,y)} \log p_{model}(y|\mathbf x, y <K+1)
\\\\ &L_{unsupervised}=-\Bbb E_{\mathbf x \sim p_{data}(\mathbf x)} \log[1- p_{model}(y=K+1|\mathbf x)] - \Bbb E_{\mathbf x \sim G} [\log p_{model}(y=K+1|\mathbf x)]\end{aligned}$$

其中求期望实际上是经验期望也就是均值损失。其中 $L_{unsupervised}$ 就是标准 GAN 的 objective，在 $L_{unsupervised}$ 中作替换 $D(\mathbf x)=1-p_{model}(y=K+1|\mathbf x)$，就更明显了,于是有
$$L_{unsupervised}=-\Bbb E_{\mathbf x \sim p_{data}(\mathbf x)} \log D(\mathbf x) - \Bbb E_{z \sim noise} \log (1-D(G(z)))$$

最小化 $L_{supervised}$ 和 $L_{unsupervised}$ 的最优解是满足 $\exp[l_j(\mathbf x)]=c(\mathbf x) p(y=j,\mathbf x), \ \forall j \in K+1$ 以及 $\exp[l_{K+1}(\mathbf x)]=c(\mathbf x) p_G(\mathbf x)$，其中 $c(\mathbf x)$ 是待定的系数函数。训练 G 以近似真实的数据分布，一种训练方法是最小化 GAN objective，使用这里的分类器作为判别器 D，这种方法引入了 G 和分类器之间的相互作用，经验表明，在半监督学习中，使用特征匹配 GAN 可以很好的优化 G。

分类器输出维度为 K+1 是过参数化的，由于输出向量中每个元素值均减去同一个值 $l_j(\mathbf x)\leftarrow l_j(\mathbf x)-f(\mathbf x)$，对 softmax 的值并不影响，所以可固定 $l_{K+1}(\mathbf x)=0, \ \forall \mathbf x$，于是 $L_{supervised}$ 变为具有 K 个类别的原始分类器的标准监督损失，此时判别器 D 为 $D(\mathbf x)=\frac {Z(\mathbf x)} {Z(\mathbf x)+1}$，其中 $Z(\mathbf x)=\sum_{k=1}^K \exp [l_k(\mathbf x)]$。