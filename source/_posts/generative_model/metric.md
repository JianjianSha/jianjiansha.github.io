---
title: 生成模型的衡量指标
date: 2022-08-12 15:33:50
tags: generative model
mathjax: true
---

# 1. IS

Inception score

**图片质量**

使用 Inception 网络对图像进行分类（针对 ImageNet 数据集），计算图像 $x$ 对应标签 $y$ 的概率 $p(y|x)$。记全部分类数量为 $C$，那么图像分类网络最后一般是 softmax 层，输出一长度为 $C$ 的向量，表示各分类的概率。

如果 Inception 网络能以较高的概率预测图片的分类，即某一分类的预测概率应该明显大于其他分类的预测概率，这就说明图片质量较高，反之说明图片质量较低。

**多样性**

计算边缘概率

$$p(y)=\int_z p(y|x=G(z))dz \tag{1}$$

如果生成图像的多样化很好，那么标签 $y$ 的分布 $p(y)$ 具有较高的熵，即 $p(y)$ 分布更均匀。

总结：

1. 图片质量。给定一个图片 $x$，Inception 分类网络预测其分类为 $y$，且预测概率 $p(y|x)$ 越大越好，即条件熵越小越好。
2. 图片多样性。此时需要考虑标签的边缘分布 $p(y)$，而非条件分布，边缘分布越均匀，那么多样性越好，即熵越大越好。

综合图片质量和多样性，给出 IS 计算公式

$$\text{IS}(G)=\exp \{ E_{x \in p} D_{KL}(p(y|x)||p(y))\} \tag{2}$$

$G$ 表示生成器 Generator，$x$ 表示生成图像，$y$ 表示图像某标签。

解释：

$$\begin{aligned}E_{x \in p} D_{KL}(p(y|x)||p(y))&=E_{x\in p} E_{p(y|x)}\log p(y|x)-\log p(y)
\\\\ &=E_{p(x,y)} \log p(y|x) - E_{p(x,y)}\log p(y)
\\\\ &=H(y)-H(y|x)
\end{aligned} \tag{3}$$

条件熵越小，标签分布的熵越大，那么 (3) 式越大，也就是 IS 值越大，根据前面的总结的两点内容可知，**IS 越大越好** 。

**算法描述**：

生成数据集所有图像 $\{x_i\}_{i=1}^N$ 经过 Inception 预测，得到预测分类 $\{y_i\}_{i=1}^N$，以及每个预测概率 $\{p(y_i=c|x_i)\}_{i=1,c=1}^{N,C}$ ，每个数据 $x_i$ 得到 $C$ 个分类预测概率，得到一个概率矩阵 $P_{N, C}$

边缘分布 $p(y=c)=\frac 1 N \sum_{i=1}^N p(y_i=c|x_i)$，即，概率矩阵得列和 $\sum P_{:,c}$

期望计算过程：

$$E_{p(x,y)}\log p(y)=\sum_{x,y} p(x)p(y|x)\log p(y)=\frac 1 N\sum_{i=1}^N\sum_{c=1}^C p(y_i=c|x_i)\log p(y_i=c)$$

$$E_{p(x,y)} \log p(y|x)=\frac 1 N \sum_{i=1}^N \sum_{c=1}^C p(y_i=c|x_i) \log p(y_i=c|x_i)$$

# 2. FID

Fréchet Inception Distance

同样使用 Inception 网络。在原 Inception 分类网络中，最后一层使用 linear 层， 将 feature channel 改为分类数量 $C=1000$，我们去掉这个 linear，得到一个高层 feature，维度记为 $n$。对于真实图像数据，这个高层 feature 满足某个分布，对于生成模型生成的样本，经 Inception 也得到一个高层 feature，这个 feature 也满足某个分布，我们的目标是使得两个高层特征的分布尽量相同。假设相同，那么生成图像的真实性和多样性和训练数据相同。使用 FID 衡量两个分布的距离，显然 **FID 越小越好**，表示两个分布越相近。

**Fréchet Distance**

考虑高斯分布。高斯分布可以使用期望和协方差矩阵来确定，那么当期望和协方差矩阵相同时，两个高斯分布就相同。高斯分布之间的距离则使用均值和协方差矩阵计算。假设高层 feature 维度为 $n$，那么均值维度为 $n$，协方差矩阵维度为 $n\times n$，FID 计算如下

$$\text{FID}(x,g)=\|\mu_x-\mu_g\|_2^2 + Tr(\Sigma_x+\Sigma_g-2(\Sigma_x\Sigma_g)^{1/2})$$

其中 $x$ 表示真实图片，$g$ 表示生成图片。

解释：

$\|\mu_x-\mu_g\|_2^2$ 越小，说明特征均值越接近，说明图像质量较高。

$$\begin{aligned}Tr(\Sigma_x+\Sigma_g-2(\Sigma_x\Sigma_g)^{1/2})&=Tr(\Lambda_x +\Lambda_g - 2(\Lambda_x\Lambda_g)^{1/2})
\\\\ &=\sum_{i=1}^n (\sqrt {\lambda_{xi}}-\sqrt {\lambda_{gi}})^2
\end{aligned}$$

故 $Tr(\Sigma_x+\Sigma_g-2(\Sigma_x\Sigma_g)^{1/2})$ 越小，说明在 n 个维度上，真实图像数据的方差和生成图像数据的方差均接近，方差接近表示熵接近，多样性接近。

通过上面两点分析可知， FID 越小越好。

```python
act = get_activations(...)  # 获取 InceptionV3 高层特征，shape (batch_size, 2048)
mu = np.mean(act, axis=0)   # (2048,)
sigma = np.cov(act, rowvar=False) # rowvar=False -> 每列表示一个变量，共 2048 个变量
                                  #每个变量有 batch_size 个取值，sigma shape (2048,2048)
```

对真实图片和生成图片，分别经过上述过程得到 $(\mu_x, \Sigma_x)$ 和 $(\mu_g, \Sigma_g)$ ，然后计算 Frechet distance 代码如下，

```python
diff = mu1-mu2

stdcov, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
if not np.isfinit(stdcov).all():
    offset = np.eye(sigma1.shape[0]) * eps
    stdcov = scipy.linalg.sqrtm((sigma1+offset).dot(sigma2+offset))

if np.iscomplexobj(stdcov): # 存在复数
    # 如果虚部不全部很小
    if not np.allclose(np.dianonal(stdcov).imag, 0, atol=1e-3):
        m = np.max(np.abs(stccov.imag)) # 取最大虚部
        raise ValueError('Imaginary component {}'.format(m))
    stdcov = stdcov.real

tr = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(stdcov)
fid = diff.dot(diff) + tr
```