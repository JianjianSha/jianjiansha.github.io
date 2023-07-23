---
title: Beta 混合模型
date: 2023-07-13 18:25:25
tags: machine learning
mathjax: true
---

与高斯混合模型 GMM 类似，各分量还可以是 Beta 分布，这就是 Beta 混合模型（BMM）。

Beta 分布的 PDF 为

$$p(x;\alpha, \beta) = \frac {\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} x ^ {\alpha -1} (1- x) ^ {\beta -1}, \quad x \in [0, 1] \tag{1}$$


BMM 定义

$$p(x|\theta) = \sum _ {k=1} ^ K \pi _ k p(x|\theta _ k) \tag{2}$$

其中 $0 \le \pi _ k \le 1$ 且有 $\sum _ {k=1} ^ K \pi _ k = 1$ ，$\theta _ k$ 表示第 $k$ 个 Beta 分布的参数。

对数据集 $X = \lbrace x _ n \rbrace _ {n=1} ^ N$，使用 BMM 拟合，最大似然为

$$l(X|\theta) = \prod _ {n=1} ^ N \sum _ {k=1} ^ K \pi _ k p(x _ n | \theta _ k ) \tag{3}$$

为了计算方便，通常取对数似然，得

$$L(X|\pi, \alpha , \beta ) = \sum _ {n=1} ^ N \log \left( \sum _ {k=1} ^ K \pi _ k p(x _ n | \alpha _ k, \beta _ k) \right) \tag{4}$$

求最大对数似然估计，可以通过对参数求偏导解得，但是由于 log 内部有求和项，所以无法求得最佳解析解，下面介绍通过 EM 算法求数值解。

本文中，有时候将模型具体的参数 $\pi, \alpha , \beta$ 统一称为参数 $\theta$，且 $p(x)$ 与 $p(x|\theta)$ 含义相同，而 $p(x|\theta _ k)$ 和 $p(x|\alpha _ k, \beta _ k)$ 表示第 $k$ 个分量。

## 1. EM 求解 BMM

使用 EM 求解 GMM 时，我们知道，给定 responsibility，可以得到各参数的闭式解（参考文章 [GMM](/2021/11/13/ml/GMM)），这里 responsibility 为后验概率

$$r_{nk}= p(z_k=1|x _ n)=\frac {p(x _ n, z _ k =1)}{p (x _ n)} = \frac {\pi _ k p(x _ n |\alpha _ k, \beta _ k)}{p(x _ n)}\tag{5}$$

然而对于 BMM，给定 responsibility 也无法得到参数的闭式解。

<details><summary>没有闭式解的原因</summary>

求单个数据对参数 $\alpha _ k$ 的梯度，

$$\begin{aligned}\frac {\partial p(x|\theta)}{\partial \alpha _ k} &=\sum _ {j=1} ^ K \pi _ j \frac {\partial p(x|\theta _ j)}{\partial \alpha _ k}
\\\\ &= \pi _ k \frac {\partial p(x|\theta _ k)}{\partial \alpha _ k}
\\\\ &=\pi _ k (1-x) ^ {\beta -1} [ f(\alpha _ k, \beta _ k)' x ^ {\alpha -1}+ f(\alpha _ k, \beta _ k) x ^ {\alpha - 1} \log x ]
\\\\ &= \pi _ k x ^ {\alpha - 1} (1-x) ^ {\beta -1}[f(\alpha _ k, \beta _ k)' + f(\alpha _ k, \beta _ k) \log x]
\end{aligned}$$

其中

$$f(\alpha _ k, \beta _ k)=\frac {\Gamma(\alpha _ k +\beta _ k)}{\Gamma (\alpha _ k) \Gamma(\beta _ k)}$$

另一方面，

$$f(\alpha _ k, \beta _ k)' = \frac {\Gamma '(\alpha _ k + \beta _ k) \Gamma(\alpha _ k)-\Gamma(\alpha _ k + \beta _ k) \Gamma '(\alpha _ k)}{\Gamma ^ 2 (\alpha _ k) \Gamma(\beta _ k)}$$

Gamma 函数以及其导数分别为，

$$\Gamma(x) = \int _ 0 ^ {\infty} e ^ {-t} t ^ {x-1} dt
\\\\ \Gamma'(x) = \int _ 0 ^ {\infty} e ^ {-t} t ^ {x-1} \log t  \ dt$$

代入上式，得

$$\frac {\partial p(x|\theta)}{\partial \alpha _ k} = \left(\frac {\Gamma '(\alpha _ k + \beta _ k)}{\Gamma (\alpha _ k + \beta _ k)} - \frac {\Gamma '(\alpha _ k)}{\Gamma (\alpha _ k)} +  \pi _ k \log x \right) p(x|\alpha _ k, \beta _ k)$$

数据集对数似然对参数 $\alpha _ k$ 的梯度为，

$$\begin{aligned}\frac {\partial L}{\partial \alpha _ k} &= \sum _ {n=1}^N \frac {\partial \log p(x _ n |\theta)}{\partial \alpha _ k} = \sum _ {n=1} ^ N \frac 1 {p(x _ n | \theta)} \frac {\partial p(x _ n |\theta)}{\partial \alpha _ k} 
\\\\ &= \sum _ {n=1} ^ N \left(\frac {\Gamma '(\alpha _ k + \beta _ k)}{\Gamma (\alpha _ k + \beta _ k)} - \frac {\Gamma '(\alpha _ k)}{\Gamma (\alpha _ k)} +  \pi _ k \log x \right) \frac {p(x|\alpha _ k, \beta _ k)} {p(x _ n |\theta)}
\\\\ &=  \sum _ {n=1} ^ N \left(\frac {\Gamma '(\alpha _ k + \beta _ k)}{\Gamma (\alpha _ k + \beta _ k)} - \frac {\Gamma '(\alpha _ k)}{\Gamma (\alpha _ k)} +  \pi _ k \log x \right) \frac {r _ {nk}} {\pi _ k}
\end{aligned}$$

令上式梯度为 0，由于 $\Gamma'$ 和 $\Gamma$ 中变量 $\alpha _ k$ 存在于积分内，所以无法求得 $\alpha _ k$ 的闭式解。

同理，对于 $\beta _ k$ 也是如此，无法求得闭式解。
</details>

<br/>

但是对于混合参数 $\pi _ k$，由于其不存在于 $\Gamma$ 函数中，所以可以求其闭式解，过程与 [GMM](/2021/11/13/ml/GMM) 一文中解法相同，这里仅给出结论如下，

$$\pi _ k ^ {new} = \frac {N _ k}{N} = \frac {\sum _ {n=1} ^ N r _ {nk}} N \tag{6}$$

整体的求解思路为：

1. 初始化参数 $\theta=(\pi, \alpha, \beta)$，计算 responsibility 矩阵

2. 按 (6) 式计算混合参数 $\pi$

3. 保持 $\pi$ 不变，数值法求混合分量参数 $\alpha, \beta$ 的最大似然估计

    SGD 数值法求解：将数据集 $X$ （或者以 mini-batch 方式）按 (4) 式计算负对数似然，然后使用 pytorch 等自动求解梯度框架，反向传播计算梯度，然后更新参数，这个过程执行 `epochs` 次。

4. 跳至步骤 `1`，继续迭代更新。

**# 参数的初值设置**

数值法需要设置参数的初值，混合参数 $\pi$ 是有范围的即，非负，且和为 `1`，那么可以设置 $\pi _ k = 1/K$ 。混合分量的参数 $\alpha, \beta$ 要求为正实数，所以不太容易找到合适的初值。一种方法是使用矩方法（Method of Moments）得到初值。

下面以一个简单的例子予以说明。

BMM 使用两个分量 $K=2$，第一个为 `Beta(1,5)`，从中生成 `40` 个数据，第二个为 `Beta(10, 2)`，从中生成 `60` 个数据。

混合权重参数初值选为 $\pi _ 1 = \pi _ 2 =0.5$。

将数据排序，然后分为两半，得到两个新的数据集，每个数据集用于拟合一个 Beta 分量，计算矩估计。

已知 Beta 分布的期望和方差为，

$$\mu = \frac {\alpha} {\alpha + \beta}, \quad V = \frac {\alpha \beta} {[(\alpha + \beta) ^ 2 (\alpha + \beta + 1)]} \tag{7}$$

使用样本期望和样本方差代替分布的期望和方差，即

$$\overline x = \frac {\alpha} {\alpha + \beta}, \quad s ^ 2=\frac {\alpha \beta} {[(\alpha + \beta) ^ 2 (\alpha + \beta + 1)]} \tag{8}$$

解 (8) 式，得到 $\alpha, \beta$ 的估计，

$$\beta = \frac {\alpha (1-\overline x)}{\overline x}, \quad \alpha = \overline x \left[\frac {\overline x (1- \overline x)} {s ^ 2} -1 \right] \tag{9}$$

以下是示例代码，

```python
import numpy as np
K = 2
a1, b1 = 1, 5
a2, b2 = 10, 2
n1, n2 = 40, 60
p1 = np.random.beta(a1, b1, (n1,))
p2 = np.random.beta(a2, b2, (n2,))
p = np.concatenate([p1, p2])
p = np.sort(p)
p1 = p[:(n1+n2)//2]
p2 = p[(n1+n2)//2:]

def mom(data):  # method of moment
    mu = np.mean(data)
    var = np.var(data, ddof=1)  # 样本方差，除以 n-1
    im = mu * (1 - mu) / var - 1
    alpha = mu * im
    beta = (1 - mu) * im
    return alpha, beta

s = mom(p1) # (alpha, beta) for 50% smaller data
l = mom(p2) # (alpha, beta) for 50% larger data
```

这样就得到两个分量的参数初值。

**# MLE 数值法**

保持混合权重参数值不变，使用 MoM 估计的分量分布参数，然后开始获取分量分布参数的 MLE 。

定义一个类，实现 BMM，

```python
import torch
import torch.optim as optim
from torch.distributions.beta import Beta

class BMM:
    def __init__(self, alpha, beta, pi):
        # (K, )
        self.alpha = alpha
        self.beta = beta
        self.pi = pi
        self.beta = Beta(alpha, beta)
    
    def log_prob(self, x):
        '''x: (N, )'''
        # (N, K)
        p = self.beta.log_prob(x).exp().float()
        p = torch.matmul(p, self.pi)    # (N, )
        return torch.log(p)     # 使用 log ，计算稳定
    
    def responsibility(self, x):
        # p(x_n|alpha_k, beta_k)
        p = self.beta.log_prob(x).exp().float() # (N, K)
        num = p * self.pi.view(-1, 1)       # (N, K)
        denom = torch.matmul(p, self.pi)    # (N, )
        return num / denom.view(-1, 1)      # (N, K)
```

损失函数为负对数似然（reduce 为 `mean`，求均值）。调用示例为，

```python
data = torch.from_numpy(p)  # array 转为 tensor
data = data.repeat(2, 2).T
pi = torch.ones((2,)) / K
alpha = torch.tensor([s[0], l[0]], requires_grad=True)
beta = torch.tensor([s[1], l[1]], requires_grad=True)
bmm = BMM(alpha, beta, pi)

optimizer = optim.Adam([alpha, beta], lr=1e-1)

for i in range(300):
    optimizer.zero_grad()
    loss = bmm.log_prob(data)
    loss = -loss.mean()
    loss.backward()
    optimizer.step()
    bmm = BMM(alpha, beta, pi)

print('Fact:', (a1, b1), (a2, b2))
print('MoM:', s, l)
print('MLE:', (alpha[0].item(), beta[0].item()), (alpha[1].item(), beta[1].item()))
```

输出结果为，

```sh
Fact: (1, 5) (10, 2)
MoM: (0.6438045439935532, 1.5347664957218974) (31.44035776827936, 3.8002839879732564)
Iter: (0.9055941719028626, 3.419914290216871) (15.605259617435333, 2.3167238773896996)
pi: [0.41805458, 0.5819454] # 这个结果是下方代码输出
```

可见，经过 MLE 之后，分量分布参数值明显改善了很多。

**# 根据 responsibility 计算混合权重参数**

根据 (5) 和 (6) 式更新混合权重参数 $\pi$，直接看代码实现，

```python
r = bmm.responsibility(x)   # (N, K)
pi = r.mean(0)
print('pi:', list(pi))
```

接着就是使用上面更新后的 $\pi$ 值，然后继续执行 MLE 更新 Beta 分布的参数值，然后再根据 responsibility 更新混合权重参数 $\pi$，这样一直迭代更新。

