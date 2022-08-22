---
title: 相似性度量
date: 2022-04-28 13:40:21
tags: machine learning
---

# 1. 欧氏距离

$$D=\|\mathbf x_1-\mathbf x_2\|_2=[(\mathbf x_1-\mathbf x_2)^{\top}(\mathbf x_1-\mathbf x_2)]^{1/2}$$

# 2. 曼哈顿距离
又称城市街区距离

$$D=\|\mathbf x_1-\mathbf x_2\|_1=\sum_i |x_{1i}-x_{2i}|$$

# 3. 切比雪夫距离

$$D=\|\mathbf x_1-\mathbf x_2\|_{\infty}=\max_i \ |x_{1i}-x_{2i}|$$

# 4. 闵可夫斯基距离

$$D=\|\mathbf x_1-\mathbf x_2\|_p, \quad p \in \mathbb R^+$$

# 5. 标准化欧氏距离

数据各维度标准化后再计算欧氏距离，标准化过程为

$$x=\frac {x-\mu} v$$

标准化欧氏距离为

$$D=\left(\sum_i \left(\frac {x_{1i}-x_{2i}} {v_i}\right)^2\right)^{1/2}$$

# 6. 马氏距离

Mahalanobis Distance

有 n 个样本 $\mathbf x_1, \ldots, \mathbf x_n$，且 $\mathbf x_i \in \mathbb R^d$（不一定独立同分布），协方差矩阵记为 $S$，协方差计算公式

$$V=E[(X-E[X])(X-E[X])^{\top}]$$

这 m 个样本的均值为 $\mu$，样本协方差计算为 

$$S=\frac n {n-1} V$$

```python
# 计算样本协方差
import numpy as np
n=10
d=2
# initialize n d-dim samples
x = np.random.randn(d, n)
# calc cov matrix
s = np.cov(x)
```

定义样本向量 $\mathbf x$ 到 $\mu$ 的马氏距离为

$$D(\mathbf x)=\sqrt {(\mathbf x-\mu)^{\top} S^{-1}(\mathbf x-\mu)}$$

$\mathbf x_i, \ \mathbf x_j$ 之间的马氏距离为 

$$D(\mathbf x_i, \mathbf x_j)=\sqrt {(\mathbf x_i -\mathbf x_j)^{\top} S^{-1}(\mathbf x_i-\mathbf x_j)}$$

当 $\mathbf x_1, \ldots, \mathbf x_n$ 独立同分布，那么 $S=I$ 为单位矩阵，此时马氏距离变为欧氏距离。

# 7. 夹角余弦

$$D=\cos \theta=\frac {\mathbf x_1^{\top}\mathbf x_2}{|\mathbf x_1| \cdot|\mathbf x_2|}$$

# 8. 汉明距离
> 两个等长字符串对应位不同的数量（或占总长的比例）

```python
x = 1
y = 4
s = x ^ y
c = 0 # 统计数量，汉明距离
while s > 0:
    if s & 1 == 1:
        c += 1
    s = s >> 1
```

# 9. Jaccard 相似系数

两个集合 $A, B$，Jaccard 相似系数

$$J(A, B)=\frac {|A \cap B|} {|A \cup B|}$$

Jaccard 距离

$$J_{\delta}(A, B)=1-J(A,B)$$

## 10. 相关系数

$$\rho_{XY}=\frac {Cov(X,Y)}{\sqrt {D(X) \cdot D(Y)}}$$

相关距离

$$D=1-\rho_{XY}$$

## 11. 信息熵

定义

$$E(X)=-\sum_i p_i \log p_i$$

交叉熵可看作一种距离，记 $p$ 为预测概率分布，$q$ 为真实概率分布，那么交叉熵为

$$CE=-\sum_i q_i \log p_i$$