---
title: Noise-contrastive estimation
date: 2023-08-06 20:10:47
tags: contrastive learning
mathjax: true
---

论文：[Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)

# 1. 简介

假设从一个未知的分布 $p _ d (\cdot)$ 中采样，得到一个样本 $\mathbf x \in \mathbb R ^ n$。我们使用函数族 $\lbrace p _ m(\cdot) \rbrace _ {\alpha}$ 建模，模型参数为 $\alpha$，假设最优解为 $\alpha ^ {\star}$ 。

模型的任意解 $\hat \alpha$ 均需要满足归一化条件

$$\int p _ m (\mathbf u; \hat \alpha) d \mathbf u= 1 \tag{1}$$

通过一个配分函数

$$p _ m (\cdot; \alpha) = \frac {p _ m ^ 0 (\cdot;\alpha)}{Z(\alpha)}, \quad Z(\alpha)=\int p _ m ^ 0(\mathbf u; \alpha)d \mathbf u \tag{2}$$

这样非归一化分布 $p _ m ^ 0 (\cdot; \alpha)$ 就没有归一化条件限制，但是 $Z(\alpha)$ 中的积分则是难以处理的，除非 $p _ m ^ 0 (\cdot; \alpha)$ 有解析解。

一种方法是将 $Z(\alpha)$ 看作模型的一个额外的参数，但是这种方法没办法使用 MLE（最大似然估计）求模型的参数解，因为求最大对数似然的时候，会使得 $Z(\alpha)$ 趋于 0，从而使得观察数据集对数似然最大，

$$L(\alpha, c)=\sum _ {i=1} ^ T \log p _ m (\mathbf x _ i; \alpha) =\sum _ {i=1} ^ T  \log p _ m ^ 0 (\mathbf x _ i;\alpha) + c$$

其中 $Z(\alpha)$ 作为模型的一个额外参数 $c=-\log Z(\alpha)$，显然为了最大化上式，容易令 $\beta \rightarrow 0$ 。

所以，需要找到一种方法，直接使用 $p _ m ^ 0(\cdot;\alpha)$ 估计模型，而不需要计算 $Z(\alpha)$ 中的积分。最近的解决方法有 contrastive divergence，score matching 等。

本文，作者提出一种新的非归一化模型，比以上提到的两种方法好。基本思想是通过学习判别数据 $\mathbf x$ 和噪声 $\mathbf y$ ，这种方法称为 noise-contrastive estimation（NCE）。

# 2. NCE

根据 (2) 式得 $\log p _ m(\cdot;\theta) = \log p _ m ^ 0(\cdot;\alpha) + c$，参数为 $\theta=(\alpha, c)$ 。

记观察数据集为 $X=(\mathbf x _ 1, \ldots, \mathbf x _ T)$，人工生成的噪声为 $Y=(\mathbf y _ 1, \ldots, \mathbf y _ T)$，噪声分布为 $p _ n(\cdot)$。最大化如下目标函数可以求解模型，

$$J _ T(\theta)=\frac 1 {2T} \sum _ {t=1} ^ T \log h(\mathbf x _ t;\theta)+\log [1-h(\mathbf y _ t; \theta)] \tag{3}$$

其中 

$$h(\mathbf u;\theta)=\frac 1 {1+\exp[-G(\mathbf u; \theta)]} \tag{4}$$

$$G(\mathbf u;\theta)=\log p _ m (\mathbf u; \theta) - \log p _ n(\mathbf u) \tag{5}$$

$p _ n (\mathbf u)$ 是我们事先选好的一个分布用于生成噪声，故可以看作是已知的。

合并观察数据和噪声之后的数据集为 $U=(\mathbf u _ 1, \ldots, \mathbf u _ {2T})$，从以上目标函数中可以看出，对于真实数据，希望 $h(\mathbf x; \theta)$ 越大越好；对于噪声数据，希望 $1-h(\mathbf y;\theta)$ 越大越好。而 $h (\mathbf u;\theta)=\sigma (G(\mathbf u;\theta))$，所以：

1. 对于真实数据，希望 $p _ m(\mathbf u;\theta)$ 尽量大
2. 对于噪声数据，希望 $p _ m(\mathbf u;\theta)$ 尽量小
3. 以上两点分析中，给定一个数据 $\mathbf u$，$p _ n(\mathbf u)$ 是固定不变的，我们要优化的是参数 $\theta$


## 2.1 与监督学习的联系

(3) 式在监督学习中也出现过，例如判别区分真实数据 $X$ 和噪声数据 $Y$，

1. 如果 $\mathbf u \in X$，那么标签 $C _ t = 1$
2. 如果 $\mathbf u \in Y$，那么标签 $C _ t = 0$

由于真实数据和噪声数据一样多，所以 $P(C=1)=P(C=2)=1/2$，标签条件概率为

$$p(\mathbf u|C=1;\theta)=p _ m(\mathbf u; \theta) \quad p(\mathbf u|C=0)=p _ n(\mathbf u) \tag{6}$$

于是后验分布为

$$\begin{aligned}p(C=1|\mathbf u;\theta)&= \frac {p(\mathbf u|C=1;\theta)p(C=1)}{p(\mathbf u|C=1;\theta)p(C=1)+p(\mathbf u|C=0;\theta)p(C=0)}
\\\\ &= \frac {p _ m(\mathbf u;\theta)}{p _ m(\mathbf u;\theta)+ p _ n(\mathbf u)}
\\\\ &=h(\mathbf u;\theta) \end{aligned} \tag{7}$$

$$p(C=0|\mathbf u;\theta) = 1-h(\mathbf u;\theta) \tag{8}$$

目标函数为对数似然，

$$\begin{aligned}l(\theta) &= \frac 1 {2T}\sum _ {t=1} ^ {2T} C _ t \log p(C _ t =1|\mathbf u _ t; \theta) + (1-C _ t) \log p(C=0|\mathbf u _ t;\theta)
\\\\ &= \frac 1 {2T} \sum _ {t=1} ^ T \log h(\mathbf x _ t;\theta) + \log [1-h(\mathbf y _ t;\theta)]
\end{aligned} \tag{9}$$

噪声分布应该与真实数据分布非常接近，否则的话，分类问题太容易，使得模型没有学习太多关于真实数据的信息

# 3. 仿真实验

## 3.1 非监督任务

假设数据 $\mathbf x \in \mathbb R ^ 4$ 来自独立成分分析（ICA）模型，

$$\mathbf x = A \mathbf s \tag{10}$$

4 个独立源组成 $\mathbf s$，均来自 Laplace 分布，期望为 `0`，方差为 `1` ，那么数据分布 $p _ d (\cdot)$ 为

$$\log p _ d (\mathbf x)=-\sum _ {i=1} ^ 4 \sqrt 2 |\mathbf b _ i ^ {\star} \mathbf x| + (\log |\det B ^ {\star} - \log 4|) \tag{11}$$

其中 $\mathbf b _ i ^ {\star}$ 为矩阵 $B ^ {\star} =A ^ {-1}$ 的行。

非归一化模型为

$$\log p _ m ^ 0 (\mathbf x; \alpha)=-\sum _ {i=1} ^ 4 \sqrt 2 |\mathbf b _ i  \mathbf x| \tag{12}$$

模型参数为 $\alpha \in \mathbb R ^ {16}$ 就是 $(\mathbf b _ 1, \ldots, \mathbf b _ 4)$ 。

归一化模型为，

$$\log p _ m(\mathbf x;\theta) = \log p _ m ^ 0(\mathbf x; \alpha) + c \tag{13}$$

其中 $c$ 是归一化模型参数之一。

模型参数的真实值为 

$$\alpha = (\mathbf b _ i ^ {\star}) _ {i=1} ^ 4$$

$$c = \log |\det B ^ {\star} - \log 4|$$

**# 估计方法**

选用噪声数据 $\mathbf y$ 来自高斯分布，其期望和协方差矩阵与 $\mathbf x$ 的相同，

$$\log p _ n (\mathbf y) = -\frac 1 2 (\mathbf y - \mu) ^ {\top} \Sigma ^ {-1}(\mathbf y - \mu) + const \tag{14}$$

通过最大化目标函数 (3) 式，可以求解参数 $\theta$ 的最优解。

## 3.2 监督任务

监督任务，将真实数据的 label 看作 $\mathbf x$。以图像分类任务为例，使用 MNIST 数据集，真实数据则为图像 label 即 one-hot 向量，真实数据分布也要改成条件数据分布 $p _ m (\mathbf x |\mathbf I;\theta)$ ，其中 $\mathbf I$ 为图像数据。

噪声数据按均匀分布生成 one-hot 向量 $\mathbf y$，噪声分布为 $p _ n (\mathbf y |\mathbf I)=1/10$（因为总共就 10 个 one-hot 向量）。

于是

$$h (\mathbf u;\theta) = \frac {p _ m (\mathbf x |\mathbf I;\theta)} {0.1 + p _ m (\mathbf x |\mathbf I;\theta)} \tag{15}$$

显然这个值范围为 $[0,1)$ 。

模型使用 MLP，最后一层输出 units 数量为 `10`（MNIST 有 10 个分类），最后用 sigmoid激活。

每次训练一批真实的样本时，产生同样数量的噪声样本进行训练。（相当于对于每张图片，先把真实标签放进去训练，再把噪声标签放进去训练）。

```python
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import pylab
images = mnist.train.images
input_num = 784
hidden_num = 40
output_num = 10
input_shape = [None,784]
n_classes = [None,10]
x = tf.placeholder(tf.float32,input_shape)
label = tf.placeholder(tf.float32, n_classes)
W = tf.Variable(tf.random_normal([input_num,hidden_num],0,0/input_num))
b = tf.Variable(tf.random_normal([hidden_num],0,1))
layer1 = tf.nn.relu(tf.matmul(x,W) + b)
W1 = tf.Variable(tf.random_normal([hidden_num,output_num],0,0/hidden_num))
b1 = tf.Variable(tf.random_normal([output_num],0,1))
layer2= tf.matmul(layer1,W1) + b1
y = tf.nn.sigmoid (layer2)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 对于标签为i的样本，这里计算的就是第i个输出，不考虑其他输出。
h = tf.reduce_sum(y * label, reduction_indices=[1]) / (0.1 + tf.reduce_sum(y * label, reduction_indices=[1]))
loss_true = - tf.reduce_mean(tf.log(h))
loss_false= - tf.reduce_mean(tf.log(1 - h))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss_true)
train_false_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss_false)
accs = []
for _ in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(40)
    sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
    for i in range(1):
        false_ys = np.zeros([40, 10])
        for j in range(40):
            false_ys[j, np.random.random_integers(0, 9)] = 1
        sess.run(train_false_step, feed_dict={x: batch_xs, label: false_ys})
    #Test
    if _ % 100 == 0:
        accs.append(sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(label, 1)), dtype=tf.float32)), feed_dict={x: mnist.test.images, label: mnist.test.labels}))
print(accs)
plt.plot(accs)
plt.show()
```