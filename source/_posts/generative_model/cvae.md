---
title: Conditional VAE
date: 2022-08-23 11:49:57
tags: generative_model
mathjax: true
---

使用文本控制生成的图片内容。

# 1. 背景

VAE 的目标是最大化 $\log P(X)$，而由于真实数据 $X$ 的分布未知，所以改求最大化

$$ELBO=\log P(X) - D_{KL}(Q(z|X)||P(z|X))=E_{Q(z|X)} [\log P(X|z)] - D_{KL} [Q(z|X)||P(z)] \tag{1}$$

其中 $Q(z|X)$ 为编码器输出分布，作为 $P(z|X)$ 的变分近似。(1) 式可按 KL 散度展开变换进行推导证明

现在对于数据 $X$，其关联条件为 $z$，那么编码器现在输入为 $X$ 和 $c$，即 $Q(z|X,c)$，解码器则为 $P(X|z,c)$ ，于是 ELBO 目标函数变为

$$ELBO=\log P(X|c) - D_{KL}[Q(z|X,c)||P(z|X,c)]=E_{Q(z|X,c)} [\log P(X|z,c)]-D_{KL} [Q(z|X,c)||P(z|c)] \tag{2}$$

# 2. CVAE

条件变量 $c$ 可以是任何随机变量，例如分类分布，回归目标中的高斯分布，甚至与数据 $X$ 分布相同，分布参数不同，例如图像修复任务。

以 mnist 数据集为例。条件 $c$ 是长度为 10 的 one-hot vector，由于 mnist 足够简单，所以 $X$ 与 $c$ 使用 concatenate 进行信息合并，

```python
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X_train, y_train = mnist.train.images, mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels

m = 50
n_x = X_train.shape[1]  # 784
n_y = y_train.shape[1]  # 10
n_z = 2
n_epoch = 20


# Q(z|X,y) -- encoder
X = Input(batch_shape=(m, n_x))
cond = Input(batch_shape=(m, n_y))
inputs = merge([X, cond], mode='concat', concat_axis=1)

h_q = Dense(512, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)
```

encoder 使用 3 个 fc layer，对每个数据得到 2 维 vector，分别表示 $\log \mu$ 和 $\log \sigma^2$，即隐变量 $z$ 为一维高斯分布。

decoder 过程：

1. 根据 encoder 输出的高斯分布的 $\log \mu, \ \log \sigma^2$ 采样得到 $z$ 的值
2. $z$ 与 $c$ 也是 concatenate
3. decoder 网络由 2 个 fc layer 组成，其中第二个 fc 使用 sigmoid 激活
4. 输出 784 维的 vector

```python
def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., std=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sample z ~ Q(z|X,y)
z = Lambda(sample_z)([mu, log_sigma])
z_cond = merge([z, cond], mode='concat', concat_axis=1) # <--- NEW!

# P(X|z,y) -- decoder
decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(784, activation='sigmoid')

h_p = decoder_hidden(z_cond)
outputs = decoder_out(h_p)
```

损失函数与 VAE 中完全相同，即 ELBO 的负数（最大化 ELBO 等价于最小化 -ELBO），

$$-ELBO=D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z)) - \frac 1 L \sum_{l=1}^L \log p_{\theta}(\mathbf x|\mathbf z^{(l)}) \tag{3}$$

其中 

$$\begin{aligned}D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z))&=-\int q_{\phi}(\mathbf z)[\log p_{\theta}(\mathbf z)-\log q_{\phi}(\mathbf x)] d\mathbf z
\\&=-\frac 1 2 \sum_{j=1}^J (1+ \log \sigma_j^2 - \mu_j^2 -\sigma_j^2)
\end{aligned}$$

$\frac 1 L \sum_{l=1}^L \log p_{\theta}(\mathbf x|\mathbf z^{(l)})$ 表示试验 L 次，即先后采样 L 个 $\mathbf z$ 值，每个 $\mathbf z$ 经 decoder 得到 $\mu \in \mathbb R^D$ （对于 mnist，D=784），求 $\mathbf x$ 对 $\mathbf z$ 的条件似然——对数条件概率，

$$\log p_{\theta}(\mathbf x|\mathbf z)=\log \mathcal N(\mathbf x;\mu, \sigma^2)=-\sum_{i=1}^D \frac 1 {2\sigma_i^2}(x_i - \mu_i)^2+C \tag{4}$$

事实上，更多的是使用交叉熵 $-p\log q$ 来代替 $-\log p_{\theta}(\mathbf x|\mathbf z)$，这里 $p$ 指真实分布，$q$ 指模型分布，对于某个 location 而言，交叉熵为

$$-p\log q=-[x_i \log \mu_i + (1-x_i) \log (1-\mu_i)] \tag{5}$$

实验中，取 $L=1$ 已经足够。代码如下，

```python
def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z,y)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl
```

# 3. CVAE-GAN

论文：[CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training](https://arxiv.org/abs/1703.10155)

![](/images/generative_model/cvae_1.png)

图 1. 从图中可以看出，先是 CVAE，输出 $\mathbf x'$，然后分别经分类网络和判别器，得到 $\mathbf x'$ 的分类 $c$ 和是真实数据的概率 $y$。 E,G,C,D 分别表示 encoder，generative，classification 以及 discriminative 网络。

参考：

1. https://agustinus.kristia.de/techblog/2016/12/17/conditional-vae/