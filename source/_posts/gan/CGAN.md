---
title: CGAN/DCGAN
date: 2019-07-29 17:00:43
tags: GAN
p: gan/CGAN
mathjax: true
---
# CGAN
论文 [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
<!-- more -->
在 [GAN](gan/2019/07/23/gan) 中我们知道 GAN 经过训练，生成器 G 可以根据一个随机噪声输入生成与训练集样本非常相似的样本（判别器 D 无法判别），但是 G 生成样本的标签是无法控制的，以 mnist 数据集为例，给 G 一个随机噪声输入，G 生成的样本图像可能表示数字 1，也可能是其他数字，GAN 无法控制，GAN 只能做到 G 生成样本图像很逼近真实样本图像。然而，使用额外信息来限制模型则可以控制数据生成过程，这个额外信息可以是分类标签或是其他形式的数据，于是本文的 CGAN 应运而生。

## Conditional Adversarial Nets
GAN 中的 G 和 D 均使用额外信息 y 进行条件限制，则得到 CGAN。额外信息 y 可是是分类标签或者其他形式的数据。以 mnist 训练集为例，通常选择图像的分类标签作为额外信息 y。

预先已知的输入噪声 z 和 图像分类标签 y 合并一起作为 G 的输入（这是本文所用的最简单的方法，这种处理方式可以很容易地使用传统的 GAN 网络而不需要重新设计网络）。训练样本数据 x 以及对应的图像分类标签 y 合并到一起作为 D 的输入。（G 和 D 的结构可以与 GAN 中保持一致，也可以将部分 fc 替换为 conv/deconv）

训练目标函数为，
$$\min_G \max_D V(D,G)=\Bbb E_{x \sim p_{data}(x)} [\log D(x|y)] + \Bbb E_{z \sim p_z(z)}[1-\log (1-D(G(z|y)))]$$

图 1 为 CGAN 过程示意图，
![](/images/CGAN_fig1.png)

~~这里引用一个代码片段来进行说明~~ 请参考下方 PyTorch 实现代码。
```python
import tensorflow as tf
y_dim=10    # one-hot vector for mnist-label
z_dim=100   # length of noise input vector
y=tf.placeholder(tf.float32, shape=[None,y_dim], name='label')
x=tf.placeholder(tf.float32, shape=[None,28,28,1], name='real_img')
z=tf.placeholder(tf.float32, shape=[None,z_dim], name='noise')

# G 的输入由 noise 与 label 合并，单个输入 vector 长度由原来的 100 变成 110
x_for_g=tf.concat([z,y], axis=1)    # [batch_size, 100+10]
# 然后与 GAN 中 G 的处理相同

# D 的输入由 real_img 与 label 合并
new_y=tf.reshape(y,[batch_size,1,1,y_dim])
new_y=new_y*tf.ones([batch_size,28,28,y_dim])   # [batch_size,28,28,10]
x_for_d=tf.concat([x,new_y],axis=-1)    # [batch_size,28,28,1+10]
# 然后与 GAN 中 D 的处理相同
```

## PyTorch 代码实现

代码来自 [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)

**Generator**
```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 可学习的，将分类label 转换为 embedding vector，更好的与 img 数据融合
        self.label_emb = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # 不必过多考虑为何第一个 layer 不使用 BN，可能是因为浅层，网络 input
            #   已经经过 normalization。当然如果这里使用 BN，我觉得也可以
            *block(latent_dim + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        '''
        noise: (B, latent_dim)
        labels: (B,)  -> label index
        '''
        gen_input = torch.cat((self.label_emb(labels), noise), -1)  # (B, latent_dim+num_classes)
        img = self.model(gen_input)
        img = img.view(img.shape[0], *img_shape)
        return img      # (B, C, H, W)
```

**Discriminator**
```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)

        # 不用考虑各 layer 的 output units 数量是否合适，这仅仅是使用
        #    mnist 的一个简单例子
        self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(img_shape)),512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forwad(self, img, labels):
        '''
        img: 输入 image tensor，可以是真实 images，也可以是生成 images
        labels: (B,)， label index
        '''
        B, C, H, W = img.shape
        d_in = torch.cat((img.view(B, -1), self.label_emb(labels)), -1)
        validity = self.model(d_in).view(-1)     # (B,)
        return validity
```

**训练**

```python
# (\hat p-p)^2  为什么不用交叉熵/NLL 损失？
adversarial_loss = nn.MSELoss()

G = Generator()
D = Discriminator()

optimizer_G = optim.Adam(G.parameters(), lr=1e-3)
optimizer_D = optim.Adam(D.parameters(), lr=1e-3)

for epoch in num_epochs:
    for i, (imgs, labels) in enumerate(dataloader):
        B, C, H, W = imgs.shape

        valid = torch.ones(B)
        fake = torch.zeros(B)

        # ==================== 训练 G ===================
        optimizer_G.zero_grad()
        z = torch.randn(B, latent_dim)
        gen_labels = torch.randint(0, num_classes, (B,))
        gen_imgs = G(z, gen_labels)
        validity = D(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()

        # ==================== 训练 D ===================
        optimizer_D.zero_grad()
        validity_real = D(imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)
        validity_fake = D(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
```

## 实验
### Unimodal
使用 mnist 数据集，分类标签 y 使用长度为 10 的 one-hot 向量。CGAN 的结构和训练方法介绍略，这部分可以查看原文。图 2 显示了生成样本，每一行使用一个标签作为模型的限制条件。
![](/images/CGAN_fig2.png)

### Multimodal
对应一到多映射，即每个图像可以有多个不同的标签。例如 Flickr 数据集，包含图像和对应的 UGM（user-generated metadata）。UGM 通常更具有描述性，并且语义上与人类使用自然语言描述图像更为接近，而不仅仅是标记图中的目标。不同的用户可能使用不同的词汇来描述相同的概念，因此使用一个高效的方法来规范化这些标签显得尤其重要。概念词嵌入（word embedding）在此情况下非常有用，因为语义相似的词其词向量非常接近。

根据图像特征，我们可以使用 CGAN 生成 tag-vectors 以进行对图像自动打标签。使用 AlexNet 在 ImageNet 上训练网络，网络的最后一个 fc 层输出单元为 4096 个，这个输出作为最终的图像表示。为了得到词表示，我们从 YFCC100M 数据集的 metadata 中收集 user-tags，title 和 descriptions 作为文本预料，经过预处理和文本清洗，使用 skip-gram 模型进行训练，得到长度为 200 的词向量，我们忽略词频低于 200 的词，最终得到的词典大小为 247465。生成器 G 生成样本为 tag 特征向量，额外信息 y 为图像特征（上述的 4096 向量）。

实验使用 MIR Flickr 25000 数据集，使用上述卷积模型和语言模型（AlexNet，skip-gram）分布抽取图像特征和 tag 特征。数据集中前 15000 的样本作为训练集。训练阶段，数据集中没有 tag 的图像被忽略掉，而如果图像拥有多个 tag，那么对于每个 tag 均分别使用一次这个图像。

evaluation 阶段，对于每个图像生成 100 个样本（tag 特征向量），然后对每个生成样本，使用余弦相似度计算词典中与样本最接近的 20 个词，然后再所有 100 个样本中（我理解的是在 2000 个词中）选择 top 10 最常见的词作为图像的 tags。由于这部分实验没有看到源码，故其余部分的介绍略过，详情可参考原论文。

# DCGAN
论文 [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

这篇文章主要是将卷积层、BN 以及 ReLU 引入 GAN 网络，解决以往 GAN 中使用 MLP 带来的训练不稳定，输出很奇怪的 image 等问题。

~~没有官方代码，但是 github 上有很多实现，都非常简单易懂，例如 [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)。~~

## 网络

![](/images/gan/DCGAN1.png)

图 2. DCGAN 的生成器网络结构。

100 维度的均匀分布随机变量，通过 MLP 输出后 reshape 为 $(1024,4,4)$ 的 tensor，然后经过 4 个 反卷积上采样，得到 $(3, H, W)$ 的 image 输出。

细节说明：

1. 将 pooling 替换为具有 stride 的卷积（在判别器 D 中）或反卷积（生成器 G 中）
2. G 和 D 均使用 BN
4. G 中使用 ReLU 代替 LeakyReLU，不过最后一层即输出层仍然使用 Tanh 激活
5. D 中全部使用 LeakyReLU
6. G 中，`latent_dim` 维度的噪声经过一个全卷积层线性变换为 `1024*4*4` 的 tensor，然后 reshape 为 `(1024, 4, 4)` 后作为全卷积网络的输入。

## PyTorch 代码实现

**Generator**

```python
class Generator(nn.Module):
    # 由于 mnist 图片较小，整个上采样率为 4
    def __init__(self):
        super(Generator, self).__init__()

        # 这里 mnist 的 size 为 28*28，较小，如果较大图片，可以 img_size // 16
        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),    # 上采样没有采用反卷积
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # opt.channels = 1，灰度图片， =3 为彩色图片
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
```

**Discriminator**

```python
class Discriminator(nn.Module):
    # 整个下采样为 16，对于 28*28 的 mnist image，最终输出的 spatial size 为 1*1
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            # 下采样 =2
            # Conv + LeakyReLU + Dropout +? BN
            # 可以试试 Conv +? BN + LeakyReLU
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)       # (B, 128, 1, 1)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)  # (B, 1)

        return validity

```