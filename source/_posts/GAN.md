---
title: GAN
date: 2019-07-23 10:15:08
tags: GAN
mathjax: true
---
论文 [Generative Adversarial Nets]()

# GAN
## 原理
生成对抗网络 GAN：一个生成模型 G 和一个判别模型 D，G 尽可能模拟真实的数据分布，D 尽可能的区分样本是模型生成的还是真实的。下文以图像数据为例说明。

定义一个输入噪声随机变量 z，其分布为 $p_z(z)$，G 根据 z 生成图像 $G(z;\theta_g)$，我们假设 G 是一个多层感知机 MLP 网络，网络参数为 $\theta_g$。D 也是一个 MLP $D(x;\theta_d)$ 输出是一个标量，表示 x 是真实图像的概率。训练 D 使其对输入 x 预测正确的概率最大化，即当 x 来自真实的训练数据时，$G(x)$ 尽可能大，当 x 来自 G 生成样本时，预测概率 $D(G(z))$ 尽可能小；而训练 G 目的是为了让 $D(G(x))$ 尽可能大，或者说让 $\log(1-D(G(z)))$ 尽可能小，于是目标函数为，

$$\min_G \max_D V(D,G)=\Bbb E_{x \sim p_{data}(x)}[\log D(x)] + \Bbb E_{z \sim p_z(z)}[\log(1-D(G(z)))] \qquad (1)$$

（对 D 而言，这是一个 log 似然函数，D 希望它越大越大，所以求最大值；而 G 却希望 D 的 log 似然函数越小越好，所以求最小值）

这是一个二人零和博弈。图 1 是训练过程示意图，训练使用迭代的，数值计算的方法。
![](/images/GAN_fig1.png)

图 1 中 D 模型分布为蓝色虚线，数据 x 的分布 $p_x$ 为黑色点线，G 模型分布 $p_g$ 为绿色实线（黑绿曲线上某一点分别表示此 x 值处真实数据的概率密度和生成数据的概率密度）。下面的水平线为随机噪声变量 z 的定义域，在其上对 z 均匀采样，上面水平线是 x 的定义域，向上箭头表示映射过程 x=G(z) （G 生成过程）。  
(a) 是收敛附近的对抗情况：此时 $p_g,\ p_{data}$ 两者相似，D 分类不完全准确。  
(b) 在内层循环中，训练 D 判别样本，训练过程收敛于 $D^{\ast}(x)=\frac {p_{data}(x)}{p_{data}(x)+p_g(x)}$。  
(c) D 的梯度可指引 G(z) 移动到更容易被分类为真实数据的区域，即，G 更新后，更加逼近真实数据分布。  
(d) 经过几次训练，G 和 D 到达一个平衡点，此时 $p_g=p_{data}$，D 无法再区分这两个分布，即，$D(x)=1/2$。

训练算法如下，
![](/images/GAN_alg1.png)

k 次 D 的优化与一次 G 的优化交替进行，这可以使得 G 变化缓慢，而 D 维持在最优解附近。

实际应用中，(1) 式可能无法提供足够的梯度来更新 G。训练初期，G 性能较差，生成样本与真实训练样本区别较大，所以 D 可以较高的置信度判别，此时，$\log (1-D(G(z)))$ 达到饱和（log 曲线右端较为平坦），于是我们改为训练 G 以最大化 $\log D(G(z))$，最终训练能到达相同的 G 和 D 的平衡点，但是训练初期的梯度较大（log 曲线的左端较为陡峭）。

## 理论分析
已知噪声随机变量 z 的分布 $p_z$ 时，可以获得 G 的模型分布，根据算法 1，如果 G 模型的假设空间和训练时间足够，G 可以拟合真实数据分布 $p_{data}$。现在我们来证明 $p_g=p_{data}$ 是 (1) 式的全局最优解。
### 全局最优解
__Proposition 1.__ 对于任意的 G，D 的最优解为
$$D_G^{\ast}(x)=\frac {p_{data}(x)}{p_{data}(x)+p_g(x)} \qquad (2)$$
__证明：__  给定任意 G，D 的训练准则是最大化 V(G,D)  
$$\begin{aligned} V(G,D)&=\int_x p_{data}(x) \log D(x) dx+\int_z p_z(z) \log (1-D(g(z))) dz
\\\\ &=\int_x p_{data}(x) \log D(x)+p_g(x) \log(1-D(x))dx \end{aligned}$$
$\forall (a,b) \in \Bbb R^2 \setminus \{0,0\}$，函数 $y \rightarrow a \log y+b \log(1-y)$ 在 (0,1) 区间上当 $y=\frac a {a+b}$ 时有最大值（梯度为 0 求解得到），所以要使得 V(G,D) 最大，那么对于每个 x 值，都要使 D(x) 达到最大，即 (2) 式。证毕。

D 的训练目标函数可以看作是条件概率 $P(Y=y|x)$ 的最大 log 似然函数（或者是 binary cross-entropy），其中当 x 来自 $p_{data}$ 时 y=1，当 x 来自 $p_g$ 时 y=0。得到 D 的最优解 $D_G^{\ast}$ 后 (1) 式变为，  

$$\begin{aligned} C(G)&=\max_D V(G,D)
\\\\ &=\Bbb E_{x \sim p_{data}}[\log D_G^{\ast}(x)] + \Bbb E_{z \sim p_z} [\log(1-D_G^{\ast}(G(z)))]
\\\\ &=\Bbb E_{x \sim p_{data}}[\log D_G^{\ast}(x)] + \Bbb E_{x \sim p_g} [\log(1-D_G^{\ast}(x))]
\\\\ &=\Bbb E_{x \sim p_{data}} \left[\log \frac {P_{data}(x)} {p_{data}(x)+p_g(x)} \right]+\Bbb E_{x \sim p_g} \left[\log \frac {p_g(x)} {p_{data}(x)+p_g(x)}\right] \qquad(4) \end{aligned}$$

__Theorem 1.__ 当且仅当 $p_g=p_{data}$ 时， C(G) 有全局最优解 -log4。  

__证明：__ 

1. 充分性  
令 $p_g=p_{data}$，根据 (2) 式有 $D_G^{\ast}(x)=1/2$，然后根据 (4) 式有，
$$C(G)=\Bbb E_{x \sim p_{data}}[-\log 2]+\Bbb E_{x \sim p_g}[-\log 2] \equiv -\log 4$$
2. 必要性  
   $$\begin{aligned}C(G)&=C(G)+\Bbb E_{x \sim p_{data}}[\log 2]+\Bbb E_{x \sim p_g}[\log 2]  -\log 4 \\\\ &=-\log4 +\Bbb E_{x \sim p_{data}}\left[\log \frac {P_{data}(x)} {\frac {p_{data}(x)+p_g(x)} 2} \right]+\Bbb E_{x \sim p_g} \left[\log \frac {p_g(x)} {\frac {p_{data}(x)+p_g(x)} 2}\right] \\\\ &=-\log4+KL \left(p_{data} \| \frac {p_{data}+p_g} 2 \right)+KL \left(p_g \| \frac {p_{data}+p_g} 2 \right) \\\\ &=-\log4 + 2\cdot JSD(p_{data} \| p_g) \end{aligned}$$
   其中 KL 表示 Kullback-Leibler 散度，JSD 表示 Jensen-Shannon 散度。由于 JSD 非负，且仅在 $p_g=p_{data}$ 时取得最小值 0，所以 C(G)=-log4 时，$p_g=p_{data}$。  

证毕。

### 算法 1 的收敛
上一小节我们分析了全局最优解是存在的，并且取得全局最优解的条件是 $p_g=p_{data}$。__Proposition 2__ 表明基于算法 1 的更新是有效的，训练可以收敛到全局最优解。

__Proposition 2.__ 如果 G 和 D 有足够的模型空间，且在算法 1 每次迭代中给定 G 的情况下判别器可以达到最优解，且以调优（使更小） G 的训练标准 C(G) 更新 $p_g$ 
$$\Bbb E_{x \sim p_{data}}[\log D_G^{\ast}(x)] + \Bbb E_{x \sim p_g} [\log(1-D_G^{\ast}(x))] \qquad(5)$$
那么，$p_g$ 趋于 $p_{data}$。

__证明：__

考虑 $V(G,D)=U(p_g,D)$ 是 $p_g$ 的函数，$p_g$ 可根据 (5) 式标准进行优化。注意到 $U(p_g,D)$ 是 $p_g$ （定义域）上的凸函数，不同 D 形成的凸函数集合的上确界（它也是一个凸函数）的 __次导数__ 包含了此凸函数集合在某个 D 值取得最大值所对应函数的导数，也就是说，给定任意 $p_g$（它是函数自变量），D 是可变参数，（在任意自变量 $p_g$ 处）上述结论均成立。用数学语言描述就是：

- 如果 $f(x)=\sup_{\alpha \in \mathcal A} f_{\alpha}(x)$，且 $f_{\alpha}(x)$ 对任意 $\alpha$ 在 x 上均为凸，那么当 $\beta=\arg \sup_{\alpha \in \mathcal A} f_{\alpha}(x)$ 时有 $\partial f_{\beta}(x) \in \partial f(x)$。

$V(G,D)=U(p_g,D)$ 相当于上述的上确界函数，不能保证在 $p_g$ 定义域上处处严格可导，但是这个上确界函数也是一个凸函数，保证了其具有全局唯一最优解。而上面这个结论 “在任意 $p_g$ 处，其次导数包含了在某个 D 值取得最大值所对应函数的导数”，即，“包含了在 D 取最优解 D* 时 V(G,D) 的导数”，而这个导数正是对 (5) 式求导，于是可以使用这个导数进行梯度上升/下降法更新 $p_g$，并且这个更新将会使得 $p_g$ 趋于 $p_{data}$（参考 Theorem 1）。证毕

对 (5) 式求导与算法 1 中的梯度本质相同，只是似然函数的期望改为批 SGD 中各样本损失的均值（没办法，数值计算使然），注意第一个期望在更新 $p_g$ 时不起作用，为什么这么讲？因为更新 $p_g$ 时，D 已经被固定，此时第一个期望与 $p_g$ 无关。

实际应用中，对抗网络使用 $G(z;\theta_g)$ 表示 $p_g$ 的分布，其中 $\theta_g$ 是 G 模型参数，在选定 G 的网络模型如 MLP 时，$\theta_g$ 就决定了 $p_g$ 的分布，故以上有所对 $p_g$ 的更新其实都转为对  $\theta_g$ 的更新，例如，使用 MLP 作为 G 的模型，目标函数 (1) 式中的 $p_g$ 分布替换为某个 batch 中的生成样本分布，$p_{data}$ 则替换为 batch 中的真实样本分布，简单点说，目标函数 (1) 变为 batch 中所有样本的 log-likelihood function 的均值，包含真实数据和生成数据两部分的log 似然函数，具体可参见下文的代码分析。

## 实验
实验介绍和结果分析略。在这里，我们重点看一下源码 [adversarial](http://www.github.com/goodfeli/adversarial)

> 声明：本源码使用库 Theano 和 Pylearn2，而我从来没接触过这两个库，代码分析全凭函数名、变量名和类名等。github 上也有 GAN 的其他实现如 [generative-models](https://github.com/wiseodd/generative-models)，代码通俗易懂，读者可自行查阅。

从 github 上 clone 这个仓库，进入 adversarial 本项目的根目录。以 mnist 数据集为例说明。

首先看下 mnist.yaml 这个文件，
```yaml
!obj:pylearn2.train.Train {         # 训练配置
    dataset: &train !obj:pylearn2.datasets.mnist.MNIST {    # 训练使用 mnist 数据集
        which_set: 'train',                                 # 使用 train 数据的前 50000 条
        start: 0,
        stop: 50000
    },
    model: !obj:adversarial.AdversaryPair {                 # GAN：G & D
        generator: !obj:adversarial.Generator {             # G
            noise: 'uniform',                               # noise 分布使用均匀分布
            monitor_ll: 1,
            mlp: !obj:pylearn2.models.mlp.MLP {
            layers: [
                     !obj:pylearn2.models.mlp.RectifiedLinear { # 带 ReLu 的 FC 层
                         layer_name: 'h0',
                         dim: 1200,                             # 本层 output units 数量
                         irange: .05,
                     },
                     ...
                     !obj:pylearn2.models.mlp.Sigmoid {     # FC 层后接 sigmoid
                         init_bias: !obj:pylearn2.models.dbm.init_sigmoid_bias_from_marginals { dataset: *train},
                         layer_name: 'y',
                         irange: .05,
                         dim: 784                               # 784=28x28，为 mnist 单个样本大小
                     }
                    ],
            nvis: 100,                                          # G 的噪声随机变量的向量维度
        }},
        discriminator:                                          # D
            !obj:pylearn2.models.mlp.MLP {
            layers: [
                     ...
                     !obj:pylearn2.models.mlp.Sigmoid {
                         layer_name: 'y',
                         dim: 1,                                # 输出为标量
                         irange: .005
                     }
                    ],
            nvis: 784,                                          # 输入向量维度
        },
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {      # 优化算法
        ...
        cost: !obj:adversarial.AdversaryCost2 {                 # 损失实现类
            scale_grads: 0,
            #target_scale: 1.,
            discriminator_default_input_include_prob: .5,
            discriminator_input_include_probs: {
                'h0': .8
            },
            discriminator_default_input_scale: 2.,
            discriminator_input_scales: {
                'h0': 1.25   
            }
            },
        ...
    },
    ...
}
```
可以明显知道，训练使用 mnist 的 `train` 数据集中前 50000 个数据，模型类实现为 adversarial.AdversaryPair，生成器类为 adversarial.Generator，其内部封装了一个 MLP，判别器类直接使用 MLP。损失实现类为 adversarial.AdversaryCost2。这些类的实现均位于 `__init__.py` 中。这里主要分析一下 AdversaryCost2（其他类的实现均比较简单明了）。

首先看一下生成样本和目标函数 `get_samples_and_objectives`，
```python
g=model.generator       # model is an instance of AdversaryPair
d=model.discriminator
X=data                  # 真实数据（来自训练样本）的 batch
m=data.shape[space.get_batch_axis()]    # 获取 batch 的大小，即批样本数量
y1=T.alloc(1,m,1)       # 长度为 m 的全 1 向量，代表真实数据的 label
y0=T.alloc(0,m,1)       # 长度为 m 的全 0 向量，代表生成数据的 label
# 1. 生成 m 个噪声作为 G 模型的输入 z
# 2. G 前向传播生成 m 个样本 S
S,z,other_layers=g.sample_and_noise(m,
    default_input_include_prob=self.generator_default_input_include_prob,   # 1
    default_input_scale=self.generator_default_input_scale,                 # 1
    all_g_layers=(self.infer_layer is not None)                         # False
)
if self.noise_both !=0:     # 真实数据和生成数据均添加一个噪声干扰
    ...
# D 前向传播，分别得到真实数据的预测 label 和生成数据的预测 label
y_hat1 = d.dropout_fprop(...)       # 参数略
y_hat0 = d.dropout_fprop(...)
# D 的目标损失。d.layers[-1] 为 Sigmoid 层，其目标损失为 KL 散度
d_obj = 0.5*(d.layers[-1].cost(y1,y_hat1)+d.layers[-1].cost(y0,y_hat0))
# G 的目标损失。G 希望 D 的判别结果 y_hat0 与真实 label y1 越小越好  
g_obj = d.layers[-1].cost(y1,y_hat0)
if model.inferer is not None:       # 模型推断器
    ...
else:
    i_obj = 0
return S, d_obj, g_obj, i_obj       # 返回生成样本，D 损失和 G 损失
```
再来看计算梯度函数 `get_gradients` 的实现部分，
```python
g=model.generator
d=model.generator
S,d_obj,g_obj,i_obj = self.get_samples_and_objectives(model,data)   # 调用上面分析的函数
g_params = g.get_params()
d_params = d.get_params()
# 计算损失对各参数的梯度
d_grads = T.grad(d_obj,d_params)
g_grads = T.grad(g_obj,g_params)
if self.scale_grads:    # 缩小 g_grads
    S_grad = T.grad(g_obj, S)   # G 损失对生成样本（也就是 G 的输出）的梯度
    # S_grad 的平方和的平方根的倒数作为缩小比例
    scale = T.maximum(1.,self.target_scale/T.sqrt(T.sqr(S_grad).sum()))
    # 缩小 g_grads
    g_grads = [g_grad * scale for g_grad in g_grads]

# 保存各模型参数与其对应的梯度
rval = OrderDict()
rval.update(OrderedDict(safe_zip(d_params, [self.now_train_discriminator * dg for dg in d_grads])))
rval.update(OrderedDict(safe_zip(g_params, [self.now_train_generator * gg for gg in g_grads])))

updates = OrderDict()
if self.alternate_g:
    updates[self.now_train_generator]=1. - self.now_train_generator
return rval, updates
```
最终的更新操作由 Pylearn2/Theano 库完成。

以上代码片段中，目标函数为损失，与 log 似然函数相差一个负号，所以上文分析中某些求最大值的地方变为求最小值，然后使用随机梯度下降更新模型参数，这与算法 1 中的情况完成相同。另外，对 `g_grads` 进行 scale 缩小，一种可能的原因是，

生成样本 $S=\theta_g \cdot z$，损失对 $\theta_g$ 的梯度满足
$$\nabla_{\theta_g}L=\nabla_S L \cdot \frac {\partial S}{\partial \theta_g}$$

记生成样本 S 经过 D 的输出为 y_0，即，$y_0=\theta_d \cdot S$，于是
$$\nabla_S L=\frac {dL}{dy_0}\cdot \theta_d$$
可以看出在计算损失对 G 模型参数的梯度之前，$\nabla_S L$ 这个梯度已经经过 D 中各层的传播：
1. 如果其 L2 范数大于 1，那么再经过 G 中各层反向传播时，极有可能出现梯度爆炸，即 $\nabla_{\theta_g}L$ 很大， 导致训练不稳定，所以需要将其进行 scale 缩小，缩小的比例正好能使 $\nabla_S L$ 的 L2 范数为指定值 `self.target_scale`（默认为1）
2. 如果其 L2 范数小于等于1，则对梯度不做 scale 缩小操作。

当然，还有其他损失实现类，具体请查阅源码，不再讨论。

## 总结
给定一个预先已知分布的噪声随机变量 z，G 根据 z 生成图像 G(z)，D 将 G(z) 与训练样本区分开来。训练过程根据 (1) 式交替优化 D 和 G，使得 G 尽可能拟合真实数据分布，而 D 提高判别能力，最终 G 分布与真实分布相同，D 无法判别模型分布和真实数据分布。