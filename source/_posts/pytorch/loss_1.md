---
title: Loss 1
p: pytorch/loss_1
date: 2021-01-12 17:35:17
tags: PyTorch
mathjax: true
---

前面介绍了 [交叉熵损失](2021/1/12/dl/x_ent_loss)，本篇就 PyTorch 中的各种 [Loss](https://pytorch.org/docs/stable/nn.html#loss-functions) 进行分解并掌握其用法。
<!-- more -->

# L1Loss
基于L1 范数的损失，单个样本的L1损失为 $l_n=|x_n-y_n|$，其中 `n` 为批样本中的样本索引，$x_n$ 为预测值，$y_n$ 为 GT，L1 损失适用于<b>回归</b>问题。

# MSELoss
均方差（L2范数平方）损失，单个样本损失的计算公式为 $l_n=(x_n-y_n)^2$。适用于<b>回归</b>问题。

# NLLLoss
负对数似然损失，适用于<b>分类</b>问题。对于单个样本，似然函数为
$$\mathcal L=\prod_{i=1}^C x_i^{y_i}$$
其中输出向量 $\mathbf x = (x_1,...,x_C)$ 表示每个分类的预测概率，GT 向量为 $\mathbf y=(y_1,...,y_C)$，如果是单标签分类，$\mathbf y$ 为 one-hot，如果是多标签分类，$\mathbf y$ 中可能有多个元素值为 1。负对数似然则为，
$$l=-\sum_{i=1}^C y_i \log x_i$$

实际在 PyTorch 中，NLLLoss 层的输入 Tensor 的 shape 以及 GT target 的 shape 与上面有所不同，以单标签多分类为例，网络输出 Tensor 的 shape 可以是 $(N,C)$，其中 N 表示批大小，C 表示通道也是类别数。GT target 的 shape 为 `N`，其中每个元素值的范围 `[0,C-1]`，表示某个样本的类别索引，NLLoss 层的输入已经表示样本各分类的概率对数（由`LogSoftmax`得到），负对数似然为
$$L=(l_1,...,l_N), \quad l_n=- x_{n,y_n}$$

如果给定参数`weight`，那么其必须是 1-D tensor，长度与类别数`C` 相等，用于给每个类别增加一个权重，参考 [交叉熵损失](2021/1/12/dl/x_ent_loss) 中的 [$\alpha$ 均衡交叉熵](2021/1/12/dl/x_ent_loss#Balanced-Cross-Entropy)，这在非均衡数据集上较为有效。此时有
$$l_n=- w_{y_n}  x_{n,y_n}$$

类签名
```python
torch.nn.NLLLoss(weight: Optional[torch.Tensor]=None, size_average=None, ignore_index: int=-100, reduce=None, reduction: str='mean')
```

`size_average` 和 `reduce` 这两个参数已经过时。`reduction `用于指定批样本的损失向量是否归约（均值或求和）。

`ignore_index` 如果指定，那么当 GT target 值等于 `ignore_index` 时，将会忽略对应的损失贡献。

input 通过 forward 方法指定，input 表示每个分类的概率对数，这可以通过 `LogSoftmax` 得到，input 的 shape 可以是 $(N,C)$，或者是 $(N,C,d_1,...,d_K)$，对于后者，其 target 的 shape 则为 $(N,d_1,...,d_K)$，此时的（未归约）损失 shape 也是 $(N,d_1,...,d_K)$，相比较于前者，后者就是扩展了维度而已，对于 $(d_1,...d_K)$ 中按像素级地计算负对数似然损失。


# CrossEntropyLoss
交叉熵损失，适用于分类问题。PyTorch 中，这个类（layer）合并了 `LogSoftmax` 和 `NLLLoss`，所以这个 layer 的 input 为为归一化的各分类的原始得分，input 的 shape 可以是 $(N,C)$ 或 $(N,C,d_1,...,d_K)$。target 的 shape 则为 $(N,)$ 或 $(N,d_1,...,d_K)$。
以 input 的 shape 为 $(N,C)$ 为例，此 layer 的损失计算可表示为（单个样本）
$$l_n=-\log \left(\frac {\exp x_{n,y_n}}{\sum_j \exp x_{n,j}}\right)$$
其中 $y_n \in [0,C-1]$ 为第 n 个样本的类别索引，$\sum_j$ 为某个样本对 C 个类别的求和。

除了增加了一个 `LogSoftmax` 的计算，其他均与 NLLoss 层类似，故类签名中的参数介绍略。

# PoissonNLLLoss
Poisson 损失一般用于服从 poisson 分布的计数数据回归的问题，例如下周教堂人数预测。Poisson 分布如下
$$P(X=k)=\frac {\lambda^k e^{-\lambda}} {k!}$$
随机变量 X 的期望 $E[X]=\lambda$。我们的预测值 $x$ 就是对期望 $\lambda$ 的预测，target 值就是真实的计数值（例如事件发生的次数，教堂的人数等），target 值用 $y$ 表示，也就是上式中的 $k$，于是单个样本的负对数似然可表示如下：
$$l= -\log P(y|x) =-\log \frac {x^{y} e^{-x}} {y!}=x-y \log x+ \log(y!)$$
最后一项可以忽略或者适应 Stirling 公式近似求解。因为是一个常数，所以即使忽略掉，也不影响反向传播的计算。

类签名：
```python
PoissonNLLLoss(log_input: bool=True, full: bool=False, size_average=None, eps: float=1e-8, reduce=None, reduction: str='mean')
```
`log_input` 指明 forward 的输入 input 是否经过了 log 处理，如是 True，那么上式损失计算应改为 $l=e^x - yx$，否则损失计算式为 $l=x-y \log(x+eps)$。在 Poisson 回归中，假定期望的对数符合线性模型，所以很多时候是对期望的 log 值进行预测，即 `log_input=True`，此时 target 值也要经过 log 处理。

> 程序中为了防止计算数值上下溢，往往会采用 log 处理

`full` 指示是否添加最后一项 $\log(y!)$。如需要添加，那么使用 Stirling 公式近似，Stirling 公式为
$$n! \sim \sqrt{2 \pi} n^{n+1/2} e^{-n}$$
于是有
$$\log(n!)=\frac 1 2 \log(2 \pi n)+ n \log n - n$$

forward 方法的 input 的 shape 是 $(N, *)$，其中 $*$ 表示对维度的扩展，且损失计算都是在 $*$ 维度上按像素级进行计算，故 target 的 shape 也是 $(N, *)$。如果 `reduction` 参数为 `none`，那么输出 shape 也是 $(N, *)$，否则将输出 Tensor 中所有值按 求和或平均 进行归约，最终得到一个标量值。

# KLDivLoss
KL 散度用于度量两个分布之间的差异。KL 散度损失适用于<b>回归</b>问题。

根据 KL 散度计算损失，KL 散度计算如下，
$$D(P||Q)=\sum P(x) \cdot \log \frac {P(x)}{Q(x)}$$
$$D(P||Q) = \int_x p(x) \log \frac {p(x)}{q(x)} dx$$

预测分布越接近真实分布，那么两者之间的 KL 散度应该越小，所以 KL 散度可以作为一种损失。
PyTorch 中的类签名：
```python
KLDivLoss(size_average=None, reduce=None, reduction: str='mean', log_target: bool=False)
```
`log_target` 指示 target 是否经过 log 处理。

forward 方法中，参数 input 表示预测概率，且经过 log 处理，input 的 shape 为 $(N,*)$，其中 $*$ 表示单个样本的所有维度。KL 散度损失按像素级计算（可看作是连续分布的离散采样），
$$l=y \cdot (\log y - x)$$
其中 $x$ 表示随机变量某个值对应的预测概率，且经过 log 处理，$y$ 表示这个随机变量在这个值处的真实概率。

forward 方法的输出结果的 shape 与 input 相同，为 $(N,*)$，如果 `reduction` 不为 `none`，那么输出结果将按 求和或平均 归约为一个标量值。

# BCEWithLogitsLoss
PyTorch 中这个 layer 合并了 Sigmoid 层和 BCELoss 层，由于 BCELoss 层计算单个样本的 BCE 损失为，
$$l=-[y \log x + (1-y) \log (1-x)]$$
其中 $y \in \{0,1\}$ 表示样本的真实分类，$x\in [0,1]$ 表示样本的预测概率，通常使用 Sigmoid 层来将处于实数域的前一个 layer 输出值压缩到 $[0,1]$ 之间，故为了少写一个 Sigmoid 层，将这两者合并为单个 layer： `BCEWithLogitsLoss`。所以这个 layer 的输入是原始的未归一化的各类别的得分，单个样本的损失为，
$$l_n=-w_n [y_n \log \sigma(x_n) +(1-y_n) \log (1-\sigma(x_n))]$$
这里，批样本中每个样本有各自的一个权重因子 $w_n$。

如果是多标签多分类问题，那么对于每个类别，均独立进行二分类（正或负），记类别索引为 $c$，那么单个样本的损失为
$$l_n=\sum_{c=1}^C l_{n,c}=-\sum_{c=1}^C w_n [y_{n,c} \log \sigma(x_{n,c}) +(1-y_{n,c}) \log (1-\sigma(x_{n,c}))]$$
其中 $y_{n,c} \in \{0,1\}$，$x_{n,c} \in \mathbb R$。

还可以对正类样本增加一个权重因子 $p_c$，用于权衡最终的召回率和精度，于是上式变为
$$l_n=\sum_{c=1}^C l_{n,c}=-\sum_{c=1}^C w_n [p_c y_{n,c} \log \sigma(x_{n,c}) +(1-y_{n,c}) \log (1-\sigma(x_{n,c}))]$$
当 $p_c >1$ 时召回率增大，$p_c<1$ 时 精度增大。$p_c$ 可以取类别 $c$ 下 负样本与正样本数量比，如此可认为正负例相等。

forward 方法中 input 的 shape 为 $(N,*)$，其中 $N$ 为批大小，$*$ 表示单个样本的维度大小，损失按像素计算，故 target 和未归约的 output 的 shape 均为 $(N,*)$，如果对 output 按求和或平均归约，则 output 为一个标量值。

> 这个 layer 比起 Sigmoid 和 BCELoss 两个 layer，在数值计算上更加稳定（能避免数值上下溢），因为使用了 `log-sum-exp` 技巧。

适用于<b>分类</b>问题。
# SmoothL1Loss
对 L1Loss 的改进，当 L1 范数低于一定值时，使用差的平方项来代替误差，这是因为当预测值越接近真实值时，损失的梯度应该越小，从而减缓参数的更新幅度。SmoothL1Loss 按像素计算，计算式为，
$$l_i=\begin{cases} \frac 1 {2 \beta} (x_i-y_i)^2 & |x_i - y_i| < \beta \\ |x_i-y_i|-\frac 1 {2 \beta} & \text{otherwise}  \end{cases}$$
适用于<b>回归</b>问题。