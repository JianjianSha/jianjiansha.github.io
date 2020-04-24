---
title: PyTorch 中的损失
p: pytorch/PyTorch-loss
date: 2019-10-29 09:46:16
tags: PyTorch
mathjax: true
---
常用的损失我们都很熟悉了，这里总结一下一些不太常用的损失。
<!-- more -->
## 1. PoissonNLLLoss
PyTorch 文档中关于 PoissonNLLLoss 的计算式为
```
target ~ Poisson(input)
loss(input, target)=input - target * log(input) + log(target!)
```
那么如何理解这个计算式呢？

我们知道 $X \sim Poisson(\lambda)$ 分布为
$$P(X=k)=\frac {\lambda^k e^{-\lambda}} {k!}$$
其中 $\lambda$ 是参数，也是 Poisson 分布的期望。

`target` 就是这里的 `k` 表示事件发生的次数，`input` 可以是某个网络最后的计算输出，我们令其表示 Poisson 分布的期望，这样 `input` 与 `target` 就关联起来了。

我们用 `y` 代替 `k` 表示真实值 `target`，`x` 代替 $\lambda$ 表示计算值 `input`，Poisson 分布的似然函数为
$$Likehood(y|x)=\prod_{i=1}^m P(y|x) = \prod_{i=1}^m \frac {x^y e^{-x}} {y!}$$
于是损失函数为
$$Loss=-\log [Likehood(y|x)]=-\sum_{i=1}^m (-x+y \log x - \log(y!))=\sum_{i=1}^m (x-y \log x+ \log(y!) )$$
上式与 PyTorch 文档说明的计算式一致。

`log(target!)` 可使用 Stirling 公式近似得到，Stirling 公式为
$$n! \sim \sqrt{2 \pi} n^{n+1/2} e^{-n}$$
于是有
$$\log(n!)=\frac 1 2 \log(2 \pi n)+ n \log n - n$$
与文档中 PoissonNLLLoss 的参数 `full` 的解释一致，
```
target * log(target) - target + 0.5 * log(2 \pi target)
```

## 2. KLDivLoss
KL 散度损失，根据 PyTorch 文档说明，非归约版本的损失为，
$$l(x,y) = L=\{l_1,...,l_N\}, \quad l_n=y_n \cdot (\log y_n - x_n)$$
其中输入 `x` 包含概率对数，目标值 `y` 是概率（没有取对数），`y` 和 `l` 均与 `x` 具有相同的 shape。

根据如下 KL 散度计算公式，不难理解 KLDivLoss 就是根据 KL 散度定义进行计算。
$$D(P||Q)=\sum P(x) \cdot \log \frac {P(x)}{Q(x)}
\\\\ D(P||Q) = \int_x p(x) \log \frac {p(x)}{q(x)} dx$$
（上面是离散型概率分布，下面连续型概率分布）

## 3. HingeEmbeddingLoss
用于测量两个输入相似或者不相似，相似则 `y=1`，不相似则 `y=-1`，两个输入之间的距离可以使用 $L_1$ 或者 $L_2$ 距离，hinge embedding loss 计算如下：
$$l_h(p_1,p_2)=\begin{cases} L_n(p_1,p_2) & y=1 \ (p_1 \equiv p_2)
\\\\ \max \{0, \Delta - L_n(p_1, p_2)\}       & y=-1 \ (p_1 \ne p_2)
\end{cases}$$
例如使用 $L_1$ 距离，$L_1(p_1,p_2)=\|\mathbf x_1 - \mathbf x_2\|_1$，其中 $\mathbf x$ 是特征向量。

PyTorch 文档中关于 HingeEmbeddingLoss 计算描述为，对于第 n 个样本的损失为：
$$l_n=\begin{cases} x_n, & y_n=1
\\\\ \max\{0, \Delta-x_n\}, & y_n=-1 \end{cases}$$
其中 $\Delta$ 为间距，$x_n$ 为距离。

根据 HingeEmbeddingLoss，如果两个输入相似（匹配），那么我们希望其距离 $x_n$ 越小越好，如果不相似，那么希望其距离越大越好，但是如果超过间距 $\Delta$，那么不在我们关心范围内。

## 4. MultiLabelMarginLoss
这是一个关于 multi class multi classification 问题。

`x` tensor 的 size 为 `(N,C)`，其中 N 表示 batch 大小，C 为分类数量，`y` tensor 的 size 也是 `(N,C)`，对于 mini-batch 中某个样本而言，损失为
$$loss(x,y)=\sum_{ij} \frac {\max (0, 1- (x[y[j]]-x[i]))} {x.size(0)}$$
因为指定了某个样本，所以此时 `x,y` 均为一个 vector，`x.size(0)=y.size(0)=C` 表示分类数量，`y` 的值表示某个分类的下标，即 $0 \le y[j] \le C-1$，故上式的下标必须满足条件 
$$i \in \{0,...,x.size(0)-1\}, \ j \in \{0,...,y.size(0)-1\}, \ 0 \le y[j] \le x.size(0)-1, \ i \ne y[j]$$
注意根据这个条件确定 loss 中求和项下标 `i,j` 的过程如下：
1. 根据 $j \in \{0,...,y.size(0)-1\}, \ 0 \le y[j] \le x.size(0)-1$  确定 `y` 的可取值集合 $\mathcal Y$
2. 然后确定 `x` 的可取值集合为 $\{0,...,x.size(0)-1\} \setminus \mathcal Y$

以 PyTorch 文档中的例子进行说明
```python
>>> loss = nn.MultiLabelMarginLoss()
>>> x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
>>> # for target y, only consider labels 3 and 0, not after label -1
>>> y = torch.LongTensor([[3, 0, -1, 1]])
>>> loss(x, y)
>>> # 0.25 * ((1-(0.1-0.2)) + (1-(0.1-0.4)) + (1-(0.8-0.2)) + (1-(0.8-0.4)))
tensor(0.8500)
``` 
首先确定有效的 `y` 值集合为 $\{3,0\}$，然后可确定 `x` 的 `i` 下标集合为 $\{1,2\}$，代入 loss 计算式得到损失为 `0.8500`。

最后简单的解释下损失这么计算的合理性：
1. 当 $i=y[j]$ 时，表示得到了正确的分类，正确的分类自然损失应为 0，故损失计算要求 $i \ne y[j]$。
2. 当 $i \ne y[j]$ 时，由于 `y[j]` 为真实分类，故 `i` 应该是错误分类，所以 $x[y[j]]$ 越大越好，$x[i]$ 越小越好，对应 $1-(x[y[j]]-x[i])$ 就越小，损失也越小，当然损失不能为负，所以使用 $\max (0,*)$ 进行截断。

## 5. SoftMarginLoss
用于优化多标签二分类的logistic 损失，计算如下
$$loss(x,y)=\sum_i \frac {\log (1+\exp (-y[i]*x[i]))} {x.nelement()}$$

其中 `y` 中元素取值为 1 或 -1。

为简单起见，考虑标量值 `x,y` （单标签），根据 logistic 函数
$$f(x)=\frac 1 {1+e^{-x}}$$
似然函数为 
$$L(x,y)=\mathbb I(y=1)f(x)+\mathbb I(y=-1)(1-f(x))=(1+e^{-yx})^{-1}$$
故损失函数为
$$loss(x,y)=-\log L(x,y) = \log (1+\exp(-yx))$$

## 6. CrossEntropyLoss
`input` 为各分类的原生得分，未进行归一化，size 为 `(N,C)`，`target` 的 size 为 `(N,)`，其中元素值表示分类 id。损失计算如下
$$loss(x,y=c)=-\log\frac {\exp(x_c)} {\sum_j^C \exp(x_j)}$$

以单个样本为例，`x` 包含 C 个分类得分的 vector，`y` 是 one-hot vector，长度也为 C，值为 1 的元素下标为 true 分类 id，记为 `c`。根据交叉熵公式
$$H(p,q)=-\sum_i p(x_i) \log q(x_i)$$
单个样本的交叉熵损失为
$$L=-\sum_{i=1}^C y_i \log x_i = - y_c \log x_c $$
其中 $x_c$ 经过 Softmax 归一化。


## 7. MultiLabelSoftMarginLoss

这是 SoftMarginLoss 的多分类版本，即多标签多分类，标签数量小于等于分类数量，为了使 `target` 的向量长度一致，设置标签数量为分类数量，对于不存在的标签，则设置对应的 target 值为 -1，例如分析某电影数据，分类总共为 4 类，如下

```
romance, comedy, horror, action
```

那么，对于任意一部电影，其标签最多有 4 个，如果不足 4 个，那么缺失的分类设置标签 target 值为 0 或者 -1，这里使用 0 表示，例如某电影分类为 `comedy,action`，那么其 target 值按以上顺序为 `(-1, 1, -1, 1)`。

MultiLabelSoftMarginLoss ，对某个样本而言，损失计算如下
$$loss(x,y)=-\frac 1 C * \sum_i y[i] * \log((1+\exp(-x[i]))^{-1})+(1-y[i])*\log \frac {exp(-x[i])}{1+\exp(-x[i])}$$
其中 $i \in \{0,...,x.nElement()-1\}$，`x` 元素数量为标签数量，$y[i] \in \{0,1\}$。可见每个 channel 独立做一次二分类，采用的 logistic 回归损失，然后所有 channel 做一次平均。


## 8. CosineEmbeddingLoss
损失用于测试两个输入之间相似度，target 标签为 1 或者 -1，1 表示两个输入相似，-1 表示不相似。损失计算如下
$$\text{loss}(x,y)=\begin{cases} 1- \cos (x_1, x_2), & y=1
\\\\ \max(0, \cos(x_1,x_2)-\text{margin}), & y=-1 \end{cases}$$

显然，
1. 当 `y=1` 时，$x_1, x_2$ 越相似，损失越小
2. 当 `y=-1` 时，$x_1,x_2$ 越相似，损失越大


## 9. MarginRankingLoss
测量两个输入的排列等级的损失。输入 $x_1, x_2$ 通常是 1D mini-batch 的 tensor，目标 label 是 1D mini-batch 的 tensor，包含 1 或 -1 两种值。损失计算：
$$\text{loss}(x,y)=\max(0, -y*(x_1-x_2) + \text{margin})$$
1. 当 `y=1` 时，期望 $x_1$ 排列等级比  $x_2$ 高，所以当 $x_1 > x_2$ 时，损失应该为 0
2. 增加了 `+margin` 这一项，说明 $x_1$ 不但要大于 $x_2$，还要超过至少 `margin` 的量才可以忽略损失
3. 当 `y=-1` 时，情况相反


## 10. MultiMarginLoss
多分类的带 margin 的 hinge 损失。输入 x 是一个 2D 表示 mini-batch 的 tensor，size 为 `(N,C)`， target 是一个 1D 的 size 为 `(N,)` 的 tensor，各元素值为分类 id。

对于单个样本而言，输入 input 则为 vector，target 为标量，表示这个样本的真实分类 id，损失计算如下
$$\text{loss}(x,y)=\frac {\sum_i \max [0, \text{margin}-(x[y]-x[i])]^p} {x.size(0)}$$
输入 `x` 的元素值表示各分类的得分，显然真实分类 id `y` 的得分越大越好，其他分类 id 的得分越小越好，增加 `margin` 项，说明真实分类 id 的得分必须超过其他分类 id 的得分且拉开至少 `margin` 的差距才忽略损失，否则计入损失。

## 11. TripletMarginLoss
给定三个输入，计算三者损失，三个输入记为 `a,p,n`，分别表示 anchor，positive 和 negative examples，所有输入 tensor 的 size 均为 `(N,D)`。

对于第 i 个样本，损失计算如下
$$L(a,p,n)=\max \{d(a_i, p_i)-d(a_i,n_i) + \text{margin}, 0\}$$
其中 $d$ 表示距离 $d(x_i,y_i)=\|\mathbf x_i - \mathbf y_i\|_p$。从损失计算式中可见，我们希望 anchor 与 positive 的距离近，anchor 与 negatvie 的距离远。增加 `margin` 项进一步强制 anchor 与 positive 更近，或，anchor 与 negative 的距离更远。

## 12. CTCLoss
