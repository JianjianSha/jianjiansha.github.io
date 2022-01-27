---
title: Connectionist Temporal Classification (2)
date: 2021-12-04 18:57:10
tags: deep learning
mathjax: true
---

上一篇文章介绍了 CTC 的算法和损失函数，现在继续讨论 CTC 的推断部分。

<!--more-->

# 推断

训练好模型后，可以对一个新输入进行推断，求最可能的输出序列，

$$Y^{\star}=\arg \max_Y p(Y|X)$$

一种方法是在每个时刻分别独立地求最有可能的输出，即

$$A^{\star}=\arg\max_A \prod_{t=1}^T p_t(a_t |X)$$

然后 $Y=B(A^{\star})$ 得到最终的输出序列。

这种方法某些情况下效果很好，尤其当每个时刻的输出概率向量中，大部分 概率 mass 集中在单个元素上。但是这种方法没有从全局角度进行考虑，有时候得到的输出序列并非概率最大的那个，例如

假设 RNN 输出 $[a,a,\epsilon]$ 和 $[a,a,a]$ 的概率均比 $[b,b,b]$ 的概率低，但是前两者的概率和比第三者的概率高，如果采用上面的那个简单方法，得到输出序列为 $[b]$，然而实际应该是 $[a]$。

我们采用 beam search 方法来解决这个问题。常规的 beam search 方法如图 1 所示，

![](/images/dl/CTC4.png)

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<center>图 1 （来源于 ref 1）</center>

将常规的 beam search 加以修改，使得适合本问题：将连续重复的值进行合并，并移除 $\epsilon$，在每个时间步，对每个 prefix 序列，累加所有有关的标签序列，这些标签序列均 collapsed 到这个 prefix，如图 2，

![](/images/dl/CTC5.png)
图 2. CTC beam search 算法，输出字母表为 $\{\epsilon, a, b\}$。

需要注意的是，在 `T=3` 时，prefix 为 `[a]`，proposed extension 为 `a` 时，输出（即叠加后的 prefix）可以是 `[a]`，也可以是 `[a, a]`，后者由 $[a, \epsilon, a]$ 转化而来，也就是说此时 prefix 实际是 $[a, \epsilon]$：ending in $\epsilon$。所以，对于 prefix，我们需要在内部存储其 ending in $\epsilon$ 和 not ending in $\epsilon$ 两个概率（prefix 概率为这两个概率之和）。如图 3 所示，

![](/images/dl/CTC6.png)

图 6. `T=2` 时（图中间部分），prefix 为 $[a]$，proposed extension 分别取 $\epsilon$ 和 $a$ 时，均输出 $[a]$，然而，前者实际上是 ending in $\epsilon$，看 `T=3` 部分中蓝色节点 $[a]$ 的下方较小地显示出来。

整个代码实现参考这个代码片段 [gist](https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0) 。

# 参考
1. [Sequence Modeling With CTCT](https://distill.pub/2017/ctc/)