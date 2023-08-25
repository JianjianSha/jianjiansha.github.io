---
title: V-MoE 论文解读
date: 2023-08-23 18:23:25
tags: transformer
mathjax: true
---

# 1. 介绍

稀疏 Mixture of Experts 网络（MoEs）在 NLP 中展现了优秀效果。
本文提出 Vision MoE，记为 V-MoE，这是一个稀疏版本的 ViT，相较于以往的“密集”网络结构，V-MoE 在保持 SOTA 性能的同时，推理时间降为一半左右。V-MoE 的核心思想是将 ViT 中 transformer 的部分前馈层（即，MLP）替换为稀疏 MoE ，如图 1 的顶部一行，V-MoE 有 $L$ 个 transformer blocks，其中间隔性的有一半的 block 中的 MLP 被替换为稀疏 MoE 。

![](/images/transformer/VMoE_1.png)
<center>图 1.</center>

图 1 下半部分是稀疏 MoE 的结构，关于稀疏 MoE 的介绍请接着往下看。

# 2. 视觉 MoE

**MoE 条件型计算**

单个 MoE layer 有 $E$ 个 experts，每个 expert 为一个 MLP ，记某个 expert 函数为 $e _ i: \mathbb R ^ D \rightarrow \mathbb R ^ D$，每个 expert 有一个权重 $g _ i : \mathbb R ^ D \rightarrow \mathbb R$，于是 MoE layer 可表示为

$$\text{MoE}(\mathbf x)=\sum _ {i=1} ^ E g _ i(\mathbf x) \cdot e _ i (\mathbf x) \tag{1}$$

称 $\mathbf g: \mathbb R ^ D \rightarrow \mathbb R ^ E$ 为 routing 函数。expert 函数为 MLP，于是 

$$e _ i = \text{MLP} _ {\theta _ i} (\mathbf x)=\mathbf W _ 2 ^ i \text{ReLU}(\mathbf W _ 1 ^ i \mathbf x) \tag{2}$$

其中 $\theta _ i = (\mathbf W _ 1 ^ i , \mathbf W _ 2 ^ i)$ 为参数。

(1) 式和 (2) 式中的 $\mathbf x \in \mathbb R ^ D$ 均表示一个 token（的向量），一个 token 对应 image 的一个 patch （回顾一下，ViT 将一个 image 切分为固定数量的 patches）。

稀疏 MoE 的稀疏性是指部分 experts 不用参与计算，即对应的 $g _ i = 0$ ，从而相应的 expert 函数不需要计算，节省了计算。

**routing**

routing 函数为 $\mathbf g(\mathbf x)=\text{TOP} _ k (\text{softmax} (\mathbf W \mathbf x + \epsilon))$，其中 $\text{TOP} _ k$ 是取 top k 大的值，其他值则置为 0 。随机变量 $\epsilon \sim \mathcal N(\mathbf 0, 1/E ^ 2)$ 。

根据 $\mathbf g: \mathbb R ^ D \rightarrow \mathbb R ^ E$，可知 $\mathbf W \in \mathbb R ^ {E \times D}$ 。

现在再看图 1 下左，一个 image 由 4 个 tokens 组成（这里仅用作示例说明），每个 token 的颜色逐渐变浅，不同的 image 使用不同的颜色。再看图 1 下右，MoE layer 中每个 expert 部署在不同的 device 上。对每个 token 进行 routing，假设选择 $k=1$，这表示每个 token 被路由到 1 个 expert，见图中虚线箭头标示。实线箭头表示将一组 images 的所有 tokens 分发到不同的 expert 设备上，经过 MLP 输出后，再根据 router 恢复 tokens 顺序。

**重点：** 不同 expert 的模型参数均不同，也就是说，同一个 image 中的不同 tokens 可能会经过不同的 MLP，但是计算 $\text{TOP} _ k$ 时，对所有 images 的所有 tokens 均使用相同的权重矩阵 $\mathbf W \in \mathbb R ^ {E \times D}$ 进行映射。

**expert's 缓存容量**

为了让各个 experts 负载均衡，从而提高硬件利用率，本文定义了一个 expert 的缓存容量 $B _ e$，

$$B _ e = \text{round} (\frac {k NPC} E) \tag{3}$$

其中 $N$ 为 batch size（一个 batch 中 images 数量），$P$ 为一个 image 中 tokens 数量，$k, E$ 见上文定义。$C$ 为容量率（capacity ratio），这几个参数都是超参数，预先设定的。

根据 (3) 式可知，一个 MoE layer 中所有 experts 的容量之和 $B _ e \cdot E$ 应该等于一个 batch 中所有 tokens 总共被分配到 expert 的次数 $kNP$。增加一个 C，用于调节 expert 的容量，即 $B _ e \cdot E = kNPC$ 。

如果 router 分配超过 $B _ e$ 个 tokens 到一个 expert 上，那么仅有 $B _ e$ 个 tokens 被 expert 处理，其余 tokens 的信息也没有完全丢失，因为还有 residual 连接，如图 1 顶部。如果 router 分配不足 $B _ e$ 个 tokens 到 expert 上，那么不足的部分使用 zero 填充。

图 1 中，每个 device 上平均的 mini batch 含有 $N=3$ 个 image，每个 image 分成 $P=4$ 个 tokens，每个 expert 的容量为 $16$，那么容量率为 $C = 16/3/4=4/3$ 。

## 2.1 源码解读

[code](https://github.com/google-research/vmoe)