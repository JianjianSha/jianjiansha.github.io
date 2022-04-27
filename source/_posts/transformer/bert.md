---
title: BERT
date: 2022-03-31 10:51:09
tags: transformer
mathjax: true
---
# 1. 简介

## 1.1 参数量
$BERT_{BASE}$ $L=12, H=768, A=12$
$BERT_{LARGE}$ $L=24, H=1024, A=16$

参考 Transformer 结构，L 是 Encoder 中 block 的数量，即 《Attention is all you need》 中的 $N \times$，$A$ 是 multi-head attention 中的 head 数量，$H$ 是隐藏层的单元数，即模型宽度，也就是 《Attention is all you need》 中的 $d$。

于是，multi-head attention 中 单个 head 的模型宽带为 $H/A=64$。

1. embedding， 一个矩阵 $\mathbb R^{V\times H}$，其中 $V$ 表示词汇集大小，$H$ 为 embedding 维度，由于 Transformer 中有 identity shortcut 连接，所以 embedding 维度必须与 attention 输出维度相同，故 embedding 维度也是 $H$ 。

2. Attention 中 $Q,K,V$ 的映射矩阵均为 $W \in \mathbb R^{H \times H}$，single/multi-head attention 均为这个大小。

3. Attention 中输出 $O$ 的映射矩阵也是 $W \in \mathbb R^{H \times H}$

4. attention 上方是 feed-forward layer，这是一个由两个 全连接 组成的 layer，Bert 论文指出，feed-forward/filter size 为 $4H$，所以这两个 全连接 的维度转换应该是 $H \rightarrow 4 H, \ 4H \rightarrow H$，参数量为 $2 \times 4\times H \times H=8H^2$

于是一个 Transformer block 的（可学习）参数量为 $4H^2+8H^2=12H^2$

$L$ 个 Transformer block 的参数量为 $12LH^2$，总参数量为 $VH+12LH^2$，

论文中指出所用词汇集大小为 $V=30000$，然后根据 Bert base 的超参数 $L=12,H=786$，计算出总参数量为 $112542624$ （Bert large 模型的参数可类似地计算出来）。


## 1.2 词嵌入

例如以空格进行分词，那么对于大数据集，token 数量会特别大，导致模型参数集中在嵌入层，为了解决这个问题，使用 WordPiece embedding。具体而言：使用一个 word 的子序列（字符串的前 n 个字符构成的子串），这个子序列有可能就是这个 word 的词根。参考 [Google’s neural machine translation system: Bridging the gap between human and machine translation](https://arxiv.org/abs/1609.08144)

## 1.3 特殊 token
[CLS] 表示句子的整体信息，这个 token 的最终表示，即最后一个 Transformer block 的输出 （是一个 $H$ 长度的向量）就是这个句子信息的 BERT 表征。

每个句子末尾添加 [SEP] token 表示句子结束。

BERT 的输入序列可以是单个句子，也可以是句子对（两个句子）。输入是单个句子是，输入序列是 “[cls]”、文本序列的标记、以及特殊分隔词元 “[sep]”的连结。当输入为文本对时，BERT输入序列是“[cls]”、第一个文本序列的标记、“[sep]”、第二个文本序列标记、以及“[sep]”的连结

## 1.4 输入表示

每个 token 由三个 embedding 构成：
1. word embedding。这就是 word 自身的词嵌入表示
2. segment embedding。由于 BERT 使用两个句子作为输入，那么每个句子的 token 均需要一个额外 embedding 表示是第一个句子还是第二个句子。
3. position embedding。与原生 Transformer 相同，由于 attention 中各输入是未知无关的，所以需要额外增加一个位置信息的 embedding。

三种 embedding 的维度均为 $H$，相加得到最终的 embedding 表示（维度仍为 $H$）。

# 2. 预训练 BERT
## 2.1 Masked LM

标准的条件语言模型只能进行从左到右或者从右到左地训练，数学模型为
$$p(x_i|x_1,...,x_{i-1}) \\
p(x_i|x_{i+1},...,x_T)$$

双向条件的训练的数学模型为
$$p(x_i|x_1,...,x_T)$$

这使得每个 word 可以看见自己，导致模型可以平凡地预测目标值。

为了训练出一个深度双向的表征，作者随机对 token 进行掩码处理，然后预测这些被掩盖的 token，这称为 `masked LM`，这些 masked token 的最后一个 Transformer block 的输出特征，输入到一个 fc 层，输出向量表示词汇表各 word 的得分，然后使用 softmax 进行预测。

每个输入序列的 15% 的 wordpiece 被 mask，即使用 `[MASK]` 替换原来的 token。对于 `[CLS], [SEP]` 则不参与 mask。

但是这种 mask 处理也会带来不匹配问题，在 fine-tuning 阶段，`[MASK]` 可不会出现。因为 fine-tuning 阶段的目标不同，并非用于预测被 mask 的 token，所以不会有 `[MASK]`。

为了解决这个不匹配问题，对于每个被选中的待 mask 的 token：

1. 80% 训练时间使用 `[MASK]` 替换
2. 10% 训练时间使用一个有效的随机 token 替换
3. 10% 训练时间保持不变，就是原来的 token

## 2.2 NSP

Next Sentence Prediction(NSP)


许多下游任务例如 问答（QA）以及 自然语言推断（NLI）均基于两个句子间关系的理解。为了训练出一个能理解两个句子间关系的模型，作者设计了一个二分类的下一句预测任务，即给两个句子作为输入，预测第二个句子是否是第一个句子的下一句。

训练时，每个训练样本中的句子 A 和 B，50% 的概率下 B 是 A 的下一句，即 label 为正，另外 50% 的概率下 label 为负。将 `[CLS]` 的最终向量用于预测是否是下一句。

**预训练数据：**

使用 BooksCorpus 和 English Wikipedia 数据集。作者使用 document-level 的语料而非 sentence-level 语料。Document 中，可以获得连续的两个句子（在 document 中自然连续，不一定要有语义上的某种关联）。

# 3. 实验

## 3.1 GLUE

输入序列（单个句子或者句子对），使用 `[CLS]` 的最终特征向量 $C \in \mathbb R^H$，作为输入序列的聚合表征。额外增加的分类层是一个 fc 层和 Softmax，fc 层参数 $W \in \mathbb R^{K \times H}$，其中 $K$ 表示分类数量。

`batch_size=32`

对所有 GLUE 任务 fine-tune `epoch=3` 轮。

使用 $5e^{-5}, 5e^{-5}, 4e^{-5}, 3e^{-5}, 2e^{-5}$ 不同的学习率，最终选择在 Dev set 验证数据集上最好的那个。

BERT 与其他模型在 GLUE 上的性能比较看原论文 Table ，这里不列出来了。

## 3.2 SQuAD v1.1
这是 Standford 的一个 QA 问答数据集。给定一个问题和一个段落（来自 Wikipedia，且包含了问题的答案），这个任务目标是预测答案在段落中的起始截止位置。

如图 1，

![](/images/transformer/BERT1.png)

<center>图 1 </center>

将问题和段落分别作为输入序列中的 A 和 B。

引入一个 start 向量 $S \in \mathbb R^H$ 和一个 end 向量 $E \in \mathbb R^H$。这两个向量相当于两个 fc 层的权重参数，将每个 word 的最后的 hidden 向量映射到一个非归一化得分，这两个 fc 层的权重参数均在各个 word 中共享。

例如向量 $S$，将 word `i` 的最后一个 Transformer block 的输出向量 $T_i$ 映射为得分 $S \cdot T_i$，那么经 softmax 后可得 word `i` 为答案的 start 的概率为

$$P_i=\frac {e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}} \tag{1}$$

其中 $j$ 的范围是输入序列中 $B$ 的下标范围（or 整个输入序列范围？）。

对于 $E$ 同样处理，即 word `j` 作为答案 end 的得分为 $E\cdot T_j$。于是，word `i` 和 word `j` 构成一个答案 span 的得分为 $S \cdot T_i+E\cdot T_j$，具有最大得分的 word pair，且 $j \ge i$ 就是最终预测的答案 span。

训练的目标函数是正确的 start 和 end 的对数似然之和，即最大化下式的值（梯度上升）

$$\log (P_s \cdot P_e)=\log P_s + \log P_e \tag{2}$$

其中 $s, \ e$ 分别是 gt start 和 gt end 的下标。

fine-tune `3` 个 epoch，`batch_size=32`，学习率为 $5e-5$。

## 3.3 SQuAD v2.0
SQuAD v2.0 任务是对 SQuAD v1.0 的扩展，使得对应的段落中有可能不存在答案，这更加接近现实。

若段落中没有问题的答案，那么答案的 start 和 end 均在 `[CLS]` 这一位置，于是答案 start 和 end 的概率空间需要包含 `[CLS]` 的位置。

预测时，gt 对应的得分为 $s_{null}=S\cdot C+E\cdot C$，其中 $C$ 是 `[CLS]` 对应的最后一个 Transformer block 的输出特征向量，同时计算出最佳 non-null 的得分 $\hat s_{ij}=\max_{j \ge i} S\cdot T_i+E\cdot T_j$，那么当 

$$\hat s_{ij} > s_{null}+\tau$$

时，预测为这个最佳 non-null 的答案 span，否则，预测为 null 答案（即不存在答案）。这里 $\tau$ 给出几个值，选择其中某个值使得 F1 最大。增加 $\tau$ 这个阈值是为了使预测更加准确。

训练阶段，依然使用 (2) 式进行训练，其中没有答案的情况下， gt label $s, \ e$ 均为 `[CLS]` 的位置。（1）式中分母的求和项，也需要包含 `[CLS]` 的得分。

## 3.4 SWAG

SWAG 数据集介绍：

给定一个部分描述，例如 “她打开汽车引擎盖”，那我们可以想到下一个场景可能是 “然后她检查了引擎”。SWAG 就是用于常识性推断这样一类任务的数据集，包含 113k 的问题描述，每个问题有四个选项，任务就是从中选择一个可能下个场景出现的。

在 SWAG 上 fine-tune  BERT 时，构造一个四输入序列，每个序列包含问题（序列A）和选项（序列B），对应的 Transformer 输入 Tensor 的 shape 为 `(batch_size, 4, max_length, embedding_dim)`，输出 Tensor 的 shape 为 `(batch_size, 4, max_length, H)`，其中 $H$ 就是上文所提的模型宽度，四个序列的 `[CLS]` token 对应的输出向量分别乘以一个权重向量（这就是任务相关的额外添加的 fc 层），得到这四个序列的得分，然后使用 softmax 进行分类。


训练 `3` 个 epoch，学习率为 $2e-5$，`batch_size=16`。

每个数据集上 BERT 与其他 models 的性能，详见论文，这里不再说明。