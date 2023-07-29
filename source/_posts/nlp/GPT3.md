---
title: GPT3
date: 2023-07-26 12:00:37
tags:
    - NLP
    - GPT
mathjax: true
---

论文：[Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)

# 1. 简介

NLP 任务通常是在大语料上预训练，然后在具体任务上 fine-tuning，但是 fine-tuning 的数据集获取有时候仍较为困难。人类在执行语言任务时仅需要少数示例甚至是简单的指令，目前的 NLP 系统仍达不到这样的水平，所以作者继续增大模型，使其具有 task-agnostic few-shot 性能，即本文的 GPT3 。


1. task-agnostic：训练阶段，不知道训练模型用于哪个具体任务

2. few-shot：（inference 阶段）给模型提供 任务描述以及一些示例。

    例如告诉模型：任务是找出一组数序列里面的素数，如果素数数量大于 3，那么仅找出最大的 3 个素数。一个示例是 `2,3,4,5,6`，输出 `2,3,5`，再例如一个示例 `2,3,4,5,6,7,8,9`，输出 `3,5,7` 。

    one-shot：仅给模型提供一个示例。

    zero-shot：不给模型提供示例。例如描述任务：“请告诉我这个句子描述的是开心的还是伤心的事情”


3. in-context learning：利用 context 信息学习。inference 阶段，给 context window 中填充一些示例，模型根据这些示例匹配出相应的模式（这种模式是在训练阶段学习获得），然后以此模式完成任务。

    大模型能学习多种技能，并具有识别多种模式的能力。训练阶段，学习可根据给定的 context words 预测下一个单词，这同时也让模型具有了识别模式的能力。

图 1 是 zero-shot, one-shot 和 few-shot 与传统的 fine-tuning 的对比，前三者是没有梯度更新的，而 fine-tuning 有梯度更新。

![](/images/nlp/GPT3_1.png)

<center>图 1. </center>

# 2. 方法

## 2.1 模型

GPT3 与 GPT2 结构相同，包含了 GPT2 修改过的初始化（residual layer 的权重修改为 $1/\sqrt N$），预归一化（layer norm 前置），以及可逆分词（tokenization），不同点为，

1. GPT3 有 96 个 layers，每个 layer 的 head 数量为 96
2. transformer 维度为 12888
3. context size 为 2048
4. 密集 和 locally banded 稀疏注意力模式交替使用


作者为了对比，设计了 8 个大小不同的模型，其中最大的模型参数就是 GPT3 。

## 2.2 训练集

模型更大，数据集也需要更大，Common Crawl 包含近万亿个单词，虽然这个数据集大小足够用来训练最大模型（GPT3），但是如果不对数据集进行筛选，训练出来的模型质量并不好。提高数据集质量分 3 步：

1. 使用高质量引用语料，从 CommonCrawl 中根据相似度筛选

    使用 WebText 作为高质量文档，训练一个分类器，用以区分文档来自 WebText 还是 Comom Crawl。然后使用这个训练好的分类器从 Common Crawl 中重新采样，方法是优先选择被分类器预测得分高的那些文档。分类器使用逻辑回归 $\sigma$，文档特征使用 Spark standard tokenizer 和 Hashing TF 。
    
    Common Crawl 中的文档通过分类器后的得分如果满足下式，那么就保留，否则丢弃

    ```python
    alpha = 9
    np.random.pareto(alpha) > 1 - document_score
    ```

2. 文档级别去重，避免冗余，并保持验证集的完整性，从而可以测量是否过拟合

    为了进一步提高模型质量，防止过拟合，使用 fuzzy 去重：两个文档高度重叠，那么去除其中一个。每个数据集内部进行 fuzzy 去重（而非跨数据集去重），使用 Spark MinHashLSH 方法去重。

    除了去重，作者还将在 benchmark 数据集中出现过的文本移除。

3. 将高质量引用语料加入训练集，增强 CommonCrawl，并增加其多样性

    高质量语料：
    - WebText 的一个扩展版本，通过网络爬虫抓取链接页面数据
    - 网络书，Books1 和 Books2
    - 英文 Wikipedia

![](/images/nlp/GPT3_1.png)

<center>表 1. 训练 GPT3 的混合数据集</center>

表 1 中，第一列为数据集名称，第二列为数据集中 tokens 数量，第三列表示从数据集中随机选择 example ，随机选择各数据库的权重（即，各数据库不是等概率被选中），第四列表示每训练 300 B 个 tokens，各数据库被全部遍历的次数，例如每训练 300B 个 tokens，相当于有 `300B * 3% = 9B` 个 tokens 来自 Wikipedia，而 Wikipedia 总共才 3B 个 tokens，所以相当于 Wikipedia 数据库被轮了 `9B / 3B = 3` 次。

## 2.3 训练过程

大模型使用大的 batch size，但是学习率较小。

1. Adam 优化器参数 $\beta _ 1 = 0.9, \ \beta _ 2 = 0.95, \ \epsilon=10 ^ {-8}$ 

2. 将梯度的 global norm 限制在 1.0 以内 。

    通常 clip 梯度都是对各个参数独立进行的，但是 clip global norm，则是将所有参数的梯度（tensor list）进行 rescale，使得 scaled 梯度的 norm 之和为 1.0 。过程如下，

    ```python
    global_norm = sqrt(sum([l2norm(t)**2 for t in gradient_tensors]))
    gradient_tensors *= 1.0 / global_norm
    ```

3. 每训练 260 billion tokens，学习率 cosine 衰减到 10%，即衰减率 $r$ 为 

    $$r = \frac 1 2 [1+\cos(\pi i / T)] (1-\alpha) + \alpha$$

    其中 $\alpha = 0.1, \ T = 260$

4. 训练最开始的 375 million tokens 时，对学习率线性 warmup 。

## 2.4 Evaluation

few-shot learning，对 evaluation 数据集中的每个 example，都从具体任务的训练集中随机选择 K 个 examples 作为条件，使用 1 或者 2 个换行符将 examples 分隔。 LAMBADA 和 Storycloze 没有带监督的训练集，所以从 development set（验证集）中选择 K 个 exampls 作为条件，然后在 test set 上评估（evaluate）。Winograd 只有一个数据集，那么就从这个数据集上取 K 个条件 examples，评估也在这个集合上进行。

K 的取值范围为 0 到模型 context window size 允许的最大值 。context size 为 `2048`，所以允许的 examples 数量从 10 到 100 不等。

对那些从给定的几个选项中选择一个正确 completion 的任务，K 个 examples 为 `<context> <completion>`，后面还有一个 example：`<context>` （只有 context，没有 completion）。