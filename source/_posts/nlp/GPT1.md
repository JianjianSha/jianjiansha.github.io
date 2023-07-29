---
title: GPT1
date: 2023-07-25 10:46:28
tags:
    - NLP
    - GPT
mathjax: true
---


论文：[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

源码：[openai/finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm)

# 1. 介绍

SOTA 的 NLP 模型通常都是在特定任务上使用监督数据集训练而得，但是这会带来两个限制：

1. 需要大量的标注数据，而这往往不可得

2. 难以泛化应用到其他任务

本文作者证明 NLP 的任务（文本蕴含，问答，语义相似度，文本分类等）可以通过 Generative Pre-training Transformer（GPT）一个非监督模型（在大量多样的非标注文本数据集上非监督训练），然后在具体的特定任务上监督的微调（fine-tuning），这种方法也称为半监督方法。

非监督的预训练使用大的未打标语料，目的是训练得到通用表征。下游的特定任务称为 target 任务。

# 2. 框架

训练过程分两个阶段：

1. 学习一个具有高容量的语言模型，使用未标注的大语料。
2. 使用标注数据对模型 fine-tuning，使得适应特定任务

## 2.1 非监督预训练

非监督语料由 token 序列构成 $\mathcal U = \lbrace u _ 1, \ldots, u _ n \rbrace$ ，标准语言模型的目标是最大化对数似然函数，

$$L _ 1 (\mathcal U ) = \sum _ i  \log P(u _ i |u _ {i-k}, \ldots, u _ {i-1}, \Theta ) \tag{1}$$

(1) 式条件概率 $P$ 使用神经网络模型，其参数为 $\Theta$ 。

对于位置为 $i$ 的 token，$i \in [1,n]$，这个 token 的条件概率与前面 $k$ 个 token 有关，由于 GPT1 的上下文窗口（context window）size 固定为 `512`，所以当：

1. $i \le 512$ 时，$k=i-1$
2. $i > 512$ 时，$k=512$



本文使用多层 Transformer decoder 作为语言模型，这是 Transformer 的一个变体。模型对输入的上下文 tokens 应用一个 multi-head self-attn，然后应用 position-wise 前馈层（实际上是包含两个 FC 的 MLP），multi-head self-attn 层和前馈层组成一个 transformer block（参见 [attention](/2022/01/17/transformer/self_attention) 一文中的图 3），一共使用 `n` 个 transfromer blocks，输出 target tokens，整个过程使用数学符号表示如下，

$$\begin{aligned} h _ 0 &= U W _ e + W _ p
\\\\ h _ l &= \text{transformer\_ block} (h _ {l-1}), \ \forall l \in [1,n]
\\\\ P(u) &= \text{softmax}(h _ n W _ e ^ {\top})
\end{aligned} \tag{2}$$

其中 $U = (u _ {-k}, \ldots, u _ {-1})$ 是 tokens 的上下文向量，$n$ 是 transformer layers 数量，$W _ e$ 是 token 向量的 embedding matrix，也就是说将 token vector 转为 embedding vector ，$W _ p$ 是 position embedding matrix 。

记单个 token 上下文向量 size 为 $c$（例如 one-hot vector，此时 $c=V$ 为词汇表 size，下文则直接全部使用 $V$），token embedding 向量 size 为 $d$（将 one-hot vector 转为 `d-dim` 词嵌入向量），那么 $U \in \mathbb R ^ {k \times V}$，$W _ e \in \mathbb R ^ {V \times d}$，$W _ p \in \mathbb R ^ {k \times d}$ ，所以 $h _ 0 \in \mathbb R ^ {k \times d}$ 。

transformer block 的输入为上一个 layer 的输出 $h _ {l-1}$ ， 通过 Q, K, V 三个矩阵将 $h _ {l-1}$ 做一个线性变换，Q, K, V 三个矩阵维度均为 $d \times d$，所以输出与输入相同，整个 transformer block 的输出 size 也保持不变，故 $h _ l \in \mathbb R ^ {k \times d}$，最后 $P(u)$ 的 size 为 `k x V` 。

**# 参数量计算**

（参考 [Bert](/2022/03/31/transformer/bert) 一文的参数量计算。）

GPT1 的汇表数量为 40478，所以 $c=V=40478$（[词汇表文件](https://github.com/openai/finetune-transformer-lm/blob/master/model/encoder_bpe_40000.json)）， 模型维度为 $d=768$，transformer block 中前馈网络由两个 FC 层组成，第一个 FC 层输出为 $4d=3072$，那么

1. 得到 $h _ 0$ 的 layer 参数为 $W _ e, \ W _ p$，数量为 $V \times d + k \times d = (40478 + 512) * 768$
2. transformer block 参数量为 $12 \times 12  \times d ^ 2=144*768*768$

以上两个数量相加得到 `116,414,976`，即参数量为 `~117M`。

## 2.2 监督微调

根据 (1) 式训练好模型后，将模型参数通过监督微调适应 target 任务。记有标注的数据集 $\mathcal C$ ，其中每个数据样本包含一个输入 token 序列 $x ^ 1, \ldots, x ^ m$，以及一个 label $y$ 。

输入经过上面预训练的模型，最后的 transformer block 输出为 $h _ l ^ m$ ，这个输出再通过一个线性 layer 得到输出 $\hat y$ ，用于预测 $y$ ，

$$P(y|x ^ 1, \ldots, x ^ m) = \text{softmax} (h _ l ^ m W _ y) \tag{3}$$

其中 $W _ y$ 为最后输出的线性 layer 的参数。 优化目标为最大化下式，

$$L _ 2 (\mathcal C)=\sum _ {x, y} \log P(y|x ^ 1, \ldots, x ^ m) \tag{4}$$

在 fine-tuning 阶段，作者发现将预训练的优化目标包含进来有助于学习，原因是：

1. 提高监督模型的泛化程度
2. 加速收敛

所以，优化目标改为最大化下式，

$$L _ 3 (\mathcal C) = L _ 2 (\mathcal C) + \lambda L _ 1 (\mathcal C) \tag{5}$$

## 2.3 任务特定的输入转换

一些任务如文本分类，我们可以按上面的方法直接微调，然而有些任务，例如问答或者文本蕴含，其输入结构为 有序句子对，或者三元组（文档，问题，答案），而上述模型是在文本序列上预训练，所以需要做修改。

我们将结构化输入转换为一个有序序列，使得模型能够处理，这种调整不需要改变模型架构。如图 1，给出了输入转换的具体操作，

![](/images/nlp/GPT1_1.png)

<center>图 1. 输入转换操作步骤</center>

图 1 左边部分是 transformer 结构，最后输出是文本预测（token 预测）或再加一个线性 layer 输出文本分类。

1. 文本蕴含。根据前提 `p` 能否推出假设 `h`，可以有三种结果（蕴含，冲突，中立）。

    此任务中，将 `p` 和 `h` 连接起来（concatenate），中间使用 `Delim` 分隔。Delim 可以使用 `$` 符号。

2. 相似度。 两个文本之间的相似度计算。

    这两个文本不应该有顺序，也就是说，没办法确定哪个在前哪个在后，既如此，就两种情况都考虑，最后得到两个 $h _ l ^ m$，再 element-wise 相加，最后送入一个线性 layer，得到输出结果。

3. 问答和常识推理。这类任务的输入为三元组：1. 上下文文档 `z`；2. 问题 `q`；3. 可能的答案集 $\lbrace a _ k \rbrace$ 。处理输入的方法是：将上下文文档、问题与每个可能的答案 concatenate，得到 $[z;q;\$; a _ k]$，注意只有一个 Delim。每个 concatenated 序列独立地通过模型并线性映射，最后一起进行 softmax。

concatenated 序列在首尾需要加上 start token 和 end token $(\langle s \rangle, \langle e \rangle)$ 。图 1 中的 `Extract` 应该是 end token ？

# 3. 实验

**# 数据集**

使用 BooksCorpus 数据集，包含 7000 个独立的未发布书籍，体裁多样，包括冒险，玄幻和浪漫。一个关键点是它包含长段的连续文本，使得通用模型可以学习到长距依赖。


**# 模型细节**

使用 transformer 中的 decoder，decoder 由 12 个 transformer blocks 组成，每个 transformer block 中有一个多头 self-attention，维度总共为 `768`，`12` 个 heads。 position-wise 前馈层（两个 FC layer），隐藏层单元数量为 `3072` （其实就是 768*4）。

**# 一个微调实例**

文章：[Next Word Prediction using GPT-1](https://medium.com/@prerana1298/next-word-prediction-using-gpt-1-ae999acfe3de#:~:text=GPT%20%2D1%20is%20trained%20on,It%20develops%20a%20language%20model%20.)

源码：[Next-word-Prediction-using-Swiftkey-Data
](https://github.com/kurchi1205/Next-word-Prediction-using-Swiftkey-Data/blob/main/GPT%20Model.ipynb)

# 4. 源码分析

## 4.1 官方源码

使用文章顶部所说的 openai 组织提供的项目代码进行说明，以下是一个 fine-tuning 的例子。

**# 数据集**

ROCStories

数据文件

```python
storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, 
                                                                                                  comps1, 
                                                                                                  comps2, 
                                                                                                  ys, 
                                                                                                  test_size=n_valid, 
                                                                                                  random_state=seed)
```

这两个文件均为 csv 文件，包含 7 列，第一行为列名，为

```
InputStoryId：唯一 id
InputSentence1：第一个句子（`.` 符号结尾）
InputSentence2：第二个句子（`.` 符号结尾）
InputSentence3：第三个句子（`.` 符号结尾）
InputSentence4：第四个句子（`.` 符号结尾）
RandomFifthSentenceQuiz1：测试句子1
RandomFifthSentenceQuiz2：测试句子2
AnswerRightEnding：哪个测试句子为正例（取值为 1 或 2 ）
```

这是一个分类任务，预测哪个测试句子是可以根据四个输入句子推理出来。

**# 数据准备**

```python
# 训练集，验证集，测试集
# trX1：[ ' '.join(InputSentence1, InputSentence2, InputSentence3, InputSentence4) ] -> [ [int] ]
# trX2: [ RandomFifthSentenceQuiz1 ]    -> [ [int] ] , 每个句子转为 a list of token ids
# trX3: [ RandomFifthSentenceQuiz2 ]    -> [ [int] ] , 每个句子转为 a list of token ids
# trY: [ AnswerRightEnding ]            -> [ int ], 每个元素取值 0 或 1
(trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3) = encode_dataset(rocstories(data_dir), encoder=text_encoder)
```

增加三个特殊的 token 到词汇表，

```python
encoder['_start_'] = len(encoder)
encoder['_delimiter_'] = len(encoder)
encoder['_classify_'] = len(encoder)
clf_token = encoder['_classify_']
n_special = 3
```

我们的输入序列的形式为 

`[<start id>] + trX1[i] + [<delim id>] + trX2[i] + [<cls id>]` （记为序列 1，第一种序列，下同）

或者

`[<start id>] + trX1[i] + [<delim id>] + trX3[i] + [<cls id>]` （记为序列 2，第二种序列，下同）


对于一个具体的要进行 fine-tuning 的数据集，我们修改 context window size（训练 GPT1 时为 512），context window size 为上面所说输入序列的长度最大值，但是为了防止某个句子长度太大导致输入序列长度超过 512，那就不合适了，毕竟 GPT1 只学习了 512 距离以内的注意力，所以我们先设置一个句子的最大长度，超过这个最大长度的句子都将被截断（简单处理，后面的直接扔掉），根据上面输入序列的规则，为使得输入序列长度 `<= 512`，那么单个句子的长度上限为

$$l_m = \lfloor (512 - 3) / 2 \rfloor = 512 / 2 - 2$$

那么 context window size 设置为

$$l_c = \max _ {i} \left[ \min (l _ i ^ {(1)}, l _ m ) \right] + \max _ {i, j=2,3} \left [ \min (l _ i ^ {(j)}, l _ m) \right] + 3$$

其中 $i$ 表示遍历所有训练集、验证集和测试集的数据，上标 $j=1,2,3$ 分别表示输入句子，测试句子 1 和测试句子 2 ，$l _ i ^ {(j)}$ 表示某个句子的 token 列表 size 。相关代码为，

```python
max_len = n_ctx//2-2
n_ctx = min(max([len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]+...
```

将训练集、验证集和测试集的句子（concat）转为输入序列，

```python
# trX: (N, 2, n_ctx, 2)，分别表示：
#     数据集数量 N，2 种类型的序列，序列长度 n_ctx，序列中 token 在词汇表中的 id 和 token 在序列中的位置 index
# trM: (N, 2, n_ctx)，mask，序列中填充的位置为 0
trX, trM = transform_roc(trX1, trX2, trX3)
vaX, vaM = transform_roc(vaX1, vaX2, vaX3)
if submit:
    teX, teM = transform_roc(teX1, teX2, teX3)
```

这里 position embedding 也使用可学习的位置嵌入向量，而非三角函数编码，词汇表中 token 也是使用可学习的嵌入向量表示，那么统一起来，就是全部使用可学习的嵌入向量，将位置（从 `0` 到 `n_ctx-1`）看作是特殊的 id，为了与词汇表中 token id 统一，那么设置位置 id 为 `V, V+1, ... , V+n_ctx-1`（ V表示词汇表 size，包含了 3 个特殊 token），于是 token id 和 位置 index 均是从嵌入矩阵中查找相关的嵌入向量，故嵌入矩阵 size 为 `(V + n_ctx, n_embd)` （ 其中 `n_embd` 为嵌入向量 size ）。


**# 训练**

训练语句调用，

```python
# 支持使用多 GPU 训练，每个 GPU 训练一批数据，那么所有 GPU 的一批为
n_batch_train = n_batch * n_gpu
X_train = tf.placeholder(tf.int32, [n_batch_train, 2, n_ctx, 2])# 输入序列
M_train = tf.placeholder(tf.float32, [n_batch_train, 2, n_ctx]) # mask（mask padded points）
X = tf.placeholder(tf.int32, [None, 2, n_ctx, 2])
M = tf.placeholder(tf.float32, [None, 2, n_ctx])
train, logits, clf_losses, lm_losses = mgpu_train(X_train, M_train, Y_train)
```

训练方法定义，

```python
def mgpu_train(*xs):
    # xs: 包含输入序列，mask，以及 label
    gpu_ops = []
    gpu_grads = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)    # (x1,m1,y1), ... ,(xn,mn,yn) ，n 个 GPU
    for i, xs in enumerate(zip(*xs)):   # i=0,1,...,n-1
        do_reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
            # 二分类预测得分，二分类损失，self token 预测的损失
            clf_logits, clf_losses, lm_losses = model(*xs, train=True, reuse=do_reuse)
            if lm_coef > 0: # 两个损失的 tradeoff 因子
                train_loss = tf.reduce_mean(clf_losses) + lm_coef*tf.reduce_mean(lm_losses)
            else:   # 只使用二分类损失
                train_loss = tf.reduce_mean(clf_losses)
            params = find_trainable_variables("model")  # 模型所有参数
            grads = tf.gradients(train_loss, params)    # 计算梯度
            grads = list(zip(grads, params))
            gpu_grads.append(grads)                     # 搜集所有 GPU 的梯度和对应参数
            gpu_ops.append([clf_logits, clf_losses, lm_losses]) # 收集所有 GPU 的模型输出
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]    # [all_gpu_logits, all_gpu_clf_loss, all_gpu_lm_loss]
    grads = average_grads(gpu_grads)
    grads = [g for g, p in grads]
    # Adam 优化器更新参数
    train = opt_fns[opt](params, grads, lr, partial(lr_schedules[lr_schedule], warmup=lr_warmup), 
                         n_updates_total, l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2, b1=b1, b2=b2, e=e)
    return [train]+ops
```

上面代码中，我们先重点看模型的前向传播过程，

```python
def model(X, M, Y, train=False, reuse=False):
    # X: (batch, 2, n_ctx, 2) 含义参考上面的 数据准备 一节
    # M: (batch, 2, n_ctx)
    # Y: (batch, )
    with tf.variable_scope('model', reuse=reuse):
        # 统一后的嵌入矩阵
        we = tf.get_variable("we", [n_vocab+n_special+n_ctx, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        # 将嵌入矩阵进行 dropout，防止过拟合
        we = dropout(we, embd_pdrop, train)

        X = tf.reshape(X, [-1, n_ctx, 2])   # (batch * 2, n_ctx, 2)，类型 1 和 2 的输入序列合并到一个 batch 维度
        M = tf.reshape(M, [-1, n_ctx])

        h = embed(X, we)        # 查找嵌入向量，(batch * 2, n_ctx, n_embd)
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)  # transformer decoder 输出

        # 预测   <sent>   <delim>  <comp>  <cls>   score->(1/2?)
        #       feat0     feat1   feat2   feat3   feat4
        #        |         |       |       |       |
        #     ========================================
        #     <start>   <sent> <delim>  <comp>  <cls>

        # ======================================== 计算 self token 预测 损失 ====================================
        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)   # decoder 输出特征与 We 转置做矩阵相乘，得到 token 的预测
                                                            # X[:, 1:, 0]，输入 token 作为 target，计算交叉熵损失
        # lm_losses: (batch * 2 * (n_ctx-1))
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])   # (batch * 2, n_ctx-1)
        # 使用 mask 作为权重，计算加权平均（因为填充的位置不参与计算损失，所以需要 mask 掉）
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1) # (batch * 2, )
        # ======================================== 计算 self token 预测 损失 ====================================

        # ======================================== 计算 分类（1/2） 预测 损失 ====================================
        clf_h = tf.reshape(h, [-1, n_embd]) # (batch * 2 * n_ctx, n_embd)
        # tf.argmax -> 找出 <cls> token 在每个输入序列中的 index
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        # 获取 decoder 输出中 <cls> token 位置的特征，例如上面注释 demo 中的 feat4，向量维度为 n_embd
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*n_ctx+pool_idx) # (batch * 2, n_embd)

        clf_h = tf.reshape(clf_h, [-1, 2, n_embd])  # (batch, 2, n_embd)
        if train and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            # 对分类位置的特征做 dropout，两种类型的输入序列，第一种输入序列的分类特征为 clf_h[:,0,:]
            # 第一种输入序列的分类特征为 clf_h[:,1,:]，两者均为 feat map (batch, n_embed)
            # 如果随机选择到 feat map 上 (b, i) 点 dropout，那么两个 feat map 在 (b, i) 处的值同时 dropout，
            # 即，clf_h[b,:,i] = 0
            clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, n_embd]) # (batch * 2, n_embd)， shape 还原回来
        clf_logits = clf(clf_h, 1, train=train) # <cls> 的输出特征经过一个 linear layer，输出 (batch * 2, 1)
        clf_logits = tf.reshape(clf_logits, [-1, 2])    # (batch, 2)    # 分类输出预测得分，非归一化
        # 二分类，计算交叉熵损失，(batch, 2)
        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        return clf_logits, clf_losses, lm_losses
```

最后我们看一下 decoder 的定义代码，

```python
def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, train=train, scale=scale)   # transformer layer
        n = norm(x+a, 'ln_1')                                       # residual add & norm
        m = mlp(n, 'mlp', nx*4, train=train)                        # MLP
        h = norm(n+m, 'ln_2')                                       # residual add & norm
        return h
```

这里有一点与预训练不同，输入序列全部可见，所以 decoder 中没有使用 attention-mask 。



