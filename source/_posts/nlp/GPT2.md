---
title: GPT2
date: 2023-07-25 14:26:17
tags:
    - NLP
    - GPT
mathjax: true
---

论文：[Language Models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

源码：[openai/gpt-2](https://github.com/openai/gpt-2)

# 1. 简介

通常 NLP 中的各类任务，均使用有监督的数据集训练相应的语言模型。作者则认为，可以在一个具有数百万网页的数据集（WebText）上没有显示监督的训练语言模型，意思是，只要模型容量大，训练数据集足够大，那么即使非监督训练，也能训练出一个强大的语言模型。本文模型称为 GPT2，基于 transformer 结构，模型参数量为 `1.5B`（15亿）。

# 2. 方法

单一任务的概率模型为 _p(output|input)_，而适用于多个不同任务的通用系统的模型则为 _p(output|input,task)_ ，其中 _task_ 这个条件通常由模型的网络架构实现，即，针对不同的任务设计不同的 encoder 和 decoder，但是最近有研究 (McCann 2018 [<sup>1</sup>](#refer-anchor-1)) 表明，语言本身可以灵活地指定 任务、输入、输出 这三者为符号序列，例如，

1. 翻译训练样本可以写为 `(translate to french, english text, french text)`
2. 阅读理解任务的训练样本为 `(answer the question, document, question, answer)`

MaCann 论证了训练单个模型 MQAN，可以用于多个不同的任务中，并且不需要显示地监督，也就是说不需要明确指出哪个符号表示输出，于是有监督学习目标与无监督学习目标相同，两者地全局最小点也应该相同。所以问题变为，我们是否能够优化无监督目标使其收敛。


**# 重要概念**

1. zero-shot 学习和 zero-shot 任务迁移

零次学习（zero-shot learning） 是零次任务迁移的一种特殊情况，指的是推断阶段没有样本，仅给定指令。例如将英语翻译为法语，给模型一个英文句子，以及一个单词——法语和一个提示符 `:`，那么就可以假设模型知道这是一个翻译任务，然后输出对应的法语。


## 2.1 训练集

使用大且多样的数据集，例如 Common Crawl，它是从网络上抓取文本而得。然而网络语料存在很多数据问题，为了保证语料中的文档质量，作者选择抓取 Reddit 中超链接所定位的网页内容，得到的数据集称为 `WebText`。去掉 WebText 中的 Wikipedia 文档，因为 Wikipedia 文档在其他数据集中也常见，这就导致训练集和测试集有部分数据重叠，从而使得问题复杂。

## 2.2 Input Representation

一个通用的语言模型应该可以计算任何字符串的概率，并且可以生成任何字符串。词汇表之外的 tokens 限制了模型可以生成的字符串。一种方法是将 unicode 字符串表示为 UTF-8 的字节序列，但是在大数据集上， byte-level 的语言模型没有 word-level 的语言模型性能好。

Byte Pair Encoding (BPE) 是一种介于 charactor 和 word 之间的编码方案，但是 BPE 也有一些缺点，作者对此进行适当修改。

## 2.3 模型

与 GPT1 类似，有一些修改，

1. layer norm 移到每个 transfomer block（有时候也称 transformer layer） 之前，即，对输入先做 layer norm，然后是 multi-head self-attn，然后是 residual-add，然后是前馈网络也将 norm 前置（norm+MLP+residual），而正常的 decoder 中 layer norm 是放在 residual-add 之后，前馈网络之前。在最后一个 transformer block 之后，再增加一个 layer norm 。整体结构如下文图 4 。

2. 初始时，residual layer 的权重修改为 $1/\sqrt N$，其中 $N$ 是 residual layers 数量。

3. 词汇表大小增大到 `50,257`

4. context size 从 `512` 增大到 `1024`

5. batch size 增大到 `512`

||GPT1|GPT2|GPT3|
|--|--|--|--|
|Parameters|117M|1.5B|175B|
|Decoder Layers| 12|48|96|
|Context Size|512|1024|2048|
|Hidden Layer|768|1600|12288|
|Batch Size|64|512|3.2M|
|Dataset|Common Crawl<br/>BookCorpus |Common Crawl<br/>BookCorpus<br/>WebText|Common Crawl<br/>BookCorpus<br/>Wikipedia<br/>Books<br/>Articles|

表 1. 各版本 GPT1 参数

# 3. 实验

作者训练了四个语言模型（如表 2），参数量分别为 117M（GPT1），345M，762M 以及 1.5B（GPT2），从前往后，模型增大，perplexity（混乱度？）指标降低。

|Parameters|Layers|model dim| Corrected Parameters|
|--|--|--|--|
|117M|12|768|124M|
|345M|24|1024|355M|
|762M|36|1280|774M|
|1542M|48|1600|1.5B|

表 2. 4 个模型的参数量，作者后来说论文中参数量（第一列）计算错误，而应该是第四列。

# 4. 总结

GPT2 为非监督语言模型，模型容量比 GPT1 更大，可以实现 zero-shot 任务迁移。当然，我们也可以继续在下游任务数据集上 fine-tuning ，这与 GPT1 是类似的，所以不再详细说明，但是如何实现 zero-shot 任务迁移，还是有必要搞清楚。

## 4.1 机器翻译

如图 1（图源 [<sup>2</sup>](#refer-anchor-2)），

![](/images/nlp/GPT2_1.png)
<center>图 1. GPT2 用于机器翻译</center>

图 1 左边部分是训练数据集的数据展示，当然这是一个 labeled 的数据集，用于机器翻译中将英文翻译为法文，但是我们要将它看作是无监督的数据集，将 `x`（英文）和 `y` 连接起来，中间使用一个特殊的 token `<to-fr>` 分隔（这个特殊 token 是自定义的，不是绝对固定，但是要保证在 train 和 eval 阶段一致）。

图 1 右边部分是 GPT2 的 transformer decoder（省略了 embedding layer），对于位置 4 `position #4`，前面的内容为 `how are you <to-fr>`，后面的被 mask 了，此位置预测应该是 `comment`，然后下一个位置 `position #5`，再根据 `how are you <to-fr> comment` 预测输出应为 `allez-vous`，依次进行下去。

对于非监督训练集，则样本不像上面的数据集那样每个样本均为 `<english sentence> <to-fr> <french sentence>`，而是文档中的句子，例如 图2，一句话中有英文句子以及对应的法文句子，以及相关的词 `say in French`，`wrote in French`，`translates as` 等。由于 GPT2 模型容量大，在大数据集中，GPT2 能够学习到翻译的能力，在 eval 时，输入应该类似于 `<english sentence> translate to/as french` 或者 `translate to/as french: <english sentence>`，而不能使用 `<to-fr>` 这种在训练阶段没见过的 token 。

![](/images/nlp/GPT2_2.png)
<center>图 2. WebText 训练集中有关英翻法和法翻英的句子示例</center>

## 4.1 文本总结

与机器翻译任务类似，见图 3（图源 [<sup>2</sup>](#refer-anchor-2)），注意 `<summarize>` 这样的 token 与机器翻译中的 `<to-fr>` 类似，要在训练和 eval 阶段一致。

![](/images/nlp/GPT2_3.png)
<center>图 3. GPT2 用于文本总结</center>

# 5. 源码解析

## 5.1 官方源码

官方源码中没有关于训练的源码。这里仅以 sample 实现为例分析源码。

以 `124M` 模型为例说明，下载模型使用如下命令，

```python
python download_model.py 124M
```

从 `hparams.json` 文件中可见超参数为 

```sh
n_vocab: 50257  # 词汇表 size
n_ctx: 1024     # context window size
n_embd: 768     # attention model dimension
n_head: 12      # number of heads of multi-head self-attention
n_layer: 12     # number of attention layers in decoder
```


![](/images/nlp/GPT2_4.png)
<center>图 4. transformer layer 结构</center>

模型定义代码如下，其中

`wpe` 为 $W _ p \in \mathbb R ^ {k \times d}$ （参见 [GPT1](/2023/07/25/nlp/GPT1) 中的 `## 2.1` 一节中 (2) 式以及相关解释），`wte` 为 $W _ t \in \mathbb R ^ {V \times d}$，其中 $k$ 为 context window size，$d$ 为模型维度，$V$ 为词汇表 size 。

```python
def model(hparams, X, past=None, scope='model', reuse=False):
    '''
    past: 当前的 context 信息。第 1 步（初始）时，为 None，表示没有 context 信息
            以后第 i 步时，其 shape 为 (batch, n, 2, heads, i-1, d//heads)
            n 为 transformer layer 数量，n=12
            2 表示 k，v 两个变量的值
            heads：为多头自注意力模型的 head 数量，heads=12
            d：model dimension，d=768
    '''
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)
        # position embedding matrix
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.01))
        # word embedding matrix
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.02))
        # context 的序列长度（不包括当前 timestep 的 token）
        past_length = 0 if past is None else tf.shape(past)[-2]
        # (batch, sequence, n_embd), 这里 n_embd = model_dim
        h = tf.gather(wte, X) + tf.gather(wpe, position_for(X, past_length))    # h0，参考 GPT1 一文的式 (2)
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        for layer, past in enumerate(pasts):
            # 计算 hl, l=1,...,n,  (batch, sequence, d)
            # present: concat(k, v) of l-th layer, (batch, 2*heads, sequence, d//heads)
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['presents'] = tf.stack(presents, axis=1)    # (batch, 2*n*heads, sequence, d//heads)
        h = norm(h, 'ln_f') # 最后一个 transformer layer 之后，再增加一个 norm
        h_flat = tf.reshape(h, [batch * sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)   # 归一化预测得分，(batch*sequence, n_vocab)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
```

上述代码非常简单，基本上是按 [GPT1](/2023/07/25/nlp/GPT1) 中的 (2) 式计算，但是我仔细阅读源码发现输入序列与原始的 transformer 还是不一样的，原始 transformer 其实就是 [GPT1](/2023/07/25/nlp/GPT1) 中的 (2) 式，如下，

```sh
Process 1
# 以下输入、输出指的是 transformer decoder 的输入、输出
# i=1, 输入为 {'<|endoftext|>'}，输出为 `<tok1>`
# i=2, 输入为 {'<|endoftext|>', '<tok1>'}，输出为 {'<tok1>', '<tok2>'}
# i=3, 输入为 {'<|endoftext|>', '<tok1>', '<tok2>'}，输出为 {'<tok1>', '<tok2>', '<tok3>'}
# ...
# i=1025, 输入为 {'<tok1>', ..., '<tok1024>'}，输出为 {'<tok2>', ..., '<tok1025>'}
# ...
```

但实际代码并非按照上面这个过程实现，而是

```sh
Process 2
# 以下每一步均记录 decoder 中各 layer 的 k，v（q,k,v中的 k,v)，12个layers的 k,v 记为 ks，vs，
#   记录结果保存到变量 presents，
#   并且每一步的输出均附加到 output 后面（output初始为 ['<|endoftext|>']）
# i=1, 输入：{'<|endoftext|>'}，输出：'<tok1>'，presents：[(ks1, vs1)]，output：['<|endoftext|>', '<tok1>']
# i=2, 输入：{'<tok1>'}，输出：'<tok2>'，presents：[(ks1, vs1), (ks2, vs2)]，output：['<|endoftext|>', '<tok1>', '<tok2>']
# i=3, 输入：{'<tok2>'}，输出：'<tok3>'，presents：[(ks1, vs1), (ks2, vs2), (ks3, vs3)]，output：['<|endoftext|>', '<tok1>', '<tok2>', '<tok3>']
# ...
# i=1025 -> exit
```

Process 2 中每一步表示执行上面的代码中 `model(hparams, X, past, ...)` 方法，其中输入就是 `X`，所以 `X.shape=(batch, 1)` ，当前步骤输出的 presents 传给给下一步的 `past` 参数，所以 context 信息没有放在输入 `X` 上，而是放在 `past` 上，也就是说 context 信息其实利用的是 context token 对应的 `k, v` 值，过程如图 5 所示，

![](/images/nlp/GPT2_5.png)
<center>图 5. 利用 context 信息</center>

## 5.2 非官方实现

[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

### 5.2.1 数据集

项目中提供了几个数据集，这里以 `openwebtext` 为例。openwebtext 是一个文档列表，图 6 显示了一部分文档，每个文档（text）的 word 数量都比较多（1K 以上），

![](/images/nlp/GPT2_6.png)
<center>图 6. openwebtext 训练集的前 1.01M 个文档</center>

```python
enc = tiktoken.get_encoding("gpt2") # 获取 GPT2 的分词器，每个 GPT 版本的词汇表可能不同
def process(example):   # 一个文档就是一个 example
    ids = enc.encode_ordinary(example['text'])  # 将 text tokenize，并得到 id 序列
    ids.append(enc.eot_token)   # add the end of text token, e.g. 50256 for gpt2 bpe
    out = {'ids': ids, 'len': len(ids)}
    return out

# 处理训练集
outs = [process(ex) for ex in train_dataset]
arr_len = np.sum([out['len'] for out in outs])  # 所有文档的 token 数量总和
arr = np.memmap('train.bin', dtype=np.uint16, mode='w+', shape=(arr_len, ))

idx = 0
total_batches = 1024    # 分成 1024 批，按批写入文件
size = arr_len // total_batches  # 批大小
if arr_len % total_batches:
    size += 1

for batch_idx in range(total_batches):
    batch = outs[idx : idx+size]
    batch_size = len(batch)
    arr[idx : idx+batch_size] = np.concatenate([out['ids'] for out in batch])
    idx += len(batch_size)
arr.flush()
```

根据上述代码，得到的 `train.bin` 文件就是：

1. 将文档 tokenize，得到 token id 列表，列表最后增加一个 eot_token id，列表记作 `ids`
2. 所有文档的 `ids` concatenate，得到一个超长列表，一维向量，可以看作是一个超大文档。

使用如下语句读取 `train.bin`，

```python
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
```

**# 数据加载**

训练时，不按文档，随机选择这个超大文档中的一部分，即一段连续的 tokens，称为一个 block，输入一个 block，输出为预测这个 window 向后移一位后新 block 的概率。block size 为 context window size，对于 GPT2，这个 size 为 1024 。


```python
block_size = 1024

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # 随机获取 batch_size 个 block 的 start id
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    # (batch_size, block_size)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # (batch_size, block_size)
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    ... # x.to(device), y.to(device)
```


## 5.2.2 训练

```python
# 通常，计算每一个 batch 的 loss 后，就要使用梯度更新参数
# 由于 batch 太大，而 GPU 显存不够，那么只能累加多个 batch 梯度，然后一次性更新到参数
# 这样就实现大的 batch
gradient_accumulation_steps = 5 * 8
batch_size = 12
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdytpe)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))

X, Y = get_batch('train')
while True:
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)  # loss: reduce-mean
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:    # clip 梯度，防止梯度太大，导致训练不稳定
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

<details><summary>large batch 分为 small batch 原理</summary>

假设一个大的 batch size 为 $B$，那么 reduce-mean loss 为

$$L _ B = \frac 1 B \sum _ {i=1} ^ B l _ i \tag{1}$$

反向传播，计算梯度为 

$$\nabla _ {W} L _ B = \frac {\partial L _ B}{\partial W} \tag{2}$$

将大 batch 划分为 `n` 个小 batch，即 $B = n b$，那么第 `j` 个小 batch 的 reduce-mean loss 为

$$L _ {b _ j} = \frac 1 b \sum _ {k _ j =1} ^ b l _ {k _ j} , \quad j =1,\ldots, n\tag{3}$$

反向传播，计算梯度为

$$\nabla _ W L _ {b _ j} = \frac {\partial L _ {b _ j}}{\partial W} \tag{4}$$
满足关系 

$$\begin{aligned} L _ B = \frac 1 B \sum _ {j=1} ^ n b L _ {b _ j} =\sum _ {j=1} ^ n \frac 1 n L _ {b _ j} 
\\\\ \nabla _ W L _ B = \sum _ {j=1} ^ n \frac 1 b \nabla _ W L _ {b _ j}
\end{aligned} \tag{4}$$

</details>
<br/>

### 5.2.3 模型

nanoGPT 配置如下，

```python
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
```

模型定义如下，

```python
def GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            # token embedding matrix
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embeding maxtrix
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias)
        ))
        # transformer decoder 输出特征经过一个 FC，转为 vocab 中各 token 的预测得分
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
```

上述过程参见  [GPT1](/2023/07/25/nlp/GPT1) 中的 (2) 式 。

直接看 GPT 前向传播时对输入序列如何处理，

```python
def forward(self, idx, targets=None):
    '''
    idx: (batch_size, block_size)，注意 block_size 就是上下文窗口 size
    target: 将 idx 中各上下文窗口向后移动一个 token
    '''
    device = idx.device
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = self.transformer.wte(idx) # token embeddings, (b, t, n_embd)
    pos_emb = self.transformer.wpe(pos) # position embeddings, (t, n_embd)

    # (b, t, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb)    # drop to avoid overffing

    for block in self.transformer.h:
        x = block(x)
    
    # (b, t, n_vocab)
    x = self.transformer.ln_f(x)    # 最后在增加一个 layer norm，参见上面 2.3 一节对 GPT2 的修改说明

    if targets is not None:
        logits = self.lm_head(x)    # convert to predict tokens
        # 忽略 token id 为 -1 的样本
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, [-1], :])
        loss = None
    return logits, loss
```

由于 GPT 使用 transformer decoder，训练时左边位置的 token 应该看不到右边位置的 token，也就是说，计算一个 token 出现的概率，应该与这个 token 之前的 tokens 有关，即 $p(u _ i | u _ {i-k}, \ldots, u _ {i-1} )$ 。来看代码如何实现，

```python
# 一个 transformer layer(block)
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # layer norm 前置
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # 这个 causal 表示要考虑因果关系，即当前只看到左边的 tokens，看不到右边的 tokens
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

考虑因果关系的 multi-head self-attention 的实现代码如下，

```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # 下三角
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
```

看到这里，其中各 layer 的定义应该已经很熟悉了，我们看看前向传播的实现，

```python
def forward(self, x):
    # T = 1024 上下文窗口 size
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)     # (B, n_ctx, n_embd)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    if self.flash:
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
    else:
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # Q K^T / sqrt(n_embd), shape is (B, heads, n_ctx, n_ctx)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # 上三角（主对角线除外）位置值设为 -inf
        att = F.softmax(att, dim=-1)    # attention 矩阵中每一行表示一个 token 对序列中其他 token 的注意力
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    # output projection
    y = self.resid_dropout(self.c_proj(y))
    return y    # (B, n_ctx, n_embd)
```

上述代码中，先不看 CUDA 中的 attention 快速实现，而是使用原始 tensor 操作实现 attention（`not self.flash` 分支），先求 $Q T ^ {\top} / \sqrt {d}$，这里 $d$ 是 model dimension，然后使用下三角 mask，因为 attention 矩阵中，第 `i` 行表示窗口中第 `i` 对窗口中所有 token 的注意力，显然只有左边的 tokens 能被注意到，即 $\ge i$ 的位置值应为为 0，但是这里是非归一化的注意力值，所以使用 $-\infty$，经过 softmax 归一化后则为 0，softmax 归一化沿着每一行进行，故 softmax 的参数 `dim=-1` 。
 

# REF


<div id="refer-anchor-1"></div>

- [1] The natural language decathlon: Multitask learning as question answering

<div id='refer-anchor-2'></div>

- [2] [The Illustrated GPT-2 (Visualizing Transformer Language Models)](http://jalammar.github.io/illustrated-gpt2/)