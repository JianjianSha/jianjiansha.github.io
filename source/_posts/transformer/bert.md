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

于是一个 Transformer block 的（可学习）参数量为 $4H^2 + 8H^2 = 12H^2$

$L$ 个 Transformer block 的参数量为 $12LH^2$，总参数量为 $VH+12LH^2$，

论文中指出所用词汇集大小为 $V=30000$，然后根据 Bert base 的超参数 $L=12,H=768$，计算出总参数量为 $VH+12LH^2=107,974,656$ （Bert large 模型的参数可类似地计算出来）。


## 1.2 词嵌入

例如以空格进行分词，那么对于大数据集，token 数量会特别大，导致模型参数集中在嵌入层，为了解决这个问题，使用 WordPiece embedding。具体而言：使用一个 word 的子序列（字符串的前 n 个字符构成的子串），这个子序列有可能就是这个 word 的词根。参考 [Google’s neural machine translation system: Bridging the gap between human and machine translation](https://arxiv.org/abs/1609.08144)

## 1.3 特殊 token
[CLS] 表示句子的整体信息，这个 token 的最终表示，即最后一个 Transformer block 的输出 （是一个 $H$ 长度的向量）就是这个句子信息的 BERT 表征，可用于对句子（单词序列）的分类任务。

每个句子末尾添加 [SEP] token 表示句子结束。

BERT 的输入序列可以是单个句子，也可以是句子对（两个句子）。输入是单个句子是，输入序列是 “[cls]”、文本序列的标记、以及特殊分隔词元 “[sep]”的连结。当输入为文本对时，BERT输入序列是“[cls]”、第一个文本序列的标记、“[sep]”、第二个文本序列标记、以及“[sep]”的连结。



## 1.4 输入表示

每个 token 由三个 embedding 构成：
1. word embedding。这就是 word 自身的词嵌入表示
2. segment embedding。由于 BERT 使用两个句子作为输入，那么每个句子的 token 均需要一个额外 embedding 表示是第一个句子还是第二个句子。
3. position embedding。与原生 Transformer 相同，由于 attention 中各输入是未知无关的，所以需要额外增加一个位置信息的 embedding。


三种 embedding 的维度均为 $H$，相加得到最终的 embedding 表示（维度仍为 $H$）。如图 2 所示，

![](/images/transformers/BERT2.png)

图 2.

## 1.5 BERT

BERT 有两个阶段：预训练和精调。

1. 预训练使用无标签数据进行训练，masked 语言模型（MLM），训练时，有一定概率对句子中的词进行 mask，然后训练预测这个 masked 的词。
2. 精调，对 BERT 使用预训练模型进行初始化，然后使用有标签数据进行学习，并对 bert 模型的所有参数进行微调。







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


假设无表情句子为 `my dog is hairy`，经过随机 mask，假设选择了第 4 个 token （hairy）进行 mask，mask 流程为：

- 80% 的训练时间，将这个 masked word 替换为 `[MASK]` token，即 `my dog is hairy` $\rightarrow$ `my dog is [MASK]`
- 10% 的训练时间，将这个 masked word 替换为词汇表中一个随机 word，例 `my dog is hairy` $\rightarrow$ `my dog is apple`
- 10% 的训练时间，保持 masked word 不变，取消对这个 word 进行 mask，即 `my dog is hairy` $\rightarrow$ `my dog is hairy`。这么做的目的是引导 token 特征趋于表征实际的观察 word，这个位置的 token 本来是随机的每个 word 等概率出现，或者均为一个 `[MASK]` 特殊token，而将 10% 的训练时间内使用这个位置的实际 word，相当于对这个真实 word 进行 bias。

分析：

- Transformer encoder 学习每个 input token 的上下文表征。
- 输入序列中只有 15% 的 word 被选择，然后再以 10% 的概率被随机替换为另一个 word，所以被随机替换的概率为 1.5%，这个概率足够小，不会有损模型对句子的理解。


## 2.2 NSP

Next Sentence Prediction(NSP)


许多下游任务例如 问答（QA）以及 自然语言推断（NLI）均基于两个句子间关系的理解。为了训练出一个能理解两个句子间关系的模型，作者设计了一个二分类的下一句预测任务，即给两个句子作为输入，预测第二个句子是否是第一个句子的下一句。

训练时，每个训练样本中的句子 A 和 B，50% 的概率下 B 是 A 的下一句，即 label 为正，另外 50% 的概率下 label 为负。将 `[CLS]` 的最终向量用于预测是否是下一句。

**预训练数据：**

使用 BooksCorpus 和 English Wikipedia 数据集。作者使用 document-level 的语料而非 sentence-level 语料。Document 中，可以获得连续的两个句子（在 document 中自然连续，不一定要有语义上的某种关联）。

NSP 任务数据集举例说明：

```
Input = [CLS] the man went to [MASK] store [SEP]
        he bought a gallon [MASK] milk [SEP]
Label = IsNext

Input = [CLS] the man [MASK] to the store [SEP]
        penguin [MASK] are flight ##less birds [SEP]
Label = NotNext
```



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

# 4. 下游任务

由于 BERT 在通用语料上训练时间太长，我们可以直接利用别人预训练好的模型，然后对我们的有监督训练任务进行 fine tuning。

## 4.1 文本分类

对句子（一条文本）进行处理：

1. 去掉 stop words
2. （可选）分词
3. 根据词汇表将 word 转为 word id
4. 根据设置的长度 `pad_size`，对短句子进行 padding，对长句子进行截断
5. 记录句子的长度，短句子记录其 padding 之前的长度，长句子则记录长度为 `pad_size` 即截断之后的长度。
6. 记录句子的 mask 向量，长度为 `pad_size`，mask 向量中元素为 1 表示使用句子中的 token，元素值为 0，表示是一个填充 token。
7. 记录句子的分类标签 id

**训练阶段**

DataLoader，每次 iteration 返回一批数据，格式为

```python
# x: (batch_size, pad_size)，一批句子，每个句子是 word_id 组成的 list
# word_id：[0,len(V))，这里 V 是词汇表，第一个词汇必须是 [PAD]，表示填充
# seq_len: (batch_size,) 句子实际长度
# mask: (batch_size, pad_size)，记录哪些 word_id 是填充的
# y: (batch_size,) 句子分类 id，与 word_id 一样，都是整型
(x, seq_len, mask), y
```

原生 bert 输出为序列中各个 token 的特征，shape 为 `(batch_size, pad_size, embed_dim)`，文本分类取序列第一个 token 即 `[CLS]` 的特征，

```python
# ======== BertPooler: start =========
first_token_feat = hidded_states[:,0]   # (batch_size, embed_dim)
output = dense(first_token_feat)        # 使用一个 fc layer，调整特征，以适应下游任务，
                                        # output_dim 不变，shape 依然为 (batch_size, embed_dim)
output = act(output)                    # 使用 Tanh 激活
# ======== BertPooler:   end =========

output = dense2(output)                 # 最后使用一个 fc layer，输出 units 为 分类数量
# 最后这个 output 就是各分类的非归一化概率，(batch_size, num_classes)
# 使用 F.softmax(output) 即可得到归一化概率
```

# 5. BERT 代码说明

以下内容来源 [Pytorch-Transformers——源码解读](https://www.cnblogs.com/dogecheng/p/11907036.html)，为了防止文章不见了，这里转载一下。

## 5.1 Model相关
### 5.1.1 BertConfig
BertConfig 是一个配置类，存放了 BertModel 的配置。比如：

- vocab_size_or_config_json_file：字典大小，默认30522
- hidden_size：Encoder 和 Pooler 层的大小，默认768
- num_hidden_layers：Encoder 的隐藏层数，默认12
- num_attention_heads：每个 Encoder 中 attention 层的 head 数，默认12

完整内容可以参考：https://huggingface.co/transformers/v2.1.1/model_doc/bert.html#bertconfig

### 5.1.2 BertModel
实现了基本的Bert模型，从构造函数可以看到用到了embeddings，encoder和pooler。

**BERT 模型的输入**：

下面是允许输入到模型中的参数，模型至少需要有1个输入： input_ids 或 input_embeds。

1. `input_ids` token id sequence，包括 `[CLS]` 和 `[SEP]` 的 id。(batch_size, pad_size)
2. `token_type_ids` 就是上面所说的 segment embedding，表示 token 对应的句子 id，值为 0（第一个句子） 或 1（第二个句子）。(batch_size, pad_size)

Bert 的输入需要用 [CLS] 和 [SEP] 进行标记，开头用 [CLS]，句子结尾用 [SEP]

```sh
# 两个句子：
tokens：[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
token_type_ids：0   0  0    0    0     0       0   0   1  1  1  1   1   1

# 一个句子：
tokens：[CLS] the dog is hairy . [SEP]
token_type_ids：0   0   0   0  0     0   0

# 经过 PAD
tokens：[CLS] the dog is hairy . [SEP] [PAD] [PAD] [PAD]
```

- attention_mask 可选。各元素的值为 0 或 1 ，避免在 padding 的 token 上计算 attention（1不进行masked，0则masked）。形状为(batch_size, sequence_length)。
- position_ids 可选。表示 token 在句子中的位置id。形状为(batch_size, sequence_length)。形状为(batch_size, sequence_length)。
- head_mask 可选。各元素的值为 0 或 1 ，1 表示 head 有效，0无效。形状为(num_heads,)或(num_layers, num_heads)。
- input_embeds 可选。替代 input_ids，我们可以直接输入 Embedding 后的 Tensor。形状为(batch_size, sequence_length, embedding_dim)。
- encoder_hidden_states 可选。encoder 最后一层输出的隐藏状态序列，模型配置为 decoder 时使用。形状为(batch_size, sequence_length, hidden_size)。
- encoder_attention_mask 可选。避免在 padding 的 token 上计算 attention，模型配置为 decoder 时使用。形状为(batch_size, sequence_length)。
- encoder_hidden_states 和 encoder_attention_mask 可以结合论文中的Figure 1理解，左边为 encoder，右边为 decoder。

论文《Attention Is All You Need》：https://arxiv.org/pdf/1706.03762.pdf

> 如果要作为 decoder ，模型需要通过 BertConfig 设置 is_decoder 为 True

```python
def __init__(self, config):
    super(BertModel, self).__init__(config)
    self.config = config

    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.pooler = BertPooler(config)

    self.init_weights()
def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
            head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
```

### 5.1.3 BertPooler
在Bert中，pool的作用是，输出的时候，用一个全连接层 <font color="red">将整个句子的信息用第一个token来表示</font>，源码如下

每个 token 上的输出大小都是  hidden_size (在BERT Base中是768)
```python
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

所以在分类任务中，Bert只取出第一个token的输出再经过一个网络进行分类就可以了，就像之前的文章中谈到的垃圾邮件识别

https://www.cnblogs.com/dogecheng/p/11615750.html#_lab2_2_1

### 5.1.4 BertForSequenceClassification
BertForSequenceClassification 是一个已经实现好的用来进行文本分类的类，一般用来进行文本分类任务。构造函数如下

```python
def __init__(self, config):
    super(BertForSequenceClassification, self).__init__(config)
    self.num_labels = config.num_labels

    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    self.init_weights()
```

我们可以通过 num_labels 传递分类的类别数，从构造函数可以看出这个类大致由3部分组成，1个是Bert，1个是Dropout，1个是用于分类的线性分类器Linear。

Bert用于提取文本特征进行Embedding，Dropout防止过拟合，Linear是一个弱分类器，进行分类，如果需要用更复杂的网络结构进行分类可以参考它进行改写。

他的 forward() 函数里面已经定义了损失函数，训练时可以不用自己额外实现，返回值包括4个内容

```python
def forward(...):
    ...
    if labels is not None:
        if self.num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs
    return outputs  # (loss), logits, (hidden_states), (attentions)
```

其中 hidden-states 和 attentions 不一定存在

### 5.1.5 BertForTokenClassification
BertForSequenceClassification 是一个已经实现好的在 token 级别上进行文本分类的类，一般用来进行序列标注任务。构造函数如下。

代码基本和 BertForSequenceClassification 是一样的

```python
def __init__(self, config):
    super(BertForTokenClassification, self).__init__(config)
    self.num_labels = config.num_labels

    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    self.init_weights()
```

不同点在于 BertForSequenceClassification 我们只用到了第一个 token 的输出（经过 pooler 包含了整个句子的信息）

下面是 BertForSequenceClassification 的中 forward() 函数的部分代码

```python
outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

pooled_output = outputs[1]
```

bert 是一个 BertModel 的实例，它的输出有4个部分，如下所示
```python
def forward(...):
    ...
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
```
从上面可以看到 BertForSequenceClassification 用到的是 pooled_output，即用1个位置上的输出表示整个句子的含义

下面是 BertForTokenClassification 的中 forward() 函数的部分代码，它用到的是全部 token 上的输出。

```python
outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

sequence_output = outputs[0]
```

### 5.1.6 BertForQuestionAnswering
实现好的用来做QA（span extraction，片段提取）任务的类。

有多种方法可以根据文本回答问题，一个简单的情况就是将任务简化为片段提取

这种任务，输入以 上下文+问题 的形式出现。输出是一对整数，表示答案在文本中的开头和结束位置

![](/images/transformers/BERT3.png)

图片来自：https://blog.csdn.net/weixin_37923278/article/details/103006269

参考文章：https://blog.scaleway.com/2019/understanding-text-with-bert/

example：

        文本：

        Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the                Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica          of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France              where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3          statues and the Gold Dome), is a simple, modern stone statue of Mary.

        问题：
        The Basilica of the Sacred heart at Notre Dame is beside to which structure?
        答案：
        start_position: 49，end_position: 51（按单词计算的）

        49-51 是 the Main Building 这3个单词在句中的索引

下面是它的构造函数，和 Classification 相比，这里没有 Dropout 层

```python
def __init__(self, config):
    super(BertForQuestionAnswering, self).__init__(config)
    self.num_labels = config.num_labels

    self.bert = BertModel(config)
    self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    self.init_weights()
```

模型的输入多了两个，start_positions 和 end_positions ，它们的形状都是 (batch_size,)

start_positions 标记 span 的开始位置（索引），end_positions 标记 span 的结束位置（索引），被标记的 token 用于计算损失

> 即答案在文本中开始的位置和结束的位置，如果答案不在文本中，应设为0

除了 start 和 end 标记的那段序列外，其他位置上的 token 不会被用来计算损失。

```python
def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
            inputs_embeds=None, start_positions=None, end_positions=None):
    ...
```

可以参考下图帮助理解

![](/images/transformers/BERT4.png)

> 图片来自Bert原论文：https://arxiv.org/pdf/1810.04805.pdf

从图中可以看到，QA 任务的输入是两个句子，用 [SEP] 分隔，第一个句子是问题（Question），第二个句子是含有答案的上下文（Paragraph）

输出是作为答案开始和结束的可能性（Start/End Span）

### 5.1.7 BertForMultipleChoice

实现好的用来做多选任务的，比如SWAG和MRPC等，用来句子对判断语义、情感等是否相同

下面是它的构造函数，可以到看到只有1个输出，用来输出情感、语义相同的概率

```python
def __init__(self, config):
    super(BertForMultipleChoice, self).__init__(config)

    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, 1)

    self.init_weights()
```
一个简单的例子

example：

        tokenizer = BertTokenizer("vocab.txt")

        model = BertForMultipleChoice.from_pretrained("bert-base-uncased")

        choices = ["Hello, my dog is cute", "Hello, my cat is pretty"]

        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)         # 形状为[1, 2, 7]

        labels = torch.tensor(1).unsqueeze(0)

        outputs = model(input_ids, labels=labels)

BertForMultipleChoice 也是用到经过 Pooled 的 Bert 输出，forward() 函数同样返回 4 个内容

```python
def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
    ...
    pooled_output = outputs[1]
    ...
    return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)
```

### 5.1.8 tokenization
对于文本，常见的操作是分词然后将 词-id 用字典保存，再将分词后的词用 id 表示，然后经过 Embedding 输入到模型中。

Bert 也不例外，但是 Bert 能以 字级别 作为输入，在处理中文文本时我们可以不用先分词，直接用 Bert 将文本转换为 token，然后用相应的 id 表示。

tokenization 库就是用来将文本切割成为 字或词 的，下面对其进行简单的介绍

### 5.1.9 BasicTokenizer
基本的 tokenization 类，构造函数可以接收以下3个参数

- do_lower_case：是否将输入转换为小写，默认True
- never_split：可选。输入一个列表，列表内容为不进行 tokenization 的单词
- tokenize_chinese_chars：可选。是否对中文进行 tokenization，默认True

tokenize() 函数是用来 tokenization 的，这里的 tokenize 仅仅使用空格作为分隔符，输入的文本会先进行一些数据处理，处理掉无效字符并将空白符（“\t”，“\n”等）统一替换为空格。如果 tokenize_chinese_chars 为 True，则会在每个中文“字”的前后增加空格，然后用 whitespace_tokenize() 进行 tokenization，因为增加了空格，空白符又都统一换成了空格，实际上 whitespace_tokenize() 就是用了 Python 自带的 split() 函数，处理前用先 strip() 去除了文本前后的空白符。whitespace_tokenize() 的函数内容如下：

```python
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens
```

用 split() 进行拆分后，还会将 标点符号 从文本中拆分出来（不是去除）

example：① → ② → ③

          ① "Hello, Marry!"
          ② ["Hello,", "Marry!"]
          ③ ["Hello", ",", "Marry", "!"]

### 5.1.10 WordpieceTokenizer

WordpieceTokenizer 对文本进行 wordpiece tokenization，接收的文本最好先经过 BasicTokenizer 处理

> wordpiece简介：https://www.jianshu.com/p/60fc9253a0bf

简单说就是把 单词 变成一片一片的，在BERT实战——基于Keras一文中，2.1节我们使用的 tokenizer 就是这样的：

它将 “unaffable” 分割成了 “un”, “##aff” 和 “##able”

他的构造函数可以接收下面的 3 个参数

- vocab：给定一个字典用来 wordpiece tokenization
- unk_token：碰到字典中没有的字，用来表示未知字符，比如用 "[UNK]" 表示未知字符
- max_input_chars_per_word：每个单词最大的字符数，如果超过这个长度用 unk_token 对应的 字符 表示，默认100

**tokenize()函数**

这个类的 tokenize() 函数使用 贪婪最长匹配优先算法（greedy longest-match-first algorithm） 将一段文本进行 tokenization ，变成相应的 wordpiece，一般针对英文

example ：

          input = "unaffable"  → output = ["un", "##aff", "##able"]

### 5.1.11 BertTokenizer
一个专为 Bert 使用的 tokenization 类，使用 Bert 的时候一般情况下用这个就可以了，构造函数可以传入以下参数

- vocab_file：一个字典文件，每一行对应一个 wordpiece
- do_lower_case：是否将输入统一用小写表示，默认True
- do_basic_tokenize：在使用 WordPiece 之前是否先用 BasicTokenize
- max_len：序列的最大长度
- never_split：一个列表，传入不进行 tokenization 的单词，只有在 do_wordpiece_only 为 False 时有效

我们可以使用 `tokenize()` 函数对文本进行 tokenization，也可以通过 `encode()` 函数对 文本 进行 tokenization 并将 token 用相应的 id 表示，然后输入到 Bert 模型中

BertTokenizer 的 `tokenize()` 函数会用到 WordpieceTokenizer 和 BasicTokenizer 进行 tokenization（实际上由 `_tokenize()` 函数调用）

`_tokenize()` 函数的代码如下，其中basic_tokenizer 和 wordpiece_tokenizer 分别是 BasicTokenizer 和 WordpieceTokenizer 的实例。

```python
def _tokenize(self, text):
    split_tokens = []
    if self.do_basic_tokenize:
        for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
    else:
        split_tokens = self.wordpiece_tokenizer.tokenize(text)
    return split_tokens
```

使用 `encode()` 函数将 tokenization 后的内容用相应的 id 表示，主要由以下参数：

- text：要编码的一个文本（第一句话）
- text_pair：可选。要编码的另一个文本（第二句话）
- add_special_tokens：编码后，序列前后是否添上特殊符号的id，比如前面添加[CLS]，结尾添加[SEP]
- max_length：可选。序列的最大长度
- truncation_strategy：与 max_length 结合使用的，采取的截断策略。
    - 'longest_first'：迭代减少序列长度，直到小于 max_length，适用于输入两个文本的情况，处理后两个文本的序列长度之和是 max_length
    - 'only_first'：仅截断第一个文本
    - 'only_second'：仅截断第二个文本
    - 'do_not_truncate'：不截断，如果输入序列大于 max_length 则会报错
- return_tensors：可选。返回 TensorFlow （传入'tf'）还是 PyTorch（传入'pt'） 中的 Tensor，而不是返回 Python 列表，前提是已经装好 TensorFlow 或 PyTorch

注意 encode 只会返回 token id，Bert 我们还需要输入句子 id，这时候我们可以使用 `encode_plus()`，它返回 token id 和 句子 id

`encode()` 实际上就是用了 encode_plus，但是只选择返回 token_id，代码如下

```python
...
encoded_inputs = self.encode_plus(text,
                                  text_pair=text_pair,
                                  max_length=max_length,
                                  add_special_tokens=add_special_tokens,
                                  stride=stride,
                                  truncation_strategy=truncation_strategy,
                                  return_tensors=return_tensors,
                                  **kwargs)

return encoded_inputs["input_ids"]
```

`encode_plus()` 的参数与 encode 是一样的，可以根据实际需求来选择需要使用的函数