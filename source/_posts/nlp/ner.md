---
title: 命名实体识别
date: 2023-08-03 14:01:41
tags: NLP
mathjax: true
---

命名实体识别 (Named Entity Recognition,简称NER) 

**# 命名实体识别标注**

标签类型的定义：

```sh
定义 全称 备注
B Begin 实体片段的开始
I Intermediate 实体片段的中间
E End 实体片段的结束
S Single 单个字的实体
O Other/Outside 其他不属于任何实体的字符(包括标点等)
```

**# BIO标注模式**

将每个元素标注为`B-X`、`I-X` 或者 `O`。其中 `B-X` 表示此元素所在的片段属于 X 类型并且此元素在此片段的开头，`I-X` 表示此元素所在的片段属于 X 类型并且此元素在此片段的中间位置，`O` 表示不属于任何类型。

命名实体识别中每个 token 对应的标签集合如下:
```sh
LabelSet = {O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG}
```

其他几个 tag 标注方式：

1. IOB1: 标签I用于文本块中的字符，标签O用于文本块之外的字符。如果同类型的块相邻，那么第二个块的第一个 word 使用 `B-XXX` 标注，例如
    ```sh
    ...
    I-XX1
    B-XX1   # 这里使用 B，则表示开启下一个 phrase，如果使用 I，那么表示与上面的word 还处于同一phrase中
    ...
    I-XX2
    I-XX3   # 这里 word 与上一个word 显然处于不同phrase中。如果要转为 IOB2，那么这里必须将 I 改为 B
    ```
2. IOB2: 每个文本块都以标签B开始，除此之外，跟IOB1一样。
3. IOE1: 标签I用于独立文本块中，标签E仅用于同类型文本块连续的情况，假如有两个同类型的文本块，那么标签E会被打在第一个文本块的最后一个字符。
4. IOE2: 每个文本块都以标签E结尾，无论该文本块有多少个字符，除此之外，跟IOE1一样。
5. START/END （也叫SBEIO、IOBES）: 包含了全部的5种标签，文本块由单个字符组成的时候，使用S标签来表示，由一个以上的字符组成时，首字符总是使用B标签，尾字符总是使用E标签，中间的字符使用I标签。
6. IO: 只使用I和O标签，显然，如果文本中有连续的同种类型实体的文本块，使用该标签方案不能够区分这种情况。

**# 实体识别标签**

```sh
PERSON: 人
NORP：民族，宗教或政治团体
FAC：建筑，机场，高速，桥等
ORG：公司，机构，研究所等
GPE：国家，城市，州
LOC：非 GPE 地点，山脉，水体
PRODUCT：目标物体，车辆，食物等
EVENT：有名字的飓风，战斗，战争，体育赛事等
WORK_OF_ART：书名，歌曲等
LAW：有名字的法律文书/文档
LANGUAGE：语言
DATE：日期或一段时间
TIME：小于一天的时间
PERCENT：百分比
MONEY：货币值
QUANTITY：数量
ORDINAL：序数，第一、第二等
CARDINAL：基数
```

**# 数据集**

1. [FEW-NERD：A Few-shot Named Entity Recognition Dataset](https://arxiv.org/abs/2105.07464)

2. [CLUENER2020](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/cluener_public)

3. [MSRA](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/MSRA)

4. [人民网（04年](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/people_daily)

5. [微博命名实体识别数据](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/weibo)

6. [BosonNLP NER数](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/boson)

7. [影视-音乐-书籍实体标注数](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/video_music_book_datasets)

8. [中文医学文本命名实体识别 2020CCKS](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/2020_ccks_ner)

9. [电子简历实体识别数据集](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/ResumeNER)

10. [医渡云实体识别数据](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/yidu-s4k)

11. [简历实体数据](https://github.com/jiesutd/LatticeLSTM/tree/master/data)

12. [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)

13. [Few-NERD 细粒度数据集](https://github.com/thunlp/Few-NERD/tree/main/data)

以 CoNLL-2003 为例说明。

数据文件包含 4 列，以单个空格分开，每个单词占用一行，每个句子后使用一个空行。格式为

```sh
# 单词 POS 句法块tag 命名实体tag
U.N. NNP I-NP I-ORG
official     NN   I-NP  O 
Ekeus        NNP  I-NP  I-PER 
heads        VBZ  I-VP  O 
for          IN   I-PP  O 
Baghdad      NNP  I-NP  I-LOC 
.            .    O     O 
```

数据集[下载地址](http://www.cnts.ua.ac.be/conll2003/ner.tgz) 。

将 IOB1 标注方式改为 IOB2 标注方式，代码为，

```python
def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':  # O tag 不需要修改
            continue
        split = tag.split('-')  # 必须是 X-XXX 的形式，第一个 X 是 I or B
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        # ====================================
        # at here, tag must starts with I
        # ====================================
        # (1) i==0 or tags[i-1] == 'O' means current is the start of a new phrase
        #   so its tag should be B-XXX
        elif i == 0 or tags[i - 1] == 'O':  # convert IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        # i!=0 当前不是第一个 word，
        elif tags[i - 1][1:] == tag[1:]:    # current word and the previous word are in the same phrase
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True
```

训练集的 X 为 `[[word]]`，内层为 word list，表示一个句子， Y 为 `[[tag]]`，内层表示一个句子中各单词的命名实体 tag 。

**# 模型**

1. 传统深度学习

    - RNN+CRF
    - CNN+CRF
    - LSTM+CRF
    - BiLSTM+CRF
    - BiLSTM+CNN+CRF

2. 大模型预训练

    - Bert
    - Bert+BiLSTM+CRF