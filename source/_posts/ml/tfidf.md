---
title: TF-IDF
date: 2022-08-12 13:47:45
tags: machine learning
---

# 1. 概念

TF-IDF: Term Frequency-Inverse Document Frequency，词频-逆文本频率。

一个文档中如果某个词出现频率越高，那么说明这个词对这个文档越重要。一个词如果在越多的文档中出现，那么这个词应该越不重要。

如果某个单词在一篇文章中出现的TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类


**TF**

$n_{ij}$，表示第 `i` 个单词 $t_i$ 在第 `j` 个文档 $d_j$ 中出现的频次，归一化后的词频 TF 则为

$$tf_{ij}=\frac {n_{ij}}{\sum_k n_{kj}} \tag{1}$$

**IDF**

第 `i` 个词 $t_i$ 的逆文件频率为总文件数量除以含有这个词的文件数量，然后再取对数，

$$idf_i=\log \frac {|D|}{|\{j: t_i \in d_j\}|} \tag{2}$$

为了避免除 0 错误，可改为

$$idf_i=\log \frac {|D|}{|\{j: t_i \in d_j\}|+1} \tag{3}$$

**TF-IDF**

$$\text{TF-IDF}_{ij}=tf_{ij} \cdot idf_i \tag{4}$$

# 2. 代码实现

## 2.1 sklearn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 语料可以先经过jieba分词，使用空格分隔后才能使用 sklearn tf-idf 
x_train = ['TF-IDF 主要 思想 是','算法 一个 重要 特点 可以 脱离 语料库 背景',
           '如果 一个 网页 被 很多 其他 网页 链接 说明 网页 重要']
vectorizer = CountVectorizer(max_features=10) # 指定保留最高频的 10 个关键词
tf_idf_transformer = TfidfTransformer()
# 获取词频矩阵，即上面的 n_ij
X = vectorizer.fit_transform(x_train)
# 计算每一个词的TF-IDF，即 (4) 式
tf_idf = tf_idf_transformer.fit_transform(X)
# 转换为数组
x_train_weight = tf_idf.toarray()
print(x_train_weight)
```

直接使用 `TfidfVectorizer`，

```python
from sklearn.feature_extraction.text import TfidfVectorizer

data = ["I have a pen", "I have an apple"]
tfidf = TfidfVectorizer(data)
res = tfidf.fit_transform(data)
print(res)
print(res.toarray())
print(tfidf.vocabulary_)
```