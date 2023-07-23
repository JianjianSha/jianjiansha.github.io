---
title: 'GloVe: Global Vectors for Word Representation'
date: 2023-07-18 16:10:56
tags: 
    - NLP
    - word embedding
mathjax: true
---

论文：[GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162.pdf)

源码：[stanfordnlp/GloVe](https://github.com/stanfordnlp/GloVe)

# 1. 介绍

GloVe 是一种获取单词的向量表示的无监督学习方法。

# 2. GloVe 模型

先建立几个概念。

1. 记 单词-单词共现次数矩阵 为 $X$，其中 $X _ {ij}$ 表示单词 `j` 出现在单词 `i` 的上下文中的次数。

2. 令 $X _ i = \sum _ k X _ {ik}$ ，表示所有单词出现在单词 `i` 的上下文中的次数

3. 令 $P _ {ij} = P(j|i) = X _ {ij} / X _ i$ 表示单词 `j` 出现在单词 `i` 的上下文中的概率

我们举例说明共现概率的意义和作用。假设我们对物体（热力学）相态感兴趣，例如单词 `i=ice` （冰块） 和 `j=steam`（蒸汽），另外使用一个探测单词 `k`，

1. `k` 与 `ice` 有关，与 `steam` 无关，例如 `k=solid`，我们希望 $P _ {ik} /P _ {jk}$ 较大

2. `k` 与 `steam` 有关，与 `ice` 无关，例如 `k=gas`，$P _ {ik} /P _ {jk}$ 应该较大

3. `k` 如果是 `water` 或者 `fashion`，前者与 `ice` 和 `steam` 均有关，后者则均无关，那么 $P _ {ik} /P _ {jk}$ 应该接近 1 。

表 1 展示了这些共现概率以及它们之间的比值，从表 1 可见与我们上面分析的三种情况一致。比起概率的绝对值，概率的比值更好地区分两个单词相关还是无关。

|概率和比值|k=solid|k=gas|k=water|k=fashion|
|--|--|--|--|--|
|P(k\|ice)|1.9e-4|6.6e-5|3.0e-3|1.7e-5|
|P(k\|steam)|2.2e-5|7.8e-4|2.2e-3|1.8e-5|
|P(k\|ice)/P(k\|steam)|8.9|0.085|1.36|0.96|

<center>表 1. 共现概率</center>

根据以上分析可知，共现概率地比值比概率自身值更适合用于学习词向量。记

$$F(w _ i, w _ j, \tilde w _ k) = \frac {P _ {ik}}{P _ {jk}} \tag{1}$$

其中 $w \in \mathbb R ^ d$ 是词向量，$\tilde w \in \mathbb R ^ d$ 是上下文的向量表示，$\tilde w _ k$ 表示单词 `k` 的上下文，共现概率是单词 `i` 或者 `j` 出现在 `k` 的上下文中的概率。

由于向量空间是线性的，我们要求函数 $F$ 仅依赖两个词向量之差，于是 (1) 式变为

$$F(w _ i - w _ j, \tilde w _ k) = \frac {P _ {ik}}{P _ {jk}} \tag{2}$$

(2) 式右侧是标量，左侧是向量，由于 $F$ 可以是具有复杂参数的函数，例如神经网络，这会混淆输入向量，所以干脆先将两个输入向量做点积运算，避免输入向量被 $F$ 以不希望的方式混合，

$$F((w _ i - w _ j) ^ {\top} \tilde w _ k) = \frac {P _ {ik}}{P _ {jk}} \tag{3}$$

$F$ 在群 $(\mathbb R, +)$ 和 $(\mathbb R _ {>0}, \times)$ 之间同态，即

$$F((w _ i - w _ j) ^ {\top} \tilde w _ k) =\frac {F(w _ i ^ {\top} \tilde w _ k)}{F(w _ j ^ {\top} \tilde w _ k)} \tag{4}$$

<details><summary>同态定义</summary>

对代数系统 $\langle A, \star \rangle$， $\langle B, \circ \rangle$ 以及映射 $f$，如果 $\forall x, y \in A$，有 $f(x \star y) = f(x) \circ f(y)$，则称 $f$ 为 $\langle A, \star \rangle$ 到 $\langle B, \circ \rangle$ 的同态映射，简称 同态。
</details>

(4) 式结合 (3) 式，可知

$$F(w _ i ^ {\top} \tilde w _ k) = P _ {ik} = \frac {X _ {ik}}{X _ i} \tag{5}$$

(4) 式的解是 $F = \exp$，于是

$$w _ i ^ {\top} \tilde w _ k = \log (P _ {ik}) = \log (X _ {ik}) - \log (X _ i) \tag{6}$$

(6) 式右侧的 $\log (X _ i)$ 与 `k` 无关，所以可以使用 $b _ i$ 代替，另外为了让 (6) 式具有对称性，增加一个 $\tilde b _ k$，于是 (6) 式变为

$$w _ i ^ {\top} \tilde w _ k + b _ i + \tilde b _ k = \log (X _ {ik}) \tag{7}$$

(7) 式不是良好定义，因为 log 参数不能为 0，一种方法是 $\log (X _ {ik}) \rightarrow \log (1+X _ {ik})$ 。

有些共现次数很小甚至为 0，这对应着噪声并且所携带的信息很少，在词汇表和语料较大时，$X$ 中 0 元素数量甚至达到 $75 \sim 95 \%$ ，故引入一个权重函数 $f(X _ {ij})$ 解决这个问题，使用回归模型，和平方差损失，于是带权重的损失为

$$J = \sum _ {i,j=1} ^ V f(X _ {ij}) (w _ i ^ {\top} \tilde w _ j + b _ i + \tilde b _ j - \log X _ {ij}) ^ 2 \tag{8}$$

其中 $V$ 表示词汇表大小。权重函数满足：

1. $f(0)=0$。 如果 $f$ 是一个连续型函数，那么当 $x \rightarrow 0$ 时，$f$ 要消失（接近 0 的速度）足够快，即 $\lim _ {x \rightarrow 0} f (x) \log ^ 2 x$ 有限。

2. $f(x)$ 非递减，这样共现次数较小（$x$ 较小）的权重不会太大。

3. $f(x)$ 在大 $x$ 处的值相对小，这样共现次数较大的权重也不会太大。

有很多函数满足以上属性，但是作者发现了一类较好的函数，如下

$$f(x) = \begin{cases} (x/x _ {max}) ^ {\alpha} & x < x _ {max}\\\\ 1 & o.w. \end{cases} \tag{9}$$

实验中使用 $x _ {max} = 100$，且 $\alpha = 3/4$ 比 $\alpha =1$ 的性能好。

(9) 式函数曲线如图 1 所示，

![](/images/nlp/glove_1.png)

<center>图 1. alpha=3/4</center>

# 3. 其他模型

本节介绍了其他模型，以及这些模型与 GloVe 的关系。

skip-gram 或者 ivLBL 使用 $Q _ {ij}$ 表示单词 `j` 在单词 `i` 的上下文出现的概率，假设 $Q _ {ij}$ 是一个 softmax，那么

$$Q _ {ij} = \frac {\exp (w _ i ^ {\top} \tilde w _ j)}{\sum _ {k=1} ^ V \exp (w _ i ^ {\top} \tilde w _ k)} \tag{10}$$

目标损失为

$$J = - \sum _ {i \in corpus, j \in context(i)} \log Q _ {ij} \tag{11}$$

上式表示最大化概率乘积，也就是所有组 单词-单词 的共现概率乘积，这与之前的训练集最大似然类似。

(11) 式计算语料中的每个单词 `i`，实际上语料中单词 `i` 会出现多次，其上下文中的单词 `j` 也可能会出现多次，导致 (11) 式中求和重复计算了，可以将相同的 `i-j` pair 提取出来，即，给 $Q _ {ij} ^ {X _ {ij}}$，取对数如下，

$$J = - \sum _ {i=1} ^ V \sum _ {j=1} ^ V X _ {ij} \log Q _ {ij} \tag{12}$$

由于 $X _ i = \sum _ k X _ {ik}$ 以及 $P _ {ij} = X _ {ij} / X _ i$，代入上式得

$$\begin{aligned} J &=-\sum _ {i=1} ^ V \sum _ {j=1} ^ V X _ i P _ {ij} \log Q _ {ij}
\\\\ &= -\sum _ {i=1} ^ V X _ i\sum _ {j=1} ^ V P _ {ij} \log Q _ {ij}
\\\\ &=\sum _ {i=1} ^ V X _ i H(P _ i, Q _ i)
\end{aligned} \tag{13}$$

实际上，使用 (13) 式这个目标函数训练效果并不好，一种解决方法是使用平方差来替代交叉熵损失，于是目标损失变为

$$\hat J = \sum _ {i,j} X _ i (\hat P _ {ij} - \hat Q _ {ij}) ^ 2 \tag{14}$$

其中 $\hat P _ {ij} = X _ {ij}$， $\hat Q _ {ij} = \exp (w _ i ^ {\top} \tilde w _ j)$，也就是说使用非归一化分布，但是 $X _ {ij}$ 值通常会特别大，这使得 (14) 式优化起来也不容易，所以先求对数，然后再求平方差，

$$\begin{aligned} \hat J &= \sum _ {i,j} X _ i (\log \hat P _ {ij} - \log \hat Q _ {ij}) ^ 2
\\\\ &=\sum _ {i,j} X _ i (w _ i ^ {\top} \tilde w _ j - \log X _ {ij}) ^ 2
\end{aligned} \tag{15}$$

最后，作者又指出，权重因子 $X _ i$ 不是最佳选择，所以还是使用 $f(X _ {ij})$ 作为权重因子，于是

$$\hat J = \sum _ {i,j} f(X _ {ij}) (w _ i ^ {\top} \tilde w _ j - \log X _ {ij}) ^ 2 \tag{16}$$

# 3. 源码

源码提供了四个工具，分别是：

1. `vocab_count` 。输入语料是由空格分隔的 tokens（对于原始语料可以使用 Stanford Tokenizer 切分），输出是 unigram counts。根据词汇表大小或者最小频次，可以对词汇表过滤。

2. `cooccur` 。建立单词-单词共现统计。

3. `shuffle` 。对单词-单词共现统计进行 shuffle 操作，类似于深度学习中对数据集 shuffle。

4. `glove` 。 训练 GloVe 模型。

前三个工具的实现代码这里不贴出来了，有兴趣的可直接看源码。下面给出 glove 的训练代码，并适当的做注释。

## 3.4 glove

对训练方法代码简单的说明。

首先需要明确，根据 (8) 式，模型参数为 $\lbrace w _ i, b _ i, \tilde w _ i, \tilde b _ i \rbrace _ {i=1} ^ V$，这四个变量分别表示：1. word vector; 2. word bias; 3. context word vector; 4. context word bias。

向量 $w _ i$ 和 $\tilde w _ i$ 长度均为 $d$，而 bias $b _ i$ 和 $\tilde b _ i$ 均为标量，$V$ 是词汇表大小，所以参数数量一共为

$$2 V(d+1) \tag{31}$$

假设参数存储在 tensor $W$ 中，那么 `W.shape` 应该是 `(2V, d + 1)`，前 `V` 行存储 word vector 和 word bias，后 `V` 行存储 context word vector 和 context word bias

```c++
// 训练方法
int train_glove() {
    long long a, file_size;
    int save_params_return_code;
    int b;
    FILE *fin;
    real total_cost = 0;    // double 类型

    fin = fopen(input_file, 'rb'); // 读取 shuffled 共现统计文件
    fseeko(fin, 0, SEEK_END);   // 定位到文件尾
    file_size = ftello(fin);    // 当前文件当前位置
    // CREC 包含两个单词id(int)，以及共现统计值(double)
    num_lines = file_size / (sizeof(CREC)); // 得到单词-单词共现数量
    fclose(fin);
    // 初始化 glove 模型的参数。此方法下方会给出有关注释
    initialize_parameters();
    // 保存模型参数初值
    if (save_init_param) { save_params_return_code = save_params(0); }
    // 分配用于保存线程句柄的数组
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    // linrs_per_thread：存储每个线程负责 单词-单词 pairs 的数量
    lines_per_thread = (long long *)malloc(num_threads * sizeof(long long));

    for (b = 0; b < num_iter; b++) {  // 迭代次数
        total_cost = 0;     // 总的损失
        // 设置每个线程负责的 单词-单词 pairs 数量
        for (a = 0; a < num_threads - 1; a++) lines_per_thread[a] = num_lines / num_threads;
        lines_per_thread[a] = num_lines / num_threads + num_lines % num_threads;
        // 保存所有线程的编号
        long long *thread_ids = (long long*)malloc(sizeof(long long) * num_threads);
        // 设置线程编号，从 0 开始
        for (a = 0; a < num_threads; a++) thread_ids[a] = a;
        // 创建每个线程，线程执行函数为 glove_thread，函数参数为线程编号
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, glove_thread, (void *)&thread_ids[0]);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);// 等待线程
        for (a = 0; a < num_threads; a++) total_cost += cost[a];
        free(thread_ids);
    }
    free(pt);
    free(lines_per_thread);
    return save_params(-1); // 保存最后一次迭代训练后的参数值
}
// 初始化模型参数
void initialize_parameters() {
    if (seed == 0) { seed = time(0); }
    srand(seed);        // 设置随机种子
    long long a;
    // 计算 glove 模型参数数量，参考 (8) 式 和 (31) 式
    long long W_size = 2 * vocab_size * (vector_size + 1);
    // 申请 glove 模型参数数组内存，每个参数为 double(real) 类型
    a = posix_memalign((void **)&W, 128, W_size * sizeof(real));
    // 使用梯度下降法更新，gradsq 为参数的梯度平方
    a = posix_memalign((void **)&gradsq, 128, W_size * sizeof(real));
    if (load_init_param) {} // 加载保存的参数初值
    else {
        for (a = 0; a < W_size; ++a) {
            // 每个参数均从 [-0.5, 0.5] 采样，然后除以 vector_size
            W[a] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
        }
    }
    if (load_init_gradsq) {}
    else {
        for (a = 0; a < W_size; ++a) {
            gradsq[0] = 1.0;
        }
    }
}
// 保存模型参数
int save_params(int nb_iter) {
    long long a, b;
    char format[20];
    // 存储输出文件名，输出梯度平方文件名
    char output_file[MAX_STRING_LENGTH+20], output_file_gsq[MAX_STRING_LENGTH+20];
    // 保存一个单词
    char *word = malloc(size(char) * MAX_STRING_LENGTH + 1);
    FILE *fid, *fout;
    FILE *fgs = NULL;

    if (use_binary > 0 || nb_iter == 0) { // 保存为 bin 文件
        // 设置 output_file，文件后缀为 `.bin`
        ...
        fout = fopen(output_file, "wb");
        for (a = 0; a < 2 * vocab_size * (vector_size + 1); a++)
            fwrite(&W[a], sizeof(real), 1, fout);
        fclose(fout);
        if (save_gradsq > 0) {  // 需要存储梯度平方
            // 设置 output_file_gsq，文件后缀为 `.bin`
            ...
            fgs = fopen(output_file_gsq, "wb");
            for (a = 0; a < 2 * vocab_size * (vector_size + 1); a++)
                fwrite(&gradsq[a], sizeof(real), 1, fgs);
            fclose(fgs);
        }
    }
    if (use_binary != 1) {  // 保存为 txt 文件
        // 设置 output_file，文件后缀为 `.txt`
        ...
        if (save_gradsq > 0) {
            // 设置 output_file_gsq，文件后缀为 `.txt`
            ...
            fgs = fopen(output_file_gsq, "wb");
        }
        fout = fopen(output_file, "wb");
        fid = fopen(vocab_file, "r");
        if (write_header) fprintf(fout, "%lld %d\n", vocab_size, vector_size);
        for (a = 0; a < vocab_size; a++) {
            if (fscanf(fid, format, word) == 0) {return 1;}  // 读取当前单词
            if (strcmp(word, "<unk>") == 0) {return 1;}   // 词汇表中不能有 <unk>
            fprintf(fout, "%s", word);  // 将当前单词写入输出文件
            if (model == 0) {   // 保存 word vector 和 context word vector
                for (b = 0; b < (vector_size + 1); b++)
                    fprintf(fout, "%lf", W[a * (vector_size + 1) + b]);
                for (b = 0; b < (vector_size + 1); b++)
                    fprintf(fout, "%lf", W[(vocab_size + a) * (vector_size + 1) + b]);
            } else if (model == 1) {    // 仅保存 word vector，且不包含 bias 项
                for (b = 0; b < vector_size; b++)
                    fprintf(fout, "%lf", W[a * (vector_size + 1) + b]);
            } else if (model == 2) { // 保存 word vector + context word vector
                for (b = 0; b < vector_size; b++)
                    fprintf(fout, "%lf", W[a * (vector_size + 1) + b] +
                                         W[(vocab_size + a) * (vector_size + 1) + b]);
            }
            fprint(fout, "\n");
            if (save_gradsq > 0) {} // 保存梯度平方
        }
        ...
    }
}
```

训练模型使用多线程，每个线程使用部分 单词-单词 共现统计进行训练，类似于一个线程就是一个 minibatch。代码如下。

输入是经过 shuffle 的共现统计文件，每一行表示一个 单词-单词 pair 的共现统计频次。假设输入文件有 `num_lines` 行，线程数量为 `num_threads`，那么编号为 `id` 的线程对应的输入数据的起始行位置是 `num_lines / num_threads * id` ，距离输入文件开始位置的偏移量还需要在乘以 `sizeof(CREC)`，因为每个 单词-单词 pair 的统计频次数据占用 `sizeof(CREC)` 个字节。

实际上输入是 bin 文件，所以不是按行，而是按 `CREC` 类型，每个 `CREC` 对象对应一个 单词-单词 pair 共现统计频次。

根据 (8) 式，损失对参数的梯度为，

$$\begin{aligned}\frac {\partial J/2}{\partial w _ i} &= f(X _ {ij}) (w _ i ^ {\top} \tilde w _ j + b _ i + \tilde b _ j - \log X _ {ij}) \tilde w _ j 
\\\\ \frac {\partial J/2}{\partial w _ i} &= f(X _ {ij}) (w _ i ^ {\top} \tilde w _ j + b _ i + \tilde b _ j - \log X _ {ij})
\end{aligned}\tag{32}$$




```c++
// 线程函数，使用部分共现统计数据训练模型参数
void *glove_thread(void *vid) {
    long long a, b, l1, l2;
    long long id = *(long long*)vid;
    CREC cr;
    real diff, fdiff, temp1, temp2;
    FILE *fin;
    // input_file: shuffled 共现统计文件
    fin = fopen(input_file, "rb");
    // 定位到本线程对应的输入数据位置处
    fseeko(fin, (num_lines / num_threads * id) * (sizeof(CREC)), SEEK_SET);
    cost[id] = 0;
    // 存储单词 `i` 的参数的梯度，和单词 `j` 的参数的梯度
    real *W_updates1 = (real*)malloc(vector_size * sizeof(real));
    real *W_updates2 = (real*)malloc(vector_size * sizeof(real));
    for (a = 0; a < lines_per_thread[id]; a++) {    // 遍历本线程的那部分 单词-单词 pairs
        fread(&cr, sizeof(CREC), 1, fin);   // 按 CREC 大小读取，读取一个 CREC 大小
        if (feof(fin)) break;
        // word id `0` 表示 <unk>
        if (cr.word1 < 1 || cr.word2 < 1) {continue;}

        // l1 单词 `i` 在权重数组中的起始位置
        // l2 单词 `j` 在权重数组中的起始位置
        l1 = (cr.word1 - 1LL) * (vector_size + 1);
        l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + 1)

        // 准备计算损失
        diff = 0;   // 参考 (8) 式括号内的差
        // 计算 wi^T · wj
        for (b = 0; b < vector_size; b++) diff += W[b + l1] * W[b + l2];
        diff += W[vector_size + l1] + W[vector_size + l2] - log(cr.val);
        // 使用 f(Xij) 系数。得到 f(Xij) * ()
        fdiff = (cr.val > x_max) ? diff : pow(cr.val / x_max, alpha) * diff;
        // 得到 f(Xij) * () * ()。 增加一个 1/2，为了求导方便
        cost[id] += 0.5 * fdiff * diff;

        // Adaptive gradent updates
        real W_updates1_sum = 0;
        real W_updates2_sum = 0;
        // 计算参数的梯度
        for (b = 0; b < vector_size; b++) {
            // 参考 (32) 式，eta 表示学习率
            temp1 = fmin(fmax(fdiff * W[b + l2], -grad_clip_value), grad_clip_value) * eta;
            temp2 = fmin(fmax(fdiff * W[b + l1], -grad_clip_value), grad_clip_value) * eta;

            W_updates1[b] = temp1 / sqrt(gradsq[b + l1]);
            W_updates2[b] = temp2 / sqrt(gradsq[b + l2]);
            W_updates1_sum += W_updates1[b];
            W_updates2_sum += W_updates2[b];
            gradsq[b + l1] += temp1 * temp1;
            gradsq[b + l2] += temp2 * temp2;
        }
        if (!isnan(W_updates1_sum) && !isinf(W_updates1_sum) && !isnan(W_updates2_sum) && !isinf(W_updates2_sum)) {
            // 更新参数
            for (b = 0; b < vector_size; b++) {
                W[b + l1] -= W_updates1[b];
                W[b + l2] -= W_updates2[b];
            }
        }
        // calc gradient of bias and update bias
        W[vector_size + l1] -= checknan(fdiff / sqrt(gradsq[vector_size + l1]));
        // checknan: if nan/inf, return 0; else return input
        W[vector_size + l2] -= checknan(fdiff / sqrt(gradsq[vector_size + l2]));
        fdiff *= fdiff;
        gradsq[vector_size + l1] = fdiff;
        gradsq[vector_size + l2] = fdiff;
    }
    free(W_updates1);
    free(W_updates2);
    fclose(fin);
    pthread_exit(NULL);
}
```

**# 问题**

多线程，每个线程会从参数矩阵中读取参数的值，然后计算参数的梯度，最后更新回参数矩阵中。某个单词可能会出现在多个线程中，那么为何不给参数矩阵加锁？