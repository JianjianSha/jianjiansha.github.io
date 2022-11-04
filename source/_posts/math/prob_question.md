---
title: 概率题总结
date: 2022-09-01 17:50:26
tags: probability
mathjax: true
---

一、 [知乎某抽样概率题](https://www.zhihu.com/question/550787499)

> 1、针对含 n 个不同元素的集合，每次随机抽样一个样品，检查后放回，共抽样 n 次。问：被检查的不同元素个数期望是多少？对于特定元素，未被抽到检查的概率是多少？

> 2、同第 1 种情况，但抽到的样品有 p 概率被破坏（消耗），即只有 1-p 的概率放回，n 个元素的集合共抽样 n 次。抽查到的期望与未抽到的概率是多少？

> 3、同第 2 种情况，但每次抽样 k 个，共抽 n/k 次（总共仍是 n 次）。那么抽到的期望与未抽到的概率是多少，与情况 2 相同吗？

> 4、在第2或3情况下，如果有 m 个不同的抽查员同时独立并行操作，每人抽查 n 次。结果汇总，被至少一个抽查员抽到的不同元素的个数期望是多少？对某个元素未被所有抽查员抽到的概率是多少？

1. 问：被检查的不同元素个数期望是多少？对于特定元素，未被抽到检查的概率是多少？

    将 n 次抽样看作是一个整体过程，在这个过程中，元素 i **未被抽到的概率** 为 $(1−1/n)^n$ ，被抽到的概率为 $1-(1−1/n)^n$ ，记随机变量 $X_i=1$ 表示元素 i 被抽到， $X_i=0$ 表示元素 i 未被抽到，那么 $\mathbb E[X_i]=p(X_i=1)=1−(1−1/n)^n$ ，
    被抽到的 **不同元素数量的期望** 为 

    $$\mathbb E[X]=\sum_{i=1}^n \mathbb E[X_i]=n−n(1−\frac 1 n)^n$$ 

2. 抽到的样品有 p 概率被破坏（消耗），即只有 1-p 的概率放回

    元素 i，第 1 次未被抽到的概率为 $C_0^0 \frac {n−1}n$ ，第 2 次未抽到的条件概率为  $C_1^0 p \frac {n−2}{n−1}+C_1^1 (1−p)\frac {n−1} n$ ，第三次未抽到的条件概率为 $\sum_{i=0}^2 C_2^i p^{2−i}(1−p)^i \frac {n+i−3}{n+i−2}$ ，第 k+1 次未抽到的条件概率为

    $$p(k+1|k)=\sum_{i=0}^k C_k^i p^{k−i}(1−p)^i \frac {n+i−k−1}{n+i−k}, \quad k=1,\ldots,n−1$$

    其中初始条件为 $p(k=1)=C_0^0 \frac {n−1}n$ 。

    元素 i 在 n 次抽样中 **全部未被抽到的概率** 为

    $$p(X_i=0)=p(1)\prod_{k=1}^{n−1}p(k+1|k)$$

    剩下的分析与前面相同，n次抽样中，被抽到的概率为 $p(X_i=1)=1−p(X_i=0)$ ，于是 目标期望 为

    $$\mathbb E[X]=\sum_{j=1}^n \mathbb E[X_j]=\sum_{j=1}^n \left(1−p(1)∏_{k=1}^{n−1}p(k+1|k)\right)$$


以下我们从期望的定义出发，进行求解。

第 i 次抽到的元素变量记为 $e_i$ ，概率为 $1/n$ ，给元素编号 1到 n，那么 $ei∈\{1,2,…,n\}$ 
对于一次完整的抽样过程，概率

$$p(e_1,e_2,⋯,e_n)=(\frac 1 n)^n $$

记函数 $C(e_1,e_2,⋯,e_n)$ 表示计算其中不同元素的数量，函数 

$$C(e_1,e_2,⋯,e_n)=y $$

的解的数量记为 $m(y)$ ，每个解对应的事件互斥，那么 $p(y)=m⋅(\frac 1 n)^n$ ，于是目标期望为

$$\mathbb E=∑_{y=1}^n yp(y)=∑_{y=1}^n ym(\frac 1n)^n=\frac 1 {n^n}\sum_{y=1}^n ym$$

所以问题是如何求 $m(y)$ 。

y=1 时，先选1个整数，为 $C_n^1$ 种可能，然后 $e_i$ 均相同，有 $1^n$ 种可能，故 $m(1)=C_n^1⋅1^n$ 
y=2 时，先选两个整数，为 $C_n^2$ ，然后给 $e_i$ 赋值， $2^n$ 种可能，但是要去掉全部为某个整数的情况， $C_2^1$ 个可能，即 $2^n−C_2^1$ ，故 $m(2)=C_n^2(2^n−C_2^1)$ 
y=3 时，选 3 个整数，为 $C_n^3$ ，然后给 $e_i$ 赋值， $3^n$ 种可能，但是要减去：

- 所有 $e_i$  全部为 1 个值， $C_3^1$ 种可能
- 所有 $e_i$ 只有两个不同值，根据上面分析为 $C_3^2(2^n−C_2^1)$

于是 $m(3)=C_n^3(3^n−C_3^2(2^n−C_2^1)−C_3^1)$

看出规律了，令 $m(y)=C_n^yq(y)$ ，那么

$$\begin{aligned}q(1)&=1^n \\
q(2)&=2^n−C_2^1=2^n−C_2^1q(1) \\
q(3)&=3^n−C_3^2(2^n−C_2^1)−C_3^1=3^n−C_3^2q(2)−C_3^1q(1) \\
\cdots \\
q(y)&=y^n−∑_{i=1}^{y−1} C_y^iq(i) \end{aligned}$$

综上，目前期望为

$$\mathbb E=∑_{y=1}^n ym⋅(\frac 1n)^n=\frac 1{n^n}∑_{y=1}^n yC_n^yq(y) $$

```python
def yuansu1_def(n=10):
    '''
    根据期望的定义计算解析解
    '''
    q_cache = [0] * n
    mu = 0
    def q(y):
        if y == 1:
            return 1
        r = y ** n
        for i in range(1, y):
            r -= comb(y, i) * q_cache[i-1]
        return r
    for i in range(1, n+1):
        q_cache[i-1] = q(i)
        mu += i * comb(n, i) * q_cache[i-1]
    return mu / (n**n)
    
def yuansu1_num(n=10, N=200000):
    '''
    数值模拟
    N: 试验次数
    '''
    x = np.random.randint(0, n, (N, n))
    c = np.apply_along_axis(np.bincount, 1, x, minlength=n)
    c = np.sum(c > 0, axis=1)
    a, b = np.unique(c, return_counts=True)
    return np.sum(a * b) / N

def yuansu1(n=10):
    '''
    计算期望
    '''
    return n-n*((1-1/n)**n)

def yuansu2(n=10, p=0.5):
    '''
    计算期望，有概率 p 不放回
    '''
    num = (n-1)/n
    for k in range(1, n):
        s = 0.0
        for i in range(k+1):
            s += comb(k, i) * (p**(k-i))*((1-p)**i)*(n+i-k-1)/(n+i-k)
        num *= s
    return n * (1-num)

def yuansu2_num(n=10, p=0.5, N=1000000):
    '''
    数值模拟
    '''
    counts = [0] * n
    for _ in range(N):
        s = [i for i in range(n)]
        xs = [0] * n
        for i in range(n):
            x = random.choice(s)
            xs[i] = x
            if random.random() < p:
                s.remove(x)
        j = len(np.unique(xs))
        counts[j-1] += 1
    num = 0
    for i, c in enumerate(counts):
        num += (i+1) * c

    return num / N
```