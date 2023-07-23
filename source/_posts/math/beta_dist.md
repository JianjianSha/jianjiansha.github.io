---
title: Beta 分布
date: 2023-07-14 13:46:43
tags: math
mathjax: true
summary: 介绍 Beta 分布
---


# 1. 定义

二项分布中，试验成功次数 $x$ 是随机变量，而试验总数 $n$ 和单次试验成功的概率 $p$ 是常数。

现在将单次试验成功的概率作为随机变量，而试验总数 $n$ 和 $n$ 次试验中成功次数为常量。记 $n$ 次试验中成功次数记为 $\alpha$，失败次数记为 $\beta$，显然有 $\alpha + \beta = n$，单次试验成功的概率记为 $x$，那么 $x$ 的概率密度函数为，

$$f(x)=\frac {(n-1)!}{(\alpha -1)! (\beta-1)!}x ^ {\alpha-1} (1-x) ^ {\beta -1} \tag{1}$$

如果 $\alpha, \ \beta$ 不在局限于自然数，那么有

$$f(x)=\frac {\Gamma(\alpha + \beta)}{\Gamma(\alpha)+\Gamma(\beta)}x ^ {\alpha-1} (1-x) ^ {\beta -1} \tag{2}$$


# 2. 性质

1. 期望

    $$\mu = \frac {\alpha} {\alpha + \beta} \tag{3}$$

2. 方差

    $$V = \frac {\alpha \beta}{(\alpha + \beta) ^ 2 (\alpha + \beta + 1)} \tag{4}$$

# 3. MoM 拟合

使用 method of moment 方法拟合 Beta 分布。假设数据为 $X=\lbrace x _ n \rbrace _ {n=1} ^ N$，使用样本期望和样本方差代替分布的期望和方差，

$$\overline x = \frac {\alpha} {\alpha + \beta}, \quad s ^ 2=\frac {\alpha \beta} {[(\alpha + \beta) ^ 2 (\alpha + \beta + 1)]} \tag{5}$$

解 (8) 式，得到 $\alpha, \beta$ 的估计，

$$\beta = \frac {\alpha (1-\overline x)}{\overline x}, \quad \alpha = \overline x \left[\frac {\overline x (1- \overline x)} {s ^ 2} -1 \right] \tag{6}$$

注意这里样本方差是除以 $N-1$ 。

# 4. MLE 拟合

见参考链接 2 。

# 5. 应用

BMM 应用于图像分类，参考论文 [BETA MIXTURE MODELS AND THE APPLICATION TO IMAGE CLASSIFICATION](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2009/pdfs/0002045.pdf) 。
# ref

1. https://real-statistics.com/binomial-and-related-distributions/beta-distribution/

2. https://real-statistics.com/distribution-fitting/distribution-fitting-via-maximum-likelihood/fitting-beta-distribution-parameters-mle/