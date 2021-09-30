---
title: Perceptron for Halfspaces
date: 2021-09-15 10:40:35
tags: machine learning
p: ml/halfspace
mathjax: true
---

关于线性可分二分类问题的讨论
<!--more-->

# 算法
数据集为 $S=\{(\mathbf x_i, y_i)\}_{i=1}^m$，数据集可分（不可分的情况这里不讨论），仿射函数为

$$L_d=\{h_{\mathbf w, b}: \mathbf w \in \mathbb R^d, b \in \mathbb R\}=\{\langle \mathbf w, \mathbf x \rangle+b: \mathbf w \in \mathbb R^d, b \in \mathbb R\}$$


假设类 为 

$$HS_d=\text{sign} \circ L_d$$

令 $\mathbf w'=(b, w_1, \cdots w_d)$，$\mathbf x'=(1, x_1, \cdots x_d)$，那么可写为齐次形式

$$h_{\mathbf w, b}(\mathbf x) = \langle \mathbf w', \mathbf x' \rangle$$

以下讨论如不特别说明，均使用齐次形式进行说明。现在，我们要求 ERM（经验损失最小化）预测器，那么就是要求一个 $\mathbf w^*$，使得 $\forall i = 1, \cdots , m$，有 $\text{sign}(\langle \mathbf w^*, \mathbf x_i \rangle)=y_i$，即

$$y_i \langle \mathbf w^*, \mathbf x_i \rangle > 0, \quad \forall i = 1, \cdots, m$$ (1)

由于假设了数据集是线性可分的，$\mathbf w^*$ 必然存在。

## Perceptron 迭代算法
在迭代时刻 $t$，参数记为 $\mathbf w^{(t)}$，初始时为 $\mathbf w^{(1)}=\mathbf 0$。在 $t$ 迭代时，寻找一个样本使得 $\mathbf w^{(t)}$ 将其分类错误，即 $y_i \langle \mathbf w^{(t)}, \mathbf x_i \rangle \le 0$，那么迭代更新为

$$\mathbf w^{(t+1)}=\mathbf w^{(t)}+y_i \mathbf x_i$$

由于

$$y_i \langle \mathbf w^{(t+1)}, \mathbf x_i \rangle=y_i \langle \mathbf w^{(t)}+y_i \mathbf x_i, \mathbf x_i \rangle=y_i \langle \mathbf w^{(t)}, \mathbf x_i \rangle+\|\mathbf x_i \|^2$$

样本 $(\mathbf x_i, y_i)$ 被 $\mathbf w^{(t)}$ 分类错误，故上式右侧第一项 $\le 0$，而 $\| \mathbf x_i \|^2 \ge 0$，所以是朝着正确的方向更新 $\mathbf w$。

__求解步骤__

输入：训练集 $(\mathbf x_1, y_1), \cdots , (\mathbf x_m, y_m)$

初始化: $\mathbf w^{(1)}=(0,\cdots,0)$

$\quad \text{for} \ t=1,2,\dots$

$\quad \quad \text{if} \ (\exists \ i \ s.t. \ y_i \langle \mathbf w^(t), \mathbf x_i \rangle \le 0) \ \text{then}$

$\quad \quad \quad \mathbf w^{(t+1)}=\mathbf w^{(t)}+y_i \mathbf x_i$

$\quad \quad \text{else}$

$\quad \quad \quad \text{output} \ \mathbf w^{(t)}$


我们还可以证明在有限的迭代次数后能得到符合条件的 $\mathbf w$。

令 $B=\min \{\|\mathbf w\|: \forall \ i \in [m], \ y_i \langle \mathbf w, \mathbf x_i \rangle \ge 1\}$， $R=\max_i \| \mathbf x_i \|$，那么上述算法最多在 $(RB)^2$ 次迭代后结束，且结束时有 $\forall \ in \in [m], \ y_i \langle \mathbf w^{(t)}, \mathbf x_i \rangle > 0$。

注意， $\{\|\mathbf w\|: \forall \ i \in [m], \ y_i \langle \mathbf w, \mathbf x_i \rangle \ge 0\}$  没有最小值，因为如果 $\mathbf w$ 满足条件，那么对 $\forall \ 0<\gamma <1$， $\gamma \mathbf w$ 也满足条件，而 $\|\gamma \mathbf w\|^2=\gamma^2 \|\mathbf w\|^2 < \|\mathbf w \|^2$。

但是 $\{\|\mathbf w\|: \forall \ i \in [m], \ y_i \langle \mathbf w, \mathbf x_i \rangle \ge 1\}$ 有最小值，因为 $|y_i\langle \mathbf w, \mathbf x_i \rangle|=\|\mathbf w \|\cdot\|\mathbf x_i\| \cdot|\cos \theta_i|\ge 1 \Rightarrow \|\mathbf w \| \ge 1/(\|\mathbf x_i\| \cdot|\cos \theta_i|)$  对 $\forall i \in [m]$ 均成立，式 (1)只是限制了 $\mathbf w$ 的方向，故一定存在某个 $\mathbf w$，使得 $\|\mathbf x_i\| \cdot|\cos \theta_i|$ 最大。

下面证明在最多 $(RB)^2$ 的迭代次数后算法结束。

__证：__

令 $\mathbf w^*$ 为某个满足 $B$ 的解，即 $\forall \ i \in [m]$，均有 $y_i \langle \mathbf w^*, \mathbf x_i \rangle \ge 1$，且 $\mathbf w^*$ 范数最小。

由于 $\mathbf w^{(1)}=\mathbf 0$，故 $\langle \mathbf w^*, \mathbf w^{(1)}\rangle=0$，那么

$$\langle \mathbf w^*, \mathbf w^{(t+1)}\rangle - \langle \mathbf w^*, \mathbf w^{(t)}\rangle=\langle \mathbf w^*, \mathbf w^{(t+1)}-\mathbf w^{(t)}\rangle=\langle \mathbf w^*, y_i \mathbf x_i\rangle=y_i\langle \mathbf w^*, \mathbf x_i\rangle \ge 1$$

于是

$$\langle \mathbf w^*, \mathbf w^{(T+1)} \rangle=\sum_{t=1}^T \left( \langle \mathbf w^*, \mathbf w^{(t+1)}\rangle-\langle \mathbf w^*, \mathbf w^{(t)}\rangle \right) \ge T$$

另有

$$\|\mathbf w^{(t+1)}\|^2=\|\mathbf w^{(t)}+y_i \mathbf x_i\|^2=\|\mathbf w^{(t)}\|^2+2y_i \langle \mathbf w^{(t)}, \mathbf x_i\rangle+y_i^2\| \mathbf x_i\|^2 \le \|\mathbf w^{(t)}\|^2+R^2$$

上面最后一个不等式中，由于 $t+1$ 次更新时，使用的是令 $\mathbf w^{(t)}$ 分类错误的样本 $(\mathbf x_i, y_i)$，故 $2y_i \langle \mathbf w^{(t)}, \mathbf x_i\rangle \le 0$，且根据上面定义，有 $\| \mathbf x_i \|^2 \le R$，再根据 $\|\mathbf w^{(1)}\|^2=0$所以有

$$\|\mathbf w^{(T+1)} \|^2 \le TR^2$$

故

$$\frac {\langle \mathbf w^*, \mathbf w^{(T+1)} \rangle} {\|\mathbf w^*\| \cdot \|\mathbf w^{(T+1)}\|} \ge \frac T {B \sqrt T R}=\frac {\sqrt T} {BR}$$

根据 Cauchy-Schwartz 不等式，上式最左边项 $<1$，于是有 $1 > \sqrt T / (BR)$，即迭代次数满足关系

$$T \le (RB)^2$$

## VC 维

## 齐次形式

$\mathbb R^d$ 中齐次 Halfspance 类的 VC 维等于 $d$。

__证：__

要证明 $VCdim(\mathcal H)=d$，需要满足两个条件：

- 存在样本集 $C$，其大小为 $|C|=d$，此样本可以被 $\mathcal H$ shattered。 
- 任意一个大小为 $|C|=d+1$ 的样本集均不能被 $\mathcal H$ shattered。

首先，考虑标准正交基向量 $\mathbf e_1, \cdots, \mathbf e_d$（one-hot），这个向量集可以被齐次 Halfspace shattered，这里的假设类具有如下形式

$$\mathcal H=\{\text{sign}(\langle \mathbf w , \mathbf x\rangle): \mathbf w \in \mathbb R^d\}$$

对于任意的 $(y_1, \cdots, y_d)$，令 $\mathbf w=(y_1, \cdots, y_d)$，就可以得到 $h(\mathbf e_i)=\text{sign}(\langle \mathbf w , \mathbf x\rangle)=y_i$，所以 $\mathbf e_1, \cdots, \mathbf e_d$ 被 $\mathcal H$ shattered，第一个条件满足。

另一方面，令任意集合 $\mathbf x_1, \cdots, \mathbf x_{d+1} \in R^d$，向量数据大于维度，故这 $d+1$ 个向量必然线性相关，即 $\sum_{i=1}^{d+1} a_i \mathbf x_i=\mathbf 0$，其中 $a_1, \cdots, a_{d+1}$ 不全为 $0$，记 $I=\{i:a_i>0\}$，$J=\{j:a_j<0\}$，所以 $I, \ J$ 至少有一个集合不为空。

- $I, \ J$ 均不为空，那么

$$\sum_{i \in I}a_i \mathbf x_i = \sum_{j \in J} |a_j|\mathbf x_j$$

假设 $\mathbf x_1, \cdots, \mathbf x_{d+1}$ 被 $\mathcal H$ shattered，那么存在一个 $\mathbf w$，使得 $\langle \mathbf w, \mathbf x_i \rangle >0, \ \forall \ i \in I$，且 $\langle \mathbf w, \mathbf x_i \rangle <0, \ \forall \ i \in J$，于是有

$$0 < \sum_{i \in I} a_i \langle \mathbf w, \mathbf x_i \rangle=\langle \mathbf w, \sum_{i \in I} a_i \mathbf x_i \rangle=\langle \mathbf w, \sum_{j \in J} |a_j| \mathbf x_j \rangle=\sum_{j \in J} |a_j| \langle \mathbf w, \mathbf x_j \rangle <0$$

矛盾，也就是说不存在这样的 $\mathbf w$，即 $\mathbf x_1, \cdots, \mathbf x_{d+1}$ 不能被 $\mathcal H$ shattered。

- $I=\emptyset, \ J \neq \emptyset$，那么

$$\sum_{j \in J} |a_j|\mathbf x_j=0$$

假设 $\mathbf x_1, \cdots, \mathbf x_{d+1}$ 被 $\mathcal H$ shattered，那么存在一个 $\mathbf w$，使得 $\langle \mathbf w, \mathbf x_i \rangle <0, \ \forall \ i \in J$，于是有

$$0=\langle \mathbf w, \sum_{j \in J} |a_j|\mathbf x_j \rangle= \sum_{j \in J} |a_j| \langle \mathbf w, \mathbf x_j \rangle <0$$

矛盾，也就是说不存在这样的 $\mathbf w$，即 $\mathbf x_1, \cdots, \mathbf x_{d+1}$ 不能被 $\mathcal H$ shattered。

- $I \neq \emptyset, \ J = \emptyset$，情况与上一点相同。

综上，任意 $\mathbf x_1, \cdots, \mathbf x_{d+1}$ 不能被 $\mathbf H$ shattered，故 $VCdim(\mathbf H)=d$，证毕。

## 非齐次形式
$\mathbb R^d$ 中非齐次 Halfspance 类的 VC 维等于 $d+1$。

__证：__

令大小为 $d+1$ 的集合 $C=(\mathbf 0, \mathbf e_1, \cdots, \mathbf e_d)$，假设类为

$$\mathcal H=\{\text{sign}(\langle \mathbf w, \mathbf x \rangle + b): \mathbf w \in \mathbb R^d\}$$

对于任意的 label 值 $(y_1, \cdots, y_{d+1})$，参数 $\mathbf w, b$ 满足条件
$$y_1 \cdot b > 0$$
$$y_i(w_i+b)>0, \ \forall i \in [d]$$

均有解，所以找到一个集合 $C$ 可以被 $\mathcal H$ shattered。

然后，对任意向量集合 $\mathbf x_1, \cdots, \mathbf x_{d+2}$，假设能被 $\mathcal H$ shattered，我们记 $\mathbb R^{d+1}$ 中的齐次形式的 halfspace 类假设的一般形式为 $\mathcal H'=\{\text{sign}(\langle \mathbf w', \mathbf x'\rangle): \mathbf w' \in \mathbb R^{d+1}\}$，显然令 $\mathbf w'=(b, w_1, \cdots , w_d)$，且 $\mathbf x'=(1,x_1, \cdots, x_d)$ 就得到 $\mathcal H$ （$\mathbb R^d$  中的非齐次类），也就是说 $\mathcal H \subset \mathcal H'$，于是集合 $\mathbf x_1, \cdots, \mathbf x_{d+2}$ 也应该能被 $\mathcal H'$ shattered，然而，根据前面齐次情况的讨论，$\mathbb R^{d+1}$ 中，任意向量集合 $\mathbf x_1, \cdots, \mathbf x_{d+2}$ 不能被 $\mathcal H'$ shattered，产生矛盾，故假设错误。

综合，$$VCdim(\mathcal H)=d+1$$

