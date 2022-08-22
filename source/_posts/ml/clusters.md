---
title: 聚类
date: 2022-08-11 18:54:16
tags: machine learning
---

聚类算法属于无监督算法。

# 1. K-means

给定样本集合 $\mathcal D=\{x_i\}_{i=1}^N$，聚类簇数量 $k$，每个簇集合记为 $C_i,\ i=1,\ldots, k$，那么最小化目标函数

$$L=\sum_{i=1}^k \sum_{x \in C_i} \|x-\mu_i\|_2^2 \tag{1}$$

其中 $\mu_i=\frac 1 {C_i}\sum_{x \in C_i} x$ 是簇 $C_i$ 的中心。

采用迭代近似求解。

---
算法 1：k-means
输入：数据集 $x_1,\ldots, x_N$，簇数量 $k$

在数据集中随机选择 $k$ 个数据作为初始的簇中心 $\mu_1,\ldots, \mu_k$

**for** $t=1,\ldots$ **do**

&emsp; $C_i=\{\mu_1\}, \ i=1, \ldots, k$

&emsp; **for** $j=1,\ldots, N$ **do**

&emsp;&emsp; $i^{\star}:=\arg\min_i \|x_j-\mu_i\|_2^2$

&emsp;&emsp; $C_i := C_i \cup \{x_j\}$

&emsp; **end**

&emsp; $\mu'_i=\frac 1 {C_i}\sum_{x \in C_i} x, \quad i=1,\ldots, k$

&emsp; **if** $\forall i, i\in [k]$，$\mu'_i=\mu$ （两者相等）

&emsp;&emsp; **break**

&emsp; **else**

&emsp;&emsp; $\mu_i:=\mu'_i, \quad i=1,\ldots, k$

---

k-means 算法优点：

1. 实现简单
2. 可解释性强

缺点：

1. $k$ 值依赖先验知识
2. 初始的聚类簇中心选择不同，聚类结果可能也不同
3. 噪声敏感
4. 得到的是局部最优解
5. 数据为高维时，计算L2范数较慢，可先做 PCA 降维
6. 算法假设簇是凸的，且各向同性，故对细长簇或不规则流形簇聚类效果较差。

# 2. DBSCAN

首先来看几个概念。

**密度**：基于密度的思想。每个密集区域表示一个聚类簇。

**核心点**：那么某个点的邻域半径 R 内样本点的数量大于等于 minpoints，这个点就是核心点。位于邻域内但是不是核心点的样本点称为边界点。既不是核心点也不是边界点的叫做噪声点。


**密度直达**：若 P 为核心点，Q 在 P 邻域内（Q 可以为 P），那么称 P 到 Q 密度直达。

**密度可达**：如果存在核心点 $P_1, P_2,\ldots, P_{n-1}$，以及最后一个点 $P_n$， 且 $P_i$ 到 $P_{i+1}$ 密度直达，$i=1,\ldots, n-1$，那么称 $P_1$ 到 $P_n$ 密度可达。

**密度相连**：如果存在核心点 $S$，使得 $S$ 到 $P$ 和 $Q$ 均密度可达，那么称 $P$ 和 $Q$ 密度相连。

密度直达和密度可达不具有对称性，密度相连具有对称性。

例如 $P$ 到 $Q$ 密度直达，但 $Q$ 可能不是核心点（邻域半径 R 内有 P 点，但是总样本点数量小于 minpoints），故 $Q$ 到 $P$ 不一定密度直达。

**算法思想**：由密度可达关系导出最大密度相连的样本集合，即为最终聚类的一个簇。

---
算法2：DBSCAN

输入：数据集 $\mathcal D=\{x_i\}_{i=1}^N$，邻域半径 $\epsilon$，构成核心点的邻域内样本点数量阈值 $m_0$
输出：聚类簇 $C_i$

$K=()$ 核心点列表
$N=()$ 邻域集合列表

___
**for** $i=1,\ldots, N$ **do**

&emsp; $N(x_i):=F(\mathcal D, x_i)$ # 调用 $F$ 函数获取 $x_i$ 的邻域样本集合，建立与 $x_i$ 的映射关系

&emsp; **if** $|N_i| \ge m_0$ **do**

&emsp; &emsp; $K:=K\cup (x_i,)$ # 记录核心点

&emsp; **end if**

**end for**

$C:=()$ # 用于保存聚类簇的列表

$\mathcal T:=\mathcal D$ # 初始化尚未探测的点集合
___

**while** $K \ne \emptyset$ **do**

&emsp; $i \sim \mathcal U[0, |K|-1] \cap \mathbb N$，$k:=K(i)$ # 随机取一个核心点 $k$

&emsp; $P=\{k\}$，$Q=\mathcal T \setminus P$  # 以核心点 $k$ 初始点(种子点)，探索所有密度相连的点，保存到 P，Q 为尚未探测的点集合

&emsp; **while** $P \ne \emptyset$ **do**

&emsp;&emsp; $p:=P(1)$，$N_p:=N(p)$，$P:=P\setminus \{p\}$

&emsp;&emsp; **if** $|N_p| \ge m_0$ **do** # 当前点 $p$ 是核心点

&emsp;&emsp;&emsp; $\Lambda=N_p \cap Q$ # 获取 $p$ 与之密度直达的且尚未探测的点

&emsp;&emsp;&emsp; $P:=P\cup \Lambda$   # $\Lambda$ 内的点肯定与 $k$ 密度相连，$k$ 到这些点密度可达

&emsp;&emsp;&emsp; $Q:=Q \setminus \Lambda$ # $\Lambda$ 为已探测点集合，从尚未探测集合中将它们除去

&emsp;&emsp; **end if**

&emsp; **end while** # 从 $k$ 已经探索出所有密度相连的点，与 $k$ 不密度相连的点则在 Q 中

&emsp; $C:=C \cup \{\mathcal T \setminus Q\}$ # 获取所有与 $k$ 密度相连的点，形成一个聚类簇

&emsp; $K:=K \setminus (T \setminus Q)$ # 从核心点集合 $K$ 中去掉当前聚类簇 $\mathcal T \setminus Q$ 中的核心点

&emsp; $\mathcal T:= Q$ # 更新尚未探测的点集

**end while**
___

1. 任取一个数据 $p \in \mathcal D$
2. 若 $p$ 为核心点，那么找出所有从 $p$ 密度可达的数据点，形成一个簇
3. 若 $p$ 非核心点，那么重新取一个数据点
4. 重复 `2,3` 步骤，直到所有点被处理。

---

DBSCAN 算法优点：

1. 可以对任意形状的稠密数据集进行聚类（例如狭长分布的数据集）
2. 聚类后可检测异常点，对异常点不敏感
3. 不需要设置聚类簇数量，以及初始的聚类中心点

缺点：

1. 需要设置超参数聚类半径 $\epsilon$ 和核心点邻域内点数量阈值 $m_0$。显然当样本密度不均匀，$\epsilon,\ m_0$ 不容易选择。可以通过对这两个参数进行调节，以获得较好的聚类结果。
2. 聚类时间较长。