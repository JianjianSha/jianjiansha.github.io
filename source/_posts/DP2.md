---
title: Dynamic Programming (2)
date: 2019-08-14 14:36:05
tags: math, DP
mathjax: true
---
上一篇文章 [Dynamic Programming (1)](2019/08/07/DP1) 介绍了动态规划的原理，以及几种常见模型的 DPFE。这篇文章则主要介绍一些实际应用。

## 最佳分配问题 ALLOT
最佳分配问题简称为 ALLOT，描述了如何讲有限资源分配给一些用户，损失（或者利益，损失对应最小化，利益对应最大化）与用户以及分配到的资源量有关。ALLOT 也可以看作是背包问题 KSINT 的一个变种。

假设有 M 单位的资源要分配给 N 个用户，记 C(k,d) 表示分配 d 单位资源给用户 k 时的损失，分配决策按阶段进行，即第一阶段分配 $d_1$ 单位资源给用户 1，第二阶段分配 $d_2$ 单位资源给用户 2，依次进行，定义状态 (k,m) 为阶段 k 时剩余 m 单位资源，阶段 k 分配之前的资源量为 m，阶段 k 的分配损失为 C(k,d)，下一阶段状态为 (k+1,m-d)，于是根据 [Dynamic Programming (1)](2019/08/07/DP1) 中式 (1.19) 可知 DPFE 为
$$f(k,m)=\min_{d \in \{0,...,m\}} \{C(k,d)+f(k+1,m-d)\} \quad (2.1)$$
目标是求 f(1,M)，基本条件为 f(N+1,m)=0，其中 $m \ge 0$，这表示资源可以不用全部瓜分完。

现在假设有 M=4，N=3，且
$$(C_{k,d})_{k\in \{1,2,3\};d\in \{0,...,4\}}=\begin{pmatrix}\infty & 1.0 & 0.8& 0.4 & 0.0 \\\\ \infty & 1.0& 0.5 & 0.0 & 0.0 \\\\ \infty & 1.0 & 0.6 & 0.3 & 0.0 \end{pmatrix}$$
那么，f(1,M)=1.0+0.5+1.0=2.5，最佳分配序列为 $d_1=1,d_2=2,d_3=1$。我们可以根据式 (2.1) 逐步展开计算，下面是代码实现，
```python
def allot(cache=True):
    M=4
    N=3
    max_float=1e8
    cost=[[max_float, 1.0, 0.8, 0.4, 0.0],
          [max_float, 1.0, 0.5, 0.0, 0.0],
          [max_float, 1.0, 0.6, 0.3, 0.0]]
    
    if cache:
        cache_dict = {}
    def allot_inner(k,m):
        if cache and (k,m) in cache_dict:
            return cache_dict[(k,m)]
        if k>= N: return [], 0

        min_f=max_float
        min_ds = []
        for d in range(m+1):
            ds,f=allot_inner(k+1,m-d)
            temp=cost[k][d]+f
            if min_f > temp:
                min_f = temp
                min_ds = [d]+ds
        if cache and k > 1:
            cache_dict[(k,m)]=(min_ds,min_f)
        return min_ds, min_f
    ds, f=allot_inner(0,M)
    print("min cost:",f,"opt allotments:", ds)
```



## 所有结点对的最短路径问题 APSP
在图中寻找从任一起点 s 到任一终点 t 之间的最短路径，记图中结点数量为 N，为简单起见，我们假设任意结点对之间没有非正长度的环，并且没有自环（self-loop）。

可以利用 [Dynamic Programming (1)](2019/08/07/DP1) 中的最短路径模型，将 (s,t) 看作变量，求得一系列的最短路径值，然后再求最小值即可，但是这其中肯定存在一些重复计算，所以这里我们讨论更高效率的计算方法。

__Relaxation__ 

定义 F(k,p,q) 为从 p 到 q 之间的最短路径长度，k 表示从 p 到 q 最多走 k 步（从一个节点到下一个节点为一步）。借助 [Dynamic Programming (1)](2019/08/07/DP1) 中式 (1.27)，DPFE 为
$$F(k,p,q)=\min \{F(k-1,p,q), \min_{r \in succ(p)} \{b(p,r)+F(k-1,r,q)\}\} \quad(2.2)$$
其中 r 是 p 的直接后继节点。基本条件为:
1. $F(k,p,q)=0, k \ge 0, p=q$，表示当 p 就是 q 时，最多走 k 步，最短路径长度为 0。
2. $F(0,p,q)=\infty, p \ne q$， 表示当 p 不为 q 时，最多走 0 步，最短路径长度为无穷大。

虽然我们假定没有自环，但是我们依然可以令 $b(p,p)=0$（实际路径中我们可以去掉环即可），那么式 (2.2) 可简化为
$$\begin{aligned}F(k,p,q)&=\min \{F(k-1,p,q)+b(p,p), \min_{r \in succ(p)} \{b(p,r)+F(k-1,r,q)\}\} \\\\ &=\min_{r \in succ(p)\cup \{p\}} \{b(p,r)+F(k-1,r,q)\} \qquad(2.3) \end{aligned}$$


__Floyd-Warshall__ 

式 (2.2) 这个 DPFE 是一种分而治之的思想：从 p 到 q 最多走 k 步的路径，可以分为从 p 走一步到 r 以及从 r 最多走 k-1 步到 q 两个子路径。还有一种替代方案是从 p 到 r 并且从 r 到 q，其中 p 到 r 的步数不再固定为 1，但是从 p 出发，到达 q 总共最多经过 k 个点，r 就是这 k 个中间点，不妨记这 k 个点为 $\{1,2,...,k\}$，那么求从 p 到 q 并使用 $\{1,2,...,N\}$ 作为可能的中间点的最短路径就是 p 到 q 的全局最短路径。DPFE 为
$$F(k,p,q)=\min \{F(k-1,p,q), F(k-1,p,k)+F(k-1,k,q)\} \qquad(2.4)$$

为了便于理解，我们作如下说明：
1. 将 N 个节点编号为 $V=\{1,2,...,N\}$，$p,q \in V$
2. $F(k,p,q)$ 表示从 p 到 q 且以 $\{1,2,...,k\}$ 作为可能的中间节点
3. 问题的求解目标为 $F(N,p,q)$
4. 如何理解式 $(2.4)$？
   - p 到 q 的路径不经过中间点 k，即，使用 $\{1,2,...,k-1\}$ 作为可能的中间节点
   - 或者 p 到 q 的路径经过中间点 k，即，分为两个子路径 p 到 k 和 k 到 q，这两个子路径均使用 $\{1,2,...,k-1\}$ 作为可能的中间节点
5. 式 $(2.4)$ 这个递归操作需要条件 k>0。k=0 时为基本条件 $F(0,p,q)=0, p=q$，以及 $F(0,p,q)=b(p,q), p\ne q$。前者表示当 p q 为同一节点时，不使用任何中间节点，损失为 0；后者表示当 p q 不同时，不使用任何中间节点，损失为 $b(p,q)$，需要注意这里 q 是 p 的直接后继，如果不是，那么有 $F(0,p,q)=\infty, p \notin succ(p) \cup \{p\}$
6. 中间点序列 $\{1,2,...,k\}$ 可能会包含 p 和 q，如果包含了的话，由于我们假定所有的环都是正的，所以再求序列最小值的，带环的路径均会被过滤掉，而自环 $b(p,p)=0$，不会影响最短路径的长度，如果路径中出现自环，去掉即可（去掉连续重复的节点，只保留一个）。

式 (2.2) 中 r 是 p 的后继节点，最多可取 $N-1$ 个节点（假设图中其他节点均为 p 的后继节点，就对应 $N-1$），k 最大为 $N-1$ 步，p 和 q 均各有 N 个取值，所以式 (2.2) 的时间复杂度为 $O(N^4)$，类似地，式 (2.4) 的时间复杂度为 $O(N^3)$。

实际中要解决 APSP 问题，可以根据式 (2.2) 求出矩阵序列 $\{F^{(1)},F^{(2)},...,F^{(N-1)}\}$，任一矩阵 $F^{(k)}$ 维度为 $N \times N$，$F_{p,q}^{k}$ 表示从 p 到 q 最多走 k 步的最短路径长度，然后求 $\min_{p,q} F_{p,q}^{(N-1)}$ 就是 APSP 的解。

__矩阵乘法__

为了借鉴矩阵乘法的思想，我们首先将式 (2.2) 作变换，
$$\begin{aligned} F(k,p,q)&=\min \{F(k-1,p,q), \min_{r \in succ(p)} \{b(p,r)+F(k-1,r,q)\}\}
\\\\ &= \min_{r \in succ(p)} \{b(p,p)+F(k-1,p,q), b(p,r)+F(k-1,r,q)\}
\\\\ &= \min_{r \in succ(p) \cup \{p\}} \{b(p,r)+F(k-1,r,q)\}
\\\\ &= \min_{r \in \{1,2,...,N\}} \{b(p,r)+F(k-1,r,q)\} \qquad(2.5) \end{aligned}$$
其中，$b(p,r)$ 是事先给定的任意两节点之间的距离，若两节点之间没有边 edge 相连，则距离为 $\infty$，这里称所有节点对之间的距离组成的矩阵为权重矩阵 $W_{N \times N}$。根据式 (2.2) 的基本条件，不难得知 $F^{(0)}$ 矩阵对角线全 0，其余元素均为 $\infty$：

$$F^{(0)}=\begin{bmatrix}0 & \infty & \cdots & \infty
\\\\                    \infty & 0  & \cdots & \infty
\\\\                    \vdots & \vdots & \ddots & \vdots
\\\\                    \infty & \infty & \cdots & 0 \end{bmatrix}_{N \times N}$$

$$W=\begin{bmatrix}0 & w_{12} & \cdots & w_{1N}
\\\\                    w_{21} & 0  & \cdots & w_{2N}
\\\\                    \vdots & \vdots & \ddots & \vdots
\\\\                    w_{N1} & w_{N2} & \cdots & 0 \end{bmatrix}_{N \times N}$$

根据式 (2.5)，已知 $F^{(k-1)}$ 求 $F^{(k)}$ 的代码为
```python
import sys
F_k=[[None]*N]*N
for p in range(0,N):
  for q in range(0,N):
    F_k[p][q]=sys.info.float_max
    # F_k[p][q]=0
    for r in range(0,N):
      F_k[p][q]=min(F_k[p][q],W[p][r]+F_k_1[r][q])
      # F_k[p][q]=F_k[p][q]+W[p][r]*F_k_1[r][q])
```
从上面代码片段可见，与矩阵乘法（注释部分）完全一个模样，而我们的目的是为了计算 $F^{(N-1)}$，中间的其他 $F^{(k)}$ 矩阵如无必要，可以不用计算出来，比如下面，
$$\begin{aligned} F^{(1)}&=W \circ F^{(0)}=W
\\\\ F^{(2)}&=W \circ F^{(1)}=W^2
\\\\ F^{(3)}&=W \circ F^{(2)}=W^3
\\\\ &\vdots
\\\\ F^{(N-1)}&=W \circ F^{(N-2)}=W^{(N-1)} \end{aligned} \quad(2.6)$$

$\circ$ 表示某种运算符，比如矩阵乘法或者这里的最小值计算，我们改为如下序列计算，
$$\begin{aligned} F^{(1)}&=W
\\\\ F^{(2)}&=W^2=W \circ W
\\\\ F^{(4)}&=W^4=W^2 \circ W^2
\\\\ &\vdots
\\\\ F^{2^{\lceil log(N-1) \rceil}}&=W^{2^{\lceil log(N-1) \rceil}} =W^{2^{\lceil log(N-1) \rceil-1}} \circ W^{2^{\lceil log(N-1) \rceil-1}} \end{aligned} \quad(2.7)$$
注意上面 $2^{\lceil log(N-1) \rceil}$ 中的向上取整 $\lceil \cdot \rceil$ 很重要，这保证了 $2^{\lceil log(N-1) \rceil} \ge N-1$，从而 $F^{2^{\lceil log(N-1) \rceil}} \le F^{(N-1)}$ （element-wise comparison）。

因为结合顺序无关紧要，才使得我们可以从式 (2.6) 可以改写为式 (2.7)，例如
$$F^{(4)}=W \circ F^{(3)}=W \circ (W \circ F^{(2)})=\cdots =W \circ(W \circ (W \circ W)) \stackrel{*}=(W \circ W) \circ (W \circ W)=W^2 \circ W^2$$
将 $\circ$ 替换为 $\min$，即 $\min (W, \min(W, \min(W,W)))=\min(\min(W,W), \min(W,W))$，注意这里的 $\min$ 不是 element-wise operator，就跟矩阵乘法不是矩阵点乘一样。当然，以上内容只是帮助理解，不是式 (2.6) 可以变换为式 (2.7) 的严格证明。

好了，有了式 (2.7) 就可以更快的计算出 $F^{(M)}, M \ge N-1$，由于 $F^{(k)}$ 单调减，并收敛于 $F^{(N-1)}$，于是 $F^{(M)}$ 就是全局最短路径长度矩阵。

使用矩阵乘法加速的算法代码为
```python
import sys

def fast_apsp():
  k=1
  F_prev=W
  while k<N-1:
    F_next=[[sys.info.float_max]*N]*N
    for p in range(0,N):
      for q in range(0,N):
        for r in range(0,N):
          F_next[p][q]=min(F_next[p][q], F_prev[p][r]+F_prev[r][q])
    F_prev=F_next
    k*=2
  return F_prev
```

__Floyd-Warshall__ 的代码

考虑式 (2.4)，注意 $F^{(k)}$ 中的 k 表示路径可以经过中间节点 $\{1,2,...,k\}$，所以 $F^{(0)}$ 表示不经过任何中间节点的两点之间最短路径长度矩阵，所以根据基本条件不难得到
$$F^{(0)}=W=\begin{bmatrix}0 & w_{12} & \cdots & w_{1N}
\\\\                    w_{21} & 0  & \cdots & w_{2N}
\\\\                    \vdots & \vdots & \ddots & \vdots
\\\\                    w_{N1} & w_{N2} & \cdots & 0 \end{bmatrix}_{N \times N}$$

根据式 (2.4)，不难写出原始的 __Floyd-Warshall__ 算法的代码为
```python
F_prev=F_0
def floyd_warshall():
  for k in range(0,N):
    F_next=[[None]*N]*N
    for p in range(0,N):
      for q in range(0,N):
        F_next[p][q]=min(F_prev[p][q], F_prev[p][k]+F_prev[k][q])
    F_prev=F_next
  return F_prev
```


