---
title: Receptive Field
date: 2021-02-18 10:51:00
p: dl/receptive_field
tags: deep learning, CNN
mathjax: true
---

# Size

对于一个 fully CNN 的网络，第 `k` layer 的 receptive field 大小为，
$$l_k=l_{k-1} + ((f_k-1)*\prod_{i=1}^{k-1} s_i)$$
其中，$l_{k-1}$ 表示第 `k-1` layer 上的 receptive field 大小，$f_k$ 是第 `k` layer 的 filter 大小，$s_i$ 是第 `i` layer 上的 stride 大小。这是自底向上计算，从 $l_1$ 开始，$l_1=f_1$。



还有一种自顶向下的计算方法。假设总共有 `L` 个 layer，每个 layer 的输出 feature map 记为 $f_l, \ l=1,...,L$，每个 layer 的 filter 大小为 $k_l$，stride 大小记为 $s_l$，记 $r_l$ 为最后一个 layer 关于 feature map $f_l$ 的 receptive field 大小，也就是说，$r_l$ 表示 $f_l$ 上多少个像素点对 $f_L$ 的一个像素点有贡献（这里仅考虑一维 feature map，如果是多维，那么分别独立考虑即可）。那么易知，$r_L=1$，

$r_{L-1}=k_L$，这个也很好理解，上一层 feature map 中，$k_L$ 个像素点对应本层（最后一层）一个像素点。考虑一般情况，已知 $r_l$，求 $r_{l-1}$。

首先假设 $k_l=1$，这样情况就简单些，若 $s_l=1$，那么 $r_{l-1}=r_l$，若 $s_l>1$，那么 $r_{l-1}=s_l \cdot r_l -(s_l-1)$，因为 $r_l$ 中每两个像素点之间对应到 $f_{l-1}$ 上有 $s_l-1$ 个点，所以 $r_{l-1}=(s_l-1)\cdot(r_l-1)+ r_l=s_l \cdot r_l-s_l+1$。

然后当 $k_l>1$，那么需要在 $f_{l-1}$ 上增加 $k_l-1$ 个像素点，于是
$$r_{l-1}=s_l \cdot r_l + (k_l-s_l)$$
其中，$r_L=1, \ r_{L-1}=k_L$。求解上式过程如下：
$$r_{L-2}=s_{L-1} r_{L-1}+(k_{L-1}-s_{L-1})=s_{L-1}(k_L-1)+k_{L-1}$$
$$r_{L-3}=s_{L-2} r_{L-2}+(k_{L-2}-s_{L-2})=s_{L-2}s_{L-1}(k_L-1)+s_{L-2}(k_{L-1}-1)+k_{L-2}$$
$$\cdots$$
$$r_{l}=s_{l+1}\cdots s_{L-1}(k_L-1)+s_{l+1}\cdots s_{L-2}(k_{L-1}-1)+ \cdots s_{l+1}(k_{l+2}-1)+k_{l+1}=1+\sum_{j=l+1}^{L} \left[(k_{j}-1) \prod_{i=l+1}^{j-1}s_i \right]$$
其中令 $$\prod_{l+1}^{l}s_i=1$$

于是，
$$\begin{aligned} r_{l-1}&=1+\sum_{j=l}^{L} \left[(k_{j}-1) \prod_{i=l}^{j-1}s_i \right] \\ &=1+(k_l-1)+\sum_{j=l+1}^{L} \left[(k_{j}-1) \prod_{i=l+1}^{j-1}s_i \cdot s_l \right] \\&=k_l-s_l+s_l \left(1+\sum_{j=l+1}^{L} \left[(k_{j}-1) \prod_{i=l+1}^{j-1}s_i \right] \right) \\&=s_l \cdot r_l +k_l-s_l\end{aligned}$$
与前面递推式一致，说明通项式计算正确。

output feature size 的计算为，
$$w_l=\frac {w_{l-1}+2p_l-k_l} {s_l}+1$$
其中 $w$ 表示宽，高 $h$ 的计算类似（以 2D image 数据为例）。

# Region
对输出 feature map 上一点有贡献的 region （Receptive Field）大小计算如上，还有一个参数也很重要：定位这个 region 的位置。例如输出 feature map 上一点 $f_L(i,j)$，产生这个特征的输入图像上的 region 位置如何求得。

记在特征平面 $f_l$ 上这个 region 的左端和右端的坐标分别为 $u_l, \ v_l$，这里的<b>坐标从 0 开始</b>，即，第一个像素点的坐标为 `0`，在输出特征平面 $f_L$ 上有 $u_L=v_L=i$，同样地，仅考虑一维情况，对于二维情况，另一维度独立地进行类似计算可得。

同样使用递推的思想，已知 $u_l, \ v_l$，求 $u_{l-1}, v_{l-1}$。

首先从一个简单的情况开始，假设 $u_l=0$，这表示 $f_l$ 中的 region 左侧位于第一个像素点，此时 $u_{l-1}=-p_l$，即$f_{l-1}$ 左侧填充 $p_l$ 个像素；如果 $u_l=1$，那么 $u_{l-1}=s_l-p_l$，这也很好理解，从 $f_{l-1}$ 最左侧第一个像素点（填充之后为 $-p_l$）向右移动 $s_l$；如果 $u_l=2$，那么继续向右移动 $s_l$，即 $u_{l-1}=2s_l-p_l$，于是一般地，
$$u_{l-1}=u_l \cdot s_l -p_l$$
$$v_{l-1}=v_l \cdot s_l - p_l + k_l-1$$
完全式的计算过程如下：
$$u_{L-1}=u_L \cdot s_L - p_L$$
$$u_{L-2}=u_{L-1} \cdot s_{L-1}-p_{L-1}=s_{L-1}s_L u_L-s_{L-1}p_L-p_{L-1}$$
$$u_{L-3}=u_{L-2} \cdot s_{L-2}-p_{L-2}=s_{L-2}s_{L-1}s_L u_L-s_{L-2}s_{L-1}p_L-s_{L-2}p_{L-1}-p_{L-2}$$
$$\cdots$$
$$u_l=s_{l+1}\cdots s_L u_L-s_{l+1}\cdots s_{L-1} p_{L}-\cdots-s_{l+1} p_{l+2}-p_{l+1}=u_L\prod_{i=l+1}^L s_i-\sum_{j=l+1}^L p_j \prod_{i=l+1}^{j-1} s_i$$

其中，$\prod_{i=l+1}^l s_i=1$, 类似地，
$$v_l=v_L \prod_{i=l+1}^L s_i - \sum_{j=l+1}^L(1+p_j-k_j)\prod_{i=l+1}^{j-1} s_i$$

# Relation
Receptive Field size 与 region 之间的联系，
$$r_l=v_l-u_l+1$$

# Stride & Padding
定义两个变量，有效 stride 和 有效 padding，这两者分别定义如下：

$$S_l=\prod_{i=l+1}^L s_i$$

$$P_l=\sum_{j=l+1}^L p_j \prod_{i=l+1}^{j-1}s_i$$

他们的递推公式为，
$$S_{l-1}=s_l \cdot S_l$$
$$P_{l-1}=p_l+s_l \cdot P_l$$

有着这两个定义变量，region 位置公式可表示为，
$$u_l=u_L \cdot S_l - P_l$$

# Center
receptive field 的中心可由 region 位置计算得到，在第 `l` layer 上为，
$$c_l=\frac {u_l+v_l} 2$$

