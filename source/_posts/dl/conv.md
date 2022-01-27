---
title: Convolution
date: 2021-02-19 09:27:01
tags: CNN, deep learning
p: dl/conv
mathjax: true
---
# 膨胀卷积
膨胀卷积在卷积核中引入 空洞（holes），将卷积核变大，记膨胀率为 $\alpha$，卷积核大小为 $k$，那么膨胀后卷积核大小变为 $\alpha(k-1)+1$，使用膨胀后的卷积核来做卷积计算。

膨胀卷积在图像（实例）分割中应用较多，为了扩大感知区域，同时减少计算量，膨胀卷积效果较好。

Dilated Convolution 的设计是为了获取 long-range information，故对大物体比较适用，对小物体则不太适用。Dilated Convolution 一个明显的缺点是 kernel 不连续，产生栅格效应，所以又提出了 Hybrid Dilated Convolution（HDC）混合膨胀卷积。

HDC 的一般设计原则：
1. 各膨胀卷积的膨胀率不能有大于 1 的公约数（例如 [2,4,6] 公约数为 2），否则会有栅格效应
2. 膨胀率设计为锯齿状结构，例如 [1,2,5,1,2,5] 这样的循环结构
3. 膨胀率满足如下关系
$$M_i=\max[M_{i+1}-2r_i, 2r_i-M_{i+1}, r_i]$$
其中 $r_i$ 为第 `i` 层的膨胀率，$M_i$ 为第 `i` 层的最大 dilated rate，网络总共 `L` 层，$M_L=r_L$。


# 分组卷积
假设输入 feature shape 为 $(c_0,h,w)$，original filter 为 $(k,k,c_0,c_1)$，输出 feature shape 为 $(c_1,h,w)$。对于分组卷积，假设分 n 组，那么每一组输入 feature shape 为 $(c_0/n, h, w)$，每一组使用独立的卷积核， filter shape 为 $(k,k,c_0/n, c_1/n)$，于是每一组的输出 feature shape 为 $(c_1/n, h, w)$，最后所有组的输出沿着 channel 进行 concatenate，得到最终输出 feature shape $(c_1, h, w)$，这个过程中，卷积核参数数量为
$$k \times k \times \frac {c_0} n \times \frac {c_1} n \times n$$
参数数量减小。

# Bottleneck
假设输入 shape 为 $(c_0, h, w)$，输出 shape 为 $(c_1, h, w)$，那么 filter 为 $k \times k \times c_0 \times c_1$，参数数量较大，改用 bottleneck 可以缩减参数数量，即：先使用 $1\times 1 \times c_0 \times c_2$ 的 filter，然后使用 $k \times k \times c_2 \times c_2$ 的 filter，最后使用 $1 \times 1 \times c_2 \times c_1$ 的 filter，其中 $c_2 < c_1, c_0$。

# Depthwise Conv
假设输入 shape 为 $(c_0, h, w)$，每个 channel 独立进行（二维卷积），卷积 filter 为 $k \times k \times c_0$（注意这里 filter shape 中没有 $c_1$），得到 $(c_0, h, w)$ 的中间输出，然后再使用 $1 \times 1 \times c_0 \times c_1$，得到 $(c_1, h, w)$ 的最终输出。

# 可变形卷积
略（参考 [deformable conv](/obj_det/two_stage)）