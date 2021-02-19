---
title: Dilated Convolution
date: 2021-02-19 09:27:01
tags: CNN, Deep Learning
mathjax: true
---
膨胀卷积在卷积核中引入 空洞（holes），将卷积核变大，记膨胀率为 $\alpha$，卷积核大小为 $k$，那么膨胀后卷积核大小变为 $\alpha(k-1)+1$，使用膨胀后的卷积核来做卷积计算。

膨胀卷积在图像（实例）分割中应用较多，为了扩大感知区域，同时减少计算量，膨胀卷积效果较好。

Dilated Convolution 的设计是为了获取 long-range information，故对大物体比较适用，对小物体则不太适用。Dilated Convolution 一个明显的缺点是 kernel 不连续，产生栅格效应，所以又提出了 Hybrid Dilated Convolution（HDC）混合膨胀卷积。

HDC 的一般设计原则：
1. 各膨胀卷积的膨胀率不能有大于 1 的公约数（例如 [2,4,6] 公约数为 2），否则会有栅格效应
2. 膨胀率设计为锯齿状结构，例如 [1,2,5,1,2,5] 这样的循环结构
3. 膨胀率满足如下关系
$$M_i=\max[M_{i+1}-2r_i, 2r_i-M_{i+1}, r_i]$$
其中 $r_i$ 为第 `i` 层的膨胀率，$M_i$ 为第 `i` 层的最大 dilated rate，网络总共 `L` 层，$M_L=r_L$。


