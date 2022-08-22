---
title: PyTorch 方法总结
p: pytorch/PyTorch-mtd
date: 2019-11-01 11:26:25
tags: PyTorch
mathjax: true
---

# 1. Fold / Unfold
## Fold
这是 torch.nn.Fold 类。

首先我们来复习一下卷积过程，设输入 size 为 `(N,C,H,W)`，卷积 kernel size 为 `(C,h,w)`，碰撞系数为 `d`，padding 为 `p`，stride 记为 `s`，那么整个过程相当于将 `Cxhxw` 大小的数据块在输入数据中滑动，每一次滑动做一次卷积，记共有 `L` 次卷积，即，从输入数据中切分出 `L` 个数据块与卷积核做卷积，当然每个数据块的大小与卷积核相同，为 `(C,h,w)`，最后得到的输出 map 大小为
<!-- more -->
$$H_o= \frac {H - [d(h-1)+1] + 2p} {s}+1
\\\\ W_o= \frac {W - [d(w-1)+1] + 2p} {s}+1$$
因为每一次卷积得到的值均作为输出 map 上的一点，故 `L` 为
$$L=H_o * W_o=\left(\frac {H - [d(h-1)+1] + 2p} {s}+1\right) \left(\frac {W - [d(w-1)+1] + 2p} {s}+1\right)$$

好了，现在 Fold 要做的事情是反过来的，已知 fold 的输入为 `L` 个数据块，大小为 `(N,C*h*w,L)`，有关的构造参数为卷积核 size `(h,w)`，dilation，padding，stride，以及，指定最终的（Fold）输出大小 `(H,W)`，注意，Fold 做的事情是反过来的，也就是说，从 `L` 个数据块中恢复出原来普通卷积的输入 map 的大小，即 `(H,W)`，不是做完卷积之后的输出 map 的大小，记住，__Fold 的输出是普通卷积的输入__。

Fold 的这些构造参数指明了卷积核大小，以及卷积输入的大小，然后根据其（这里指 Fold）输入 `L` 个数据块的 tensor，size 为 `(N,C*h*w,L)`，恢复出卷积输入的 tensor，因为 Fold 的构造参数中指定了 卷据输入 map 的 `(H,W)`，而批大小 `N` 也已知，所以要求出通道 `C`，根据 Fold 输入 tensor 的 第二个维度值 `C*h*w` 以及 Fold 的构造参数中卷积核大小 `(h,w)` 很容易得到通道 `C`。

先使用 PyTorch 文档中的例子加以说明，
```python
>>> fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2))  # (H,W)=(4,5), (h,w)=(2,2)
>>> input = torch.randn(1, 3 * 2 * 2, 12)   # (N, C*h*w,L)
>>> output = fold(input)
>>> output.size()
torch.Size([1, 3, 4, 5])    # (N,C,H,W)
```

将数据维度从 `H,W` 扩展到更多维度，就是 PyTorch 文档中关于 `L` 的计算式了，如下
$$L=\prod_d \lfloor \frac {\text{output\_size} [d] + 2 \times \text{padding}[d]-\text{dilation}[d] \times (\text{kernel\_size}[d]-1) -1} {\text{stride}[d]} +1 \rfloor$$

__总结：__

Fold 的输入 size 为 $(N, C \times \prod(\text{kernel\_size}), L)$，输出 size 为 $(N,C, \text{output\_size}[0], \text{output\_size}[1], ...)$

## Unfold
这是 torch.nn.Unfold 类，所做的事情与 Fold 相反，根据普通卷积的输入 tensor 以及卷积核大小，dilation，padding 和 stride 等计算得到 `L` 个与卷积核做卷积操作的数据块。`L` 计算方式如上。Unfold 的输入 size 为 $(N,C,*)$，其中 * 表示多维数据，输出 size 为 $(N,C \times \prod(\text{kernel\_size}), L)$。

引用PyTorch 文档中的例子，
```python
>>> unfold = nn.Unfold(kernel_size=(2, 3))
>>> input = torch.randn(2, 5, 3, 4)
>>> output = unfold(input)
>>> # each patch contains 30 values (2x3=6 vectors, each of 5 channels)
>>> # 4 blocks (2x3 kernels) in total in the 3x4 input
>>> output.size()
torch.Size([2, 30, 4])
```

# 2. Normalization
## BatchNorm
批归一化是针对一个 mini-batch 内的数据进行归一化。首先给出归一化公式：
$$y=\frac {x-E[x]} {\sqrt{V[x]+\epsilon}} * \gamma + \beta$$

批归一化过程为：
BatchNorm Layer 的输入（mini-batch）为 $\mathcal B=\{x_{1...m}\}$，可学习参数为 $\gamma, \beta$。计算 mini-batch 的均值，方差
$$\mu_{\mathcal B} = \frac 1 m \sum_{i=1}^m x_i, \quad \sigma_{\mathcal B}^2=\frac 1 m \sum_{i=1}^m(x_i - \mu_{\mathcal B})^2$$
然后计算归一化后的值
$$\hat x_i = \frac {x_i - \mu_{\mathcal B}} {\sqrt {\sigma_{\mathcal B}^2+ \epsilon}}$$
最后进行 scale 和 shift，
$$y_i=\hat x_i \cdot \gamma + \beta$$

__小结：__ 沿着 batch 方向进行归一化

## LayerNorm
Layer 归一化是针对某个数据（样本）内部进行归一化，假设某个数据样本到达 LayerNorm 层为 $x$，无论 $x$ 是多少维的 tensor，均可以看作是 1D vector，即 $x=(x_1,...x_H)$，$H$ 是 LayerNorm 层的单元数（也是 $x$ 的特征数），于是 LayerNorm 过程为
$$\mu=\frac 1 H \sum_{i=1}^H x_i, \quad \sigma^2=\frac 1 H \sum_{i=1}^H (x_i-\mu)^2$$
于是 LayerNorm 后的值为
$$y=\frac {x-\mu} {\sqrt {\sigma^2+\epsilon}} \cdot \gamma + \beta$$

__小结：__ 

1. 沿着特征方向进行归一化（特征包含了除 batch 维度外的其他所有维度）
2. 输入 $(B, C, H, W)$，那么将单个样本 $(C, H, W)$ 做归一化

有了前面的归一化介绍，我们知道归一化过程都很类似，区别在于如何计算 $\mu, \sigma$，或者说沿着什么方向进行归一化。

## InstanceNorm
对于每个样例的每个 channel 分别计算 $\mu, \sigma$。假设输入为 $(B,C,H,W)$，那么沿着 $(H,W)$ 方向做归一化。

## GroupNorm
GroupNorm 是选择一组 channels 进行归一化，所以是介于 InstanceNorm（单个channel）和 LayerNorm （全部 channels）之间的。

输入 $(B, C, H, W)$，其中 $C=g \cdot c$，$g$ 为分组数量，那么对 $(c, H, W)$ 做归一化。

# 3. Pool
池化操作都比较简单易懂，这里介绍几个非常规的池化操作。
## FractionalMaxPool2d
引用论文 [Fractional MaxPooling](https://arxiv.org/abs/1412.6071)。

pool 操作通常是用于降低 feature map 的大小，以常规的 `2x2` max-pooling 为例，记输入大小为 $N_{in} \times N_{in}$，输出大小为 $N_{out} \times N_{out}$，那么有
$$N_{out}=\frac {N_{in}-k+2p} {s} + 1= N_{in} / 2 \Rightarrow N_{in} /N_{out} = 2$$

将 $N_{in} \times N_{in}$ 的 feature map 划分出 $N_{out}^2$ 个 pooling 区域 $(P_{i,j})$。我们用 $\{1,2,...,N_{in}\}^2$ （或 $[1,N_{in}]^2$）表示输入 feature map，pixel 使用坐标点表示，显然 pooling 区域满足
$$P_{i,j} \subset \{1,2,...,N_{in}\}, \quad (i,j) \in \{1,...,N_{out}\}^2$$

现在，我们想让 $N_{in} / N_{out} \in (1,2)$，或者为了提高速度，让 $N_{in} / N_{out} \in (2,3)$，反正，这个比例不再是整数，这就是 Fractional max-pooling（FMP）。

那么，FMP 具体是如何实现的呢？

令两个递增序列 $(a_i)_{i=0}^{N_{out}}, \ (b_i)_{i=0}^{N_{out}}$ 均以 `1` 开始，$N_{in}$ 结尾，递增量为 `1` 或者 `2`，即 $\forall i,\ a_{i+1}-a_{i} \in \{1,2\}$，那么 pooling 区域可以有如下两种表示：
$$P_{i,j}=[a_{i-1}, a_i-1] \times [b_{j-1},b_j-1], \quad i,j \in \{1,...,N_{out}\}
\\\\ P_{i,j}=[a_{i-1}, a_i] \times [b_{j-1},b_j], \quad i,j \in \{1,...,N_{out}\}$$
注意下标 `i,j` 的范围。

第一种是 `disjoint` 表示，第二种是 `overlapping` 表示。显然使用第二种表示，相邻两个 pooling 区域是有重叠的，而第一种表示则不会。

记下采样率 $\alpha = N_{in} / N_{out}$，有如下两种方法得到 $(a_i)_{i=0}^{N_{out}}$
1. `random` 方法
   
   当 $\alpha$ 给定，那么 $(a_i-a_{i-1})_{i=1}^{N_{out}}$ 这个序列中有多少个 `1` 和多少个 `2` 已经是确定的了，将适量的 `1` 和 `2` shuffle 或者 random permutation，然后可可到 $\alpha = N_{in} / N_{out}$

2. `pseudorandom` 方法
   
   经过 $(0,0), \ (N_{out}, N_{in}-1)$ 的直线，其斜率为 $\alpha$（实际上是比下采样率小一点点，但是没关系，这两个值只要同时位于 $(1,2)$ 之间即可），将这个直线沿 y 轴 平移 $\alpha \cdot u$，其中 $\alpha \in (1,2), \ u \in (0,1), \alpha \cdot u \in (0,1)$，即
   $$y=\alpha(i+u)$$
   在此直线上取点，x 值依次为 $0,1,2,...,N_{out}$，对 y 值在应用 ceiling 函数，作为 $a_i$ 的值，
   $$a_i=\text{ceiling}(\alpha(i+u)), \quad i=0,1,2,...,N_{out}$$
   验证一下 $\{a_i\}$ 序列是否满足上述条件：
   - $i=0$，$a_0=\text{ceiling}(\alpha \cdot u)$，由于 $\alpha \cdot u \in (0,1)$，故 $a_0=1$
   - $i=N_{out}$，$a_{N_{out}}=\text{ceiling}(N_{in}-1+\alpha \cdot u)=N_{in}$
   - `otherwise：` $a_{i+1}-a_i=\text{ceiling}(\alpha \cdot i+\alpha+\alpha \cdot u)-\text{ceiling}(\alpha \cdot i+\alpha \cdot u)$。验证说明如下。

   下面验证最后一种情况：
   
   记 $\alpha \cdot i+\alpha \cdot u=f \in [k,k+1)$，k 是某个整数，那么当
   - `f=k` 时，
  
        $a_{i+1}-a_i=\text{ceiling}(k+\alpha)-k=k+\text{ceiling}(\alpha)-k=\text{ceiling}(\alpha)=2$

   - `k<f<k+1` 时，

        $k+1<f+\alpha<k+3$
  
        $a_{i+1}-a_i=\text{ceiling}(f+\alpha)-k-1 \in \{1,2\}$

   至此，验证了 $(a_i)$ 序列满足条件。显然，基于直线取离散点然后应用 ceiling 函数得到的是一种伪随机序列。

# 4. ConvTranspose

## 输出大小
转置卷积，通常又称反卷积、逆卷积，然而转置卷积并非卷积的逆过程，并且转置卷积其实也是一种卷积，只不过与卷积相反的是，输出平面的大小通常不是变小而是变大。对于普通卷积，设输入平面边长 $L_{in}$，输出平面边长为 $L_{out}$，卷积核边长为 $k$，dilation 、stride 和 padding 分别为 $d, p, s$，那么有
$$L_{out}=\frac {L_{in}-(d(k-1)+1)+2p} s + 1 \tag {4-1}$$

其中 $d(k-1)+1$ 是膨胀之后的 kernel size。

对于转置卷积，令 $L_{in}^{\top}, \ L_{out}^{\top}$ 分别表示输入和输出的边长，于是有
$$L_{out}^{\top}=s(L_{in}^{\top} - 1) +d(k-1)+1 - 2p \tag{4-2}$$

可见，转置卷积的输入输出边长的关系与普通卷积是反过来的。

## 转置卷积计算
### 第一种方法
回顾一下卷积过程，以二维卷积为例，假设输入大小为 $4 \times 4$，卷积核 $3 \times 3$，不考虑 padding，且 stride 为 1，那么根据 $(4-1)$ 式输出大小为 $2 \times 2$，我们可以用卷积核在输入平面上滑窗并做卷积来理解卷积，实际计算则是根据输入矩阵得到 $4 \times 9$ 的矩阵（__部分 element 用 0 填充__），然后将卷积核展开成 $9 \times 1$ 的矩阵，然后进行卷积相乘得到 $4 \times 1$ 的输出矩阵。

我们再看转置卷积，输入大小为 $2 \times 2$，卷积核大小为 $3 \times 3$，同样地，不考虑 padding 且 stride 为 1，那么根据 $(4-2)$ 式输出大小为 $4 \times 4$，实际的计算过程为：__由于转置卷积也是一个普通卷积__，所以先将输入矩阵 zero-padding 为 $6\times 6$ 的矩阵（$6 \times 6$ 的输入矩阵经过 $3 \times 3$ 的卷积才能得到 $4 \times 4$ 的输出大小），然后与普通卷积一样地得到为 $16 \times 9$ 的矩阵，卷积核 __旋转 180°__，然后 reshape 为 $9 \times 1$ 的矩阵，通过矩阵乘法，得到矩阵大小为 $16 \times 1$，然后 reshape 为 $4 \times 4$，此即输出矩阵。

下面我们画图来展示卷积和转置卷积地过程：

![](/images/pytorch_mth_conv.png)

<center>普通卷积</center>

![](/images/pytorch_mtd_conv_t.png)

<center>转置卷积</center>


使用 Pytorch 进行验证：

```python
import torch
from torch import nn
convt = nn.ConvTranspose2d(1,1,3)
convt.bias = nn.Parameter(torch.tensor([0.]))
convt.weight = nn.Parameter(torch.tensor([[[[0.,1,2],
                                            [2,2,0],
                                            [0,1,2]]]]))
input = torch.tensor([[[[12.,12],
                        [10,17]]]])
output = convt(input)
output
# tensor([[[[ 0., 12., 36., 24.],
#           [24., 58., 61., 34.],
#           [20., 66., 70., 24.],
#           [ 0., 10., 37., 34.]]]])
```

### 第二种方法
还有另一种方法来理解计算卷积和转置卷积。还是以上面的例子进行说明。

普通卷积中，输入矩阵 reshape 为 $1 \times 16$。因为有 4 个滑窗卷积动作，所以将卷积核分别以四种不同的 zero-padding 方式得到 4 个 $4 \times 4$ 的矩阵（即，卷积核的 $3 \times 4$ 部分位于 $4 \times 4$ 矩阵的左上角、右上角，左下角和右下角，其他位置 zero-padding），然后 reshape 为 $16 \times 4$，记这个 $16 \times 4$ 的矩阵为 $K$， 得到 $1 \times 4$ 矩阵，reshape 为 $2 \times 2$ 即输出矩阵。

转置卷积中，输入矩阵大小为 $2 \times 2$（即 `[12,12,10,17]`），直接 reshape 为 $1 \times 4$，将上面的矩阵 $K$ __转置__，得到 $4 \times 16$ 的矩阵，然后矩阵相乘得到 $1 \times 16$ 矩阵，最后 reshape 为 $4 \times 4$ 即为输出矩阵。

普通卷积的过程如下图示意，转置卷积非常简单，读者可以自己画图验证。

![卷积的另一种计算过程](/images/pytorch_mtd_conv_t_1.png)<center>卷积的另一种计算过程</center>

从转置卷积得到的结果来看，很明显，转置卷积不是普通卷积的逆过程。

### dilation > 1
现在，我们的讨论还未结束，来看 `dilation` 不为 1 的情况，例如 `dilation=2`，还是使用上面的例子，对于转置卷积，此时根据 $(4-2)$ 式得到输出矩阵大小为 $6 \times 6$，将卷积核膨胀后得到 $5 \times 5$ 矩阵（间隔填充 0），并 __旋转 180°__，由于转置卷积也是一种普通卷积，所以应该将输入矩阵 zero-padding 到 $10 \times 10$ 大小才能得到 $6 \times 6$ 的输出，也就是说，输入矩阵上下左右均进行 4 个单位的 zero-padding，

记 `input` 为 $I$，zero-padding后，$I[4:6,4:6]=[[12.,12],[10,17]]$，其余位置为 `0`，膨胀后的卷积核 __旋转 180°__ 后为 $K'=[[2., 0, 1, 0, 0],[0,0,0,0,0],[0,0,2,0,2],[0,0,0,0,0],[2,0,1,0,0]]$，可以手动计算卷积后的输出矩阵，这里给出 python 代码计算示例，
```python
convt1 = nn.ConvTranspose2d(1,1,3,dilation=2)
convt1.bias = nn.Parameter(torch.tensor([0.]))
convt1.weight = nn.Parameter(torch.tensor([[[[0.,1,2],[2,2,0],[0,1,2]]]]))
output1 = convt1(input)
output1
```

### stride > 1
依然以上面的例子进行说明，假设现在 `stride=2`，根据式 $(4-2)$ 转置卷积的输出大小为 $5 \times 5$。把转置卷积看作一种普通卷积，那么其输入大小应该为 $7 \times 7$，由于 `stride=2`，所以先将 $2 \times 2$ 输入矩阵膨胀为 $3 \times 3$ 的矩阵（2*(2-1)+1=3），然后再 zero-padding 成 $7 \times 7$ 的矩阵（上下左右 padding 的数量均为 (7-3)/2=2），经过这番处理，输入矩阵变为 $I[2,2]=I[2,4]=12, \ I[4,2]=10, \ I[4,4]=17$，其余位置均为 `0`，卷积核 __旋转 180°__ 后为 $K'=[[2., 1, 0],[0,2,2],[2,1,0]]$，于是可以手动计算出卷积后的矩阵，这里给出 python 代码计算示例，
```python
convt.stride = (2,2)
output = convt(input)
output
```

### padding > 0
继续以上面的例子进行说明，假设现在 `padding=1`，根据式 $(4-2)$ 转置卷积的输出大小为 $2 \times 2$。将输入矩阵上下左右均进行 1 单位的 zero-padding，得到矩阵大小 $4 \times 4$，卷积核大小 $3 \times 3$，计算过程还是将卷积核 __旋转 180°__，卷积计算过程略，不过相信这是足够简单的事情。

以上 `dialtion > 1, stride > 1, padding > 0` 三种情况，除了可使用 python 程序验证，还可以使用 `第二种方法` 进行验证对输入矩阵以及卷积核的处理是正确的，并且，也可以使用 `第一种方法` 对输入矩阵和卷积核进行处理然后进行普通卷积计算得到输出矩阵。

# 5. Upsample
输入维度为 `minibatch x channels x [optional depth] x [optional height] x width`，即，输入可以是 3D/4D/5D。可用的算法包括 `nearest neighbor, linear, bilinear, bicubic, trilinear`。

## nearest neighbor
顾名思义，就是使用原平面上最近的一点作为上采样后的值。例如原平面 size 为 $m \times m$，在原平面上建立坐标系 S，上采样后的 size 为 $n \times n, \ n > m$，设其上点的坐标为 $(x,y), \ x,y =0,1,...,n-1$。将上采样后平面点映射到 S 中，对应坐标记为 $(x',y')$，那么有

$$\frac {x-0} {n-1-0}= \frac {x'-0}{m-1-0} \Rightarrow x' = \frac {m-1} {n-1} \cdot x$$
同理有 $y' = \frac {m-1} {n-1} \cdot y$，然后找出与点 $(i',j')$ 最近的那个整数坐标点，显然必然在以下四个点中产生 $(\lfloor x'\rfloor, \lfloor y' \rfloor), \ (\lfloor x'\rfloor, \lceil y' \rceil), \ (\lceil x'\rceil, \lfloor y' \rfloor), \ (\lceil x'\rceil, \lceil y' \rceil)$ （这四个点可能有重合），分别计算 $(x',y')$ 与这四个点的距离，距离最小的那个点的值即作为 $(x,y)$ 上采样后的值。（使用哪种距离指标，可以查看 PyTorch 底层实现代码，这里本人尚未去查看。）

## bilinear
输入必须是 4D。
### align_corners=True
双线性插值。记四个顶点为 $(x_1,y_1), \ (x_1,y_2), \ (x_2,y_1), \ (x_2,y_2)$，然后求目标点 $(x,y), \ x_1 \le x \le x_2, \ y_1 \le y \le y_2$ 的值。沿 x 轴线性插值，
$$f(x,y_1)=\frac {f_{21}-f_{11}} {x_2-x_1} \cdot (x-x_1)+f_{11}
\\\\ f(x,y_2)=\frac {f_{22}-f_{12}} {x_2-x_1} \cdot (x-x_1)+f_{12}
\\\\ f(x,y)=\frac {f_(x,y_2)-f(x,y_1)} {y_2-y_1} \cdot (y-y_1)+f(x,y_1)$$

与 `nearest neighbor` 中一样，首先将点 $(x,y)$ 映射到原平面上一点 $(x',y')$，然后四个顶点为 $(\lfloor x'\rfloor, \lfloor y' \rfloor), \ (\lfloor x'\rfloor, \lceil y' \rceil), \ (\lceil x'\rceil, \lfloor y' \rfloor), \ (\lceil x'\rceil, \lceil y' \rceil)$。用这种映射方法，显然原平面的四个 corners 和上采样后平面的四个 corners 分别对齐，这就是 `align_corners=True` 的由来。

### align_corners=False
如下图所示，显示了 `align_corners` 不同值的区别。
![](/images/pytorch_mtd_aligncorners.png)<center>图源 [pytorch 论坛](https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9)</center>

从图中可以发现，映射回原平面坐标时，坐标计算方式不同，例如上菜以后平面上一点 $(x,y)$，映射回 S 中的坐标为
$$x'=(x+0.5)/2-0.5
\\\\ y'=(y+0.5)/2-0.5$$

此后的插值方式一致（毕竟都是双线性插值），找到最近的 4 个点 $(\lfloor x'\rfloor, \lfloor y' \rfloor), \ (\lfloor x'\rfloor, \lceil y' \rceil), \ (\lceil x'\rceil, \lfloor y' \rfloor), \ (\lceil x'\rceil, \lceil y' \rceil)$ 进行双线性插值。
## linear
与 bilinear 类似，但是输入维度必须是 3D。

## trilinear
与 bilinear 类似，但是输入维度必须是 5D。

