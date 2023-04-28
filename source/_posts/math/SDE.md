---
title: 随机微分方程
date: 2022-07-25 15:24:44
tags: math
mathjax: true
---

# 1. ODE

ordinary differential equation 常微分方程为

$$\frac {dx(t)}{dt}=f(t,x) \tag{1}$$

注意 (1) 式导数函数不仅包含了自变量 $t$ ，还包含了因变量 $x$ 。

假设初始条件为 $x(t_0)=x_0$，那么有

$$x(t)=x_0 + \int_{t_0}^t f(s, x(s)) ds \tag{2}$$

例如

$$\frac {dx(t)}{dt}=a(t) x(t), \quad x(0)=x_0 \tag{3}$$

# 2. SDE

以 (3) 式为例，假设 $a(t)$ 不是一个确定性函数，而是一个随机函数，那么就得到一个随机微分方程。假设随机函数 $a(t)$ 形式为

$$a(t)=f(t) + h(t) \xi(t) \tag{4}$$

其中 $\xi(t)$ 表示一个白噪声过程。

那么随机微分方程为

$$\frac {dX(t)}{dt}=f(t)X(t) + h(t)X(t)\xi(t) \tag{5}$$

记 $dW(t)=\xi(t)dt$ ，$dW(t)$ 表示布朗运动的微分项，于是

$$dX(t)=f(t) X(t)dt + h(t) X(t) dW(t) \tag{6}$$

给出 **SDE 的一般形式**，如下

$$dX(t,\omega)=f(t,X(t,\omega)) dt + g(t, X(t, \omega))dW(t,\omega) \tag{7}$$

其中 $\omega$ 表示 $X=X(t,\omega)$ 是一个随机变量，初始条件为 $p(X(0,\omega)=X_0)=1$。

一个例子是

$$dY(t,\omega)=\mu(t)dt + \sigma(t) dW(t,\omega)$$

其中 $Y(t,\omega)$ 和 $W(t,\omega)$ 是随机变量，因为其中包含了 $\omega$ 。

类似 (2) 式那样将 (7) 式写成积分形式，

$$X(t,\omega)=X_0+\int_0^t f(s,X(s,\omega)) ds + \int_0^t g(s, X(s,\omega)) dW(s,\omega) \tag{8}$$

# 3. 随机积分

对于积分项 $\int_0^T g(t,\omega) dW(t, \omega)$，假设 $g(t,\omega)$ 的值仅在离散时间点 $t_1,\ldots, t_{N-1}$ 发生变化，$0=t_0 < t_1 < \cdot < t_{N-1}<t_N = T$，定义

$$S=\int_0^T g(t,\omega) dW(t, \omega) \tag{9}$$

为黎曼积分，$N \rightarrow \infty$

$$S_N(\omega)=\sum_{i=1}^N g(t_{i-1},\omega) (W(t_1,\omega) - W(t_{i-1},\omega)) \tag{10}$$

**$It\hat o$ 积分**

随机变量 $S$ 称为 随机过程 $g(t,\omega)$ 关于布朗运动 $W(t,\omega)$ 在区间 $[0,T]$ 上的 $It\hat o$ 积分，如果：

$$\lim_{N \rightarrow \infty} E[(S-\sum_{i=1}^N g(t_{i-1}, w)(W(t_i,\omega)-W(t_{i-1},\omega)))]=0 \tag{11}$$

对区间 $[0,T]$ 上的每个分区序列 $(t_0,t_1,\ldots,t_N)$ 在 $\max_i (t_i - t_{i-1}) \rightarrow 0$ 时 (11) 均成立。

**例子 1**

一个简单的例子是 $g(t,\omega) \equiv c$，此时

$$\begin{aligned}\int_0^T c dW(t, \omega) &= c \lim_{N\rightarrow \infty} \sum_{i=1}^N (W(t_i,\omega)-W(t_{i-1},\omega))
\\&=c(W(T,\omega)-W(0,\omega))
\end{aligned}$$

根据布朗运动，$W(T,\omega)$ 和 $W(0, \omega)$ 均为高斯随机变量，且初始值 $W(0,\omega)=0$（零值布朗运动，初始位置位于 0 值处），于是上式变为

$$\int_0^T c dW(t, \omega)=cW(T, \omega)$$

**例子 2**

$g(t,\omega)=W(t, \omega)$，故

$$\begin{aligned} \int_0^T W(t,\omega)dW(t,\omega)&=\lim_{N \rightarrow \infty} \sum_{i=1}^N W(t_{i-1},\omega)(W(t_i,\omega)-W(t_{i-1},\omega))
\\\\ &=\lim_{N\rightarrow \infty} [\frac 1 2 \sum_{i=1}^N (W^2(t_i,\omega)-W ^ 2(t_{i-1},\omega)) - \frac 1 2 \sum_{i=1}^N (W(t_i,\omega)-W(t _ {i-1},\omega))^2]
\\\\ &=-\frac 1 2 \lim_{N\rightarrow \infty} \sum_{i=1}^N (W(t_i,\omega)-W(t _ {i-1},\omega))^2 + \frac  1 2 W^2(T, \omega)
\end{aligned} \tag{12}$$

现在关注 (12) 推导右侧的第一项（求极限项）的统计量，

$$\begin{aligned}E[\lim_{N\rightarrow \infty} \sum_{i=1}^N (W(t_i,\omega)-W(t_{i-1},\omega))^2] &=\lim_{N \rightarrow \infty} \sum_{i=1}^N E[(W(t_i,\omega)-W(t_{i-1},\omega))^2]
\\\\ &= \lim_{N \rightarrow \infty} \sum_{i=1}^N (t_i - t_{i-1})\
\\\\ &= T
\end{aligned}$$


上式推导中第二个等式是根据标准布朗运动的性质有

$$W(t_i,\omega) - W(t_{i-1},\omega) \sim \mathcal N(0, t_i-t_{i-1})$$

故 $E[(W(t_i,\omega) - W(t_{i-1},\omega))^2]=t_i-t_{i-1}$

$$\begin{aligned}V[\lim_{N\rightarrow \infty} \sum_{i=1}^N (W(t_i,\omega)-W(t_{i-1},\omega))^2]&=\lim_{N \rightarrow \infty} \sum_{i=1}^N V[(W(t_i,\omega)-W(t_{i-1},\omega))^2]
\\\\&=2 \lim_{N \rightarrow \infty} \sum_{i=1}^N (t_i - t_{i-1})^2
\\\\ &\le 2\max_i (t_i -t_{i-1}) \lim_{N \rightarrow \infty} \sum_{i=1}^N (t_i - t_{i-1})
\\\\ &= 2\max_i (t_i - t_{i-1}) \cdot T
\\\\ &=0
\end{aligned} \tag{13}$$

上式推导中第二个等式是根据 

$$V[X ^ 2]=E[X ^ 4] - E ^ 2[X ^ 2]=3(t_i-t_{i-1})^2 - (t_i-t_{i-1}) ^2=2(t_i - t_{i-1}) ^2$$

最后一个等式是因为 $\max_i (t_i - t_{i-1}) \rightarrow 0$ 。

由于方差为 0，而期望为 $T$ ，故有

$$\lim_{N\rightarrow \infty} \sum_{i=1}^N (W(t_i,\omega)-W(t_{i-1},\omega))^2=T \tag{14}$$

于是 (12) 式为

$$\int_0^T W(t,\omega)dW(t,\omega)=\frac 1 2 W^2(T,\omega) - \frac 1 2 T \tag{15}$$

(15) 式与标准的积分不同，确定性的积分为 $\int_0^T x(t) dx(t)=\frac 1 2 x^2(T)$，这与 $It\hat o$ 积分相差一个 $-\frac 1 2 T$ 项，这个例子说明 **随机微积分的微分规则尤其是链式法则和积分需要重新制定** 。


**$It\hat o$ 积分的性质**

**性质1**

$$E[\int_0^T g(t,\omega) dW(t, \omega)] = 0 \tag{16}$$

证明：

$$E[\int_0^T g(t,\omega)dW(t,\omega)] = E[\lim_{N \rightarrow \infty} \sum_{i=1}^N g(t_{i-1},\omega)(W(t_i,\omega)-W(t_{i-1},\omega))]
\\\\ =\lim_{N \rightarrow \infty} \sum_{i=1}^N E[g(t_{i-1},\omega)]E[(W(t_i,\omega)-W(t_{i-1},\omega))]=0$$

第二个等式是因为 $g(t,\omega)$ 与 $W(t, \omega)$ 相互独立。

最后一个等式是因为 

$$W(t_i,\omega) - W(t_{i-1},\omega) \sim \mathcal N(0, t_i-t_{i-1})$$

**性质2**

$$V[\int_0^T g(t,\omega)dW(t,\omega)] = \int_0^T E[g^2(t,\omega)]dt \tag{17}$$

证明：

$$\begin{aligned}V[\int_0^T g(t,\omega)dW(t,\omega)]&=E[(\int_0^T g(t,\omega)dW(t,\omega))^2]
\\\\ &=E[(\lim_{N \rightarrow \infty} \sum_{i=1}^N g(t_{i-1},\omega)(W(t_i,\omega)-W(t_{i-1},\omega)))^2]
\\\\ &=\lim_{N\rightarrow \infty} \sum_{i=1}^N \sum_{j=1}^N E[g(t_{i-1},\omega)g(t_{j-1},\omega)(W(t_i,\omega)-W(t_{i-1},\omega))(W(t_j,\omega)-W(t_{j-1},\omega))]
\\\\ &=\lim_{N\rightarrow \infty} \sum_{i=1}^N  E[g^2(t_{i-1}, \omega)] E[(W(t_i,\omega)-W(t_{i-1},\omega))^2]
\\\\ &= \lim_{N\rightarrow \infty} \sum_{i=1}^N  E[ g^2(t_{i-1}, \omega) ] (t_i-t_{i-1})
\\\\ &=\int_0^T E[g^2(t,\omega)] dt
\end{aligned} \tag{18}$$

上式推导中，第一个等号是因为根据 (16)，目标积分的期望为 0。第四个等号是因为 $g(t_i,\omega)$，$g(t_j,\omega)$，$W(t_i,\omega)-W(t_{i-1},\omega)$，$W(t_j,\omega)-W(t_{j-1},\omega)$ 均相互独立（$i \ne j$）。

**线性性质**

$$\int_0^T [a_1 g_1(t,\omega)+a_2 g_2(t,\omega)]dW(t,\omega)=a_1 \int_0^T g_1(t,\omega)dW(t,\omega) +a_2 \int_0^T g_2(t,\omega)dW(t,\omega) \tag{19}$$

**$It\hat o$ 引理**

前面提到经典微积分不适应于随机微积分方程。

给定一个随机微分方程

$$dX(t)=f(t,X(t))dt + g(t,X(t))dW(t) \tag{20}$$

以及另一个过程 $y(t)$，它是关于 $X(t)$ 的函数，

$$Y(t)=\phi(t,X(t))$$

函数 $\phi(t,X(t))$ 对 $t$ 连续可微且对 $X$ 二次连续可微，那么求 $Y(t)$ 的微分

$$dY(t)=\tilde f(t,X(t))dt + \tilde g(t,X(t))dW(t) \tag{21}$$




若以传统微积分的链式法则，那么

$$dy(t)=(\phi_t(t,x)+\phi_x(t,x)f(t,x))dt + \phi_x(t,x)g(t,x)dW \tag{22}$$

在随机微积分中，根据 $p(t,X(t))$ 的泰勒展开得，

$$dY(t)=\phi_t(t,X)dt + \frac 1 2 \phi_{tt}(t,X)dt^2+\phi_x(t,X)dX(t) + \frac 1 2 \phi_{xx}(t,X)(dX(t))^2 + h.o.t. \tag{23}$$

将 (20) 式代入 (23) 式得

$$\begin{aligned}dY(t)=& \phi_t(t,X)dt+\phi_x(t,X)[f(t,X(t))dt + g(t,X(t))dW(t)]
\\\\ &+\frac 1 2 \phi_{tt}(t,X) dt^2 + \frac 1 2 \phi_{xx}(t,X)(f ^ 2(t,X(t))dt ^ 2+g ^ 2(t,X(t))dW ^ 2(t)
    \\\\ &+ 2f(t,X(t))g(t,X(t))dtdW(t)) + h.o.t.\end{aligned}\tag{24}$$

高阶微分项 $(dt,dW)$ 快速趋于 0，$dt^2 \rightarrow 0$ 以及 $dtdW(t) \rightarrow 0$。根据零初值标准布朗运动的规则有 $W(t) \sim \mathcal N(0,t)$ ，所以 $E[W^2]=t$，故

$$dW^2(t,\omega)=dt \tag{25}$$

保留 (24) 式中的一阶微分项，得

$$dY(t)=[\phi_t(t,X)+\phi_x(t,X)f(t,X(t)) + \frac 1 2 \phi_{xx}(t,X)g^2(t,X(t))] dt+\phi_x(t,X)g(t,X(t))dW(t) \tag{26}$$

比较 (21) 和 (26) 式，得

$$\tilde f(t,X(t))=\phi_t(t,X) + \phi_x(t,X) f(t,X(t))+\frac 1 2 \phi_{xx}(t,X)g^2(t,X(t))\tag{28}$$

$$\tilde g(t,X(t))=\phi_x(t,X)g(t,X(t)) \tag{29}$$

比较 (28) (29) 式与传统微积分的 (22) 式，发现多了 $\frac 1 2 \phi_{tt}(t,X)g^2(t,X(t))$，这一项称为 $It\hat o$ 校正项。

**例子**

利用 $It \hat o$ 公式解决 $\phi(t,X)=X^2$，SDE 为 $dX(t)=dW(t)$，于是得 $f(t,X)=0$，$g(t,X)=1$， $X(t)=W(t)$，偏导数为 $\frac {\partial \phi(t,X)}{\partial X}=2X$，$\frac {\partial^2 \phi(t,X)}{\partial X^2}=2$， $\phi_t(t,X)=0$，那么根据 (28) (29) 式得

$$d(W^2(t))=1dt + 2W(t)dW(t)$$

上式两边求 $[0,t]$ 区间的积分，且根据 $W(0)=0$，得到 $W(t)$ 的方程为

$$W^2(t)=1t + 2 \int_0^t W(t)dW(t)$$

变换，于是 

$$\int_0^t W(t)dW(t)=\frac 1 2 W^2(t) - \frac 1 2 t \tag{30}$$

# ref

1. Stochastic Differential Equations, Florian Herzog