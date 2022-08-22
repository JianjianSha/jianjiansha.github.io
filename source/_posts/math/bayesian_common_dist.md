---
title: 贝叶斯推断中常用分布
date: 2022-05-31 11:49:08
tags: math
mathjax: true
---


# 1 Gamma 分布

## 1.1 Gamma 函数

$$\Gamma(x)=\int_0^{\infty} t^{x-1} e^{-t}dt, \quad x>0$$

Gamma 函数满足阶乘性质：$\Gamma(x+1)=x\Gamma(x)$

## 1.2 Gamma 分布

$$\Gamma(x|\alpha,\beta)=\frac {\beta^{\alpha} x^{\alpha-1} e^{-\beta x}}{\Gamma(\alpha)}, \quad x\gt0, \ \alpha,\beta >0$$

$\alpha$：形状参数，主要决定分布曲线的形状；
$\beta$：比例参数，主要决定曲线有多陡；
$1/\beta$：scale 参数。

令 $\beta x = t$，对上式积分得

$$\int_0^{\infty} \Gamma(x|\alpha, \beta) dx =  \frac 1 {\Gamma(\alpha)}\int_0^{\infty} \beta^{\alpha} \frac {t^{\alpha-1}}{\beta^{\alpha-1}} e^{-t} d t / \beta=1$$

验证上式是一个有效的概率密度函数。

### 1.2.1 统计量

$Mean=\alpha / \beta$

$Variance = \alpha /\beta^2$

### 1.2.2 解释说明

如果一固定时间段内某随机事件发生次数的平均为 $\lambda$（即，遵循一个 Poisson 过程），那么在 $x$ 时间内此事件发生 $k$ 次的概率为 $F(x)$，其中 $F$ 是 Gamma 分布的 cdf ，且 $\alpha=k$，$\beta = \lambda$。

**例子**

假设某店铺发送一次汇票是随机事件，且平均每 15 mins 发送一次汇票，那么在 3 小时内，此店铺发送 10 次汇票的概率是多少？

答：平均每小时发生 $\lambda=4$ 次汇票发送（以小时为单位），在 $3$ （小时时间内）发生 $k=10$ 次事件的概率为 

$$F(3)=P(x<3|\alpha=10,\beta=4)=0.7586$$

## 1.3 Inverse Gamma 分布

$$f(x|\alpha, \beta)=\frac {e^{-1/(\beta x)}} {\Gamma(\alpha) \beta ^{\alpha} x^{\alpha+1}}, x > 0, \ \alpha,\beta >0$$

令 $t= 1/(\beta x)$，此时 $x=1/(\beta t)$ 对上式积分

$$\int_{\infty}^0 \frac 1 {\Gamma(\alpha)} e^{-t} \frac {(\beta t)^{\alpha+1}}{\beta^{\alpha+1}}d (\frac 1 t) =-\frac 1 {\Gamma(\alpha)}\int_{\infty}^0  t^{\alpha-1} e^{-t} dt=1$$


记 $F(x)$ 为 $\Gamma(x|\alpha, \beta)$ 的 cdf，$F'(x)$ 为  $\Gamma^{-1}(x|\alpha', \beta')$ 的分布，那么存在以下关系，

$$F'(x)=1-F(\frac 1 x|\alpha, \frac 1 {\beta})$$

对 pdf 积分可证明。这表明，

如果 $X \sim Inv-Gamma(\alpha, \beta)$，那么有 $1/X \sim Gamma(\alpha, 1/\beta)$。


## 1.4 Chi-square 分布

$n$ 个独立的随机变量 $X_1,\ldots, X_n$ 均服从标准正态分布，那么 $\mathcal X^2=\sum_{i=1}^n X_i^2 $ 服从自由度为 $n$ 的卡方分布 $\mathcal X^2(n)$，pdf 为

$$f_n(x)=\frac 1 {2 \Gamma(n/2)} \left(\frac x 2 \right)^{n/2-1} e^{-x/2}, \quad x > 0$$

令 $t= x/2$，对上式积分 $\int_0^{\infty} f_n(x/2)dx/2=1$。

## 1.5 Inverse Chi-square 分布

$$f_n(x)=\frac {x^{-n/2-1} e^{-1/(2x)}}{2^{n/2} \Gamma(n/2)}, \quad x >0$$

令 $t=1/(2x)$，对上式积分

$$\int_{\infty}^0 f_n(\frac 1 {2t})d(\frac 1{2t})=\frac {(2t)^{n/2+1} e^{-t}}{2^{n/2} \Gamma(n/2)} (-\frac 1 {2t^2} )dt=\frac 1 {\Gamma(n/2)}\int_0^{\infty} t^{n/2-1} e^{-t} dt = 1$$

验证这是一个有效的概率密度函数。

根据 cdf 定义，


$$\begin{aligned}F(X\le x)&=\int_0^x f_n(t)dt
\\&=\int_0^x \frac {t^{-n/2-1} e^{-1/(2t)}}{2^{n/2}\Gamma(n/2)}dt
\\&=\int_0^x -\frac 1 {2\Gamma(n/2)} (\frac 1 2\cdot \frac 1 t)^{n/2-1} e^{-1/2 \cdot 1/t} d(1/t)
\\&\stackrel{u=1/t}= \int_{1/x}^{\infty} \frac 1 {2\Gamma(n/2)} (\frac u 2)^{n/2-1} e^{-u/2} du
\\&=1-F_n(1/x)=F_n(X \ge 1/x)
\end{aligned}$$

其中 $F_n(x)$ 是 Chi-square 分布的 cdf，上式推导过程中，$\stackrel{u=1/t}=$ 的推导，首先负号 $-$ 使得积分变为 $\int_x^0$，然后 $u=1/t$，对应的积分范围变为 $\int_{1/x}^{\infty}$。

根据上式推导中的变换 $u=1/t$，有以下结论：

如果 $X \sim Inv-\mathcal X^2(n)$，那么  $1/X \sim \mathcal X^2(n)$。


## 1.6 Scaled 逆卡方分布

$$f(x)=\frac {(ns^2/2)^{n/2} e^{-ns^2/(2x)}}{\Gamma(n/2) x^{n/2+1}}, \quad x>0, \ s, n>0$$

自由度为 $n$，scale 参数为 $s^2$

令 $t=ns^2/(2x)$，对 $f(x)$ 积分，最终可得到 Gamma 函数，积分为 `1`，推导过程略。

与 $Inv-\mathcal X^2(n)$ 相比较，发现通过 $x:=x/(ns^2)$ 两者 $\exp$ 指数部分很相似，事实上

$$\begin{aligned}F(x)&=\int_0^x f(t) dt
\\&=\int_0^x \frac {(ns^2/2)^{n/2} e^{-ns^2/(2t)}}{\Gamma(n/2) t^{n/2+1}}dt
\\&=\int_0^x \left(\frac {t}{ns^2}\right)^{-n/2-1} \frac {\exp\{-\frac 1 {2t/(ns^2)} \}}{2^{n/2}\Gamma(n/2)} d(\frac t {ns^2})
\\&=\int_0^{x/(ns^2)} \frac {y^{-n/2-1} e^{-1/(2y)}}{2^{n/2}\Gamma(n/2)} dy
\\&=F_1( x/(ns^2))
\end{aligned}$$

其中 $F_1(\cdot)$ 是 逆卡方分布。

根据上式推导中的变换 $y=t/(ns^2)$，有以下结论：

1. 如果 $X \sim Scaled-Inv-\mathcal X^2(n,s^2)$，那么  $X/(ns^2) \sim Inv-\mathcal X^2(n)$。

2. 如果 $X \sim Scaled-Inv-\mathcal X^2(n,s^2)$，那么 $X \sim Inv-Gamma(n/2, 2/(ns^2))$，且有 $1/X \sim Gamma(n/2, ns^2/2)$。令 Inv-Gamma 中的 $\alpha=n/2, \ \beta=2/(ns^2)$ 可以得到 Scaled-Inv 卡方分布。




# 2 指数分布

**pdf**

$$f(x)=\lambda e^{-\lambda x}, \quad x \ge 0, \ \lambda>0$$

**cdf**

$$F(x)=1-e^{-\lambda x}$$

**inv-cdf**

$$F^{-1}(p)=-\log (1-p)/\lambda$$

指数分布就是 Gamma 分布在 $\alpha=1, \beta=\lambda$ 时的特例。

## 2.1 统计量

$Mean = 1/\lambda$

$Variance = 1/\lambda^2$

## 2.2 解释说明

根据 指数分布就是 Gamma 分布在 $\alpha=1, \beta=\lambda$ 时的特例 可知，如果 $\lambda$ 表示在一固定时间段内某随机事件发生的平均次数，那么在 $x$ 时间内事件发生一次的概率为 $F(x)$。

**性质1**：指数分布无记忆性。

$$P(x\ge s)=P(x\ge t+s|x\ge t)$$

**性质2**：如果 $x \sim Poisson(\lambda)$，那么事件发生的时间间隔 $t$ 服从期望为 $1/\lambda$ 的指数分布。

证：

对于 $Poisson(\lambda)$ 分布，单位时间内事件发生的次数平均为 $\lambda$，那么在固定时间 $t$ 内发生的次数平均为 $\lambda t$，因此根据 $Poisson(\lambda t)$ 分布， 时间 t 内事件不发生的概率为

$$P(x=0)=\frac {(\lambda t)^0 e^{-\lambda t}}{0!}=e^{-\lambda t}$$

那么 t 时间内事件发生一次的概率为

$$1-P(x=0)=1-e^{-\lambda t}=F(t)$$

这是指数函数的 cdf，表示在 t 时间之后，事件发生了一次。

# 3. Weibull 分布

**pdf**

$$f(x)=\frac {\beta} {\alpha}(\frac {x}{\alpha})^{\beta-1} \exp \{-(\frac x {\alpha})^{\beta}\}$$

**cdf**

$$F(x)=1-\exp \{-(\frac x {\alpha})^{\beta}\}$$

**inv cdf**

$$F^{-1}(p)=\alpha (-\log (1-p))^{1/\beta}$$

# 4. T 分布

假设 $X_1, \ldots, X_n$ 独立同分布，采样自高斯分布 $\mathcal N(\mu, \sigma^2)$，记样本均值

$$\overline X=\frac 1 n \sum_{i=1}^n X_i$$

样本方差为

$$S^2=\frac 1 {n-1} \sum_{i=1}^n (X_i-\overline X)^2$$

那么随机变量

$$\frac {\overline X - \mu} {\sigma /\sqrt n}$$

服从标准正态分布。


随机变量

$$\frac {\overline X - \mu} {S /\sqrt n}$$

服从学生 t 分布，自由度为 n-1。这里不可观测变量为 $\mu$，故 t 分布可以用于推断 $\mu$ 的置信区间。

**pdf**

$$f(t)=\frac {\Gamma(\frac {v+1} 2)} {\sqrt {v \pi} \Gamma(\frac v 2)} \left(1+\frac {t^2} v\right)^{-(v+1)/2}$$

# 5. Gumbel

Gumbel 分布是一种极值型分布。假设某随机变量服从指数型分布，每采样 n 次并取其中最大值作为目标值，显然这个目标值也是一个随机变量，遵循 Gumbel 分布。

**pdf**

$$f(x;\mu, \beta)=e^{-z - e^{-z}}, \quad z=\frac {x-\mu} {\beta}$$

**cdf**

$$F(x;\mu, \beta)=e^{-e^{(x-\mu)/ \beta}}$$

**inverse-cdf**
CDF 的反函数

$$F^{-1}(y;\mu, \beta)=\mu-\beta \log(-\log (y))$$

可使用 Gumbel-softmax 采样作为多项分布的重参数 trick，例如 $\mu=0, \beta=1$，使得目标函数中保留多项分布的参数 $\boldsymbol \theta$（K 维，K-dim），而使用均匀分布进行采样，从而将随机性转移到均匀分布中。

**多项分布的采样**

```python
def sample_with_softmax(logits, size):
    probs = softmax(logits)
    return np.random.choice(len(logits), size, p=probs)
```

**基于 gumbel 采样**

$$x = \arg \max_i \log \theta_i + G_i$$

其中 $\boldsymbol \theta=(\theta_1, \ldots, \theta_K)$ 是多项分布各分类的概率，$G_i, i=1,\ldots, K$ 表示第 `i` 次 Gumbel 采样。

步骤：

1. 生成 $K$ 个服从均匀分布 $U(0,1)$ 的独立样本 $\epsilon_1, \ldots, \epsilon_K$
2. 计算 $G_i=-\log (-\log(\epsilon_i))$
3. 计算新向量 $\mathbf v' = [v_1+G_1,\ldots, v_K +G_K]$，其中 $v_i=\log \theta_i$。
4. 计算 softmax ，

    $$\sigma(v_i')=\frac {e^{v_i'/T}}{\sum_{j=1}^K e^{v_j'/T}}$$

    其中 $T$ 表示温度，$T$ 越大上式计算结果越平滑，越小就越接近 one-hot 。

```python
def softmax(logits):
    ex = np.exp(logits)
    return ex / np.sum(ex, axis=-1, keepdims=True)

def inv_gumbel_cdf(y, mu=0, beta=1, eps=1e-20):
    return mu-beta*np.log(-np.log(y+eps))

def sample_gumbel(size):
    p = np.random.random(size)
    return inv_gumbel_cdf(p)

def sample_with_gumbel_noise(logits, size):
    noise = sample_gumbel((size, len(logits)))  # (size, K)
    return np.argmax(logits + noise, axis=1)

def differentiable_sample(logits, size, temperature=1):
    noise = sample_gumbel((size, len(logits)))
    return softmax((logits + noise)/temperature)
```

