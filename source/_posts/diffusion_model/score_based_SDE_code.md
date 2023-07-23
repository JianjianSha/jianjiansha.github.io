---
title: Score-Based Generative Modeling Through SDE：代码分析
date: 2023-05-05 17:14:53
tags: diffusion model
mathjax: true
---

本文是 [Score-Based Generative Modeling Through SDE](/2022/07/26/diffusion_model/score_based_SDE) 一文具体实现的源码分析，源码位于 [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) 。

为了表述简洁，将 《Score-Based Generative Modeling Through SDE》 一文简称为 score-based-sde 。

**准备工作**

```sh
git clone https://github.com/yang-song/score_sde_pytorch.git
```

# 1. DEMO

demo 文件为 `Score_SDE_demo_PyTorch.ipynb`。

## 1.1 Predictor Corrector sampling

以 CIFAR-10 为例说明，SDE 使用 VE-SDE， predictor 使用 ReverseDiffusionPredictor（这是 reverse-time SDE 采样，其他两种为原始采样，概率流采样），corrector 使用 LangevinCorrector，

```python
# batch_size = 64。 采样得到 8 行 8 列图像
# img_size = 32，CIFAR-10
# channels = 3, CIFAR-10
shape = (batch_size, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
snr = 0.16  # 信噪比，参见下方 (2) 式中的 r
n_steps = 1 # 单个噪声 scale 下，corrector 迭代 1 次
probability_flow = False    # 不使用概率流
# 构造一个采样函数
# scalar: 输入数据（图像像素值）范围为 [0,1]， scalar 决定是否要 rescale 到 [-1,1]
# inverse_scalar: 将数据 rescale 到 [0,1]
sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)
# score_model: 使用 NCSNPP 模型预测得分函数
x, n = sampling_fn(score_model)
show_samples(x)     # 显示生成的图像
```

接着看 `get_pc_sampler` 方法，其返回 `pc_sampler` 方法，这就是 PC 采样器。

```python
def pc_sampler(model):
    with torch.no_grad():
        # 对于 SMLD 反向过程， 采样 xT = \sigma_max * z
        # 参考 score-based-sde 文中 1.1 一节最后的部分
        x = sde.prior_sampling(shape).to(device)
        # sde.N = 1000 。反向过程转移次数
        # sde.T = 1，迭代次数 i -> i/N -> t，转为连续形式，这样 t 范围 [eps, 1]
        # eps: 1e-5，t 起始时间，使用 eps 代替 0，避免数值计算错误。参见 score_based_sde
        # 一文的 C.1 和 C.2 部分。
        timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

        for i in range(sde.N):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t
            # x corrector 输出结果，带噪声
            # x_mean corrector 输出结果，不带噪声
            x, x_mean = corrector_update_fn(x, vec_t, model=model)
            x, x_mean = predictor_update_fn(x, vec_t, model=model)
        # 将生成的数据恢复到 [0,1] 范围
        # 总共的迭代次数 = N * (corrector 迭代次数 + predictor 迭代次数)
        return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)
```

上述代码中，先执行 corrector，然后执行 predictor，我没有在论文中看到作者提及这一顺序的变化，不过由于是在循环体中，所以改变执行顺序问题不大。

### 1.1.1 corrector 实现代码


```python
def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    '''
    x: 当前 step，输入数据 x_t，(batch_size, channels, img_size, img_size)
    t: timestep，(batch_size,) 。 取值范围 [eps, 1]
    sde: SDE 类实例（三种 SDE：VE, VP, sub-VP）
    model: NCSNPP 模型
    corrector: LangevinCorrector 类
    continuous: True，表示使用连续模式
    snr: 信噪比，0.16
    n_steps: 单个噪声 scale 下的 corrector 迭代次数
    '''
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        ...
    else:   # 构建类实例
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)    # 执行 corrector 步骤


class LangevinCorrector(Corrector):
    ...
    def update_fn(self, x, t):
        '''
        x: 输入数据 x，(batch_size, channels, img_size, img_size)
        t: timestep，(batch_size,)
        '''
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr   # PC1000 采样器固定信噪比为 0.16
        if isinstance(sde, (sde_lib.VPSDE, sde_lib.subVPSDE)):
            ...
        else:   # VESDE
            alpha = torch.ones_like(t)  # (batch_size,)
        
        for i in range(n_steps):    # corrector 迭代 n_steps 次
            grad = score_fn(x, t)   # 模型输出的得分函数，s_{\theta}-----
            noise = torch.randn_like(x) # 噪声 z ~ N(0,I)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            # 参考下面 (3) 式
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step[:, None, None, None] * grad   # 参考下方 (1) 式
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise
        return x, x_mean
```

`score_fn` 用于计算得分函数 $\mathbf s _ {\theta} (\mathbf x _ t, t)$ ，也就是模型的输出。

```python
def get_score_fn(sde, model, train=False, continuous=False):
    model_fn = get_model_fn(model, train=train) # 模型前向传播计算，输出为得分函数
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(x, t):
            # subVP 必须使用连续模式
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                labels = t * 999    # time step: 0, 1, ..., N-1
                score = model_fn(x, labels) # 计算 s(xt, t)，t 作为模型的 time-embedding 输入
                # 连续模式下，计算 p(xt|x0) 分布的标准差，参见 score_based_sde 的 (B5/B6) 式
                # std 与 x0 无关，所以这里使用 0 代替了 x0
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:   # VP 的非连续情况，
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)     # 计算得分函数
                # 计算 p(xt|x0) 分布的标准差，参考 score_based_sde 的 (7) 式
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
            # 模型训练阶段，模型 target 为噪声 z （DDPM 原始采样）
            # 而得分函数对应的 target 为 \nabla_xt p(xt|x0) = -(xt-x0) / sigma^2 = - z/sigma
            # 所以模型输出 score 需要做如下处理后，才是真正的得分函数
            score = -score / std[:, None, None, None]
            return score
    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t):
            if continuous:
                # labels 是 std，即 sigma(t)，参见 score_based_sde 中的 (B1) 式
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # VE，t=0 对应最大的噪声 level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()
            score = model_fn(x, labels)

```


根据 [score-based-sde](/2022/07/26/diffusion_model/score_based_SDE) 一文中算法 1（PC 采样，VE-SDE），Corrector 迭代公式为

$$\mathbf x _ i  = \mathbf x _ i + \epsilon _ i \mathbf s _ {\theta} (\mathbf x _ i, \sigma _ i) + \sqrt {2 \epsilon _ i } \mathbf z \tag{1}$$

(1) 式右端的第二、三项分别为信号和噪声，固定信噪幅度比为 `r` （不同的采样器，作者固定信噪幅度比为不同的值，参见论文 Table 5），

$$\frac {||\epsilon _ i \mathbf s _ {\theta}(\mathbf x _ {t-1}, \sigma _ i)|| _ 2}{||\sqrt {2\epsilon _ i} \mathbf z|| _ 2} = r \tag{2}$$

解得

$$\epsilon _ i = 2  r ^ 2  \cdot \frac {\mathbf z ^ {\top} \mathbf z}{\mathbf s _ {\theta} ^ {\top} \mathbf s _ {\theta}} \tag{3}$$


### 1.1.2 Predictor 实现代码

```python
def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    '''
    构建 predictor 类实例
    x: 输入数据     (batch_size, channels, img_size, img_size)
    t: timestep     (batch_size,)
    sde: VE-SDE 类实例
    model: NCSN 模型
    predictor: 反向 diffusion 类
    probability_flow: 是否使用概率流采样
    continuous: 是否使用连续模式
    '''
    # 获取得分函数，score_fn 内部执行 model 前向传播，输出为 model 输出
    score_fn = multils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        ...
    else:
        # 创建 ReverseDiffusionPredictor 类实例
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)


class Predictor(abc.ABC):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # reverse sde
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

class ReverseDiffusionPredictor(Predictor):
    ...
    def update_fn(self, x, t):
        # 获取离散化 f(x, t)，G(t)
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)     # 噪声

        # f: 参见 score-based-sde 一文中 (39) 式。reverse-time SDE
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean
```

rsde 为 [score-based-sde](/2022/07/26/diffusion_model/score_based_SDE) 一文中的 (39) 式，离散化后数据迭代公式 (40) 式，其中包含了 f 函数和 G 函数，对于 VE，前向 SDE 为

$$d \mathbf x = \mathbf f(\mathbf x, t) dt + \mathbf G(t) d\mathbf w \tag{4}$$

其中 

$$\mathbf f (\mathbf x, t)=\mathbf 0, \ \mathbf G (t)= I \sqrt {\frac {d [\sigma ^ 2 (t) ]} {dt}} \tag{5}$$

rsde 的实现代码以及离散化代码如下，

```python
class SDE(abc.ABC):
    def reverse(self, score_fn, probability_flow=False):
        N = self.N      # 噪声 scale 数量
        T = self.T      # SDE 的 timestep 范围为 [0, 1]，故 T 为 1
        sde_fn = self.sde   # 前向 SDE 实现方法
        discretize_fn = self.discretize     # 离散化实现方法

        class RSDE(self.__class__):
            ...
            def sde(self, x, t):
                '''
                反向 SDE 中的 f 和 G 函数，连续型
                '''
                drift, diffusion = sde_fn(x, t) # SDE 中的 f G，见上面 (5) 式
                score = score_fn(x, t)
                # 反向 SDE：f <- f - G G^T s_{\theta}
                # 参见 score-based-sde 一文中 (39) 式
                drift = drift - diffusion[:, None, None, None] ** 2 * score \
                    * (0.5 if self.probability_flow else 1.)
                # 概率流 ODE 不是 SDE，是确定的转移过程，所以没有 diffusion
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion
            
            def discretize(self, x, t):
                '''
                反向 SDE f 和 G 函数，离散型
                '''
                f, G = discretize_fn(x, t)  # 得到离散化的 f G，见下方 (8) 式
                # 参见 score-based-sde 一文中 (40) 式
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) \
                    * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G
        return RSDE()

# sde_fn 和 discretize_fn 方法
class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N

    def sde(self, x, t):
        '''
        前向 SDE
        计算 SDE 中的 f(x, t) 和 G(x,t)，注意是 连续型
        '''
        sigma = self.sigma_ min * (self.sigma_max / self.sigma_min) ** t    # 见下方 (6) 式
        drift = torch.zeros_like(x) # f, G ,参考 (5, 7) 两式
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return drift, diffusion

    def discretize(self, x, t):
        # 离散化 f，G，即 f_i+1, G_i+1 。这也是 前向过程
        timestep = (t * (self.N - 1) / self.T).long()   # t [0,1] -> i [0, N-1]

        # timestep 处的 sigma
        sigma = self.discrete_sigma.to(t.device)[timestep]  # sigma_{i+1}
        # timestep-1 处的 sigma，如果 timestep-1=0,那么 sigma=0
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),    # sigma_i
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)    # （8）式
        return f, G
```

需要指出的是，`self.discrete_sigmas` 的初始化是在 $\sigma$ 的 log 空间中线性等分的，时间维度上等间距，这是因为（见 [score-based-sde](/2022/07/26/diffusion_model/score_based_SDE) 一文中 (B1) 式），

$$\sigma(t) = \sigma _ {min} (\frac {\sigma _ {max}}{\sigma _ {min}}) ^ t \tag{6}$$

取对数后，$\log \sigma(t) = k t + c$，其中 $k, \ c$ 均为常数，所以相当于时间 $t$ 等间距。

(6) 式代入 (5) 式，得

$$G (t) = \sigma (t) \sqrt {2 \log \frac {\sigma _ {max}}{\sigma _ {min}}} \tag{7}$$

对于离散化处理过程，则是 drift 和 diffusion 则是 [score-based-sde](/2022/07/26/diffusion_model/score_based_SDE) 一文中的 (40.1) 式，这里再次列出如下

$$\mathbf f _ {i+1} = \mathbf 0, \ \mathbf G _ {i+1} = I \sqrt {\sigma _ {i+1} ^ 2 - \sigma _ i ^ 2} \tag{8}$$

反向 SDE 为 [score-based-sde](/2022/07/26/diffusion_model/score_based_SDE) 一文中的 (39) 式，离散化后为 (40) 式。

## 1.2 概率流采样

获取概率流采样函数，

```python
# 调用示例
shape = (batch_size, 3, 32, 32)
sampling_fn = sampling.get_ode_sampler(sde,
                                       shape,
                                       inverse_scaler,
                                       denoise=True,
                                       eps=sampling_eps,
                                       device=config.device)
x, nfe = sampling_fn(score_model)
```

其中 `sampling_fn` 这个采样函数定义如下，

```python
def get_ode_sampler(sde, shape, inverse_scaler, denoise=False,
                    rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3, device='cuda'):
    '''
    sde: SDE 类实例
    shape: 采样生成的图像 shape (batch_size, 3, 32, 32)
    inverse_scaler: 将生成图像像素值恢复到 [0, 1] （原来 **可能** 是 [-1,1]）
    denoise: 
    '''
    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)  # 返回 x, x_mean
        return x    # 返回 x_mean，不加噪声，参考下方 (9) 式

    def drift_fn(model, x, t):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None):
        '''
        model: NCSN 模型
        z: 如果有值，那么从 z 样本开始反向转移，否则从一个已知分布中采样然后反向转移
        '''
        with torch.no_grad():
            if z is None:   # demo 中，走这个 分支
                # 采样 xT，从一个已知分布中采样 N(0, sigma_max^2 I)
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                '''t: 自变量； x: 因变量； 返回：偏微分表达式'''
                # x 本来是一维向量，numpy.narray 类型，转为 shape 的 torch.Tensor 类型
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)   # drift: (D2) 式 \tilde {\mathbf f}
                # drift 与 x shape 相同，(batch_size, channels, img_size, img_size)
                # 将 drift 平铺为 一维向量
                return to_flattened_numpy(drift)    
            
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev     # 右手端 evaluation 次数
            # solution.y    # 微分方程的解 (n, n_points)，n=batch_size * channels * img_size * img_size
            # solution.y[:,-1]  # 在 t=eps 处 x 的值
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
            if denoise:
                x = denoise_update_fn(model, x)
            x = inverse_scaler(x)   # 将数据范围恢复到 [0, 1]
            return x, nfe
    return ode_sampler
```

`ode_sampler` 函数中，参数 z 为默认值 None，所以从 $\mathcal N(0, \sigma _ {max} ^ 2 I)$ 中采样得到初始值 $\mathbf x _ T$ （参见  [score-based-sde](/2022/07/26/diffusion_model/score_based_SDE) 一文的 1.1 小节内容），然后调用 `solve_ivp` 求解微分方程（参见  [score-based-sde](/2022/07/26/diffusion_model/score_based_SDE) 一文中的 (D1) (D2) 式），这个微分方程自变量为 $t$，因变量为 $\mathbf x$，$t$ 范围为 $1 \sim 0$（因为是反向转移过程），对应 `sde.T` 和 `eps`（使用 `eps` 代替 0 是为了避免数值计算错误），$\mathbf x$ 的初值为 $\mathbf x _ T$，将 $\mathbf x$ 展开为一维向量：`to_flattened_numpy(x)`。

`ode_func` 的参数是偏微分方程的 自变量 和 因变量，返回的是偏微分表达式，使用 `drift_fn` 函数计算 (D1) 和 (D2) 式，这里不再详细说明 `drift_fn` 这个函数了，跟上面 1.1.2 小节分析类似，非常简单。

这里，`denoise` 值为 True，那么对最后的 $\mathbf x$（即 ODE 在 `eps` 处的解）再进行一次 Predictor 操作，并且不添加噪声，也就是 [score-based-sde](/2022/07/26/diffusion_model/score_based_SDE) 一文的 (40) 式并去掉包含噪声 $\mathbf z _ i$ 的项，如下，

$$\mathbf x _ i = \mathbf x _ {i+1} - \mathbf f _ {i+1}(\mathbf x _ {i+1}) + \mathbf G _ {i+1} \mathbf G _ {i+1} ^ {\top} \mathbf s _ {\theta} (\mathbf x _ {i+1}, i+1) \tag{9}$$




### 1.2.1 似然函数计算


直接看相关计算函数，

```python
def get_likelihood_fn(sde, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
    def drift_fn(model, x, t):
        score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]    # drift，参见 (D2) 式
    
    def div_fn(model, x, t, noise):
        return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)
    
    def likelihood_fn(model, data):
        '''
        data: 一批数据，(batch_size, channels, img_size, img_size)
        '''
        with torch.no_grad():
            shape = data.shape
            if hutchinson_type == 'Gaussion':
                epsilon = torch.randn_like(data)
            elif hutchinson_type == 'Rademacher':
                # -1. or 1.
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
            else:
                raise NotImplementedError()
            
            def ode_func(t, x):
                '''
                t: 自变量 - 时间
                x: 因变量，(batch_size * channels * img_size * img_size + batch_size)

                '''
                # (batch_size, channels, img_size, img_size) 数据 tensor
                sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
                # 自变量 t，批数据
                vec_t = torch.ones(sample.shape[0], device=sampler.device) * t
                # 计算得到 drift，参见 (D2) 式
                drift = mutils.to_flattened_numpy(drift_fn(model, sample, vec_t))
                # 计算 [\log pt(x)]', 参见下方的 (10) 式
                logp_grad = mutils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
                # 得到 (D3), (D4) 的导数向量
                # 前 batch_size*channels*img_size*img_size 个元素为 drift
                # 后 batch_size 个元素为 [\log pt(x)]'
                return np.concatenate([drift, logp_grad], axis=0) 
            
            # [x, 0]，一维向量，最后 batch_size 个元素为 0，前面 batch_size * channels * img_size * img_size 表示 x
            init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
            solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev     # evaluation 次数
            # zp 中，后 batch_size 个元素为 -log pT(xT)，前面的则为 xT
            zp = solution.y[:,-1]   # 得到 t=T 时刻的数据值，即 xT 和 -\log p_T(xT)，参考下方 (10) 式
            # z: 提取 xT
            z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
            # delta_logp: -log pT' (xT)，参考下方 (12) 式
            delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
            # xT 的先验分布为 N(0, sigma_max^2 I)，计算 先验分布的 log 值
            prior_logp = sde.prior_logp(z)      # log pT(xT)：先验分布的 log 值，shape 为 (batch_size,)
            bpd = -(prior + delta_logp) / np.log(2) # -log p0(x0)： (12) 式 取负，表示 负对数似然， 然后转换为 2 为底的对数
            N = np.prod(shape[1:])      # 单个数据 x 的维度
            bpd = bpd / N           # bits/dim
            offset = 7. - inverse_scaler(-1.)   # 8.
            bpd = bpd + offset                  # shift bpd by 8.
            return bpd, z, nfe
    return likelihood_fn
```

上述代码中，`drift_fn` 函数计算 drift，即 $\tilde {\mathbf f} _ {\theta}$，参见 [score-based-sde](/2022/07/26/diffusion_model/score_based_SDE) 一文的 (D2) 式，`solve_ivp` 求解的是 (D3) 和 (D4) 式，偏微分方程的初值为 `init`，这个向量包含两个部分：

1. $\mathbf x _ 0$，即 0 时刻的数据值，shape 为 `(batch_size, channels, img_size, img_size)`，放置 `init` 这个向量之前需要 flatten $\mathbf x _ 0$，这是 (D3) 式的初值

2. $\mathbf 0$ 。这个初始值参考下方 (12) 式。


上述代码包含两个微分方程（微分方程组）的求解，自变量为 $t$，因变量为 $[\mathbf x, -\log p _ t (x)]$ 组成的向量，长度为 `channels*img_size*img_size + 1`，对于一个 batch 样本而言，这个长度还要再乘以 `batch_size`。


$$\frac {d \mathbf x}{dt} = \tilde {\mathbf f}(\mathbf x, t), \quad \frac {-\partial \log  p _ t(\mathbf x)}{\partial t} = \nabla \cdot \tilde {\mathbf f} (\mathbf x, t) = \sum _ i \frac {\partial }{\partial x _ i} \tilde f _ i (\mathbf x, t) \tag{10}$$

根据 (10) 式可知，

$$-\log p _ T (\mathbf x _ T) - (-\log p _ 0 (\mathbf x _ 0)) = \int _ 0 ^ T \nabla \cdot \tilde {\mathbf f} (\mathbf x, t) dt$$

变换后为

$$\log p _ 0 (\mathbf x _ 0)=\log p _ T (\mathbf x _ T)  +  \int _ 0 ^ T \nabla \cdot \tilde {\mathbf f} (\mathbf x, t) dt \tag{11}$$

我们要求的负对数似然就是 $-\log p _ 0 (\mathbf x _ 0)$，但是使用 `solve_ivp` 求解 (10) 式时，知道 $\mathbf x$ 的初态值，而 $-\log p _ t (\mathbf x_t)$  的初态值却是未知的，而且正是我们要求解的， $-\log p _ t (\mathbf x_t)$ 的末态值是已知的（$\mathbf x (T) \sim \mathcal N(\mathbf 0, \sigma _ {max} ^ 2 I)$ ），而 $\mathbf x$ 的末态值却又未知。

为了得到 (10) 式微分方程组的初态值，作如下变换 

$$-\log p' _ t (\mathbf x)=-\log p _ t (\mathbf x) + \log p _ 0 (\mathbf x _ 0) \tag{12}$$

显然 $-\log p' _ t(\mathbf x)$ 与 $-\log p _ t(\mathbf x)$ 的导数相同，满足 (10) 式，因为 $\log p _ 0 (\mathbf x _ 0)$ 是未知但固定的量（未知常数），并且 $t=0$ 时，$-\log p _ 0 ' (\mathbf x _ 0) = 0$，所以 $-\log p _ 0 '(\mathbf x _ 0)$ 作为 (10) 式因变量的初始值，使用 `solve_ivp` 求解 ODE 后，可以得到最后状态 $t=T$ 的因变量的解，即 $-\log p' _ T(\mathbf x _ T)$ 的值，也就是上述代码中的 `zp[-shape[0]:]` 这个向量，于是根据 (12) 式，当 $t=T$ 有

$$-\log p' _ T (\mathbf x(T)) = \int _ 0 ^ T  \nabla \cdot \tilde {\mathbf f} (\mathbf x, t) dt 
\\\\ \log p _ 0 (\mathbf x _ 0) = \log p _ T (\mathbf x _ T) + (- \log p' _ T (\mathbf x _ T)) \tag{13}$$

这里 $\mathbf x _ T \sim \mathcal N(\mathbf 0, \sigma _ {max} ^ 2 I)$。

这里 `div_fn` 计算的实际上是 (10) 式的导数值。


[score-based-sde](/2022/07/26/diffusion_model/score_based_SDE) 一文的 (D6) 式用于高效计算上述 (10) 式，其中 $\epsilon$ 期望为 $\mathbf 0$，协方差为 $I$，见上述代码中的 `epsilon` 变量，有两种初始化方式：Gaussian 和 Rademacher，前者对应 $\epsilon \sim \mathcal N(\mathbf 0, I)$，后者对应二值分布 $\epsilon \sim \text{Bin}(-1, 1)$。

下面给出 `get_div_fn` 函数的代码注释，

```python
def get_div_fn(fn):
    def div_fn(x, t, eps):
        '''计算 score_based_sde 中的 (D7) 式'''
        with torch.enable_grad():
            # 设置可导，因为要计算 \nabla_x (\epsilon_T · f_{\theta}(x, t))，参见 score_based_sde 中的(D7)式
            x.requires_grad_(True)  
            # fn(x, t) 就是计算 drift: f_{\theta}(x, t)，然后计算 \epsilon · f_{\theta}(x, t)  这里 · 表示向量点乘
            fn_eps = torch.sum(fn(x, t) * eps)  
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0] # \nabla_x (\epsilon · f_{\theta}(x, t))，
        x.requires_grad_(False)
        # grad_fn_eps = \nabla_x (\epsilon · f_{\theta}(x, t)) 是标量对向量求导，导数是一个向量
        # grad_fn_eps * eps 向量点乘，然后沿 (channels, img_size, img_size) 求和
        # 返回 (batch_size,) 的向量
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
    return div_fn
```

### 1.2.2 重现

Representation，代码较简单，如下所示，

```python
likelihood_fn = likelihood.get_likehood_fn(sde, inverse_scaler, eps=1e-5)
sampling_fn = sampling.get_ode_sampler(sde, shape, inverse_scaler, denoise=True, 
                                       eps=sampling_eps, device=config.device)

# 获取一批用于 eval 的图像数据
eval_images = torch.from_numpy(eval_images).permute(0, 3, 1, 2).to(config.device)
# 计算似然，返回似然（bits/dim)、xT、微分方程 eval 次数
_, latent_z, _ = likelihood_fn(score_model, scaler(eval_images))

# 指定 xT，然后进行采样（前面是不指定 xT，xT 随机噪声采样）
x, nfe = sampling_fn(score_model, latent_z)
```

## 1.3 受控生成

### 1.3.1 图像修复

调用代码如下，

```python
train_ds, eval_ds, _ = datasets.get_dataset(config)
eval_iter = iter(eval_ds)
bpds = []

# 参考 PC 采样的相关代码注释，这里不再重复
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
snr = 0.16
n_steps = 1
probability_flow = False

pc_inpainter = controllable_generation.get_pc_inpainter(sde
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        denoise=True)
batch = next(eval_iter)
img = batch['image']._numpy()
# 得到一批数据，并转为 tensor，tensorflow 加载数据维度 (batch, height, width, channel)
# 转换维度为 (batch, channel, height, width)
img = torch.from_numpy(img).permute(0, 3, 1, 2).to(config.device)
show_samples(img)   # 显示原图

mask = torch.ones_like(img)
mask[:,:,:,16:] = 0.        # 每个图片 widht=32， 所以每个图片右边一半像素值置零，黑色
show_samples(img * mask)    # 显示右半黑色的图

x = pc_inpainter(score_model, scaler(img), mask)
show_samplers(x)        # 显示修复后的图像
```

上面代码中的 `pc_inpainter` 是图像修复函数，其代码定义如下，

```python
def get_inpaint_update_fn(update_fn):
    # update_fn：参加 score_based_sde 中的算法 1/2 中的 corrector 和 predictor
    def inpaint_update_fn(model, data, mask, x, t):
        with torch.no_grad():
            vec_t = torch.ones(data.shape[0], device=data.device) * t
            # x_mean: 未加噪声
            # x: 加了噪声。x 和 x_mean 均是反向过程得到的数据
            x, x_mean = update_fn(x, vec_t, model=model)
            # std: p(xt) 分布的标准差，边缘分布而非条件分布
            # masked_data_mean: p(xt) 分布的期望
            masked_data_mean, std = sde.marginal_prob(data, vec_t)
            # masked_data: xt，这是前向过程中得到的数据
            masked_data = masked_data_mean + torch.randn_like(x) * std[:,None,None,None]
            # 在 t 时刻，被遮挡部分使用反向的 x，未被遮挡部分使用前向的 xt
            x = x * (1. - mask) + masked_data * mask
            x_mean = x * (1. - mask) + masked_data_mean * mask
            return x, x_mean
    return inpaint_update_fn

projector_inpaint_update_fn = get_inpaint_update_fn(predictor_update_fn)
corrector_inpaint_update_fn = get_inpaint_update_fn(corrector_update_fn)

def pc_inpainter(model, data, mask):
    '''
    model: 模型
    data: 原输入图像数据 batch
    mask: 用于遮挡部分图像的掩码
    '''
    with torch.no_grad():
        # x: 图像数据被遮挡部分使用 xT 填充（xT 根据 sigma_max 噪声采样）
        x = data * mask + sde.prior_sampling(data.shape).to(data.device) * (1. - mask)
        # 时间倒序，这是因为计算 xt，反向过程必须是一步一步进行，而前向过程可以一步到位
        timesteps = torch.linspace(sde.T, eps, sde.N)   
        for i in range(sde.N):
            t = timesteps[i]
            x, x_mean = corrector_inpaint_update_fn(model, data, mask, x, t)
            x, x_mean = projector_inpaint_update_fn(model, data, mask, x, t)
        return inverse_scaler(x_mean if denoise else x)
```

图像修复还是用到了原图数据，如果原图数据未知，只有被部分遮挡的图像数据，上述方法还有效吗？

### 1.3.2 着色

COLORIZATION

```python
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
...
img = torch.from_numpy(img).permute(0, 3, 1, 2).to(config.device)
# RGB channel 求均值，还要 repeat 保证channels=3，因为network 输入要求 channels=3
gray_scale_img = torch.mean(img, dim=1, keepdims=True).repeat(1, 3, 1, 1)
gray_scale_img = scaler(gray_scale_img)     # 可能scale 到[-1,1]，取决配置
pc_colorizer = controllable_generation.get_pc_colorizer(
    sde, predictor, corrector, inverse_scaler,
    snr=snr, n_steps=n_steps, probability_flow=probability_flow,
    continuous=config.training.continuous, denoise=True
)
x = pc_colorizer(score_model, gray_scale_img)   # 上色
```

上色函数如下，

```python
# M: 3x3 的正交矩阵（正交矩阵的行列向量均是归一化向量）
invM = torch.inverse(M) # M^{-1}
def decouple(inputs):
    # 经过decouple，每个channel值均不同
    return torch.einsum('bihw,ij->bjhw', inputs, M.to(inputs.device))

def couple(inputs):
    return torch.einsum('bihw,ij->bjhw', inputs, invM.to(inputs.device))

def get_mask(image):
    # 第一个channel为1，其他channel均为0
    mask = torch.cat([torch.ones_like(image[:, :1, ...]),
                      torch.zeros_like(image[:, 1:, ...])], dim=1)
    return mask

def get_colorization_update_fn(update_fn):
    def colorization_update_fn(model, gray_scale_img, x, t):
        '''
        gray_scale_img: 灰度图
        x: 反向过程中，t 时刻的输入数据
        '''
        mask = get_mask(x)
        vec_t = torch.ones(x.shape[0], device=x.device) * t
        # 得到反向过程 t 时刻 corrector/predictor 的输出数据 xt
        x, x_mean = update_fn(x, vec_t, model=model)
        # 灰度图旋转到另一空间后，前向变换到 t 时刻的分布 pt 的期望和标准差
        masked_data_mean, std = sde.marginal_prob(decouple(gray_scale_img), vec_t)
        # 前向过程的数据 xt
        masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
        x = couple(decouple(x) * (1. - mask) + masked_data * mask)
        x_mean = couple(decouple(x) * (1. - mask) + masked_data_mean * mask)
        return x, x_mean
    return colorization_update_fn 

predictor_colorize_update_fn = get_colorization_update_fn(predictor_update_fn)
corrector_colorize_update_fn = get_colorization_update_fn(corrector_update_fn)

def pc_colorizer(model, gray_scale_img):
    # gray_scale_img: 每个 channel 值均相同（RGB 均相同，为原图像RGB三通道均值）
    with torch.no_grad():
        shape = gray_scale_img.shape
        mask = get_mask(gray_scale_img) # 第一channel 通过滤波器
        # 将灰度图旋转变换到另一个空间，进行mask，保留第一channel，然后将
        # 采样得到xT，过滤掉其第一channel，然后旋转变换到另一空间，两者合并
        # 合并后再回到原来空间
        # x: 反向过程的初始值 xT
        x = couple(decouple(gray_scale_img) * mask + \
                   decouple(sde.prior_sampling(shape).to(gray_scale_img.device)
                            * (1. - mask)))
        timesteps = torch.linspace(sde.T, eps, sde.N)
        for i in range(sde.N):
            t = timesteps[i]
            x, x_mean = corrector_colorize_update_fn(model, gray_scale_img, x, t)
            x, x_mean = predictor_colorize_update_fn(model, gray_scale_img, x, t)
    return inverse_scaler(x_mean if denoise else x)
```

### 1.3.3 根据标签生成图像

**# 计算分类器输出对输入的梯度**

（以下代码源自 [score_sde](https://github.com/yang-song/score_sde.git)）

```python
# 分类器前向传播，得到非归一化得分
def get_logit_fn(classifier, classifier_params):
    def preprocess(data):   # 分类器输入数据归一化
        image_mean = jnp.asarray([[[0.49139968, 0.48215841, 0.44653091]]])
        image_std  = jnp.asarray([[[0.24703223, 0.24348513, 0.26158784]]])
        return (data - image_mean[None, ...]) / image_std[None, ...]
    
    def logit_fn(data, ve_noise_scale):
        '''
        data: 输入数据
        ve_noise_scale: VE SDE 的 timesteps
        '''
        data = preprocess(data)     # 输入数据归一化
        logits = classifier.apply({'params': classifier_params}, data, ve_noise_scale, train=False, mutable=False)
        return logits

def get_classifier_grad_fn(logit_fn):
    def grad_fn(data, ve_noise_scale, labels):
        '''
        data: (B, 3, H, W)
        labels: (B,)
        '''
        def prob_fn(data):
            logits = logit_fn(data, ve_noise_scale) # 分类器前向传播，得到非归一化得分，(B, C)
            # log p_t(y|x)   先计算得到所有分类的概率对数 (B, C) -> 然后根据标签条件，提取对应标签的 log p，(B,)
            # 然后求和，得到一个标量。为何要求和？因为计算 标量对向量 的梯度，即：log p 对 x 的梯度，见下方详细说明
            prob = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(labels.shape[0]), labels].sum()
            return prob
        return jax.grad(prob_fn)(data)  # shape 为 (B, 3, H, W)
    return grad_fn
```

分析以上代码：

根据 [score_based_SDE](/2022/07/26/diffusion_model/score_based_SDE) 一文中的 (I3) 式，需要计算分类器输出概率对数 对 输入数据的梯度 $\nabla _ {\mathbf x} \log p _ t(\mathbf y| \mathbf x)$，这里分类器的输入除了数据 $\mathbf x$，还有 timesteps $t$。上述代码中输入数据是一个 mini batch，记 batch size 为 $B$，那么输出 `prob` 为

$$P=\sum _ {i=1} ^ B \log p _ t(y _ i|\mathbf x _ i)$$

利用深度学习框架自带的求梯度功能，计算梯度则为，例如对 mini batch 中第 `i` 个输入数据 $\mathbf x _ i$ 的梯度为

$$\nabla _ {\mathbf x _ i} P = \nabla _ {\mathbf x _ i} \log p _ t (y _ i|\mathbf x _ i)$$

**# 条件采样函数**

```python
def conditional_corrector_update_fn(rng, state, x, t, labels):
    # 计算模型输出，即模型的得分估计 s_{\theta} -> \nabla_x log p_t(x)
    score_fn = mutils.get_score_fn(sde, score_model, state.params_ema, state.model_state, train=False,
                                   continuous=continuous)
    def total_grad_fn(x, t):
        ve_noise_scale = sde.marginal_prob(x, t)[1]
        # \nabla_x \log p_t(x) + \nabla_x \log p_t(y|x)
        # 参见下方 corrector 更新过程中的方括号部分 [...]
        return score_fn(x, t) + classifier_grad_fn(x, ve_noise_scale, labels)
    if corrector is None:
        corrector_obj = NoneCorrector(sde, total_grad_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, total_grad_fn, snr, n_steps)
    return corrector_obj.update_fn(rng, x, t)

def pc_conditional_sampler(rng, score_state, labels):
    rng, step_rng = random.split(rng)
    x = sde.prior_sampling(step_rng, shape)     # 反向过程的 xT ~ N(0, sigma_max*I)
    timesteps = jnp.linspace(sde.T, eps, sde.N) # timesteps 等间隔从 1 到 0 （反向过程）

    def loop_body(i, val):
        # 当前 timestep，的输入样本 x，以及输入分布的期望 mu
        rng, x, x_mean = val
        t = timesteps[i]
        vec_t = jnp.ones(shape[0]) * t
        rng, step_rng = random.split(rng)
        x, x_mean = conditional_corrector_update_fn(step_rng, score_state, x, vec_t, labels)
        rng, step_rng = random.split(rng)
        x, x_mean = conditional_predictor_update_fn(step_rng, score_state, x, vec_t, labels)
        return rng, x, x_mean
    
    _, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body, (rng, x, x))
    return inverse_scaler(x_mean if denoise else x)
```

以上代码简单易读。反向过程与 [score_based_SDE](/2022/07/26/diffusion_model/score_based_SDE) 中的算法 1 一致（这里以 VE SDE 为例说明）。

对于 corrector，更新过程为

**for** $j=1,\ldots,M$ **do**

&emsp;&emsp;$\mathbf z \sim \mathcal N(\mathbf 0, I)$

&emsp;&emsp;$\mathbf x _ i \leftarrow \mathbf x _ i + \epsilon _ i [\mathbf s _ {\theta} ^ {\star} (\mathbf x _ i)+ \nabla _ {\mathbf x _ i} \log p _ t(y_i|\mathbf x _ i)] + \sqrt {2\epsilon _ i} \mathbf z$


上述过程中 $M$ 就是代码中的 `n_steps` 变量。