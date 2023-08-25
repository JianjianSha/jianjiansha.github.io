---
title: pytorch 常见代码片段
date: 2023-08-05 11:44:40
tags: PyTorch
---

# 1. 固定随机种子

```python
def seed_torch(seed=1029):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法
```

# 2. 模型保存

```python
# 保存整个模型，模型结构和权重参数
torch.save(model, save_model_dir)
# 保存模型权重参数
torch.save(model.state_dict, save_weight_dir)
# 加载模型
model2 = torch.load(save_model_dir)
# 加载模型参数
model3 = models.resnet152()
model3.load_state_dict(torch.load(save_weight_dir))
# 如需在 GPU 上运行模型，那么取消以下注释
# model2.cuda()
# model3.cuda()
```

## 2.1 多卡模型存储与加载

### 2.1.1 单卡保存+多卡加载

```python
model2 = nn.DataParallel(model2).cuda()
model3 = nn.DataParallel(model3).cuda()
```

### 2.1.2 多卡保存+单卡加载

**# 直接加载整个模型**

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'   #这里替换成希望使用的GPU编号

model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()
# 保存+读取整个模型
torch.save(model, save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
model4 = torch.load(save_dir).module
```

**# 加载权重**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'   #这里替换成希望使用的GPU编号
import torch
from torchvision import models

save_dir = 'resnet152.pth'   #保存路径
model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()

# 保存权重
torch.save(model.module.state_dict(), save_dir)

model5 = models.resnet152()
model5.load_state_dict(torch.load(save_dir))
model5 = nn.DataParallel(model5).cuda()
```

### 2.1.3 多卡保存+多卡加载

建议使用加载权重的方式。因为加载整个模型，如果模型保存时使用的 GPU 编号与模型加载后使用的 GPU 编号不一致，会导致错误。

```python
model = nn.DataParallel(model).cuda()
torch.save(model.state_dict(), save_dir)

model6 = models.resnet152()
model6.load_state_dict(torch.load(save_dir))
model6 = nn.DataParallel(model6).cuda()
```

如果当前只有完整的模型，那么通过参数字典赋值也可以，

```python
model = torch.load(save_dir)
model7.state_dict = model.state_dict
```

# 3. 冻结部分层参数

```python
for param in model.parameters():
    if param.name.startswith("layer"):  # 模型中名称以 ‘layer’ 开头的层参数冻结
        param.requires_grad = False
```

# 4. 可视化

## 4.1 打印网络结构

安装包

```sh
pip install torchinfo
```

使用
```python
import torchvision.models as models
from torchinfo import summary
resnet18 = models.resnet18() # 实例化模型
summary(resnet18, (1, 3, 224, 224)) # 1：batch_size 3:图片的通道数 224: 图片的高宽
```

## 4.2 可视化卷积核

```python
conv1 = dict(model.features.named_children())['3']
kernel_set = conv1.weight.detach()
num = len(conv1.weight.detach())
print(kernel_set.shape)
for i in range(0,num):
    i_kernel = kernel_set[i]
    plt.figure(figsize=(20, 17))
    if (len(i_kernel)) > 1:
        for idx, filer in enumerate(i_kernel):
            plt.subplot(9, 9, idx+1) 
            plt.axis('off')
            plt.imshow(filer[ :, :].detach(),cmap='bwr')
    break   # 仅显示 kernel_set[0]，shape 为 (64, 3, 3)
```

## 4.3 可视化特征图

**方法一**

```python
class Hook:
    def __init__(self):
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []

    def __call__(self,module, feat_in, feat_out):
        print("hooker working", self)
        self.module_name.append(module.__class__)
        self.features_in_hook.append(feat_in)
        self.features_out_hook.append(feat_out)
        return None

hook = Hook()
model = models.resnet152(pretrain=True)
model.layer4.register_forward_hook(hook)

model.eval()
_ = model(inputs)
    
out = hook.features_out_hook[0] # out: (B, C, H, W)
channels  = out.shape[1]        # channel size 
feats = out[0].cpu().clone()    # feat map of the first image

plt.figure(figsize=(20, 17))
for c in range(channels):
    if c > 99:                  # only show the first 100 channels
        break
    ft = feats[c]
    plt.subplot(10, 10, c+1) 
    
    plt.axis('off')
    #plt.imshow(ft[ :, :].detach(),cmap='gray')
    plt.imshow(ft[ :, :].detach())
```

**方法二**

使用 `from mmengine.visualization import Visualizer`，参考

https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html

## 4.4 可视化 class activation map

class activation map （CAM）的作用是判断哪些变量对模型来说是重要的，在CNN可视化的场景下，即判断图像中哪些像素点对预测结果是重要的。除了确定重要的像素点，人们也会对重要区域的梯度感兴趣，因此在CAM的基础上也进一步改进得到了Grad-CAM（以及诸多变种）。CAM和Grad-CAM的示例如下图所示：

![](/images/pytorch/ccsnippet_1.png)

安装包，

```sh
pip install grad-cam
```

一个简单的例子

```python
import torch
from torchvision.models import vgg11,resnet18,resnet101,resnext101_32x8d
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

model = vgg11(pretrained=True)
img_path = './dog.png'
# resize操作是为了和传入神经网络训练图片大小一致
img = Image.open(img_path).resize((224,224))
# 需要将原始图片转为np.float32格式并且在0-1之间 
rgb_img = np.float32(img)/255
plt.imshow(img)

from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 将图片转为tensor，shape 为 (1, 3, 224, 224)
img_tensor = torch.from_numpy(rgb_img).permute(2,0,1).unsqueeze(0)

target_layers = [model.features[-1]]
# 选取合适的类激活图，但是ScoreCAM和AblationCAM需要batch_size
cam = GradCAM(model=model,target_layers=target_layers)
targets = [ClassifierOutputTarget(preds)]   
# 上方preds需要设定，比如ImageNet有1000类，这里可以设为200
grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]
cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
print(type(cam_img))
Image.fromarray(cam_img)
```

## 4.5 使用 FlashTorch 可视化

安装包，

```sh
pip install flashtorch
```

内容来自
https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%B8%83%E7%AB%A0/7.2%20CNN%E5%8D%B7%E7%A7%AF%E5%B1%82%E5%8F%AF%E8%A7%86%E5%8C%96.html



# 5. 测试两个 ndarray 几乎相等

```python
np.testing.assert_almost_equal(actual, desired)

torch.allcose(tensor1, tensor2)
```


# 6. torch.compile

torch.compile 是加速 PyTorch 代码的最新方法！ torch.compile 通过 JIT 将 PyTorch 代码编译成优化的内核，使 PyTorch 代码运行得更快。 

**# 传入函数**

```python
def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a + b
opt_foo1 = torch.compile(foo)   # 得到优化后的函数
```

**# 装饰器**

```python
@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a + b
```

**# 传入模型**

```python
mod = MyModule()
opt_mod = torch.compile(mod)

# ======================================================
# train 是一个循环函数，完成一个 batch 数据的训练（前向+反向+更新参数）
model = init_model()
opt = torch.optim.Adam(model.parameters())
train_opt = torch.compile(train, mode="reduce-overhead")

for i, (x, y) in enumerate(dataloader):
    train_opt(x, y) # 完成前向传播，loss 计算，反省传播，更新梯度
```

**与 TorchScript 的区别**

```python
def f(x, y):
    if x.sum() < 0:
        return -y
    return y
a, b = torch.ones(2, 2), torch.ones(2, 2)
# 由于 a 和 b 已经确定，且 x.sum() > 0，所以 trace_f 固定为返回 y，永远不可能返回 -y
traced_f = torch.jit.trace(f, (a, b))  
c, d = torch.ones(3, 3), torch.ones(3, 3)
traced_f(c, d) == traced_f(-c, d)

compiled_f = torch.compile(f)
compiled_f(c, d) == 0 - compiled_f(-c, d)

script_f = torch.jit.script(f)
script_f(torch.ones(2, 2), 1)   # 报错，script_f 两个参数类型不同
script_f(torch.ones(1, 2), torch.arange(3)) # 调用正确
compiled_f(torch.ones(1, 1), 1) # 调用正确

def nf(x):              # 非 pytorch 函数，因为存在 numpy 调用语句
    y = np.array(2)
    y = torch.from_numpy(y)
    return x
# trace，script 无法正确处理非 pytorch 函数，而且 compile 可以正确处理非 pytorch 函数
```

# 7. 计时

```python
# 方法一
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

# 方法二
import torch.utils.benchmark as benchmark
def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)
t0 = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

print(t0.timeit(100))
```

# 8. AMP

自动混合精度

什么时候用torch.FloatTensor,什么时候用torch.HalfTensor呢？这是由pytorch框架决定的，在AMP上下文中，以下操作中Tensor会被自动转化为半精度浮点型torch.HalfTensor：
```sh
__matmul__
addbmm
addmm
addmv
addr
baddbmm
bmm
chain_matmul
conv1d
conv2d
conv3d
conv_transpose1d
conv_transpose2d
conv_transpose3d
linear
matmul
mm
mv
prelu
```

有些操作的输入，根据需要则需要强制转为 fp32 从而避免计算出 NaN ，强转方法为 `x = x.float()` 。


```python
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)
scaler = GradScaler()
for epoch in range(epochs):
    for i, (x, y) in enumerate(trainloader):
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()
```

支持 AMP 的 GPU 架构为：Hopper，Ampere，Turing，Volta，具体参见 https://www.nvidia.com/en-us/data-center/tensor-cores/

## 8.1 分布式训练中的 AMP

对于分布式训练，由于 autocast 是 thread local 的，需要使用，

**torch.nn.DataParallel**

```python
MyModel(nn.Module):
    @autocast()
    def forward(self, input):
        ...
        
#alternatively
MyModel(nn.Module):
    def forward(self, input):
        with autocast():
            ...


model = MyModel()
dp_model=nn.DataParallel(model)

with autocast():
    output=dp_model(input)
    loss = loss_fn(output)
```

**torch.nn.DistributedDataParallel**

与 DataParallel 相同处理。

# 9. EMA

指数移动平均/权重移动平均

n 个数据 $\theta _ 1, \ldots, \theta _ n$

EMA: $v _ t = \beta \cdot v _ {t-1} + (1-\beta) \cdot \theta _ t, \quad v _ 0 = 0$

实现 demo

```python
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay  # 相当于上述 \beta
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # 使用 EMA
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        # 不使用 EMA
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 初始化
ema = EMA(model, 0.999)
ema.register()

# 训练过程中，更新完参数后，同步update shadow weights
def train():
    optimizer.step()    # 更新模型参数
    ema.update()

# eval前，apply shadow weights；eval之后，恢复原来模型的参数
def evaluate():
    ema.apply_shadow()
    # evaluate
    ema.restore()
```

# 其他

ONNX, Tensorboard 参考 PyTorch 文档或 tutorials 。

**# 推理**

TensorRT 优点

1. Reduced Precision：将模型量化成INT8或者FP16的数据类型（在保证精度不变或略微降低的前提下），以提升模型的推理速度。
2. Layer and Tensor Fusion：通过将多个层结构进行融合（包括横向和纵向）来优化GPU的显存以及带宽。
3. Kernel Auto-Tuning：根据当前使用的GPU平台选择最佳的数据层和算法。
4. Dynamic Tensor Memory：最小化内存占用并高效地重用张量的内存。
5. Multi-Stream Execution：使用可扩展设计并行处理多个输入流。
6. Time Fusion：使用动态生成的核去优化随时间步长变化的RNN网络

使用 TensorRT，将 `.onnx` 格式文件转换为 TensorRT 格式，

```sh
trtexec.exe --onnx=resnet50.onnx --saveEngine=resnet50.engine
```

`.engine` 格式文件是一个 plan file，推理阶段先将此文件反序列化为一个 inference engine 。


