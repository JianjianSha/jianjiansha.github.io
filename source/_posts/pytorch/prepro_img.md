---
title: 图像数据在送入网络之前的处理
date: 2022-04-16 10:47:31
tags: PyTorch
summary: 总结视觉任务中对图像数据的常用处理
---

本文总结 PyTorch 和 TorchVision 中常用的图像处理的方法。

# 1. Resize

```python
CLASS torchvision.transforms.Resize(size, interpolation='bilinear', max_size=None, antialias=None)
```

参数说明：
1. `size`：如果是 (h,w)，那么图像被 resize 到这个 size；如果是 int，那么图像短边 rescale 到这个 size，长边保持相同的比例进行 rescale。
2. `interpolation`：插值方式，枚举类型，可取值为

    ```python
    BILINEAR （默认值）
    NEAREAT
    BICUBIC
    ```

3. `max_size`： resize 之后的长边最大值。
4. `antialias`：是否使用 antialias 滤镜效果，从而可得到高质量的 resized 图像（若使用，会降低处理速度）。

    如果图像是 PIL 类型，那么这个参数值被忽略，且总是使用 `anti-alias`；如果图像是 Tensor 类型，那么这个参数默认为 `False`。仅在插值模式为 `BILINEAR` 时可设置为 `True`，

__forward(img)__

`img`：图像，PIL 或者 Tensor

返回：rescaled 图像，PIL 或 Tensor

# 2. ToTensor
```python
CLASS torchvision.transforms.ToTensor
```

将一个 PIL 图像（例如通过 Image.open 得到）或者 numpy.ndarray (`HxWxC`，范围为 `[0, 255]`，类型为 `np.uint8`) 转为 torch.FloatTensor，shape 为 `CxHxW`，范围为 `[0.0, 1.0]`。

对于其他情况，`ToTensor` 返回 tensor，但是不进行 scaling（即，不归一化到 `[0.0, 1.0]` 范围内）。

# 3. Normalize

```python
CLASS torchvision.transforms.Normalize(mean, std, inplace=False)
```

参数说明：
1. `mean`：每个通道的均值序列
2. `std`：每个通道的标准差序列
3. `inplace`：bool

归一化如下：
$$\mathbf y_c = (\mathbf x_c - \mu_c)/ \sigma_c, \ c=1,\ldots, C$$

__forward(tensor)__

`tensor`：被归一化的 tensor

返回：归一化后的 tensor
