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
CLASS torchvision.transforms.Normalize(mean, std, inplace=False)-
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


# 4. default_collate

```python
from torch.utils.data.dataloader import default_collate
```

创建 DataLoader 的一个实例时，如不指定 `collate_fn` 参数，则使用默认的 `default_collate`。

例如一个 Dataset 的 `__getitem__` 函数返回

```python
def __getitem__(self, i):
    x = cv2.imread(self.images[i])
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = resize(x)
    # 返回 (x, y)
    # 其中 x 是 (c,h,w) 的 np.narray，y 是 x 的标签，类型为 int64
    return np.transpose(np.array(x, dtype=np.float32), (2, 0, 1)), self.labels[i]
```

创建 DataLoader 实例，

```python
# 未设置 collate_fn 参数
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
```

根据 DataLoader 的源码

```python
if collate_fn is None:
    if self._auto_collation:    # 设置了batch_size，那么为 True
        collate_fn = _utils.collate.default_collate
    ...
```

可知未设置 collate_fn 参数时，使用默认的 default_collate。

根据 DataLoader 的 `__iter__` 方法得到迭代器，这个方法内部调用 `_get_iterator` 方法，考虑最简单的单进程加载数据，那么返回 `_SingleProcessDataLoaderIter` 类实例，这个类的 `__next__` 方法中使用 `_next_data` 方法获取下一批数据，

```python
def _next_data(self):
    index = self._next_index()
    data = self._dataset_fetcher.fetch(index)
    ...
    return data
```

上面代码片段中，`_next_index()` 用于获取下一批数据的索引，

```python
def _next_index(self):
    return next(self._sampler_iter)`    # 这是 DataLoader 中的 BatchSampler 迭代器对象, 构造语句为
    # sampler = RandomSampler(dataset, generator=generator)
    # batch_sampler = BatchSampler(sampler, batch_size, drop_last)
    # 默认时 generator 为 None， drop_last 为 False
```

例如总共有 `10000` 个图片，索引 `0~9999`，假设随机获得的下一批数据索引为 `231, 564, 783, 239`（如何实现随机获取一批数据索引，这里先跳过，我们重点看如何预处理一批数据），然后使用 `_dataset_fetcher.fetch` 方法获取图像数据，

```python
self._dataset_fetcher = _DatasetKind.create_fetcher(
    self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last
)
```
这个 dataset fecther 是指 `_MapDatasetFetcher` 类实例，

```python
# class _MapDatasetFetcher
def fetch(self, possibly_batched_index):
    if self.auto_collation:     # 为 True，走这个判断分支
        data = [self.dataset[idx] for idx in possibly_batched_index]
    else:
        data = self.dataset[possibly_batched_index]
    return self.collate_fn(data)
```

显然 `self.dataset[idx]` 就是调用的 Dataset 的 `__getitem__` 方法，返回为一个 tuple，

```python
# tuple, (img, lbl)
return np.transpose(np.array(x, dtype=np.float32), (2, 0, 1)), self.labels[i]
```

最后调用 `self.collate_fn` 方法，也就是默认的 `default_collate` 函数，

```python
def default_collate(batch):
    elem = batch[0]     # batch is a list of tuples
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            ...     # 多进程加载才走这个分支
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    ...
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))   # tuple size: 2
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('...')   # 每个 tuple 的 size 均必须为 2
        transposed = zip(*batch)    # batch 中：所有 img 的 tuple，和所有 lbl 的 tuple
        return [default_collate(samples) for samples in transposed]
    ...
```

从上述代码可见，对于我们这里的 list of tuples，首先是使用 zip 函数，得到关于一个 batch 中所有图像数据的 tuple `(x1,x2,...,xm)`，以及对应的图像标签的 tuple `(y1,y2,...,ym)`，然后再分别应用 `default_collate` 函数，这里的某个样本的图像数据 `xi` 如果是 ndarray 实例，那么先将其使用 `torch.as_tensor` 转为 Tensor 实例，**元素值不变**，然后再次应用 `default_collate` 函数，对于 Tensor 列表，将所有 Tensor 在新增的第一个维度上进行 stack，例如列表中有 `m` 个 Tensor，shape 均为 `(3, h, w)`，那么 stack 后为一个 `(m, 3, h, w)` 的 Tensor 。

对于图像标签列表 `(y1,y2,...,ym)`，则使用 `torch.tensor(batch)` 直接将其转为一维 tensor，dtype 为 torch.int64。

总结：未对图像数据归一化。




