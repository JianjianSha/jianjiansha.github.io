---
title: 数据集说明
date: 2022-02-15 10:53:32
tags: deep learning
---

对各种常用数据集格式进行说明

<!--more-->

# 1. MNIST

## 1.1 存储说明
[官方网站](http://yann.lecun.com/exdb/mnist/)

有以下四个文件
```sh
train-images-idx3-ubyte: training set images
train-labels-idx1-ubyte: training set labels
t10k-images-idx3-ubyte:  test set images
t10k-labels-idx1-ubyte:  test set labels
```

如果下载的是 `.gz` 压缩文件，使用下面命令进行解压
```
gzip -d <file name>
```

数据说明：

1. 训练集 label 文件 `train-labels-idx1-ubyte`
    ```sh
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    ```
    labels 值为 `[0,9]`。

2. 训练集 image 文件 `train-images-idx3-ubyte`
    ```sh
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    ```

    行顺序存储，像素值范围 `[0,255]`，其中 `0` 表示背景，`255`表示前景。

测试集文件类似，只是 images 数量不同。

## 1.2 numpy 加载

```python
import numpy as np
import struct

def load_mnist(train=True):
    prefix = 'train' if train else 't10k'
    home = os.path.expanduser('~')
    root = os.path.join(home, 'data/cv/mnist')
    label_path = os.path.join(root, '%s-labels-idx1-ubyte'%prefix)
    image_path = os.path.join(root, '%s-images-idx3-ubyte'%prefix)

    with open(label_path, 'rb') as f:
        f.read(8)
        buf = f.read(-1)
        labels = np.frombuffer(buf, dtype=np.uint8).reshape(-1)

    with open(image_path, 'rb') as f:
        f.read(16)
        buf = f.read(-1)
        images = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 784)
    return images, labels
```
