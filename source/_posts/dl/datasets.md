---
title: 数据集说明
date: 2022-02-15 10:53:32
tags: deep learning
---

对各种常用数据集格式进行说明

<!--more-->

# 1. CV

## 1.1 MNIST

### 1.1.1 存储说明
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

### 1.1.2 numpy 加载

```python
import numpy as np
import struct

def load_mnist(train=True):
    '''
    load mnist from local device
    return (images, labels), with shapes of (N, 784) and (N,) respectively.
    '''
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

## 1.2 CIFAR

`CIFAR10` 包含 10 个分类：飞机（airplane）、汽车（automobile）、鸟类（bird）、猫（cat）、鹿（deer）、狗（dog）、蛙类（frog），马（horse）、船（ship）和卡车（truck）。

图片尺寸为 `32x32x3` RGB 彩色图片，数据集中一共有 50000 张训练图片和 10000 张测试图片。

[数据集下载](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

在CIFAR-10 数据集中，文件 `data_batch_1.bin`、`data_batch_2.bin` 、... 、`data_batch_5.bin` 和 `test_ batch.bin` 中各有10000 个样本。一个样本由3073 个字节组成，第一个字节为标签label ，剩下3072 个字节为图像数据。样本和样本之间没高多余的字节分割， 因此这几个二进制文件的大小都是 `3073*10000=30730000` 字节

python 版本的数据集文件 `data_batch_1, data_batch_2, ... , data_batch_5, test_batch`，使用 pickle 将数据序列化到文件，所以保存的是一个字典，我们关注其中的数据和标签，

```
{
    'labels': [6, 9, 9, ... ],
    'data': array([[59, 43, 50, ... , 140, 84, 72],
                   ...,
                   [62, 61, 60, ... , 130, 130, 131]], dtype=uint8),
    ...
}
```

### 1.2.1 数据加载

**# 借助 tensorflow**
```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

**# 本地导入**
```python
import pickle
import numpy as np

def load(fname='data_batch_1'):
    root = 'data/cv/cifar/cifar-10/'
    with open(root + fname, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    data = np.array(dict[b'data'])              # (10000, 3072)
    labels = np.array(dict[b'labels'])          # (10000,)
    return data, labels
```

**# 借助 pytorch**
```python
from torchvision import datasets
import torchvision.transforms as transforms


data_path = 'data/cv/cifar'
transform = transforms.Compose([transforms.ToTensor()]) # 归一化到 [0, 1] 之间且转为 Tensor 类型，维度 C, H, W

train_set = datasets.CIFAR10(data_path, train=True, transform=transform, download=True)
# 会自动检测目录 os.path.join(data_path, datasets.CIFAR10.base_folder) 之下是否有数据文件
# 如有且通过 md5 验证，那么使用已经存在的数据文件，否则重新下载数据文件
# 内部维护一个 data: (50000, 32, 32, 3)，targets: (50000)
# __getitem__: img (normalized tensor), target
```

## 1.3 CeleA

CeleA是香港中文大学的开放数据，包含10177个名人的202599张图片。
官网：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
下载地址（百度网盘，官方的）：http://pan.baidu.com/s/1eSNpdRG

```
CelebA中有 10,177人 的 202,599张 图片
├── Anno
│   ├── identity_CelebA.txt		//身份标记：图片名 + 编号，相同编号代表同一人，如：000001.jpg 2880
│   ├── list_attr_celeba.txt	//标记图片属性
│   ├── list_bbox_celeba.txt	//
│   ├── list_landmarks_align_celeba.txt	//对齐后的图片，人脸标记（眼鼻嘴）lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
│   └── list_landmarks_celeba.txt	//自然环境下的图片，人脸标记（眼鼻嘴）lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
├── Eval
│   └── list_eval_partition.txt	//分组：训练、验证、测试
├── Img
│    └── img_align_celeba.zip	//对齐后的图片集
└── README.txt
```

list_attr_celeba.txt中属性说明

英文	中文
5_o_Clock_Shadow	胡须
Arched_Eyebrows	柳叶眉
Attractive	有魅力
Bags_Under_Eyes	眼袋
Bald	秃顶
Bangs	刘海
Big_Lips	大嘴唇
Big_Nose	大鼻子
Black_Hair	黑发
Blond_Hair	金发
Blurry	模糊
Brown_Hair	棕色头发
Bushy_Eyebrows	浓眉
Chubby	圆脸
Double_Chin	双下巴
Eyeglasses	戴眼镜
Goatee	山羊胡子
Gray_Hair	白发
Heavy_Makeup	浓妆
High_Cheekbones	高颧骨
Male	男人
Mouth_Slightly_Open	嘴微微张开
Mustache	胡子
Narrow_Eyes	小眼睛
No_Beard	没有胡须
Oval_Face	鸭蛋脸
Pale_Skin	苍白的皮肤
Pointy_Nose	尖鼻子
Receding_Hairline	发际线高
Rosy_Cheeks	红润的脸颊
Sideburns	鬓胡
Smiling	微笑
Straight_Hair	直发
Wavy_Hair	卷发
Wearing_Earrings	戴着耳环
Wearing_Hat	戴着帽子
Wearing_Lipstick	擦口红
Wearing_Necklace	戴着项链
Wearing_Necktie	戴着领带
Young	年轻

**加载数据**

```python
# 使用 PyTorch
import torchvision.datasets as dset
import torchvision.transforms as transforms
 
root = 'data/cv/celeba'
dataset = dset.CelebA(root=root, download=True,
	                      transform=transforms.Compose([
		                      transforms.Resize(64),
		                      transforms.CenterCrop(64),
		                      transforms.ToTensor(),
		                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
print(dataset)
```

# 2. Search

## 2.1 MS MARCO

https://microsoft.github.io/msmarco/