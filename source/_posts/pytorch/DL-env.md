---
title: 深度学习环境搭建
p: pytorch/DL-env
date: 2019-09-09 16:38:11
tags: DL
---
本文仅针对 ubuntu 系统进行讨论。

搭建深度学习环境 tensorflow，pytorch 等，如需要 GPU 加速，一般选择安装 
<!-- more -->
NVIDIA cuda 工具包，以前通常需要预先安装：
1. NVIDIA driver
2. cuda
3. cudnn

# NVIDIA driver
曾经安装 NVIDIA 驱动采取的比较复杂的方法，先是 close nouveau，让系统进入命令行，然后安装事先下载好的驱动安装文件 `NVIDIA-Linux-x86_64-xxx.xxx.run`，这里使用比较简单的安装方法，打开 ubuntu 的 Software & Updates，点击 Additional Drivers，选择 `Using NVIDIA driver metapackage from nvidia-driver-xxx` 然后点击 `Apply Changes` 进行驱动安装。

# cuda & cudnn
直接使用 conda 安装 pytorch，安装过程比较简单，执行以下命令即可，
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

如果下载较慢，可使用清华源，执行命令，
```
conda config
conda config --set show_channel_urls yes

cd ~
vi .condarc
```
打开 `.condarc` 文件并添加
```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

然后执行
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

# 安装 tensorflow
```
conda install tensorflow-gpu
```
这条命令会自动安装合适的 cuda 和 cudnn

# PyTorch
下载源码
```
git clone https://github.com/pytorch/pytorch.git
```
更新源码
```
git reset --hard
git pull origin master
git submodule sync
git submodule update --init --recursive
```

安装
```
python setup.py install
```
如果不想安装，仅仅编译生成，那么执行
```
python setup.py build
```
由于我这里安装了 Clang 和 llvm，设置了 `CPLUS_INCLUDE_PATH`，导致生成的过程中 include 到 llvm 的头文件，所以可以临时屏蔽 llvm 的头文件路径，
```
export CPLUS_INCLUDE_PATH='' && python setup.py build
```
