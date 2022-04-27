---
title: 深度学习环境搭建
p: pytorch/DL-env
date: 2019-09-09 16:38:11
tags: DL
---
本文仅针对 ubuntu 系统进行讨论。

<!-- more -->

搭建深度学习环境 tensorflow，pytorch 等，如需要 GPU 加速，一般选择安装 

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
（注：或者使用 `pip install tensorflow`）

这条命令会自动安装合适的 cuda 和 cudnn

## 源码安装

源码安装基本参考[官方文档](https://www.tensorflow.org/install/source#ubuntu)

创建虚拟环境 name 为 tf，python 版本 3.10，确保 pip，wheel，numpy 等安装
```sh
conda create -n tf python=3.10
pip install -U numpy
pip install -U keras_preprocessing --no-deps
```

安装 bazel，使用 ubuntu apt 仓库安装（其他安装方式参考[这里](https://docs.bazel.build/versions/main/install-ubuntu.html)）
```sh
sudo apt install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel
```
我的 ubuntu 中已经安装了 jdk1.8_281，然而上面的最后一条命令中 `jdk1.8` 仅仅是保持传统惯例，并非表示系统必须安装 `jdk1.8`，但是我仍然推荐安装一个 jdk，版本随意，不要低于 `1.8`，安装 jdk 是为了能 Bazel 能 `build` java 代码。

**GPU支持**

确保安装以下 package

1. NVIDIA GPU drivers
2. CUDA
3. cuDNN

安装好后，将 CUPTI（包含在 CUDA toolkit 中即，安装了 CUDA 后，CUPTI 也已经安装好）的安装路径附加到 `LD_LIBRARY_PATH` 环境变量，
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```

下载 tensorflow 源码，这一步时间较长，
```sh
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

配置
```sh
./configure
```
根据提示进行下一步操作。当然这里使用虚拟环境，执行以下命令进行配置而非上面的配置命令，
```sh
python configure.py
```
区别是前者默认使用虚拟环境外部的路径，而后者使用虚拟环境内部的路径，当然无论哪种配置命令，都可以手动修改默认值。

`./configure` 命令执行结果展示
```sh
./configure
You have bazel 3.0.0 installed.
Please specify the location of python. [Default is /usr/bin/python3]: 


Found possible Python library paths:
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.6/dist-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: 
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: 
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: Y
CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with TensorRT support? [y/N]: 
No TensorRT support will be enabled for TensorFlow.

Found CUDA 10.1 in:
    /usr/local/cuda-10.1/targets/x86_64-linux/lib
    /usr/local/cuda-10.1/targets/x86_64-linux/include
Found cuDNN 7 in:
    /usr/lib/x86_64-linux-gnu
    /usr/include


Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus Each capability can be specified as "x.y" or "compute_xy" to include both virtual and binary GPU code, or as "sm_xy" to only include the binary code.
Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]: 6.1


Do you want to use clang as CUDA compiler? [y/N]: 
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: 
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
    --config=mkl            # Build with MKL support.
    --config=monolithic     # Config for mostly static monolithic build.
    --config=ngraph         # Build with Intel nGraph support.
    --config=numa           # Build with NUMA support.
    --config=dynamic_kernels    # (Experimental) Build kernels into separate shared objects.
    --config=v2             # Build TensorFlow 2.x instead of 1.x.
Preconfigured Bazel build configs to DISABLE default on features:
    --config=noaws          # Disable AWS S3 filesystem support.
    --config=nogcp          # Disable GCP support.
    --config=nohdfs         # Disable HDFS support.
    --config=nonccl         # Disable NVIDIA NCCL support.
Configuration finished
```
注意配置过程中，设置 `cuda=Y` 以启用 CUDA，并指定 CUDA 和 cuDNN 的版本，这是因为一台机器上可能装有多个 CUDA 或 cuDNN 版本。指定 CUDA 的计算力，例如我这里是 `6.1` 。 CUDA 的计算力可以到 [官网](https://developer.nvidia.com/cuda-gpus) 上查询。


**生成 pip 包**

依然是在 tensorflow 目录下，使用 `bazel build` 生成 TensorFlow 2.x 包，
```sh
bazel build [--config=option] //tensorflow/tools/pip_package:build_pip_package
```

由于在 `./configure` 中启用了 `cuda`，故生成的包支持 GPU，当然也可以特别指定支持 GPU，
```sh
bazel build --config=cuda [--config=option] //tensorflow/tools/pip_package:build_pip_package
```

如果想生成 TensorFlow 1.x 版本，那么执行，
```sh
bazel build --config=v1 [--config=option] //tensorflow/tools/pip_package:build_pip_package
```

由于 bazel 生成 tensorflow 过程中会使用大量内存，如果内存不够大，那么为 `bazel build` 添加选项 `--local_ram_resources=2048`

`GCC` 版本使用 `7.3`。

bazel build 会生成名为 `build_pip_package` 的可执行文件，运行这个可执行文件会生成一个位于 `/tmp/tensorflow_pkg` 目录下的 `.whl` 包

从发行版本分支生成：
```sh
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

从 master 分支生成，使用 `--nightly_flag` 以便获取正确的依赖，
```sh
./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg
```


重新生成时，建议先执行 `bazel clean`。

最后，安装包，
```sh
pip install /tmp/tensorflow_pkg/tensorflow-version-tags.whl
```

测试是否支持 GPU
```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')

# successful NUMA node read from SysFS had negative value (-1), 
# but there must be at least one NUMA node, so returning NUMA node zero
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
会出现上述 NUMA node 值为 -1 的警告，这个可以忽略，不用管它。代码中逻辑为，如果 NUMA node 读取值为负，那么返回 0。一种修改 NUMA node 的方法为（[原帖](https://stackoverflow.com/questions/44232898/memoryerror-in-tensorflow-and-successful-numa-node-read-from-sysfs-had-negativ)），

Annoyingly, the numa_node setting is reset (to the value -1) for every time the system is rebooted. To fix this more persistently, you can create a crontab (as root).

The following steps worked for me:
```sh
# 1) Identify the PCI-ID (with domain) of your GPU
#    For example: PCI_ID="0000.81:00.0"
lspci -D | grep NVIDIA
# 2) Add a crontab for root
sudo crontab -e
#    Add the following line
@reboot (echo 0 | tee -a "/sys/bus/pci/devices/<PCI_ID>/numa_node")
```


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

现在增加了 develop 模式，conda 环境下执行
```
cd [pytorch github project root path]
python setup.py develop
```
这样只会生成一个位于 site-packages 中的 torch 的 egg-link，可以随时修改 pytorch 源码，而不用重装 pytorch。

容器运行 pytorch-gpu
```
docker pull pytorch/pytorch:1.7.1-cuda11.0-cudnn8-level
docker run -p 9527:22 --gpus all -rm -itd --ipc=host -v /home/xx/xx:/home/xx/xx --name pytorch pytorch/pytorch:1.7.1-cuda11.0-cudnn8-level
```

# 安装 mmdetection
以 conda 虚拟环境名称 `base` 为例，其中已经安装了 PyTorch，cudatoolkit 等包，还有一些包如`matplotlib, pillow, opencv` 等图像处理相关的包也需要安装，可以使用
```
conda list
```
查看。现在要安装 mmdetection，

1. 安装 mmcv，这是 open-mmlab 一众库的基础，
```
git clone https://github.com/open-mmlab/mmcv.git
```

进入根目录
```
cd mmcv
```
以开发模式安装，
```
MMCV_WITH_OPS=1 pip install -e .
```
其中，MMCV_WITH_OPS 默认为 0，表示 cpu 模式下运行 mmcv（轻量级模式），为 1 时 启用 cuda 加速。`pip install -e .` 表示可编辑模型安装当前目录的库，等同于 `python setup.py develop`。

下载 mmdetection 源码，
```
git clone https://github.com/open-mmlab/mmdetection.git
```
同样地，以开发模式安装，
```
cd mmdetection
pip install -r requirements/build.txt
python setup.py develop
```
