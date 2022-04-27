---
title: 并行训练
date: 2022-04-27 17:36:33
tags: PyTorch
---

# 1. 指定可用 GPU
按照优先级 __从高到低__ 顺序介绍如何设置可用显卡设备。

1. 代码中指定

    ```python
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    ```

2. 在 shell 脚本中指定

    ```sh
    # file: train.sh
    source bashrc
    CUDA_VISIBLE_DEVICES=gpu_id python train.py
    ```
3. shell 中导出环境变量

    ```sh
    # file: train.sh
    source bashrc
    export CUDA_VISIBLE_DEVICES=gpu_id && python train.py
    ```

4. 命令行中指定

    ```sh
    CUDA_VISIBLE_DEVICES=gpu_id python train.py
    # or 执行 shell 脚本
    CUDA_VISIBLE_DEVICES=gpu_id sh run.sh
    ```

指定多个显卡，
```sh
# file: train.py
source bashrc
export CUDA_VISIBLE_DEVICES=gpu_id1 && CUDA_VISIBLE_DEVICES=gpu_id2 python train.py
```

# 2. 模型/数据加载到 GPU 

有以下两种方法：

1. `.cuda()`

    ```python
    model.cuda(gpu_id)  # gpu_id 为 int 类型
    model.cuda('cuda:'+str(gpu_ids))
    model.cuda('cuda:1,2') # 将模型加载到多个 GPU 上
    ```

2. `torch.cuda.set_device()`

    在定义模型之前执行以下一行代码，可以将模型和数据加载到对应 GPU 上，
    ```python
    torch.cuda.set_device(gpu_id)
    torch.cuda.set_device('cuda:'+str(gpu_ids)) # 可以多卡
    ```

    同时模型还需要执行，
    ```python
    model.cuda()    # cuda 方法不带参数
    ```

    如果 `model.cuda(...)` 方法中有参数，那么不使用 `torch.cuda.set_device()` 中指定 GPU，而使用 `model.cuda(...)` 参数指定的 GPU。

__指定可用GPU再指定加载GPU__

例如，命令行执行

```sh
CUDA_VISIBLE_DEVICES=2,3,4,5 python3 train.py
```

而代码内部指定

```python
model.cuda(1)
```

由于 GPU index 从 `0` 开始，故会将 model 加载到 GPU3 上。

# 3. 多卡并行

## 3.1 nn.DataParallel

单进程控制多 GPU

```python
torch.nn.DataParallel(model, device_ids)
```

以上 `model` 必须已经是加载到 GPU 上，且必须与 `device_ids[0]` 是同一块 GPU。例如，

```python
gpus = [0, 1, ,2, 3]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
model.cuda()
model = nn.DataParallel(model, deivce_ids=gpus, output_device=gpus[0])
```

当然，也可以不指定 `device_ids`，同时 `model.cuda()` 中不带参数，此时使用所有的 GPU 并行训练。

## 3.2 torch.distributed

torch 自动将代码分配给 n 个进程，分别在 n 个 GPU 上运行。

命令为
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

torch.distributed.launch 启动器在命令行分布式执行 python 文件，执行时，将当前进程的 index（其实就是 GPU 的 index）通过参数 `local_rank` 传递给 python。

设置 GPU 之间通信使用的后端，

```python
import torch.distributed as dist

dist.init_process_group(backend='nccl')

nprocs = torch.cuda.device_count()
batch_size = batch_size // nprocs
model.cuda()
model = torch.nn
```

使用 Distributed Sampler 划分数据集，