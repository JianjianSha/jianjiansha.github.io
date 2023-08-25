---
title: 并行训练
date: 2022-04-27 17:36:33
tags: PyTorch
---

# 1. 指定可用 GPU

指定了可用 GPU 之后，这些 GPU 对每一个进程均可用（但不一定全部用到，看情况）。

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
predictions = model(inputs)     # forward pass on multi-GPUs
loss = loss_function(predictions, labels)   # compute loss function
loss.mean().backward()      # Average GPU-losses + backward pass
optimizer.step()
predictions = model(inputs) # Forward pass with new parameters
```

当然，也可以不指定 `device_ids`，同时 `model.cuda()` 中不带参数，此时使用所有的 GPU 并行训练。或者通过环境变量设置训练所用 GPUs，
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
model = model.cuda()    # 先加载到第一个 GPU 上
model = nn.DataParallel(model)

inputs = inputs.cuda()
labels = labels.cuda()
```

![](/images/pytorch/train_parallel_1.png)

<center>图 1. DataParallel 前向和后向传播示意图</center>

注意到图 1 Forward 过程的第 4 步（图 1 右上角），将所有并行计算的结果合并到 GPU-1 上，对于分类任务，这不成问题，因为输出 size 较小，如果使用一个较大的 batch size 来训练一个语言模型，那么 GPU-1 内存压力较大。


## 3.2 torch.distributed

**# 几个概念**

1. rank：用于表示进程的编号/序号（在一些结构图中rank指的是软节点，rank可以看成一个计算单位），每一个进程对应了一个rank的进程，整个分布式由许多rank完成。
2. node：物理节点，可以是一台机器也可以是一个容器，节点内部可以有多个GPU。
3. rank与local_rank： rank是指在整个分布式任务中进程的序号；local_rank是指在一个node上进程的相对序号，local_rank在node之间相互独立。
4. nnodes、node_rank与nproc_per_node： nnodes是指物理节点数量，node_rank是物理节点的序号；nproc_per_node是指每个物理节点上面进程的数量。
5. word size ： 全局（一个分布式任务）中，rank的数量。

6. backend ：通信后端，可选的包括：nccl（NVIDIA推出）、gloo（Facebook推出）、mpi（OpenMPI）。从测试的效果来看，如果显卡支持nccl，建议后端选择nccl，，其它硬件（非N卡）考虑用gloo、mpi（OpenMPI）。

7. master_addr与master_port：主节点的地址以及端口，供init_method 的tcp方式使用。 因为pytorch中网络通信建立是从机去连接主机，运行ddp只需要指定主节点的IP与端口，其它节点的IP不需要填写。 这个两个参数可以通过环境变量或者init_method传入。

    ```python
    # 方式1：
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", 
                            rank=rank, 
                            world_size=world_size)
    # 方式2：
    dist.init_process_group("nccl", 
                            init_method="tcp://localhost:12355",
                            rank=rank, 
                            world_size=world_size)
    ```

### 3.2.1 单机多卡

torch 自动将代码分配给 n 个进程，分别在 n 个 GPU 上运行。

命令为
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

torch.distributed.launch 启动器在命令行分布式执行 python 文件，执行时，将当前进程的 index（其实就是 GPU 的 index）通过参数 `local_rank` 传递给 python。

设置 GPU 之间通信使用的后端，

```python
import torch.distributed as dist

def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        torch.distributed.barrier()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

args = parser.parse_args()
args.nprocs = torch.cuda.device_count()

dist.init_process_group(backend='nccl')

torch.cuda.set_device(args.local_rank)
model.cuda(args.local_rank)

args.batch_size = args.batch_size // nprocs

model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[local_rank])

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda(local_rank)
cudnn.benchmark = True

train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=2,
                                           pin_memory=True,
                                           sampler=train_sampler)
val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=2,
                                            pin_memory=True,
                                            sampler=val_sampler)

for epoch in range(args.start_epoch, args.epochs):
    train_sampler.set_epoch(epoch)
    val_sampler.set_epoch(epoch)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, local_rank, args)

    # evaluate on validation set
    acc1 = validate(val_loader, model, criterion, local_rank, args)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if args.local_rank == 0:
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
            }, is_best)                           
```

reference: https://github.com/tczhangzhi/pytorch-distributed/tree/master

**多机多卡**

代码为，

```python
import torch
import torchvision
import torch.utils.data.distributed
import argparse
import torch.distributed as dist
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)  # 增加local_rank
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)

def main():
    dist.init_process_group("nccl", init_method='env://')    # init_method方式修改
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data_set = torchvision.datasets.MNIST('~/DATA/', train=True,
                                          transform=trans, target_transform=None, download=True)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set,
                                                    batch_size=256,
                                                    sampler=train_sampler,
                                                    num_workers=16,
                                                    pin_memory=True)
    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    net = net.cuda()
    # DDP 输出方式修改：
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                                    output_device=args.local_rank)
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(1):
        for i, data in enumerate(data_loader_train):
            images, labels = data 
            # 要将数据送入指定的对应的gpu中
            images.to(args.local_rank, non_blocking=True)
            labels.to(args.local_rank, non_blocking=True)
            opt.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print("loss: {}".format(loss.item()))


if __name__ == "__main__":
    main()
```

假设一共有两台机器（节点1和节点2），每个节点上有8张卡，节点1的IP地址为192.168.0.1 占用的端口12355（端口可以更换），那么启动方式为：

```sh
>>> #节点1
>>>python -m torch.distributed.launch --nproc_per_node=8
           --nnodes=2 --node_rank=0 --master_addr="192.168.0.1"
           --master_port=12355 MNIST.py
>>> #节点2
>>>python -m torch.distributed.launch --nproc_per_node=8
           --nnodes=2 --node_rank=1 --master_addr="192.168.0.1"
           --master_port=12355 MNIST.py
```

**torch.distributed.barrier**

实现预训练模型的下载和读入内存，如果所有进程都分别下载一遍显然是不合理的，只需要主进程下载即可，其他进程则使用 `barrier` 进行等待，代码如下，

```python
# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:  # 其他 GPU 进程等待
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab                          
# 位置 1
args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                      num_labels=num_labels,
                                      cache_dir=args.cache_dir if args.cache_dir else None)
tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                            do_lower_case=args.do_lower_case,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
model = model_class.from_pretrained(args.model_name_or_path,
                                    from_tf=bool(".ckpt" in args.model_name_or_path),
                                    config=config,
                                    cache_dir=args.cache_dir if args.cache_dir else None)

if args.local_rank == 0:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
# 位置 2
```

主进程下载预训练模型，并加载模型权重，然后也使用 `barrier` 等待，当所有进程均到达 `barrier` 之后，则分别继续执行，即主进程从上述代码的位置 2 开始执行，其他进程则从位置 1 开始执行，其他进程直接加载预训练模型权重，而不用继续下载模型文件。

**保存和加载**

```python
## 只保存 rank0 进程上模型参数
if torch.distributed.get_rank() == 0:
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

# 或者考虑以 CPU 形式保存
torch.save(model.module.cpu().state_dict(), "model.pth")  
```

### 3.2.2 spawn 方式的多机分布式

上一小节是 launch 方式分布式。现在考虑 spawn 方式分布式。

**# 每个节点上每个进程占用一张卡**

一个节点指一台机器。

1. world_size = 节点数 x 每个节点上 GPU 数量，（全局进程数量）

2. dist.init_process_group 的参数 rank 需要根据 node 编号和 GPU 数量确定，（全局进程编号）

以下代码例子中，两个节点，`node_rank` 分别为 0 和 1，每个节点 `8` 张卡，即 `local_size=0`，所以 `world_size = 2 * local_size`。某个节点上某个卡的编号为 `local_rank`，

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--world_size", type=int)
parser.add_argument("--node_rank", type=int)
parser.add_argument("--master_addr", default="127.0.0.1", type=str)
parser.add_argument("--master_port", default="12355", type=str)
args = parser.parse_args()


def example(local_rank, node_rank, local_size, world_size):
    # 初始化
    rank = local_rank + node_rank * local_size  # 全局 rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl",
                            init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
                            rank=rank,
                            world_size=world_size)
    # 创建模型
    model = nn.Linear(10, 10).to(local_rank)
    # 放入DDP
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank) 
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    # 进行前向后向计算
    for i in range(1000):
        outputs = ddp_model(torch.randn(20, 10).to(local_rank))
        labels = torch.randn(20, 10).to(local_rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()


def main():
    local_size = torch.cuda.device_count()  # 单个节点上所有 GPU，每个 GPU 启动一个进程
    print("local_size: %s" % local_size)
    mp.spawn(example,
        args=(args.node_rank, local_size, args.world_size,),
        nprocs=local_size,
        join=True)


if __name__=="__main__":
    main()
```

启动方式：

```sh
>>> #节点1
>>>python python demo.py --world_size=16 --node_rank=0 --master_addr="192.168.0.1" --master_port=22335
>>> #节点2
>>>python python demo.py --world_size=16 --node_rank=1 --master_addr="192.168.0.1" --master_port=22335
```

**# 每个节点上单进程多卡**

注意点：

1. dist.init_process_group 的参数 rank 等于节点编号（实际上就是全局进程编号）

2. world_size = 节点的总数量（全局进程数量）

3. DDP 不需要指定 device。

```python
import torchvision
from torchvision import transforms
import torch.distributed as dist
import torch.utils.data.distributed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--master_addr", default="127.0.0.1", type=str)
parser.add_argument("--master_port", default="12355", type=str)
args = parser.parse_args()


def main(rank, world_size):
    # 一个节点就一个rank，节点的数量等于world_size
    dist.init_process_group("gloo",
                            init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
                            rank=rank,
                            world_size=world_size)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data_set = torchvision.datasets.MNIST('~/DATA/', train=True,
                                          transform=trans, target_transform=None, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set,
                                                    batch_size=256,
                                                    sampler=train_sampler,
                                                    num_workers=16,
                                                    pin_memory=True)
    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    net = net.cuda()
    # net中不需要指定设备！
    net = torch.nn.parallel.DistributedDataParallel(net)
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(1):
        for i, data in enumerate(data_loader_train):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            opt.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print("loss: {}".format(loss.item()))


if __name__ == '__main__':
    main(args.rank, args.world_size)
```

启动方式，

```sh
>>> #节点1
>>>python demo.py --world_size=2 --rank=0 --master_addr="192.168.0.1" --master_port=22335
>>> #节点2
>>>python demo.py --world_size=2 --rank=1 --master_addr="192.168.0.1" --master_port=22335
```

### 3.2.3 分布式通信的常见函数

判断底层通信库是否可用：
```python
torch.distributed.is_nccl_available()  # 判断nccl是否可用
torch.distributed.is_mpi_available()  # 判断mpi是否可用
torch.distributed.is_gloo_available() # 判断gloo是否可用
```

获取当前进程的rank
```
torch.distributed.get_rank(group=None)  # group=None，使用默认的group
```

获取任务中（或者指定group）中，进程的数量

```
torch.distributed.get_rank(group=None)   # group=None，使用默认的group 
```

获取当前任务（或者指定group）的后端。
```
torch.distributed.get_backend(group=None)  # group=None，使用默认的group 
```

**进程内指定显卡**

目前很多场景下使用分布式都是默认一张卡对应一个进程，所以通常，我们会设置进程能够看到卡数： 下面例举3种操作的API，其本质都是控制进程的硬件使用。

```
# 方式1：在进程内部设置可见的device
torch.cuda.set_device(args.local_rank)
# 方式2：通过ddp里面的device_ids指定
ddp_model = DDP(model, device_ids=[rank]) 
# 方式3：通过在进程内修改环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = loac_rank
```