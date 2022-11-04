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

reference：

1. https://github.com/tczhangzhi/pytorch-distributed/tree/master

# 4. 原理

```python
x = torch.arange(1, 4)
y = x ** 2
z = torch.sum(y)
z.backward()

```