---
title: pytorch 技巧1
p: pytorch/tricks_1
date: 2021-01-08 09:45:15
tags: PyTorch
---
假设最后一层的 input shape 为 (N,H,W)，输出为 loss 为一标量，那么最后一层 input 的梯度 shape 为 (N,H,W)，与 input Tensor 自身 shape 相同，然后继续反向传播，倒数第二层的 input shape 为 (N',H',W')，
假设某层输入input 的 shape 为 (N,H,W)，输出 output 的 shape 为 (N',H',W')，这里为了叙述简单，不考虑 batch 的维度，反向传播时，output 变量的梯度的 shape 应该与 output 变量自身相同，output 对 input 的梯度 shape 应该为
(N',H',W',N,H,W)，与 