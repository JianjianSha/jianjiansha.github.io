---
title: L-Margin Softmax Loss
date: 2024-04-02 08:11:13
tags: face recognition
---

# 1. 简介

交叉熵损失与 softmax 是 CNN 网络的常见组合，尽管简单易用，但是对特征的区分不够好，一个好的特征是类之间分离尽可能大，类内尽可能紧凑。而 softmax 与交叉熵的组合并不鼓励模型学习这种类内紧凑类间分离的特征。