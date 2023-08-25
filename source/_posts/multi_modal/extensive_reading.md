---
title: 多模态的一些论文泛读
date: 2023-08-02 13:48:49
tags: multi modal
mathjax: true
---

# 1. 图文匹配

## 1.1 CLIP

论文：[Learning Transferable Visual Models From Natural Language Supervision](https://arXiv.org/abs/2103.00020)

利用来自自然语言对图像进行监督，优点是数据集可以很大，而不需要人工标注。借鉴 GPT 的思想，使用图文对（image text pair）预训练出一个大模型，然后可以 fine-tuning，zero-shot transfer 或者 linear probe 下游任务。

这种预训练称之为 Contrastive Language-Image Pre-training (CLIP) 。


**# 预训练**

给定一个 batch size 为 `B` 的图文对，得到 $B \times B$ 的图文对，其中对角线表示正例，其余表示负例，模型输出 $B \times B$ 个预测得分，这个得分矩阵第 `i` 行表示第 `i` 个图像与 `B` 个文本的匹配得分，使用 `softmax(dim=1)` 做归一化 ，类似地，第 `j` 列表示第 `j` 个文本与 `B` 个图像的匹配得分，使用 `softmax(dim=0)` 做归一化，计算损失的伪代码如下，

```python
# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]
# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)
# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

**# image encoder**

1. ResNet-D，在 ResNet 基础之上做小修改

    输出特征 shape 为 `(B, C, H, W)`，其中 `H, W` 是最后一层 layer 的输出特征 spatial size，`C` 是其输出 channel ，然后通过一个全局平均层，得到 `(B, C)`，表示 B 个图像的特征。

2. ViT

    图像经过 ViT 输出特征 shape 为 `(B, N+1, D)`，N 为图像切割后的 patches 数量，`+1` 是表示第一个位置增加 `<class>` 这个特殊 token，使用 `<class>` 对应的特征表示图像的特征，shape 为 `(B, D)`

**# text encoder**

1. transformer decoder

    - model dimension 为 512
    - heads 数量为 8
    - transformer layer 数量为 12
    - 输出特征 shape 为 `(B, L, D)`
    - 由于文本的开头和结尾分别使用了 `<sos>` 和 `<eos>` 两个特殊的 token，所以最终使用 `<eos>` 对应的特征向量来表示整个文本的特征，故文本特征 shape 为 `(B, D)`
    
文本输出特征的 D 与图像输出特征的 C/D 可能不同，这没关系，可以通过一个线性映射 layer，将维度统一映射为 `n_embd` ，例如上面代码中的 `W_i, W_t` 。

**# zero-shot transfer**

例如图像分类，每个图像对应一个分类名，这个分类名就作为这个图像的 text。假设数据集一共有 `C` 个分类名，那么对于每一个图像，得到 `C` 个图文对，计算这 `C` 个图文对的特征向量相似度，得到 $1 \times C$ 个预测得分，然后使用 `softmax(dim=1)` 进行归一化。

```python
# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)
```

**# linear probe**

与 fine-tuning 不同，linear probe 固定预训练模型的参数，将预训练模型的输出特征（以及原本的 label）作为数据集，使用一层线性变换进行分类，训练时仅仅学习这一线性层的参数。

例如 cifar-100 数据集，提取图像特征，

```python
def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            # (B, D)
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)
    # (N, D)
    # (N, )     N 是数据集 size
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
```

然后使用逻辑回归分类器，训练并预测，代码如下，

```python
# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
```

对于样本 $(\mathbf x, \mathbf y)$，其中 $\mathbf x \in \mathbb R ^ d$，$\mathbf y \in \mathbb N^C$ 为 one-hot 向量，逻辑回归分类器参数为 $W \in \mathbb R ^ {d \times C}$ 。