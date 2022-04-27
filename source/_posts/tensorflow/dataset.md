---
title: tensorflow 中数据集构建
date: 2022-02-25 10:22:19
tags: tensorflow
categories: TensorFlow
summary: 对于各式各样的数据形式，如何高效的构建数据集对象？
---
# 1. 建立数据集

```python
tf.data.Dataset.from_tensor_slices()
```

返回 `tf.data.Dataset` 类对象。

适用于数据量较小的情况。例如装载 MNIST，

```python
(train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
for img, lbl in mnist_dataset:
    ...
```
需要注意，`train_data` 和 `train_label` 的第 `0` 维大小必须相同。

## 1.1 TFRecord

对于较大的数据集，则不能通过 `tf.data.Dataset.from_tensor_slices()` 来构建数据集，而是借助 `TFRecord` 来处理。

`TFRecord` 为 TensorFlow 中数据集存储格式，如下
```py
[
    {   # example 1 (tf.train.Example)
        'feature_1': tf.train.Feature,
        ...
        'feature_k': tf.train.Feature
    },
    ...
    {   # example N (tf.train.Example)
        'feature_1': tf.train.Feature,
        ...
        'feature_k': tf.train.Feature
    }
]
```

`TFRecord` 是一个 `tf.train.Example` 组成的 list，每个 `tf.train.Example` 对象包含若干个 feature。

**将数据集存储为 TFRecord**

```python
# 猫狗分类
data_dir = 'C:/datasets/cats_vs_dogs'
train_cats_dir = data_dir + '/train/cats'   # 猫图片目录
train_dogs_dir = data_dir + '/train/dogs'   # 狗图片目录
tfrecord_file = data_dir + 'train/train.tfrecords'

train_cat_filenames = [train_cats_dir + filename for filename in os.listdir(train_cats_dir)]
train_dog_filenames = [train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)]
train_filenames = train_cat_filenames + train_dog_filenames
train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)

with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for filename, label in zip(train_filenames, train_labels):
        image = open(filename, 'rb').read()
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'label': tf.train.Feature(int64_list=tf.train.BytesList(value=[label]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
```

**读取 TFRecord**

```python
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}
def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
    return feature_dict['image'], feature_dict['label']

dataset = raw_dataset.map(_parse_example)
```

返回的 `dataset` 是 `tf.data.Dataset` 实例。

## 1.2 预处理

**shuffle**

打乱数据集，
```python
def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None, name=None)
```
其中 `buffer_size` 指定缓冲区大小。

洗牌原理：

1. 取前 `buffer_size` 大小的数据样本，添加到缓冲区
2. 随机选取其中一个样本，作为第 `t=1` 个样本，并从缓冲区中移除这个样本
3. 从缓冲区以外的数据中取一个样本填充到缓冲区中被移除样本的位置。
4. 重复这样的操作，直到没有新样本填充到缓冲区，此时依次输出缓冲区样本

使用示例：

```python
dataset = tf.data.Dataset.range(6)
dataset = dataset.shuffle(3)
print(list(dataset.as_numpy_iterator()))
```

**batch**

```python
def batch(self, batch_size, drop_reminder=False, num_parallel_calls=None,
          deterministic=None, name=None)
```
将数据集分批迭代（而非单个样本迭代）。

参数说明：

1. `batch_size`: 批大小。样本数据在第 `0` 维度上进行 stack（非 concatenate，注意区别），如果样本数据是 `(x,y)` 的元组形式，那么分别对 `x` 和 `y` 进行 stack，得到的依然是形如 `(x,y)` 的元组。

2. `drop_remainder`：是否丢弃最后不足 `batch_size` 的样本。

3. `num_parallel_calls`：指定需要并行计算的 batch 数量。

4. `deterministic`：并行计算时产生 batch 的顺序是否确定？

**prefetch**

此方法在 GPU 计算的时候预加载下一 batch 的数据，从而充分利用计算资源。使用示例，
```python
mnist_dataset = mnist_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
```

完整示例：

```python
import tensorflow as tf
import os

num_epochs = 10
batch_size = 32
learning_rate = 0.001
data_dir = 'C:/datasets/cats_vs_dogs'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
test_cats_dir = data_dir + '/valid/cats/'
test_dogs_dir = data_dir + '/valid/dogs/'

def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)            # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label

if __name__ == '__main__':
    # 构建训练数据集
    train_cat_filenames = tf.constant([train_cats_dir + filename for filename in os.listdir(train_cats_dir)])
    train_dog_filenames = tf.constant([train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)])
    train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=-1)
    train_labels = tf.concat([
        tf.zeros(train_cat_filenames.shape, dtype=tf.int32), 
        tf.ones(train_dog_filenames.shape, dtype=tf.int32)], 
        axis=-1)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(
        map_func=_decode_and_resize, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
    train_dataset = train_dataset.shuffle(buffer_size=23000)    
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    model.fit(train_dataset, epochs=num_epochs)
```

1. ref

本文示例代码来源 [tf.wiki/zh_hans](https://tf.wiki/zh_hans/basic/tools.html)