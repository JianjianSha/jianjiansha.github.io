---
title: tensorflow 模型保存
date: 2022-06-10 11:29:57
tags: tensorflow
---

```python
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model
model = create_model()
model.summary()
```

# 1. 保存模型权重

```python
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 通过回调方式，在每一轮训练完成后保存模型权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

# 重新创建模型，然后加载权重文件
model = create_model()
model.load_weights(checkpoint_path)
```

手动保存，但是这种方式无法与 `model.fit` 兼容

```python
# 手动保存
model.save_weights('./checkpoints/my_checkpoint')

# 重新创建模型
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')
```

# 2. 保存整个模型

有两种保存格式
1. `SavedModel` (TF2.X 的默认方式)
2. `HDF5`


## 2.1 SavedModel 
```python
model.fit(train_images, train_labels, epochs=5)
model.save('./checkpoints/my_model')

# 加载模型
model2 = tf.keras.models.load_model('./checkpoints/my_model')
model2.summary()

loss, acc = model2.evaulate(test_images, test_labels, verbose=1)
```

## 2.2 HDF5

```python
model.fit(train_images, train_labels, epochs=5)
model.save('./checkpoints/my_model.h5')

# 加载模型
model2 = tf.keras.models.load_model('./checkpoints/my_model.h5')
model2.summary()

loss, acc = model2.evaulate(test_images, test_labels, verbose=1)
```