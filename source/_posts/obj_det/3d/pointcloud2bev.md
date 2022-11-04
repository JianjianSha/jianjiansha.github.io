---
title: 点云生成鸟瞰图
date: 2022-10-11 17:13:15
tags: 3d object detection
---

点云数据一般为 2 dim，shape 为 `(N, 4)`，其中 N 为 point 数量，4 表示 `(x,y,z,r)` 三维坐标 xyz 以及反射率 r。

# 1. 图像与点云坐标

如图 1，

![](/images/obj_det/3d/pointcloud2bev_1.png)

<center>图 1</center>

创建 BEV 图，需要从点云中提取 x 和 y 坐标值，

![](/images/obj_det/3d/pointcloud2bev_2.png)

图 2 [<sup>1</sup>](#refer-anchor-1)

需要注意以下几点：

1. 图像中的 x 和 y 分别对应点云中的 y 和 x
2. x 和 y 均指向相反的方向
3. 需要平移坐标，使得最小坐标值为 `(0, 0)`



# 2. 限制矩形区域

将注意力集中于点云的特殊区域，这样可以减少计算量。

矩形区域设置为距离原点两侧 10 米，前方 20 米，

```python
# 读取点云数据
points = np.fromfile('000009.bin', dtype=np.float32, count=-1).reshape([-1, 4])

side_range=(-10, 10)    # 点云中最左到最右
fwd_range=(0, 20)       # 点云中最后到最前
```

创建过滤器，进行过滤，

```python
x_points = points[:,0]
y_points = points[:,1]
z_points = points[:,2]

f_flt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
s_flt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
filter = np.logical_and(f_flt, s_flt)
indices = np.argwhere(filter).flatten()

# 保留点
x_points = x_points[indices]
y_points = y_points[indices]
z_points = z_points[indices]
```

# 3. 映射点到像素
点云数据为实数，而图像像素坐标为整数，若直接取整，则会损失大量数据，丢失分辨率。

如果计量单位是米，想要一个 5 cm 的分辨率，则

```python
res = 0.05

# 图像与点云 x 和 y 对调
x_img = (-y_points / res).astype(np.int32)
y_img = (-x_points / res).astype(np.int32)
```

# 4. 平移得左上角原点

平移使得左上角为原点，从而使得图像的 x y 坐标均非负，

```python
x_img -= int(np.floor(side_range[0] / res))
y_img += int(np.ceil(fwd_range[1] / res))
```

这里由于点云数据 -y 坐标范围为 `(side_range[0], side_range[1])`，-x 坐标范围为 `(-fwd_range[1], -fwd_range[0])`，所以第 3 步中映射到图像像素坐标后还需要平移 `-(side_range[0], -fwd_range[1])`，即 `(-side_range[0], fwd_range[1])` 。

# 5. 像素值

将点云 z 值（表示高度）填充到图像像素上去，设置一个像素值范围，小于或大于该范围则会被 clip。

调整像素值范围为 `[0, 255]`。

```python
height_range = (-2, 0.5)    # 最低到最高
pixel_values = np.clip(a=z_points, 
                       a_min=height_range[0], 
                       a_max=height_range[1])

def scale_255(a, min, max, dtype=np.uint8):
    return (((a - min) / float(max - min)) * 255).astype(dtype)

pixel_values = scale_255(pixel_values, min=height_range[0], max=height_range[1])
```

# 6. 创建图像数组

```python
x_max = 1 + int((side_range[1] - side_range[0]) / res)
y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
im = np.zeros([y_max, x_max], dtype=np.uint8)

im[y_img, x_img] = pixel_values
```

# 7. 可视化

```python
from PIL import Image

im2 = Image.fromarray(im)
im2.show()
```

为了看的更清楚，可以使用彩色图可视化，

```python
import matplotlib.pyplot as plt
plt.imshow(im, cmap='spectral', vmin=0, vmax=255)
plt.show()
```

# REF


<div id="refer-anchor-1"></div>

- [1] [Creating Birdseye View of Point Cloud Data](http://ronny.rest/tutorials/module/pointclouds_01/point_cloud_birdseye/)

