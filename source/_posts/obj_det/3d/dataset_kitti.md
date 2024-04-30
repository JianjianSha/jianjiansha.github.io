---
title: Kitti 数据集简介
date: 2022-10-12 18:14:53
tags: 3d object detection
mathjax: true
---



# 1 几种坐标系

- 图像像素坐标系：表示三维空间物体在图像平面上的投影，像素是离散化的，其坐标原点在 CCD 图像平面的左上角，u 轴平行于CCD平面水平向右，v 轴垂直于 u 轴向下，坐标使用 (u,v) 来表示。图像宽度 W ，高度 H

- 图像物理坐标系：坐标原点在CCD图像平面的中心，x,y 轴分别平行于图像像素坐标系的 (u,v) 轴，坐标用 $(x_p,y_p)$ 表示。

- 相机坐标系：以相机的光心为坐标系原点，$x_c,y_c$ 轴平行于图像坐标系的 x,y 轴，相机的光轴为 $z_c$ 轴，坐标系满足右手法则，即 z->前

- 世界坐标系（world coordinate）：也称为测量坐标系，是一个三维直角坐标系，以其为基准可以描述相机和待测物体的空间位置。世界坐标系的位置可以根据实际情况自由确定。

### 1.1 世界坐标系转相机坐标系

$$\begin{bmatrix}x_c \\\\ y_c \\\\ z_c \\\\ 1\end{bmatrix}= \begin{bmatrix} \mathbf R & \mathbf t \\\\ \mathbf 0 & 1 \end{bmatrix} \begin{bmatrix} x_w \\\\ y_w \\\\ z_w \\\\ 1 \end{bmatrix} \tag{1}$$

世界坐标系与相机坐标系之间的转换，可以将其中物体视作刚性物体，那么转换只有旋转和平移两种操作，上式中 $\mathbf R$ 是旋转矩阵，$\mathbf t$ 是平移矩阵。

### 1.1.1 平移

点 A $(x_1, y_1, z_1)$ 平移 $(t_x, t_y, t_z)$ 后的坐标为

$(x_2, y_2, z_2) = (x_1+t_x, y_1+t_y, z_1+t_z)$ 

使用矩阵表示为

$$\begin{bmatrix} x_2 \\\\ y_2 \\\\ z_2 \\\\ 1 \end{bmatrix} = \mathbf t \begin{bmatrix} x_1 \\\\ y_1 \\\\ z_1 \\\\ 1\end{bmatrix}= \begin{bmatrix}  1 & 0 & 0 & t_x \\\\ 0 & 1 & 0 & t_y \\\\ 0 & 0 & 1 & t_z\\\\ 0 & 0 & 0 & 1\end{bmatrix}\begin{bmatrix} x_1 \\\\ y_1 \\\\ z_1 \\\\ 1\end{bmatrix} \tag{2}$$

### 1.1.2 旋转

**二维平面的旋转**

如图 1 

![](/images/obj_det/3d/dataset_kitti_1.png)
<center>图 1. 二维平面中的旋转</center>

向量 $\overrightarrow  {OA}= (x_1, y_1)$ 逆时针旋转 $\theta$ 角度后为 $\overrightarrow {OB}=(x_2, y_2)$，易知

$$\overrightarrow {OA}=\overrightarrow {OA}_x + \overrightarrow {OA}_y$$

$\overrightarrow {OA}_x=(x_1, 0)$ 旋转后为 $(x_1 \cos \theta, x_1 \sin \theta)$，$\overrightarrow {OA}_y=(0, y_1)$ 旋转后为 $(-y_1 \sin \theta, y_1 \cos \theta)$，所以

$$(x_2, y_2) = (x_1 \cos \theta, x_1 \sin \theta)+ (-y_1 \sin \theta, y_1 \cos \theta)$$

写成矩阵形式为

$$\begin{bmatrix} x_2 \\\\ y_2 \end{bmatrix} = \begin{bmatrix}\cos \theta  & -\sin \theta \\\\ \sin \theta & \cos \theta \end{bmatrix}\begin{bmatrix} x \\\\ y \end{bmatrix} \tag {3}$$

**三维空间中的旋转**

记旋转矩阵为 $\mathbf R$，将三维向量分别投影到 X-Y, Y-Z, X-Z 平面上，记 $\mathbf R$ 在 XY 平面上旋转角度为 $\theta_z$，在 YZ 平面上旋转角度为 $\theta_x$，在 XZ 平面上旋转角度为 $\theta_y$ （三个角度均指逆时针旋转角度），那么 $\mathbf R$ 可由三个角度 $(\theta_x, \theta_y, \theta_z)$ 表征，对应的三个旋转为 $(\mathbf R_x, \mathbf R_y, \mathbf R_z)$。

根据前面二维平面的旋转公式 (3) 易知，

$$\mathbf R_z = \begin{bmatrix} \cos \theta_z & -\sin \theta_z & 0 \\\\ \sin \theta_z & \cos \theta_z & 0 \\\\ 0 & 0 & 1\end{bmatrix}, \ \mathbf R_y = \begin{bmatrix} \cos \theta_y & 0& \sin \theta_y  \\\\ 0 & 1 & 0 \\\\ -\sin \theta_y &0& \cos \theta_y\end{bmatrix}, \ \mathbf R_x = \begin{bmatrix} 1 & 0 & 0 \\\\ 0 & \cos \theta_x &-\sin \theta_x  \\\\ 0& \sin \theta_x & \cos \theta_x\end{bmatrix}$$

且有关系

$$\mathbf R = \mathbf R_x \mathbf R_y \mathbf R_z \tag{4}$$

### 1.1.3 刚体变换

使用 $\mathbf T$ 表示刚体变换，刚体变换=平移+旋转，可以用分块矩阵表示如下，

$$\mathbf T = [\mathbf R | \mathbf t] \tag{5}$$

为社么可以用分块矩阵表示呢？因为先后两个变换可以用矩形相乘表示，

$$\begin{aligned}\mathbf T &= \mathbf {t R}
\\\\&= \begin{bmatrix} 1 & 0 & 0 & t_x \\\\ 0 & 1 & 0 & t_y \\\\ 0 & 0 & 1 & t_z \\\\ 0 & 0 & 0 &1\end{bmatrix}\begin{bmatrix} r_1 & r_2 & r_3 & 0 \\\\  r_4 & r_5 & r_6 & 0 \\\\  r_7 & r_8 & r_9 & 0 \\\\ 0 & 0 & 0 & 1 \end{bmatrix}
\\\\ &= \begin{bmatrix} r_1 & r_2 & r_3 & t_x \\\\  r_4 & r_5 & r_6 & t_y \\\\  r_7 & r_8 & r_9 & t_z \\\\ 0 & 0 & 0 & 1 \end{bmatrix}=\begin{bmatrix} \mathbf R & \mathbf t \\\\ \mathbf 0_{1 \times 3} & 1\end{bmatrix}
\end{aligned} \tag{6}$$

使用齐次坐标 $[x, y, z, 1]^{\top}$ 就使用 (6) 式的变换矩阵，而 $[x, y, z]^{\top}$ 坐标则使用 (5) 式变换矩阵。

易知，使用 $\mathbf {R t}$ 结果与上面的一样，平移与旋转两个操作的先后顺序无关紧要。

从世界坐标系到相机坐标系的刚体变换矩阵 $\mathbf T$ 也称为 **相机外参**（camera extrinsics）

$$\begin{bmatrix} X_c \\\\ Y_c \\\\ Z_c \\\\ 1\end{bmatrix}= \mathbf T \begin{bmatrix}X_w \\\\ Y_w \\\\ Z_w \\\\ 1 \end{bmatrix}$$

## 1.2 相机坐标系到图像坐标系

空间点 P 在相机坐标系的齐次坐标为 $[x_c, y_c, z_c, 1]^{\top}$，其像点在图像坐标系的齐次坐标为 $[x_p, y_p, 1]^{\top}$，根据针孔成像原理，如图 2,3,4

![](/images/obj_det/3d/dataset_kitti_2.png)

<center>图 2. 相机的针孔成像图</center>

![](/images/obj_det/3d/dataset_kitti_3.png)
<center>图 3. 相机坐标系和图像坐标系对应关系</center>

![](/images/obj_det/3d/dataset_kitti_4.png)
<center>图 4. 考虑 Y-Z 平面的三角相似</center>



根据三角形的相似性可知

$$\begin{cases} x_p = f \frac {x_c}{z_c} \\\\ y_p = f \frac {y_c}{z_c} \end{cases}$$

矩阵变换形式为

$$z_c \begin{bmatrix} x_p \\\\ y_p \\\\ 1 \end{bmatrix} = \begin{bmatrix} f & 0 & 0 & 0 \\\\ 0 & f & 0 & 0 \\\\ 0 & 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} x_c \\\\ y_c \\ z_c \\\\ 1 \end{bmatrix} =[\mathbf K | \mathbf 0]  \begin{bmatrix} x_c \\\\ y_c \\ z_c \\\\ 1 \end{bmatrix} \tag{7}$$

$f$ 是相机焦距。

## 1.3 图像坐标系转像素坐标系

![](/images/obj_det/3d/dataset_kitti_5.png)

<center>图 5. (u,v) 像素坐标系，(X,Y) 图像坐标系</center>

像素坐标系不利于坐标变换，因此需要建立图像坐标系，其坐标轴的单位通常为毫米（mm），原点是相机光轴与相面的交点（称为主点），即图像的中心点，X 轴、Y 轴分别与 u 轴、 v 轴平行。故两个坐标系实际是平移关系，即可以通过平移就可得到。

图像坐标系转为像素坐标系：

$$\begin{bmatrix}u \\\\ v \\\\  1\end{bmatrix}= \begin{bmatrix} 1/dx & 0 & u_0 \\\\  0 & 1/dy & v_0 \\\\ 0&0&1 \end{bmatrix} \begin{bmatrix} x_p \\\\ y_p \\\\ 1 \end{bmatrix}\tag{8}$$

dx,dy 表示相邻像素之间的物理距离（例如单位长度为 mm），实际上 CCD 相机上每个像素对应一个感光点。

(8) 式中，$x=0$ 对应图像坐标系的中心点，对应 $u=u_0$ 即像素坐标系中心点。

假设图像的物理宽高为 $w, h$，那么 $\frac w {2 \cdot dx} = u_0, \ \frac h {2 \cdot dy}=v_0$ 。

## 1.4 综合

综合以上，从世界坐标系到像素坐标系的变换为

$$\begin{aligned}z_c\begin{bmatrix}u \\\\ v \\\\  1\end{bmatrix}&=\begin{bmatrix} 1/dx & 0 & u_0 \\\\  0 & 1/dy & v_0 \\\\ 0&0&1 \end{bmatrix}\begin{bmatrix} f & 0 & 0 & 0 \\\\ 0 & f & 0 & 0 \\\\ 0 & 0 & 1 & 0 \end{bmatrix}\begin{bmatrix} \mathbf R & \mathbf t \\\\ \mathbf 0_{1\times 3} & 1\end{bmatrix}\begin{bmatrix}x_w \\\\ y_w \\\\ z_w \\\\ 1\end{bmatrix}
\\\\ &=\begin{bmatrix} f_x & 0 & u_0 & 0 \\\\  0 & f_y & v_0 & 0 \\\\ 0&0&1 &0\end{bmatrix}\begin{bmatrix} \mathbf R & \mathbf t \\\\ \mathbf 0_{1\times 3} & 1\end{bmatrix}\begin{bmatrix}x_w \\\\ y_w \\\\ z_w \\\\ 1\end{bmatrix}
\end{aligned} \tag{9}$$

$\begin{bmatrix} f_x & 0 & u_0 & 0 \\\\  0 & f_y & v_0 & 0 \\\\ 0&0&1 &0\end{bmatrix}$ 为 **相机内参**（camera intrinsics）。 $\begin{bmatrix} \mathbf R & \mathbf t \\\\ \mathbf 0_{1\times 3} & 1\end{bmatrix}$ 为相机外参。相机标定就是为了求解这两个矩阵的参数。


# 2. 3D 目标检测

数据下载：

[KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php)

[KITTI 3d object](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

几种坐标系图示

![](/images/obj_det/3d/dataset_kitti_6.png)

<center>图 6. 坐标系</center>

传感器与坐标轴的关系

|Sensor|X轴正向|Y轴正向|Z轴正向|
|--|--|--|--|
|camera|右|下|前|
|LiDAR|前|左|上|
|GPS/IMU|前|左|上|



## 2.1 left color images

left color image 的图像方向：x->右，y->下，z->前


## 2.2 velodyne point clouds

velodyne 激光雷达数据。 velodyne 激光雷达指标为，角度分辨率 0.09°，距离精度 2cm。视场角（FOV）：360° 水平，26.8° 垂直。测量范围：120m 

下载的 velodyne 数据包里面都是 bin 文件，其中存储了点云数据，每个点 `(x,y,z,r)` 表示 xyz 坐标和反射强度，坐标系方向：x->前，y->左，z->上。

以 "000001.bin" 为例（其中 000001 是 id，图像，点云，标定，label 等文件），velodyne 文件内容如下，

```
7b14 4642 1058 b541 9643 0340 0000 0000
46b6 4542 1283 b641 3333 0340 0000 0000
4e62 4042 9643 b541 b072 0040 cdcc 4c3d
8340 3f42 08ac b541 3bdf ff3f 0000 0000
e550 4042 022b b841 9cc4 0040 0000 0000
10d8 4042 022b ba41 4c37 0140 0000 0000
3fb5 3a42 14ae b541 5a64 fb3f 0000 0000
7dbf 3942 2731 b641 be9f fa3f 8fc2 f53d
cd4c 3842 3f35 b641 4c37 f93f ec51 383e
dbf9 3742 a69b b641 c3f5 f83f ec51 383e
2586 3742 9a99 b741 fed4 f83f 1f85 6b3e
...
```

点云数据以浮点二进制文件格式存储，每行包含8个数据，每个数据由四位十六进制数表示（浮点数），每个数据通过空格隔开。一个点云数据由四个浮点数数据构成，分别表示点云的x、y、z、r（强度 or 反射值），点云的存储方式如下表所示，

```
pointcloud-1    |   pointcloud-2
x   y   z   r   |   x   y   z   r
pointcloud-3    |   pointcloud-4
x   y   z   r   |   x   y   z   r
...
```

代码示例，

```python
def get_lidar(filepath):
    '''filepath: lidar file(bin 文件)'''
    np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
```


## 2.3 camera calibration matrices

相机雷达标定文件，主要用于将激光雷达投影在图像上显示。以 "000001.txt" 文件为例，内容如下

    ```python
    P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
    P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
    P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
    P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03
    R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
    Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
    Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01
    ```

### 2.3.1 投影矩阵

0,1,2,3 表示相机编号。 0 为左边灰度相机，1 为右边灰度相机，2 为左边彩色相机，3 为右边彩色相机。P0~P3 是修正（rectified）后的相机投影矩阵 $R^{3 \times 4}$ ，也就是将 **第 0 相机 rectified 坐标系** 的点变换为 **第 i 个相机的图像坐标（实际是像素坐标）** 中，参考上面 (9) 式中的相机内参矩阵。

需要注意的是，Pi 除了相机内参（投影变换），还包含了一个平移操作，表示第 `i` 个相机到 `0` 号相机摄像头的距离偏移，这是因为投影变换是同一个相机的相机坐标转图像（像素）坐标，而这里是 A 相机的相机坐标转 B 相机的图像（像素）坐标，所以需要组合两个先后顺序操作：1. A 相机到 B 相机的相机坐标系平移操作 $T_{AB}$；2. 相机的投影操作 $K$。用矩阵表示为

$$\mathbf P_i=\begin{bmatrix} f_x & 0 & u_0 & -f_x b_x \\\\ 0 & f_y & v_0 & -f_y b_y \\\\ 0 & 0 & 1 & -b_z \end{bmatrix} \tag{10}$$

其中 $b_x, \ b_y, \ b_z$ 分别表示表示第 `i` 个相机到 `0` 号相机摄像头的 x，y，z 方向的距离偏移。由于四个相机均相同，故四个相机的 $f_x, f_y, u_0, v_0$  均分别相等（即相机内参矩阵均相等），只是四个相机的安装位置不同，所以需要用 $b_x, b_y, b_z$ 表示。

**例子 1**

例如 A 表示 0 号机，B 表示 2 号机，A 的（修正后）相机坐标系中某点 P 的坐标记为 $[x_a, y_a, z_a]^{\top}$， B 到 A （B 相对 A）的摄像头距离为 $T_{AB}=[b_x, b_y, b_z]^{\top}$，那么 P 平移到 B 相机坐标系中坐标为 

$$\begin{aligned} \begin{bmatrix}x_b\\\\ y_b\\\\ z_b\end{bmatrix}  &= \begin{bmatrix}x_a\\\\ y_a\\\\ z_a\end{bmatrix}-T_{AB}  \qquad (\vec {BP}=\vec {AP}- \vec {AB})
\\\\&= \begin{bmatrix}x_a-b_x\\\\ y_a-b_y\\\\ z_a-b_z\end{bmatrix}
\end{aligned}$$

B 的相机坐标系转为 B 的图像（像素）坐标系的投影变换，参考 (9) 式可知，为

$$\begin{aligned} z_a \begin{bmatrix} u_b \\\\ v_b \\\\ 1 \end{bmatrix}&=\begin{bmatrix} f_x & 0 & u_0 & 0 \\\\ 0 & f_y & v_0 & 0 \\\\ 0 & 0 & 1 & 0\end{bmatrix}\begin{bmatrix}x_b\\\\ y_b\\\\ z_a\\\\1\end{bmatrix}
\\\\&=\begin{bmatrix} f_x & 0 & u_0 & 0 \\\\ 0 & f_y & v_0 & 0 \\\\ 0 & 0 & 1 & 0\end{bmatrix}\begin{bmatrix}x_a-b_x\\\\ y_a-b_y\\\\ z_a\\\\1\end{bmatrix}
\\\\&=\begin{bmatrix} f_x & 0 & u_0 & -f_x b_x \\\\ 0 & f_y & v_0 & -f_y b_y \\\\ 0& 0&1 & 0\end{bmatrix}\begin{bmatrix}x_a\\\\ y_a\\\\ z_a\\\\1\end{bmatrix}
\end{aligned} \tag{11}$$

上式就是从 A 相机坐标变换到 B 图像（像素）坐标的变换过程，注意深度值使用的是 A 相机中的 Z 轴值 $z_a$。将上式中的变换矩阵写成 $\mathbf P_i$ ，那么上式等式关系可写为

$$\begin{bmatrix} z_au_b \\\\ z_av_b \\\\ z_b \end{bmatrix}=\mathbf P_i \begin{bmatrix}x_a\\\\ y_a\\\\ z_a\\\\1\end{bmatrix} \tag{12}$$

下文的标定类方法 `rect_to_img` 中，就是使用上式先得到具有 `z_a` 缩放因子的 B 相机的图像像素坐标，然后除以 `z_a` 得到最终 B 的图像像素坐标 $u_b, v_b$，并通过 $z_b - (-b_z)=z_a$ 得到 A 相机坐标系中点的深度值。具体参见下文的代码注释。

根据 (11) 式，可知 $z_a u_b = f_x x_a + u_0 z_a - f_x b_x$， 变换得

$$x_a = (u_b-u_0)z_a/f_x + b_x \tag{13}$$

类似地可计算出 

$$y_a=(v_b-v_0)z_a /f_y + b_y \tag{14}$$

这就是从 2 号机图像像素坐标系变换到 0 号机的相机坐标系。

### 2.3.2 其他变换矩阵

R0_rect： 0号相机的修正矩阵， size 为 $R^{3 \times 3}$，将 **第 0 相机的相机坐标系** 中的点变换为 **第 0 相机的 rectified 坐标系** 中，由于相机的不正，所以需要对相机坐标系中的各点进行修正。实际计算中常常扩展为

$\begin{bmatrix} \mathbf R_{rect}^0 & \mathbf 0_{3 \times 1} \\\\ \mathbf 0_{1\times 3} & 1\end{bmatrix}$。

Tr_velo_to_cam 是雷达到相机的旋转平移矩阵，参见上面 (6) 式。

Tr_imu_to_velo 是惯导到雷达的旋转平移矩阵

具体说明：

label 文件中的 3d bbox 坐标都是位于第 0 相机的 rectified（矫正后？）坐标，将 label 中的某点 $\mathbf x$ （rectified 坐标系）变换到相机 i 的图像坐标系中（x->右，y->下，z->前），

$$\mathbf y_i =\mathbf P_i \mathbf x \tag{15}$$

将第 0 相机（即参考相机）坐标系下的点 $\mathbf x_0$ 变换到其 rectified 坐标系中，

$$\mathbf x = \mathbf R_{rect}^0 \mathbf x_0 \tag{16}$$

将 lidar 坐标系中的点 $\mathbf x_{velo}$ 变换到相机 0 的 **相机坐标系** 中，

$$\mathbf x_0 = \mathbf T_{velo2cam} \mathbf x_{velo} \tag{17}$$

将 lidar 坐标系中的点变换到第 i 相机的 **图像坐标系** 中，

$$\mathbf y_i = \mathbf P_i \mathbf R_{rect}^0 \mathbf T_{velo2cam} \mathbf x_{velo} \tag{18}$$

将第 0 相机的 rectified 坐标系中的点变换到 lidar 坐标系，

$$\mathbf x_{velo} = \mathbf T_{velo2cam}^{-1} (\mathbf R_{rect}^0)^{-1} \mathbf x \tag{19}$$

处理标定文件的代码，

```python
def get_calib_from_file(filepath):
    '''
    读取标定文件。功能：将激光雷达坐标系中的点投影到彩色图像（P2,P3）中，参见 (18) 式
    但是 (18) 式中的变换矩阵都是扩展到 4x4 大小。
    filepath: 标定文件路径
    '''
    with open(filepath) as f:
        lines = f.readlines()
    obj = lines[2].strip().split(' ')[1:]       # 2 号相机，即左边彩色的投影矩阵left-color
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]       # 右边彩色相机的投影矩阵
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]       # 旋转矩阵
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]       # 雷达到相机的旋转平移矩阵
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

class Calibration:
    def __init__(self, filepath):
        calib = get_calib_from_file(filepath)
        self.P2 = calib['P2']               # (3, 4)
        self.R0 = calib['R0']               # (3, 3)
        self.V2C = calib['Tr_velo2com']     # (3, 4)
        
        # == 相机内参，参见上面 (9) 式 ==
        self.cu = self.P2[0, 2] # u0
        self.cv = self.P2[1, 2] # v0
        self.fu = self.P2[0, 0] # fx
        self.fv = self.P2[1, 1] # fy
        # == 相机内参，参见上面 (9) 式 ==
        # == (10) 式中的 bx 和 by ==
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)
        # == (10) 式中的 bx 和 by ==

    def cart_to_hom(self, pts):
        '''
        转为齐次坐标。(x,y,z) -> (x,y,z,1)
        pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        '''
        pts_hom = np.hstack(pts, np.ones((pts.shape[0], 1), dtype=np.float32))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        '''
        雷达坐标转为 0 号机的修正坐标。 x_rect = R0_rect · Tr_velo_to_cam · x_lidar
        pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        '''
        pts_lidar_hom = self.cart_to_hom(pts_lidar)     # (N, 4)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T)) # (N, 3)
        return pts_rect
    
    def rect_to_img(self, pts_rect):
        '''
        0 号相机修正坐标到 2 号相机图像像素坐标
        pts_rect: (N, 3), 0 号机的修正坐标
        :return pts_img: (N, 2)，2 号机的图像像素坐标
                pts_rect_depth: (N,)，0 号机修正坐标系的深度（z坐标）
        '''
        pts_rect_hom = self.cart_to_hom(pts_rect)       # (N, 4)
        # (12) 式，转为经 z_a 缩放后的 2 号相机像素坐标
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)    # (N, 3)
        # (12) 式，除以 z_a，得到像素坐标 [u,v]
        pts_img = (pts_2d_hom[:,0:2].T / pts_rect_hom[:,2]).T # (N, 2)
        # (12) 式，2号相机图像坐标系深度+2号相对0号相机的深度偏差->得到 0 号相机（修正坐标系中）的深度
        # 参考上面例子1，则是 z_a = z_b + b_z
        pts_rect_depth = pts_2d_hom[:,2] - self.P2.T[3, 2]  # (N,)
        return pts_img, pts_rect_depth
    
    def lidar_to_img(self, pts_lidar):
        '''
        雷达坐标转为 2 号机图像像素坐标
        pts_lidar: (N, 3)，雷达坐标系
        :return pts_img: (N, 2)，2 号机图像像素坐标
                pts_depth: (N, 1)， 0 号机的深度
        '''
        pts_rect = self.lidar_to_rect(pts_lidar)    # 0 号机修正坐标
        pts_img, pts_depth = self.rect_to_img(pts_rect) # 2 号机图像像素坐标
        return pts_img, pts_depth
    
    def img_to_rect(self, u, v, depth_rect):
        '''
        u: 像素坐标系 u 轴坐标, (N,)，2 号机图像像素坐标 u_b
        v: 像素坐标系 v 轴坐标, (N,)，2 号机图像像素坐标 v_b
        depth_rect: 0号机深度 z_a, (N,)，0 号机 z 坐标
        :return pts_rect: (N, 3)，0 号机相机坐标系
        '''
        # (13) (14) 两式
        x = ((u-self.cu) * depth_rect) / self.fu + self.tx      # (N,)
        y = ((v-self.cv) * depth_rect) / self.fv + self.ty      # (N,)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)  # (N, 3)
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        '''
        深度图转为 0 号机修正后的相机坐标
        depth_map: (H, W)
        :return pts_rect: (HW, 3)
        '''
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]   # (HW,)
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def corners3d_to_img_boxes(self, corners3d):
        '''
        3D 角点修正相机坐标转图像像素坐标
        corners3d: (N, 8, 3) corners in rect coord . 长方体有 8 个顶点
        :return boxes: 图像像素坐标 bounding box x1y1x2y2，(N, 4)
                boxes_corner: 图像像素坐标，3d bbox corner，(N, 8, 2)
        '''
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)    # (N, 8, 4)
        # (12) 式，转为经 z_a 缩放后的 2 号相机像素坐标，z_a 是 0 号机修正坐标系中点的 z 坐标
        # 即每个点变换后的坐标为 [z_a*u_b, z_a*v_b, z_b]
        img_pts = np.matmul(corners3d_hom, self.P2.T)   # (N, 8, 3)
        # 获取 2 号机图像像素坐标（为什么是除以 img_pts[:,:,2] ，而不是除以 corners3d_hom[:,:,2]？）
        x, y = img_pts[:,:,0] / img_pts[:,:,2], img_pts[:,:,1] / img_pts[:,:,2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(y, axis=1), np.max(y, axis=1)

        # (N, 4)
        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)   # (N, 8, 2)
        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        '''
        u: 2 号机图像像素坐标系，点的 u 坐标，(N,)
        v: 2 号机图像像素坐标系，点的 v 坐标，(N,)
        d: 点与相机之间的距离，d^2=x^2+y^2+z^2
        :return pts_rect: 相机修正后坐标系的点坐标， (N, 3)
        '''
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        # 统一到像素坐标系中，计算距离
        fd = np.sqrt((u - self.cu)**2 + (v - self.cv)**2 + self.fu**2)  # (N,)
        # 根据 (13) (14) 式
        x = ((u - self.cu) * d) / fd + self.tx      # (N,)
        y = ((v - self.cv) * d) / fd + self.ty      # (N,)
        z = np.sqrt(d**2 - x**2 - y**2)             # (N,)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return pts_rect
```

这里稍微解释一下 `camera_dis_to_rect` 方法。根据 (13) (14) 式，可根据 2 号机（因为上述代码片段中使用 P2）图像像素坐标计算出 0 号机的修正后相机坐标系中点坐标，但是需要注意 (13) 式中 $z_a$ 表示 0 号机相机修正坐标系中点的 z 坐标，这在 `camera_dist_to_rect` 是未知的，所以不能直接使用 (13) (14) 式计算 x，y 坐标。

已知 $f_x=f/dx=f_y=f/dy$，记 $\delta = dx = dy$， 根据图 4，三角形相似可知，

$$\frac {y_p}{y_a}=\frac f{z_a}= \frac l d$$

其中 $(x_a, y_a, z_a)$ 表示 0 号机的相机坐标系中点坐标，下标 a 遵循了上面例子 1 中的 A 相机，即 0 号相机。$y_p$ 是 0 号相机的图像坐标系中点的 Y 轴坐标，下标 p 表示 picture，即图像坐标系（非像素坐标系），$f$ 是焦距（X,Y 轴焦距相等）。 l 和 d 分别是小大两个相似直角三角形的斜边边长。
另外，根据 (8) 式可知

$$v=\frac {y_p}{dy} + v_0 \Rightarrow v-v_0=\frac {y_p}{\delta}$$

类似地可得 $u-u_0=x_p/ \delta$，于是 $l / \delta = \sqrt { (u-u_0)^2+(v-v_0)^2}$

所以 

$$\frac {z_a}{f_y}= \frac {z_a} {f} \delta=\frac d l \delta= d / \sqrt { (u-u_0)^2+(v-v_0)^2}$$

注意，以上三式中的 $u, v , u_0, v_0$ 是点 p 和图像中心点的图像像素坐标系中的坐标值，这个像素坐标系可以是基于 0 号相机，也可以是基于 2 号相机，因为这两个像素坐标系仅仅通过一个平移即可相互转换，所以无论采用哪个相机的像素坐标系， $u-u_0$ 和 $v-v_0$ 均保持不变。上述代码中使用的是 2 号相机像素坐标系中的坐标。

于是根据 (14) 式，有 $y_a=(v_b-v_0)z_a /f_y + b_y=(v_b-v_0)d / \sqrt { (u-u_0)^2+(v-v_0)^2}+b_y$ 。$x_a$ 的计算类似。这就是 `camera_dis_to_rect` 方法实现的数学原理。




## 2.4 training labels
以 "000001.txt" 为例，内容为

```
type  trunc occlud alpha  x1     y1     x2     y2     h    w    l     x    y    z     rotation_y
Truck 0.00  0       -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56
Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
Cyclist 0.00 3 -1.65 676.60 163.95 688.98 193.93 1.86 0.60 2.02 4.59 1.32 45.84 -1.55
DontCare -1 -1 -10 503.89 169.71 590.61 190.13 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 511.35 174.96 527.81 187.45 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 532.37 176.35 542.68 185.27 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 559.62 175.83 575.40 183.15 -1 -1 -1 -1000 -1000 -1000 -10
```

每个图片对应于一个label 文件， 每一行代表一个object, 且有16个数据，分别为

1. 第一列(字符串) : 目标类别。 KITTI总共有9各类别:Car、Van、Truck、Pedestrian、Person_sitting、Cyclist、Tram、Misc、DontCare。其中DontCare标签表示该区域没有被标注，比如由于目标物体距离激光雷达太远。为了防止在评估过程中（主要是计算precision），将本来是目标物体但是因为某些原因而没有标注的区域统计为假阳性(false positives)，评估脚本会自动忽略DontCare区域的预测结果。

2. 第2列（浮点数）：代表物体是否被截断（truncated）数值在0（非截断）到1（截断）之间浮动，数字表示指离开图像边界对象的程度。

3. 第3列（整数）：代表物体是否被遮挡（occluded）。整数0、1、2、3分别表示被遮挡的程度。0: 全部可见；1: 部分遮挡；2: 大面积遮挡；3: 未知

4. 第4列（弧度数）：物体的观察角度（alpha）。取值范围为：-pi ~ pi（单位：rad），它表示在相机坐标系下，以相机原点为中心，相机原点到物体中心的连线为半径，将物体绕相机y轴旋转至相机z轴，此时物体方向与相机x轴的夹角，如图 7 所示。实际上也就是物体朝向与切线方向的夹角。

5. 第5~8列（浮点数）：物体的2D边界框大小（bbox）。四个数分别是xmin、ymin、xmax、ymax（单位：pixel），表示2维边界框的左上角和右下角的坐标。

6. 第9~11列（浮点数）：3D物体的尺寸（dimensions）。 分别是高、宽、长（单位：米）

7. 第12-14列（浮点数）：3D物体的位置（location）。 分别是x、y、z（单位：米），特别注意的是，这里的xyz是在相机坐标系下3D物体的中心点位置。

8. 第15列（弧度数）：3D物体的空间方向（rotation_y）。 取值范围为：$[-\pi, \pi)$（单位：rad），它表示，在照相机坐标系下，物体的全局方向角（物体前进方向 Z' 与相机坐标系 Z 轴的夹角），也就是说 rotation_y 表示 绕相机 Y 轴旋转的角度 使得 Z 轴 变成物体的朝向 Z’ 轴，顺时针转动为正值，逆时针为负值。（参考 2 中说是与 X 轴的夹角，但是我怀疑说的是 LiDAR 坐标系）

    如图 8 ，我们给出 camera 坐标系下的三种旋转角度，根据右手定则确定正向：右手握住旋转所绕的轴，拇指为此轴正向，然后四指所指方向为正。例如图 11，右手握住 Y 轴，拇指向下，那么物体朝向顺时针旋转了 rotation_y 角度，与四指所指方向一致，故判断 rotation_y 为正。

9. 第16列（浮点数）：检测的置信度（score）。要特别注意的是，这个数据只在测试集的数据中有（待确认）。

![](images/obj_det/3d/dataset_kitti_7.png)

<center>图 7.</center>


![](images/obj_det/3d/dataset_kitti_8.png)

<center>图 8</center>

![](images/obj_det/3d/dataset_kitti_9.png)
<center>图 9. 来源：kitti 官网</center>

![](images/obj_det/3d/dataset_kitti_10.png)
<center>图 10. 来源：kitti 官网</center>

![](images/obj_det/3d/point_rcnn_4.png)
<center>图 11</center>


图 9 和 10 是展示了各个传感器和各自的部署位置以及相应的坐标系。

label 处理类 `Object3d` 代码，

```python
def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class Object3d:
    def __init__(self, line):
        '''line: label 文件中的一行'''
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])   # 被截断程度
        self.occlusion = float(label[2])    # 被遮挡程度
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])    # 高
        self.w = float(label[9])    # 宽
        self.l = float(label[10])   # 长
        self.pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)# 物体中心点位置 (x,y,z)
        self.dis_to_cam = np.linalg.norm(self.pos)  # 物体中心点距离相机的距离（相机坐标系）
        self.ry = float(label[14])
        self.score = float(label[15]) if len(label) == 16 else -1.0     # 检测置信度
        self.level_str = None
        self.level = self.get_obj_level()   # 3d 目标检测的难易程度。物体越高，截断越小，遮挡越小，那么检测越容易
```

## 2.5 road plane

有时候训练集中还包含道路平面的 label 文件，位于目录 `KITTI/object/training/planes` 下，这个文件夹包含 ego vehicle 的 ground planes，每个相机拍摄图片，均有一个相应的 ground plane 文件，例如 `000000.txt`，内容为

```sh
# Matrix
WIDTH 4
HEIGHT 1
-7.051729e-03 -9.997791e-01 -1.980151e-02 1.680367e+00 
```

其中 `WIDTH 4` 和 `HEIGHT 1` 表示是 gt 是 $1 \times 4$ 的矩阵，矩阵为

$$[-7.051729e-03,\ -9.997791e-01,\ -1.980151e-02, \ 1.680367e+00]$$

前三个数表示 plane 的法向量，这是根据车辆的 IMU 的数据 （roll，pitch， yaw）计算出来的，且法向量总是接近 $[0, \ -1, \ 0]$，这表示 plane 几乎平整（法向量沿 Y 轴负方向，即竖直向上）。第四个数是相机相对 ground plane 的高度。

1. 所有相机高度: 1.65m
2. velodyne 激光扫描器高度: 1.73m
3. GPS/IMU 高度: 0.93m

# 3. 车道线检测

数据集：[Road/Lane Detection](https://www.cvlibs.net/datasets/kitti/eval_road.php)

road & lane estimation benchmark 包含 289 的训练图片和 290 个测试图片。道路场景包含 3 种分类：
1. uu - urban unmarked (98/100) （98 training / 100 testing）
2. um - urban marked (95/96)
3. umm - urban multiple marked lanes (96/94)
4. urban - 以上三种的合集

如下图，

![](/images/obj_det/3d/dataset_kitti_11.jpg)

ground truth 由人工标注，有两种道路地形：

1. road - 道路区域，即，所有的 lanes（所有车道）
2. lane - ego-lane，即，当前 ego vehicle 所行驶的车道（仅在 um 分类下的图片中存在）

ground truth 仅在训练集中提供。



# Ref

1. https://zhuanlan.zhihu.com/p/493026799

2. https://blog.csdn.net/boon_228/article/details/125925739

3. 代码参考 https://github.com/sshaoshuai/PointRCNN.git