---
title: 3D 目标检测中的一些计算
date: 2022-11-21 11:28:56
tags: 3d object detection
mathjax: true
---

# 1. IoU

2D 目标检测中常用到 IOU 进行 NMS 操作。由于 2D 目标检测中 bbox 都是 axis aligned，所以计算 IOU 非常简单直观，如图 1 (a)，

$$IoU(A,B)=\frac {A \cap B}{A \cup B}=\frac {A \cap B}{|A|+|B|-A \cap B} \tag{1}$$

然而，对于 3D 目标检测，情况复杂的多。首先我们考虑将 box3d 转换为 bev，这在自动驾驶等场景中常见，因为物体只有绕 Y 轴（竖直向下）的旋转（rotate_y 角度），所以可以转换为 bev，这样就 3D IoU 就转为 2D IoU 计算了，但是由于存在朝向角，所以通常不是 axis-aligned 的 box，如图 1 (b)，

![](/images/obj_det/3d/utils_1.png)
<center>图 1. (a) axis-aligned boxes；(b) rotated boxes</center>

## 1.1 BEV rotated boxes IoU

计算 rotated boxes 的 IoU：

### 1.1.1 计算两个 boxes 的面积

$$|\mathbf a| = \sqrt {(x_2-x_1)^2+(y_2-y_1)^2}, \ |\mathbf b| = \sqrt {(x_2-x_3)^2+(y_2-y_3)^2} \tag{2}$$

于是红色 box 面积为

$$A=|\mathbf a|\times |\mathbf b| \tag{3}$$

类似地可知蓝色 box 面积为

$$B=\mathbf a'|\times |\mathbf b'| \tag{4}$$

注意这里计算长度（欧氏距离）跟选用哪个坐标系无关。

### 1.1.2 计算 overlap 区域面积

确定 overlap 区域的顶点，由于凸多边形的交仍是凸多边形，所以 overlap 是一个凸多边形，选择多边形内部某点，将多边形所有顶点绕这个内部点按顺序排序（例如逆时针），然后将多边形划分为若干个三角形，计算三角形面积之和，如图 1 (c) 所示，这样就能得到 overlap 区域面积。

需要注意的是，由于两个 box corners 的坐标所基于的坐标系不同（图 1 (b) 中已经画出），所以要计算多边形顶点坐标和面积，需要先将 box corners 坐标转换为相机坐标系。


**坐标系转换**

如图 1 (b)，红色 box corner $(x_1, y_1)$ 是基于红色虚线坐标系，box 是 axis aligned，如果基于相机坐标系例如图 1 (c) 中的坐标系，那么红色 box 显然是歪的，以相机坐标系角度看红色 box，可以认为是一个 axis-aligned 的 box 固定 box center（以中心点为轴） **顺时针** 旋转 rotation_y 角度（简记为 $\theta$），记旋转后点为 $(\hat x_1, \hat y_1)$，那么有

$$\begin{bmatrix} \hat x_1 \\ \hat y_1 \end{bmatrix}=\begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}\begin{bmatrix} x_1 \\ y_1\end{bmatrix} \tag{5}$$


通常我们可以知道中心点（BEV）坐标 $(x_c, y_c)$，以及 box 的 $l, w$，那么此时

$$x_1 = x_c - l / 2, \quad y_1 = y_c - w / 2 \tag{6}$$

也就是说， $(x_1, y_1)$ 所基于的坐标系仍与 box 的边 平行，只是坐标系原点不在 box 中心，此时（以 box 中心为坐标原点的） (5) 式变为

$$\begin{bmatrix} \hat x_1 \\\\ \hat y_1 \end{bmatrix}=\begin{bmatrix} \cos \theta & -\sin \theta \\\\ \sin \theta & \cos \theta \end{bmatrix}\begin{bmatrix} x_1-y_c \\\\ y_1-y_c\end{bmatrix}+ \begin{bmatrix} x_c \\\\ y_c \end{bmatrix} \tag{7}$$

代码实现：

```c++
// 计算 bev 下的 boxes iou（boxes 非 axis-aligned）
// box_a:  (x1, y1, x2, y2, ry)
// corner 坐标是基于物体自身坐标系的，要计算 IoU，还需要将各个 box 的
// corner 坐标转到标准坐标系下
inline float iou_bev(const float *box_a, const float *box_b) {
    float sa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
    float sb = (box_b[2] - box_a[0]) * (box_a[3] - box_a[1]);

    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / (fmaxf(sa + sb - s_oveerlap, EPS));
}

// 计算两个 box overlap 区域面积
inline float box_overlap(const float *box_a, const float *box_b) {
    float a_x1 = box_a[0], a_y1 = box_a[1], a_x2 = box_a[2], a_y2 = box_a[3], a_angle = box_a[4];
    float b_x1 = box_b[0], b_y1 = box_b[1], b_x2 = box_b[2], b_y2 = box_b[3], b_angle = box_b[4];

    Point center_a((a_x1 + a_x2) / 2, (a_y1 + a_y2) / 2);
    Point center_b((b_x1 + b_x2) / 2, (b_y1 + b_y2) / 2);

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x1, a_y2);
    box_a_corners[3].set(a_x2, a_y2);
    
    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x1, b_y2);
    box_b_corners[3].set(b_x2, b_y2);

    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
        // 将 box_a 和 box_b 每个角点均分别顺时针旋转 ry_a 和 ry_b ，得到标准坐标系下的角点坐标
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }
    box_a_corners[4] = box_a_corners[0];    // 第 5 个点，设为第一个角点，这样就形成一个闭环
    box_b_corners[4] = box_b_corners[0];    // 下方求两个 box 的边线交叉点，需要用到这个点

    Point cross_points[16];     // 存储 box_a 和 box_b 的边线交叉点
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);  // overlap 是一个多边形，其内部一点，初始化为 (0, 0)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; i < 4; j++) {
            flag = intersection(box_a_corners[i+1], box_a_corners[i], box_b_corners[j+1], box_b_corners[j], cross_points[cnt]);
            if (flag) {     // 存在交叉点，那么交叉点就说 overlap 多边形的角点
                poly_center = poly_center + cross_points[cnt];  // 更新 overlap 多边形内部点坐标
                cnt++;
            }
        }
    }

    for (int k = 0; k < 4; k++) {
        if (check_in_box2d(box_a, box_b_corners[k])) {  // 检查 box_b 的角点是否位于 box_a 内部
            poly_center = poly_center + box_b_corners[k];// 更新 overlap 多边形内部点坐标
            cross_points[cnt] = box_b_corners[k];       // box_b 角点在 box_a 内部，那么这个角点也是 overlap 多边形的角点
            cnt++;
        }
        if (check_in_box2d(box_b, box_a_corners[k])) { // 反过来一样，如果 box_a 角点在 box_b 内部，那么这个角点也是 overlap 多边形的角点
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    Point temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            // 以内部点为原心，将 overlap 多边形顶点按顺时针排序
            if (point_cmp(cross_points[i], cross_points[i+1], poly_center)) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i+1];
                cross_points[i+1] = temp;
            }
        }
    }

    float area = 0;
    for (int k = 0; k < cnt-1; k++) {
        // 将 overlap 多边形划分为 n-2 个三角形，其中 n 为顶点数量，累加三角形面积得到多边形面积
        area += cross(cross_points[k] - cross_points[0], cross_points[k+1] - cross_points[0])
    }
    return fabs(area) / 2.0;
}
```

以上代码中，box_a 和 box_b 均为长度 5 的向量，表示 (x1, y1, x2, y2, ry)，如图 2， 计算蓝色 box_a 和橙色 box_b 实线框的重叠部分面积，x1,y1,x2,y2 实际上是将 box_a  axis-align 之后（即绿色虚线框）的 corner 坐标，蓝实线框的面积与绿虚线框面积相等，所以 `iou_bev` 方法中，box_a 和 box_b 的面积计算可按 axis-aligned 等效 box 计算面积。

![](/images/obj_det/3d/utils_2.png)
<center>图 2</center>

计算 overlap 多边形面积方法为 `box_overlap`，根据 `box_overlap` 方法实现代码，总结步骤如下：

1. 将虚线框 corner 坐标转换为实现框（box_a） 的 corner 坐标，左上角顶点，顺时针旋转 ry 角度即可。
2. 计算 box_a 和 box_b 的边交点 `intersection` 方法，交点是 overlap 多边形的顶点。
3. 如果某个 box 顶点位于另一个 box 内部，那么这个 box 顶点也是 overlap 多边形的顶点。
4. 得到 overlap 多边形的所有顶点后，需要排序（例如顺时针）`point_cmp` 方法，然后将 overlap 多边形划分为若干个三角形
5. 计算所有三角形面积之和，则为 overlap 多边形的面积。`cross` 方法计算三角形面积

