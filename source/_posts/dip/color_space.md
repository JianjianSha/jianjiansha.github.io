---
title: 颜色空间
date: 2022-05-05 09:35:27
tags: DIP
categories: 数字图像处理
summary: RGB, HSV
---

# 1. RGB

RGB 颜色空间表示 R (red)，G (green), B (blue) 三种颜色。通常这三种分量的取值范围为 [0,255] 内的整数。

# 2. HSV

## 2.1 Hue
H 色相：角度度量，范围 $[0^{\circ}, 360^{\circ}]$。

从红色开始逆时针方向计算。红色为 $0^{\circ}$，绿色为 $120^{\circ}$，蓝色为 $240^{\circ}$，补色（色相相差 $180^{\circ}$）分别为：青色 $180^{\circ}$，品红 $300^{\circ}$，黄色 $60^{\circ}$。

## 2.2 Saturation

S 饱和度：表示颜色接近光谱色的程度。一种颜色可看成是某种光谱色与白色混合的结果，光谱色所占比例越大，那么颜色就越接近光谱色，颜色的饱和度就越高，颜色就深而艳。

饱和度通常取值范围为 $[0, 1]$ 内的实数， 饱和度越高，颜色就越饱和。光谱色的白光成分为 0，其饱和度最大，为 1。

## 2.3 Value

V 亮度：颜色明亮程度。通常取值范围为 $0\% \sim 100\%$，表示 由黑到白。


# 3. 转换

## 3.1 RGB -> HSV
$$R=R/255, \ G=G/255, \ B=B/255\\
M=\max(R, G, B), \ m=\min(R,G,B)\\
\Delta=M-m$$

$$H=\begin{cases}0^{\circ} & \Delta=0\\
60^{\circ} \times (\frac{G-B} {\Delta} \mod 6) & M=R\\
60^{\circ} \times(\frac{B-R} {\Delta} + 2) & M=G \\
60^{\circ} \times(\frac {R-G} {\Delta}+4) & M=B\end{cases}$$

$$S=\begin{cases} 0 & M=0 \\ \frac {\Delta} M & M \neq 0 \end{cases}$$

$$V=M$$

易知 $\frac {G-B} {\Delta}, \frac {G-B} {\Delta}, \frac {G-B} {\Delta}$ 在其相应的条件下，均位于范围 $[-1, 1]$ 内，于是

$$\frac{B-R} {\Delta} + 2 \in [1, 3], \quad \frac {R-G} {\Delta}+4 \in [3, 5]$$

$$\frac {G-B} {\Delta} \mod 6 \in \begin{cases} [5, 6) & G < B\\ [0, 1] & G > B \end{cases}$$

## 3.2 HSV -> RGB
suppose: $0 \le H < 360, \ 0 \le S \le 1, \ 0 \le V \le 1$ 根据上一节的变换关系有，

$$\Delta=V \times S\\
M=V\\
m=M-\Delta = V-\Delta$$

到此，可以求出上面三个值。

且根据 $H$ 的变换关系有

$$X=\Delta \times (1-|\frac H {60^{\circ}} \mod 2 -1|)=\begin{cases}G-B & M=R,G \ge B,m=B \\ R-B & M=G, B<R,m=B \\ B-R & M=G,B \ge R,m=R \\ G-R & M=B, R<G,m=R \\ R-G & M=B,R \ge G,m=G \\ B-G &M=R,G <B,m=G \end{cases}$$

注意 $X \ge 0$，所以上式右侧各分段表达式也必须要非负。

表达式 $1-|\frac H {60^{\circ}} \mod 2 -1|$ 在 $[0, 6)$ 范围内是周期为 2 的三角波形函数（参考下方代码片段），

```python
x = np.arange(0, 6, 0.1)
y = 1-np.abs(np.mod(x,2)-1)
x = np.split(x, [20, 40])
y = np.split(y, [20, 40])
plt.plot(x[0], y[0], x[1], y[1], x[2], y[2], color='b')
plt.show()
```


$H \in [0^{\circ}, 360^{\circ})$，每 $60^{\circ}$ 一个分段，

$$H \in \begin{cases} [0, 60^{\circ}) & M=R, G \ge B \\ [60^{\circ}, 120^{\circ}) & M=G,G<B \\ \vdots \\ [300^{\circ},360^{\circ}) & M=B,G<B \end{cases}$$

综上，可知归一化的 $R,G,B$ 为， 

$$(R,G,B)=\begin{cases}(V,m+X, m) & H \in [0, 60^{\circ}) \\ (m+X, V,m) & H \in [60^{\circ}, 120^{\circ}) \\ (m, V, m+X) & H \in [120^{\circ}, 180^{\circ}) \\ (m, m+X, V) & H \in [180^{\circ}, 240^{\circ}) \\ (m+X,m,V) & H \in [240^{\circ}, 300^{\circ})\\ (V,m,m+X) & H \in [300^{\circ}, 360^{\circ}) \end{cases}$$

最后乘以 255 即可得到各通道值，

$$(R,G,B)=(R,G,B)\times 255$$


## 3.3 代码
```python
def rgb2hsv(r,g,b):
    '''
    r,g,b: [0, 255]
    '''
    r,g,b = r/255,g/255,b/255
    M,m = max(r,g,b), min(r,g,b)
    d = M-m
    v = M
    s = 0 if M=0 else d/M
    if d == 0: h = 0
    elif r==M: h=(g-b)/d * 60
    elif g==M: h=120+(b-r)/d*60
    elif b==M: h=240+(r-g)/d*60

def hsv2rgb(h,s,v):
    '''
    h: [0, 360)
    s,v: [0, 1]
    '''
    d = v * s
    M = v
    m = M - d
    x = d * (1-abs((h/60) % 2 - 1))
    if 0 <= h < 60:
        r,g,b = v, m+x, m
    elif 60 <= h < 120:
        r,g,b = m+x, v, m
    elif 120 <= h < 180:
        r,g,b = m, v, m+x
    elif 180 <= h < 240:
        r,g,b = m, m+x, v
    elif 240 <= h < 300:
        r,g,b = m+x, m, v
    elif 300 <= h < 360:
        r,g,b = v, m, m+x
    return r * 255, g * 255, b * 255

fig = plt.figure()  # 定义新的三维坐标轴
ax = Axes3D(fig)
size = 30
points = np.linspace(0, 255, size).astype(np.int32)

for h in np.linspace(0, 360, size):
    for s in np.linspace(0, 100, size):
        for v in np.linspace(0, 100, size):
            if v < s:
                continue
            x_ = s * cos(h * pi / 180)
            y_ = s * sin(h * pi / 180)
            # z_ = -(v ** 2 - s ** 2) ** 0.5
            z_ = v
            if h == 360: h = 0
            x, y, z = hsv2rgb(h, s / 100, v / 100)
            ax.plot([x_], [y_], [z_], "ro", color=(x / 255, y / 255, z / 255, 1))

print('---')
ax.set_zlabel('r')
ax.set_ylabel('g')
ax.set_xlabel('b')
plt.show()
```