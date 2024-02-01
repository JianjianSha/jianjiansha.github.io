---
title: 隐写介绍
date: 2023-10-18 09:02:32
tags: stego
---

# 1. 频域

## 1.1 DWT+DCT+SVD

嵌入过程：

1. 对载体图像进行 DWT 分解，得到 4 个子带图像 LL LH HL HH
2. 对 LL 进行 DCT 变换，得到 DCT 系数矩阵
3. 对 DCT 系数矩阵进行 SVD，得到左奇异矩阵 U，奇异值矩阵 S 和右奇异矩阵 V
4. **重新将 S 另存为 S0**，将水印图像 $A$ 缩放到合适大小 ($n \times n$)，然后嵌入到 S 矩阵中某个事先确定的位置 

    ```python
    alpha = 0.1 # 嵌入强度系数
    S[i:i+n, j:j+n] += alpha * A
    ```
5. 合并 U S V，得到新的 DCT 系数矩阵
6. 对新的 DCT 系数矩阵进行 iDCT，得到新的 LL
7. 对 LL LH HL HH 进行 iDWT，得到嵌入水印后图像

方法说明：

1. 选择 LL 是由于 LL 比较稳定，比较容易恢复出水印，但是 LL 是低频，嵌入水印后图像失真可能较大。

    也可以尝试嵌入到高频子带，例如 HL, HH 等。

2. 可以对载体图像进行多次 DWT 变换，即，选择 2 级以上的 LL（或 HL HH 等）进行嵌入

3. 嵌入水印时，还可以将水印图像做相同的变换：DWT->DCT->SVD，然后将水印图像的奇异值矩阵 S' 嵌入到载体图像的奇异值矩阵 S 中，但是这种方法在提取水印时，需要提供水印图像 SVD 分解后的左右奇异矩阵 U' 和 V'

4. 选择奇异值矩阵 S 进行嵌入，是因为奇异值矩阵具有较强的稳定性，图像遭受较小的攻击（较小的改变，例如压缩）时，奇异值不会发生较大的改变，并且，奇异值对旋转、缩放和平移等几何攻击具有不变性。

提取过程：

1. 对嵌入水印后图像进行 DWT

2. 对 LL 进行 DCT

3. 对 DCT 系数矩阵进行 SVD，得到奇异矩阵 S

4. 根据下式计算水印数据，

    ```python
    A = (S[i:i+n, j:j+n]-S0[i:i+n,j:j+n])/alpha
    ```

代码（来源：[CSDN](https://blog.csdn.net/m0_52363973/article/details/131115784)）

```matlab
M = 512; % 原图像长度
N = 64; % 水印图像长度
K = 32; % 子块大小
 
alpha=0.1;% 嵌入强度系数
 
% 打开原图、水印图
I = imread('lena.jpg');
G = imread('a.jpg');
%W = zeros(M);
 
% 缩放、灰度化原图、改变精度
I = imresize(I,[M M]);
%I = im2double(I); % double精度转换
I = rgb2gray(I); % 灰度化处理
 
G = imresize(G,[N N]);
%G = im2double(G); % double精度转换
G = rgb2gray(G); % 灰度化处理
 
subplot(2,2,1);
imshow(I);
title('原始载体图片');
subplot(2,2,2);
imshow(G);
title('原始水印图像');
 
%Step 1
[LL,LH,HL,HH] = dwt2(G,'haar'); % 进行2维哈尔离散小波变换
[U,S,V] = svd(HH);% 对HH进行SVD分解，得到U、S、V矩阵
 
%Step 2
%进行2级离散小波变换
[LL1, LH1, HL1, HH1] = dwt2(I, 'haar');
[LL2, LH2, HL2, HH2] = dwt2(LL1, 'haar');%128*128
H0 = entropy(HH2)% 计算HH3系数的信息熵
 
%Step 3
 
%选出最优嵌入块 默认为4*4:(1,1)
optimal_block_index = 0;
 
%Step 4
%对最优嵌入块进行 DCT 变换，得到DCT系数矩阵 B
m = floor(optimal_block_index/4)+1;
n = mod(optimal_block_index, 4)+1
x = (m - 1) * K + 1;
y = (n - 1) * K + 1;
H_I = HH2(x:x+K-1, y:y+K-1);
B = dct2(H_I);
 
%Step 5
%对B进行奇异值分解,嵌入水印
[U1,S1,V1] = svd(B);
S2 = S1 + alpha * S;
B1 = U1 * S2 * V1;
H_I = idct2(B1);
HH2(x:x+K-1, y:y+K-1) = H_I;
LL1 = idwt2(LL2,LH2,HL2,HH2,'haar');
W = idwt2(LL1,LH1,HL1,HH1,'haar');
W = uint8(W);
 
%攻击
%高斯滤波攻击
%H = fspecial('gaussian',3,0.4);
%W = imfilter(W,H);
 
%G 压缩攻击
%quality = 50;
%W = imresize(W, 0.5); % 缩小图像
%imwrite(W, 'temp.jpg', 'Quality', quality); % 保存为JPEG格式
%W = imread('temp.jpg'); % 重新读取JPEG图像
%W = imresize(W, 2); % 放大图像
 
% 剪切攻击
%r = 0.3; % 剪切比例为30%
%sz = size(W);
%h1 = round(sz(1)*r); % 剪切高度
%w1 = round(sz(2)*r); % 剪切宽度
%x1 = round(rand(sz(1)-h1)); % 随机选择一行
%y1 = round(rand(sz(2)-w1)); % 随机选择一列
%W(x1+1:x1+h1, y1+1:y1+w1) = 0; % 将指定区域置为0
 
% 旋转攻击
%angle = 20; % 旋转角度为20度
%W = imrotate(W, angle, 'bilinear', 'crop');
 
%提取水印
[LL3, LH3, HL3, HH3] = dwt2(W, 'haar');
[LL4, LH4, HL4, HH4] = dwt2(LL3, 'haar');%128*128
H_I2 = HH4(x:x+K-1, y:y+K-1);
B2 = dct2(H_I2);
[Uw,Sw,Vw] = svd(B2);
Sx = (Sw - S1)/alpha;
B2 = U * Sx * V;
H_I2 = idct2(B2);
A = idwt2(LL,LH,HL,H_I2,'haar');
A = uint8(A);
subplot(2,2,3);
imshow(W);
title('嵌入水印后的载体图像');
subplot(2,2,4);
imshow(A);
title('提取出来的水印图像');
 
% 计算PSNR值
psnr_val = psnr(G, A);
 
% 显示PSNR值
fprintf('The PSNR value between the original image and reconstructed image is %f.\n', psnr_val);
 
% 计算直方图
h1 = imhist(G);
h2 = imhist(A);
 
% 根据直方图计算 NC 值
nc_val = sum(sqrt(h1 .* h2)) / sqrt(sum(h1) * sum(h2));
 
% 显示 NC 值
fprintf('The NC value between the two images is %f.\n', nc_val);
```