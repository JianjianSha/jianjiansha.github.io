---
title: Harris-Laplace
date: 2024-04-10 10:03:42
tags:
---

以下内容摘自：https://blog.csdn.net/zizi7/article/details/50132881

Harris-Laplace算法结合Harris算法和Laplace尺度空间（见LOG算法），实现了尺度不变性。

Harris-Laplace算法的基本思路很简单：

（1）首先预设一组尺度（高斯函数的方差），在每个尺度下进行Harris检测

（2）对当前图像，生成一组不同尺度下的图像集

（3）对（1）中找到的每个角点，在（2）中相同位置下比较其26个邻域的LOG相应值，如果也为局部极大值，则认为该点为最终角点，通过对应的尺度也可以得到该角点的空间尺度（“半径”）



2. 细节
（1）Harris-Laplace算法在进行特定尺度下Harris检测时不再使用近似的梯度算子（[-1 0 1]），而是使用当前尺度（方差sigma）乘以0.7（经验值）作为梯度算子高斯核的方差，相应的，该算子半径为 3*0.7*sigma

（2）生成Laplace尺度空间时，使用的是当前尺度sigma，高斯核半径为3*sigma
                        
下面为相关代码，可以先参考 [screen_cam_fbs](/2024/04/08/stego/screen_cam_fbs) 中的讲解，以下将此参考文章记为 `ref 1` 。

```c++
#define PI 3.14159
 
typedef struct myPoint
{
	int x;
	int y;
}_point;
 
typedef struct mySize
{
	int height;
	int width;
}_size;
 
typedef struct myPointR
{
	int x;
	int y;
	int radius;
}_pointR;
 
template<typename T1, typename T2>
void convolution(T1* img, T2** out, _size imgSize, T2* kernel, _size kerSize);
template<typename T1>
void dotMultiply(T1* src1, T1* src2, T1** dst, _size srcSize);
template<typename T1>
void gaussian(int* src, float** out, _size imgSize, _size gaussSize, float sita);
void strengthR(float* A, float* B, float* C, float** R, float* maxR, _size imgSize, float alpha);
void depressR(float* srcR, _point* dstR, int* numDst, _size imgSize, _size depressWin, float threshold);
void getLOGkernel(int halfWin, float sita, float** kernel);
bool hasHarrisCorner(_point* pointsSet, int numPoints, _point pos);
void _cornerHarrisLaplace(unsigned char* src, _size srcSize, _pointR* corners, int* numCorner, float alpha);
 
void main()
{
	cv::Mat img = cv::imread("../file/butterfly.jpg", 0);
	unsigned char* imgBuffer = new unsigned char[img.rows*img.cols];
	for(int i=0; i<img.rows; i++)
	{
		uchar* ptr = img.ptr<uchar>(i);
		for(int j=0; j<img.cols; j++)
			imgBuffer[i*img.cols+j] = ptr[j];
	}
 
	_size srcSize;
	srcSize.height = img.rows;
	srcSize.width = img.cols;
 
	_pointR* corners = new _pointR[srcSize.height*srcSize.width];
	int numCorners = 0;
	_cornerHarrisLaplace(imgBuffer, srcSize, corners, &numCorners, 0.04);
 
	cv::Mat colorImg;
	cv::cvtColor(img, colorImg, CV_GRAY2BGR);
	for(int i=0; i<numCorners; i++)
	{
		cv::Point center(corners[i].x, corners[i].y);
		cv::circle(colorImg, center, corners[i].radius, cv::Scalar(0,0,255));
	}
 
	cv::namedWindow("show");
	cv::imshow("show", colorImg);
	cv::waitKey(0);
}
 
template<typename T1, typename T2>
void convolution(T1* img, T2** out, _size imgSize, T2* kernel, _size kerSize)
{
	for(int i=0; i<imgSize.height; i++)
	{
		for(int j=0; j<imgSize.width; j++)
		{
			int count = 0;
			T2 sumValue = 0;
			for(int m=i-kerSize.height; m<=i+kerSize.height; m++)
			{
				for(int n=j-kerSize.width; n<=j+kerSize.width; n++)
				{
					if(m>=0 && m<imgSize.height && n>=0 && n<imgSize.width)
						sumValue += T2(img[m*imgSize.width+n])*kernel[count];
					count++;
				}
			}
			(*out)[i*imgSize.width+j] = sumValue;
		}
	}
}
 
template<typename T1>
void dotMultiply(T1* src1, T1* src2, T1** dst, _size srcSize)
{
	for(int i=0; i<srcSize.height; i++)
	{
		for(int j=0; j<srcSize.width; j++)
			(*dst)[i*srcSize.width+j] = src1[i*srcSize.width+j]*src2[i*srcSize.width+j];
	}
}
 
template<typename T1>
void gaussian(T1* src, float** out, _size imgSize, _size gaussSize, float sita)
{
	int sizeKernel = (2*gaussSize.height+1) * (2*gaussSize.width+1);
	float* gaussKernel = new float[sizeKernel];
	for(int i=-gaussSize.height; i<=gaussSize.height; i++)
	{
		for(int j=-gaussSize.width; j<=gaussSize.width; j++)
		{
			float tmp = -1*(i*i+j*j)/(2*sita*sita);
			gaussKernel[(i+gaussSize.height)*(2*gaussSize.width+1)+(j+gaussSize.width)] =
				exp(tmp)/(2*PI*sita*sita);
		}
	}
 
	convolution(src, out, imgSize, gaussKernel, gaussSize);
	delete[] gaussKernel;
}
 
void strengthR(float* A, float* B, float* C, float** R, float* maxR, _size imgSize, float alpha)
{
	*maxR = 0;
	for(int i=0; i<imgSize.height; i++)
	{
		for(int j=0; j<imgSize.width; j++)
		{
			float detM = A[i*imgSize.width+j]*B[i*imgSize.width+j] - C[i*imgSize.width+j]*C[i*imgSize.width+j];
			float traceM = A[i*imgSize.width+j] + B[i*imgSize.width+j];
			float result = detM - alpha*(traceM*traceM);
			if(result > *maxR)
				*maxR = result;
			(*R)[i*imgSize.width+j] = result;
		}
	}
}
 
void depressR(float* srcR, _point* dstR, int* numDst, _size imgSize, _size depressWin, float threshold)
{
	*numDst = 0;
	for(int i=0; i<imgSize.height; i++)
	{
		for(int j=0; j<imgSize.width; j++)
		{
			float flagValue = srcR[i*imgSize.width+j]<threshold?0:srcR[i*imgSize.width+j];
			int numPoint = 0, numPass = 0;
			for(int m=i-depressWin.height; m<=i+depressWin.height; m++)
			{
				for(int n=j-depressWin.width; n<=j+depressWin.width; n++)
				{
					if(m>=0 && m<imgSize.height && n>=0 && n<imgSize.width)
					{
						float compareValue = srcR[m*imgSize.width+n]<threshold?0:srcR[m*imgSize.width+n];
						if(flagValue > compareValue)
							numPass ++;
						numPoint ++;
					}
				}
			}
			if(numPoint == numPass+1)
			{
				_point corner;
				corner.x = j; 
				corner.y = i;
				dstR[(*numDst)++] = corner;
			}
		}
	}
}
 
void getLOGkernel(int halfWin, float sita, float** kernel)  
{  
	int winSize = 2*halfWin+1;  
	float tmp1, tmp2, sumValue = 0;  
	float powsita = sita*sita;  
	for(int i=-halfWin; i<=halfWin; i++)  
	{  
		for(int j=-halfWin; j<=halfWin; j++)  
		{  
			tmp1 = -1*(i*i+j*j)/(2*powsita);  
			tmp2 = exp(tmp1)*(i*i+j*j-2*powsita);//exp(tmp1)*(1+tmp1)/(-1*powsita*powsita);  
			sumValue += tmp2;  
			(*kernel)[(i+halfWin)*winSize+(j+halfWin)] = tmp2;  
		}  
	}  
	for(int i=0; i<winSize*winSize; i++)  
		(*kernel)[i] -= sumValue/(winSize*winSize);  
}  
 
bool hasHarrisCorner(_point* pointsSet, int numPoints, _point pos)
{
	for(int i=0; i<numPoints; i++)
	{
		if(pointsSet[i].x == pos.x && pointsSet[i].y == pos.y)
			return true;
	}
	return false;
}

// 计算 Harris-Laplace 特征点
// src  原始图像数据
// srcSize 原始图像 size
// 存储 Harris-Laplace 角点，即特征点
void _cornerHarrisLaplace(unsigned char* src, _size srcSize, _pointR* corners, int* numCorner, float alpha)
{
	float sigma_begin = 1.5;    // ref 1 中的 sigma_0
	float sigma_step = 1.2;     // ref 1 中的 \xi
	int sigma_n = 13;           // ref 1 中的 N
	float* sigma = new float[sigma_n];  // ref 1 sigma_n
	int* numcorner_n = new int[sigma_n];
	for(int i=0; i<sigma_n; i++)    
		sigma[i] = sigma_begin*pow(sigma_step, i);
 
	_point** allPoints = new _point*[sigma_n];  // 存储所有尺度空间的特征点
	for(int i=0; i<sigma_n; i++)
		allPoints[i] = new _point[srcSize.height*srcSize.width];
 
	//Harris
	for(int i=0; i<sigma_n; i++)
	{
		float sigma_d = sigma[i]*0.7;   // ref 1 中 sigma_D
		_size sizeX, sizeY;
		sizeX.height = 0; sizeY.width = 0;
		sizeX.width = int(3*sigma_d+0.5);
		sizeY.height = sizeX.width;
 
		float* kernel = new float[2*sizeX.width+1];
		for(int j=0; j<2*sizeX.width+1; j++)
		{
			int x = j-sizeX.width;
            // 设置 ref 1 中的 (13) 式
			kernel[j] = exp(-1*x*x/(2*sigma_d*sigma_d))/(sqrt(2*PI)*sigma_d*sigma_d*sigma_d)*x;
		}
 
		float* gradientX = new float[srcSize.height*srcSize.width];
		float* gradientY = new float[srcSize.height*srcSize.width];
		convolution(src, &gradientX, srcSize, kernel, sizeX);   // 计算 ref 1 中的 L_x
		convolution(src, &gradientY, srcSize, kernel, sizeY);   // 计算 ref 1 中的 L_y
 
		delete[] kernel;
 
		float* gradientXX = new float[srcSize.height*srcSize.width];
		float* gradientYY = new float[srcSize.height*srcSize.width];
		float* gradientXY = new float[srcSize.height*srcSize.width];
		dotMultiply(gradientX, gradientX, &gradientXX, srcSize);// ref 1 的 L_x ^ 2
		dotMultiply(gradientY, gradientY, &gradientYY, srcSize);// ref 1 的 L_y ^ 2
		dotMultiply(gradientX, gradientY, &gradientXY, srcSize);// ref 1 的 L_x L_y
 
		delete[] gradientX;
		delete[] gradientY;
 
		_size gaussSize;
		gaussSize.height = sizeX.width;  gaussSize.width = sizeX.width;
		float sita = sigma[i];
		float* M_A = new float[srcSize.height*srcSize.width];
		float* M_B = new float[srcSize.height*srcSize.width];
		float* M_C = new float[srcSize.height*srcSize.width];
		gaussian(gradientXX, &M_A, srcSize, gaussSize, sita);// sigma_n，计算 ref 1 的 (5) 式，即 H 矩阵各元素
		gaussian(gradientYY, &M_B, srcSize, gaussSize, sita);
		gaussian(gradientXY, &M_C, srcSize, gaussSize, sita);
 
		delete[] gradientXX;
		delete[] gradientYY;
		delete[] gradientXY;
 
		float maxR = 0;
		float* R = new float[srcSize.height*srcSize.width];
		strengthR(M_A, M_B, M_C, &R, &maxR, srcSize, alpha);// 计算 ref 1 中的 C，(6) 式
 
		delete[] M_A;
		delete[] M_B;
		delete[] M_C;
 
		float threshold = 0.1;
		threshold *= maxR;
		_size depressSize;
		depressSize.height = 1; depressSize.width = 1;
		int num;
        // 当前尺度空间中，寻找 8 连通域的极值，作为候选角点
		depressR(R, allPoints[i], &num, srcSize, depressSize, threshold);
		numcorner_n[i] = num;
 
		delete[] R;
	}
 
	// Laplace
	float** laplaceSnlo = new float*[sigma_n];
	for(int i=0; i<sigma_n; i++)
		laplaceSnlo[i] = new float[srcSize.height*srcSize.width];
 
	for(int i=0; i<sigma_n; i++)
	{
		float sita = sigma[i];
		_size kernelSize;
		kernelSize.height = 3*sita;
		kernelSize.width = 3*sita;
		float *kernel = new float[(2*kernelSize.height+1)*(2*kernelSize.height+1)];
		getLOGkernel(kernelSize.height, sita, &kernel);// 计算 ref 1 中 (14) 式
		convolution(src, &(laplaceSnlo[i]), srcSize, kernel, kernelSize);// 计算 (15) 式
 
		delete[] kernel;
	}
 
    // 根据 LoG 筛选 harris 候选角点。这里没有使用 ref 1 中的迭代法
    // 因为迭代法太慢，所以使用了简单筛选方法：只要 harris 角点是相邻 LoG 尺度空间的极值
    // 那么它就是最终的角点
	*numCorner = 0;
	for(int i=0; i<srcSize.height; i++)
	{
		for(int j=0; j<srcSize.width; j++)
		{
			for(int k=0; k<sigma_n; k++)
			{
				_point pos;
				pos.x = j; pos.y = i;
                // 此位置是否有 harris 候选角点
				bool isFind = hasHarrisCorner(allPoints[k], numcorner_n[k], pos);
				if(isFind)
				{
					if(k==0)    // 第一个尺度空间，与下一个尺度空间的 LoG response 比较
					{
                        // 此候选角点位置处，LoG 尺度空间比相邻LoG尺度空间的响应值大，则保留
						if(laplaceSnlo[k][i*srcSize.width+j]>laplaceSnlo[k+1][i*srcSize.width+j])
						{
							corners[*numCorner].x = j;
							corners[*numCorner].y = i;
							corners[*numCorner].radius = int(PI*sigma[k]+0.5);
							(*numCorner)++;
 
							break;
						}
					}
					else if(k==sigma_n-1)// 最后一个尺度空间，与前一个尺度空间 LoG response 比较
					{
						if(laplaceSnlo[k][i*srcSize.width+j]>laplaceSnlo[k-1][i*srcSize.width+j])
						{
							corners[*numCorner].x = j;
							corners[*numCorner].y = i;
							corners[*numCorner].radius = int(PI*sigma[k]+0.5);
							(*numCorner)++;
 
							break;
						}
					}
					else    // 与前后两个尺度空间的 LoG response 比较
					{
						if(laplaceSnlo[k][i*srcSize.width+j]>laplaceSnlo[k-1][i*srcSize.width+j] &&
							laplaceSnlo[k][i*srcSize.width+j]>laplaceSnlo[k+1][i*srcSize.width+j])
						{
							corners[*numCorner].x = j;
							corners[*numCorner].y = i;
                            // 特征半径为 \pi*sigma_n
							corners[*numCorner].radius = int(PI*sigma[k]+0.5);
							(*numCorner)++;
 
							break;
						}
					}
				}
			}
		}
	}
	for(int i=0; i<sigma_n; i++)
	{
		delete[] laplaceSnlo[i];
		delete[] allPoints[i];
	}
	delete[] laplaceSnlo;
	delete[] allPoints;
	delete[] sigma;
	delete[] numcorner_n;
}
```