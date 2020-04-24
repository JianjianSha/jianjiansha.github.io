---
title: PyTorch-1
p: pytorch/PyTorch-1
date: 2019-06-12 19:17:11
tags: PyTorch
categories: DL Framework
---
# 安装
一直以来就对深度学习的框架源码有着浓厚兴趣，但是由于涉及到的领域较多，C++，python，CUDA，数学等，加上时间也比较零碎，就耽搁至今，后来意识到我不可能等完全弄明白之后再来写博客记录，毕竟能力不足，所以还是边看源码边记录，不求完全搞明白，但求能从整体上有个大致的理解，如果还能整明白一些数学计算上的代码实现，那就再好不过了。
<!-- more -->
当前最流行的深度学习框架就是tensorflow和pytorch了，但是tensorflow据说代码工业化程度非常高，我等菜鸡先避其锋芒，来分析pytorch，希望能给自己带来点信心。

下载源码
```
git clone --recursive https://github.com/pytorch/pytorch
```

由于使用了子模块所以增加--recursive选项，记pytorch的root dir为$ROOT_DIR。

根据安装步骤进行自上而下的阅读。Linux下安装使用命令
```
cd pytorch
python setup.py install
```
pytorch底层计算使用C++实现，并提供了python调用接口，所以这一命令就是使用setuptools安装python包，安装依赖库及修改配置项这里均跳过，故直接看$ROOT_DIR/setup.py中的setup()方法，但是在这个方法之前先执行了build_deps()用于生成有关 caffe2 的依赖库

### build_deps()
这个方法内部关键的一步为
```
build_caffe2(...)
```
查看这个方法的定义，发现build_caffe2做了如下几件事：
1. run_cmake。执行cmake，这个命令的选项这里省略不展开，注意执行cmake这个命令的工作目录为`$ROOD_DIR/build`， cmake的Source Tree为$ROOD_DIR，这个 目录下存在top level的CMakeLists.txt
2. 在$ROOT_DIR/build下编译并安装，使用make install或者 ninja install（cmake生成的Makefile中install这个target包含了build这个步骤）
3. 将build/caffe2/proto下的所有.py文件 拷贝到caffe2/proto/下，这些.py文件是根据caffe2/proto/下的.proto文件生成

这其中最复杂的部分就是run_cmake了，先是使用cmake的-D option设置一些cmake的变量，然后对source tree应用cmake， 查看top level的CMakeLists.txt，这个文件看着好像特别庞大，实际上做的事情也就那么几种：1)设置变量，根据不同操作系统设置或修改变量；2)设置include dir以及lib dir；3）加载.cmake文件以使用其中自定义的cmake函数；4）设置C++文件编译选项；5）安装配置文件/目录到指定位置等；我们注意比较关键的语句如下：
```
add_subdirectory(c10)
add_subdirectory(caffe2)
add_subdirectory(modules)
```
这表明将c10,caffe2,modules等目录添加进build tree，这些目录下必定也有相应的CMakeLists.txt， 所以需要继续查看这些CMakeLists.txt中定义了哪些生成规则。

另外，top level 中CMakeLists.txt中有这么一行
```
include(cmake/Dependencies.cmake)
```
这个Dependencies.cmake指明安装Caffe2所依赖的各种库，其中一些库位于本项目中如`$ROOT_DIR/third_party`或$ROOT_DIR/caffe2，还有一些库则是需要预先手动安装的，举个例子：
1. 非本项目的公共库，比如添加BLAS库依赖，假设最开始设置了环境变量BLAS=OpenBLAS（环境变量的设置可参考setup.py文件头部注释）, 那么选择添加OpenBLAS库依赖，在Dependencies.cmake中代码为
```
...
elseif(BLAS STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED)
  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${OpenBLAS_LIB})
```
这个find_package告诉我们去查看`$ROOT_DIR/cmake/Modules/FindOpenBLAS.cmake`，好的我们跳过去看一下这个.cmake文件，发现其定义了OpenBLAS的头文件和库文件的搜索路径，然后根据这些搜索路径分别搜索头文件cblas.h所在目录以及库名openblas， 分别使用变量OpenBLAS_INCLUDE_DIR和OpenBLAS_LIB保存，从上面的代码片段，我们知道搜索到的库名被添加到Caffe2_PUBLIC_DEPENDENCY_LIBS中，而我们再跳至$ROOT_DIR/caffe2/CMakeLists.txt发现其中有
```
target_link_libraries(caffe2 PUBLIC ${Caffe2_PUBLIC_DEPENDENCY_LIBS})
```
这就相当于能生成-lopenblas这样的链接选项。

我们直接再看另一个库caffe2_pybind11_state的生成，因为下文会提到它，查看$ROOT_DIR/caffe2/CMakeLists.txt发现
```
add_subdirectory(python)
...
add_library(caffe2_pybind11_state MODULE ${Caffe2_CPU_PYTHON_SRCS})
install(TARGETS caffe2_pybind11_state DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")
```
其中Caffe2_CPU_PYTHON_SRCS在$ROOT_DIR/caffe2/python/CMakeLists.txt中设置， 类似地，还根据是否使用CUDA或者ROCM , 生成caffe2_pybind11_state_gpu或caffe2_pybind11_state_hip。生成这些库文件后，直接install到python的site-packages目录下的caffe2/python目录中

以上就是build_dep()这个方法介绍，接着看$ROOT_DIR/setup.py中的setup方法。

### setup()
setup方法（可以参考[setup()](https://docs.python.org/3/distutils/apiref.html)），其中几个值得说明的参数：
1. ext_modules 有5个扩展库分别如下：
- torch._C 指定了C++源文件，链接库，编译选项，链接选项和头文件/库dir
- torch._dl 非WINDOWS平台下才有，指定了C源文件
- caffe2.python.caffe2_pybind11_state
- caffe2.python.caffe2_pybind11_state_gpu
- caffe2.python.caffe2_pybind11_state_hip

后三个库在上一步中其实已经生成好了，其中caffe2.python前缀表示两级目录（package），可以在$ROOT_DIR/build/caffe2/python目录下查看。扩展模块ext_modules在build_ext这个动作中生成。

2. cmdclass，重写了build_ext, clean, install这几个action，这个action用在python setup.py <action> 命令中。install动作跟默认一致。 clean是清除编译过程中产生的临时文件，这些临时文件的pattern在.gitignore中给定。我们重点看一下build_ext这个动作对应的类build_ext，其中方法包含

- create_compile_commands这是一个自定义方法，用于将compile_commands.json中的gcc编译器改为g++，修改原因代码注释写的很清楚，使用gcc编译s时不会include c++的头文件目录。 文件compile_commands.json是根据`$ROOT_DIR/CMakeLists.txt中的set(CMAKE_EXPORT_COMPILE_COMMAND ON)`这句代码而生成，所以位于$ROOT_DIR/build目录下，这个json文件中指明了编译各个文件时的工作路径（working directory），编译指令（command）以及被编译的原文件，格式如下
```
[
{
  "directory":"<path/to/root>build/third_party/protobuf/cmake",
  "command": "/usr/bin/c++ ... -I<path/to/root>/third_party/protobuf/src ... 
                -o CMakeFiles/libprotobuf.dir/__/src/google/protobuf/arena.cc.o ...",
  "file": "<path/to/root>/third_party/protobuf/src/google/protobuf/arena.cc"
},
...
]
```
其中每个{...}块表示编译一个源文件到目标文件 .o。 将文件中gcc改为g++后重新保存为$ROOT_DIR/compile_commands.json。
- run打印各library（比如 CUDA, CUDNN, NUMPY等）的使用情况，然后执行基类同名方法的逻辑
- build_extensions 生成由ext_modules指定的python扩展库所用的方法

ext_modules中添加了5个扩展，后三个扩展在build_deps()中已经生成并安装，当然，caffe2_pybind11_state_gpu和caffe2_pybind11_state_hip是根据配置决定是否生成，配置了CUDA则生成前者，配置了ROCM则生成后者，如果均未配置，则这两个扩展均不生成。既然在build_deps()中已经生成并安装，所以这里将其从ext_modules中删除，于是build_extensions实际上只生成torch._C, torch._dl这两个扩展库。

然而，除了build_deps()方法还有其他方法可用于生成ext_modules中 的后三个扩展库，生成路径为`$ROOT_DIR/torch/lib/python3.7/site-packages/caffe2/python/`，所以需要判断在这个路径下是否存在后三个扩展库，若不在（此时就是前面所说的使用build_deps()生成），则将扩展库名称从ext_modules中予以删除， 若存在，则还需则将其拷贝到生成目录`$ROOT_DIR/torch/build/lib.linux-x86_64-3.7/`下，并修改拷贝后的文件名称，以caffe2.python.caffe2_pybind11_state为例说明，两级前缀表示目录所以最终的目录为`$ROOT_DIR/torch/build/lib.linux-x86_64-3.7/caffe2/python/`，剩余的caffe2_pybind11_state表示扩展库的文件名，还需要添加后缀名，这个后缀名由系统平台和python版本，我这里是.cpython-37m-x86_64-linux-gnu.so，于是拷贝后得到文件$ROOT_DIR/torch/build/lib.linux-x86_64-3.7/caffe2/python/caffe2_pybind11_state.cpython-37m-x86_64-linux-gnu.so ，这样使用基类的build_extensions()方法才能将其进一步安装到 python的site-packages目录下，我这里是.../miniconda3/lib/python3.7/site-packages/caffe2/python/目录。

3. packages 指定安装到python 的site-packages下的包
```
packages = find_packages(exclude=['tools', 'tools.*'])
```

由于PyTorch项目中除tools之外，只有caffe2和torch两个目录包含__init__.py，所以将caffe2和torch两个包安装到site-packages下。

现在再回头看看ext_modules中指定的5个扩展，不难得知，其中torch._C, torch._dl这两个扩展安装到site-packages/torch下，扩展包名称分别为_C, _dl（省略了文件ext后缀），而另外三个caffe2有关的扩展则根据名称（.号切分，前面都是目录名，最后一个是文件名）知道其安装在site-packages/caffe2/python下。

### 整理

以上就是pytorch安装过程，主要分为两部分:

1. 使用CMake生成c++库，对应build_deps()这个方法执行
2. 使用python的setup方法生成扩展库，主要是build_ext。

根据上面两点，重新整理一遍。
```
top-level的CMakeLists.txt中
add_subdirectory(c10)
add_subdirectory(caffe2)
```
于是先看caffe2这个目录下的CMakeLists.txt， 寻找其中的关键语句，
```
add_library(caffe2_proto STATIC $<TARGET_OBJECTS:Caffe2_PROTO>
add_library(thnvrtc SHARED ${TORCH_SRC_DIR}/csrc/jit/fuser/cuda/thnvrtc.cpp>
add_library(caffe2 ${Caffe2_CPU_SRCS})
if (TORCH_STATIC)
  add_library(torch STATIC ${DUMMY_EMPTY_FILE})
else()
  add_library(torch SHARED ${DUMMY_EMPTY_FILE})
endif()
torch_cuda_based_add_library(caffe2_gpu ${Caffe2_GPU_SRCS})
hip_add_library(caffe2_hip ${Caffe2_HIP_SRCS})
add_library(caffe2_pybind11_state MODULE ${Caffe2_CPU_PYTHON_SRCS})
add_library(caffe2_pybind11_state_gpu MODULE ${Caffe2_GPU_PYTHON_SRCS})
add_library(caffe2_pybind11_state_hip MODULE ${Caffe2_HIP_PYTHON_SRCS})
```
安装目录则寻找对应的install语句。此外，文件中还有一句
```
add_subdirectory(../torch torch)
```
（实际上caffe2目录下CMakeLists.txt中存在很多add_subdirectory，但是都是类似的处理过程，所以不一一说明，仅以torch这个目录进行说明）

于是查看torch目录下的CMakeLists.txt， 其中生成的库为
```
add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})
```
然后根据其中的
```
set(LIBSHM libshm)
set(LIBSHM_SRCDIR ${TORCH_SRC_DIR}/lib/${LIBSHM_SUBDIR})
add_subdirectory(${LIBSHM_SRCDIR})
```
继续查看torch/lib/libshm下的CMakeLists.txt，其中生成的库为
```
ADD_LIBRARY(shm SHARED core.cpp)
```
有关的库依赖，分为预装库和本项目（pytorch）内包含的库，CMake生成规则位于cmake/Dependencies.cmake文件中，仔细查看该文件发现：
- 预先装的库依赖，这些库名存在Caffe2_PUBLIC_DEPENDENCY_LIBS中。如上文所举例子OpenBLAS 那样添加g++的链接flag和 `-I<include dir>flag`。
- 本项目内包含的库。包括：
(1) tbb
```
add_subdirectory(${CMAKE_SOURCE_DIR}/aten/src/ATen/cpu/tbb)    # 添加tbb库
```
(2) qnnpack
```
# 添加 qnnpack 库
# source directory为${PROJECT_SOURCE_DIR}/third_party/QNNPACK
# output directory为${PROJECT_BINARY_DIR}/confu-deps/QNNPACK
add_subdirectory("${QNNPACK_SOURCE_DIR}" "${CONFU_DEPENDENCIES_BINARY_DIR}/QNNPACK")
list(APPEND Caffe2_DEPENDENCY_LIBS qnnpack)
```
最后一行指引CMake去QNNPACK的目录（位于third_party下）去生成qnnpack库，然后回到Dependencies.cmake中添加到Caffe2_DEPENDENCY_LIBS中。
(3) nnpack
```
# 添加 nnpack
include(${CMAKE_CURRENT_LIST_DIR}/External/nnpack.cmake)
```
跳至nnpack.cmake文件，发现其中包含
```
add_subdirectory(${NNPACK_SOURCE_DIR} ${CONFU_DEPENDENCIES_BINARY_DIR}/NNPACK)
```
找到包含NNPACK的代码目录位于third_party下，显然这个NNPACK也应该包含CMakeLists.txt文件指示CMake 生成nnpack库，然后回到Dependencies.cmake中将nnpack添加到Caffe2_DEPENDENCY_LIBS。

(4) 类似地，还添加了 cpuinfo，gflag，glog::glog，googletest，fbgemm，fp16等。这些也不一定全部使用，是否使用还得看相应配置

(5) LMDB。使用如下语句
```
find_package(LMDB)
```
所以去cmake/Modules目录下寻找FindLMDB.cmake， 在这个.cmake文件中寻找lmdb库以及lmdb.h头文件（linux中已经安装，分别位于/usr/lib/x86_64-linux-gnu和/usr/include）, 将库名称和头文件目录分别保存于变量LMDB_LIBRARIES和LMDB_INCLUDE_DIR，然后回到Dependencies.cmake，照例执行
```
include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
list(APPEND Caffe2_DEPENDENCY_LIBS ${LMDB_LIBRARIES})
```
类似的，还可以添加OPENCL，LEVELDB，NUMA，ZMQ，REDIS，OPENCV，FFMPEG，Python，MPI等。

(6) pybind11。在Dependencies.cmake添加pybind11依赖，
```
find_package(pybind11 CONFIG)# 配置模式下寻找，然而没有${pybind11_DIR}，也没有pybind11Config.cmake
if(NOT pybind11_FOUND)
  find_package(pybind11)     # 继续module模式下寻找
endif()
```
虽然存在cmake/Modules/Findpybind11.cmake，然而其中find_path并没有找到pybind11/pybind11.h这个头文件，因为我没有预先安装pybind11，CMake自然是找不到的，于是在Dependencies.cmake中直接添加
```
include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/pybind11/include)

```
(7) OPENMP
```
FIND_PACKAGE(OpenMP QUIET)
```
如果找到OpenMP，那么${OpenMP_CXX_FLAGS} 和 ${OpenMP_CXX_LIBRARIES}分别存储头文件搜索路径和库文件链接flag，生成caffe2时可以用到OpenMP，用法是在caffe2/CMakeLists.txt中，
```
target_compile_options(caffe2 INTERFACE ${OpenMP_CXX_FLAGS})
target_link_libraries(caffe2 PRIVATE ${OpenMP_CXX_LIBRARIES})
```
(8) CUDA。在Dependencies.cmake中有
```
include(${CMAKE_CURRENT_LIST_DIR}/public/cuda.cmake)
```
在这个cuda.cmake中，使用 find_library寻找cuda相关的库，找到后作为IMPORTED target进行库的添加，
```
add_library(caffe2::cuda UNKNOWN IMPORTED)
```
其他cuda有关的库类似的进行添加，包括caffe2::cudart，caffe2::cudnn，caffe2::curand，caffe2::cufft，caffe2::tensorrt， caffe2::cublas，caffe2::nvrtc，当然这些库不一定全部添加，根据配置决定添加哪些库，然后回到Dependencies.cmake中，
```
list(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::cuda caffe2::nvrtc)
```
保存到Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS，将来在caffe2/CMakeLists.txt用于链接。

(9) 其他的依赖库如NCCL，CUB，GLOO等与上述某一点说明类似，不再一一罗列。

Dependencies.cmake中有很多库是作为生成caffe2库的依赖，比如QNNPACK，对这部分库添加到Caffe2_DEPENDENCY_LIBS（或Caffe2_PUBLIC_DEPENDENCY_LIBS，Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS），这个使用下面语句（位于caffe2/CMakeLists.txt）得到链接flag
```
target_link_libraries(caffe2 PRIVATE ${Caffe2_DEPENDENCY_LIBS})
```
2. 生成python的扩展库。首先后三个有关caffe2的扩展已经在上一步中生成并安装，所以对于剩下的两个扩展予以说明。

- torch._C 链接的两个库为
```
main_libraries=['shm', 'torch_python']
```
显然前面已经生成了这两个库。而使用的源文件则为
```
main_sources=["torch/csrc/stub.cpp"]
```
- torch._dl此扩展使用源文件torch/csrc/dl.c生成 。查看这个文件，发现就是添加了<dlfcn.h>中的三个常量到torch._dl库中，如下
```
RTLD_GLOBAL=0x100
RTLD_NOW   =0x2
RTLD_LAZY  =0x1
```
这三个常量指示动态加载（比如加载torch._C）的模式，用于dlopen()方法中，增加这三个常量是为了防止python 的os 模块中没有这些flag，并且也没有python的DLFCN模块，此时可以从torch._dl中得到这些flag。相当于把torch._dl当作备胎。

### 还有...
可能，大概了解清楚PyTorch的安装过程了，毕竟安装过程我也没试过（只试过较老版本的安装），没有看到最终生成的各种文件，仅供参考吧。