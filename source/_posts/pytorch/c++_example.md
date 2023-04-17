---
title: C++ Example (PyTorch)
date: 2023-03-06 14:29:23
tags: PyTorch
summary: 使用 C++ 调用 PyTorch
---

# 1. 一个简单的例子

## 1.1 实操

```c++
// main.cpp
#include <torch/torch.h>
#include <iostream>

int main()
{
    torch::Tensor tensor = torch::zeros({2, 2});
    std::cout << tensor << std::endl;

    torch::Tensor tensor1 = torch::rand({2, 3});
    std::cout << tensor1 << std::endl;

    torch::Tensor tensor2 = torch::abs(tensor1 - 1);
    std::cout << tensor2 << std::endl;

    return 0;
}
```

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example)

# list(APPEND CMAKE_PREFIX_PATH "/home/<username>/src/pytorch/torch/share/cmake/")

# message(${CMAKE_PREFIX_PATH})
# 指定 pytorch 的安装路径，便于 cmake 寻找
list(APPEND CMAKE_PREFIX_PATH "/home/<username>/src/pytorch")   


find_package(Torch REQUIRED)
# message(${TORCH_INCLUDE_DIRS})

add_executable(main main.cpp)
target_link_libraries(main ${TORCH_LIBRARIES})

# include 目录可以不用指定
# target_include_directories(main PUBLIC ${TORCH_INCLUDE_DIRS})
set_property(TARGET main PROPERTY CXX_STANDARD 14)
```

创建 build 目录，

```
mkdir build
cd build
```

文件目录结构为

```
build
main.cpp
CMakeLists.txt
```

生成指令为

```sh
# 当前目录为 build 目录
# 这里我机器上没有 make，故使用 Ninja，如果要使用 make 生成，执行 `cmake ..` 默认就使用 make
cmake -GNinja ..    
cmake --build .
```

## 1.2 分析

### 1.2.1 头文件引用

`torch/torch.h` 位于 `/home/<username>/src/pytorch/torch/include/torch/csrc/api/include` 目录（下文全部省去前缀 `/home/<username>/src/`，`pytorch` 是从 github 上 clone 的项目源码根目录）

**# 头文件引用关系**

```
torch/torch.h -> torch/all.h -> torch/types.h
```

在 `torch/types.h` 头文件中，有如下代码，

```c++
namespace torch {
    using namespace at;
    ...
}
```

这就将 `at::` 命名空间下的函数引入 `torch::` 命名空间中，例如 `torch::abs` 实际上就是 `at::abs` ，如果一个函数在 `torch::` 和 `at::` 中均存在，那么总是使用 `torch::` 中的函数，例如 `torch::zeros` 这个函数，在 `pytorch/torch/include/torch/csrc/autograd/generated/variable_factories.h` 中也存在定义，将其修改为，

```c++
inline at::Tensor zeros(at::IntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  size = at::IntArrayRef({5, 5});   // 增加这一句代码，将 size 改为 (5,5)
  return autograd::make_variable(at::zeros(size, at::TensorOptions(options).requires_grad(c10::nullopt)), /*requires_grad=*/options.requires_grad());
}
```

那么重新生成上面那个 example，会发现 `tensor` 的 size 为 (5,5)，这说明调用的是 `torch::` 自己定义的函数，而非 `at::` 中定义的函数。（测试完毕还需要将 variable_factories.h 文件恢复为修改前状态——删除增加的那句代码）



