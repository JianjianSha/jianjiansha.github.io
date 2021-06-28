---
title: cpp-aux-tools
date: 2019-07-11 19:16:23
tags: c++
p: cpp/aux-tools
---
来看一个 c++ 程序片段
<!-- more -->
```c++
// test.cpp
int f(int i) { return 0; }
```
编译
```
gcc test.cpp -o test.o
```
查看 f 的 low-level assembler 名称（name mangling），
```
nm test.o | grep f
// 输出
// 000000000000008b T _Z4fi
```
逆过程为
```
c++filt -n _Z4fi
// 输出
// f(int)
```

反汇编
```
objdump -d test.o
```
更多 option 可查看 `objdump --help`。

查看头文件搜索路径
```
gcc -xc++ -E -v -
```

查看链接库依赖
```
ldd test.o
```

设置动态库、头文件搜索目录的相关环境变量
```
export LD_LIBRARY_PATH=/xx/xx:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/xx/xx:$CPLUS_INCLUDE_PATH
```

## 生成静态库
```
g++ -c x.cpp
ar crv libx.a x.o
```

## 生成动态库
```
g++ -shared -fPIC -o libx.so x.o
```
