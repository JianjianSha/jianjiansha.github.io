---
title: cpp-aux-tools
date: 2019-07-11 19:16:23
tags: c++
---
来看一个 c++ 程序片段
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