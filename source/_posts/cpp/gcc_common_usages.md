---
title: GCC common usages
date: 2021-08-06 09:20:22
tags: C++
---

示例程序
```c
// hello.c
#include <stdio.h>
int main(int argc, char **argv)
{
 printf(“hello world\n”);
 return 0;
}
```
# 生成可执行文件
```shell
$ gcc -o hello hello.c
```
生成可执行文件这一过程其实是分成了很多中间步骤，下面给出关键的几个步骤。

# 预处理
使用 `-E` 使得预处理之后就立即停止，不进行后面的编译
```shell
gcc -E -o hello.pp.c hello.c
```

# 编译

```shell
$ gcc -S hello.c
```
此命令在编译后立即停止，不进行汇编（汇编后生成二进制文件，即目标文件）。结果生成文件 `hello.s`。

```shell
$ gcc -c hello.c
```
编译并汇编但不进行链接，生成目标 `hello.o`。


gcc 命令选项可以查看 [官方文档](https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html)。

编译后的目标文件有多个 section，和一个符号表：
1. `Text`：可执行代码 （T）
2. `Data`: 预分配变量存储 （D）
3. `Constants`：只读数据    （R）
4. `Undefined`: 已经用到但是未定义的符号 （U）
5. `Debug`：调试信息（例如，行号）

这些条目可以通过 `nm` 或者 `readelf` 查看。

# 链接命令
链接使用如下命令，
```shell
$ gcc -o hello hello.o
```
gcc 其实是调用 `ld` 讲目标文件 `crt1.o` 和我们的 `hello.o` 链接到一起，其中 `crt1.o` 为启动目标，`crtn.o` 为结束目标，`crt1.o` 包含了 `_start` 入口点，其中会调用 `main` 函数，执行我们所实现的函数主体。这可以通过 `nm` 查看，
```shell
$ nm /usr/lib/x86_64-linux-gnu/crt1.o
```
其中会有 `U main` 这一行，表示 `main` 这个符号未定义，这是由我们自己实现 `main` 函数。

其实还有 `crti.o` 目标文件也会链接进入，这个目标文件中实现了 `_init` 和 `_finit` 函数，分别在 `main` 函数之前和 之后执行。
![](/images/cpp/C_linking_process.png)<center>C 程序链接过程 参考[文章](https://akaedu.github.io/book/ch19s02.html)</center>

# 静态库
使用 `ar` 命令生成静态库，静态库是有着全局符号表的目标文件的集合。
```shell
$ ar crv libhello.a hello.o
```
关于 `ar` 命令的更多介绍可参考[这篇文章](https://linux.die.net/man/1/ar)

当链接到一个静态库时，目标代码拷贝进最终的可执行体，所有的符号地址重新计算。

静态库抽取出目标文件，
```shell
$ ar -x libhello.a
```

# 动态库

动态库比静态库更接近于可执行体（executable），动态库相当于 executable 少了 `main` 函数，而静态库相当于多个目标文件的打包。

生成动态库命令，
```shell
$ gcc -fPIC -c hello.c
```
这里不能链接，否则编译器会报错：`main` 函数未定义。

然后链接生成动态库，
```shell
$ ld -shared hello.o -o libhello.so
```
我们也可以直接一步生成动态库，
```shell
gcc -shared -fPIC -o hello.c
```

# 链接过程
链接器分为静态链接器和动态（运行时）链接器。
1. 静态链接器 static linker 负责生成 shared library 和 executable，linux 上为 `ld`，google 还提供了一个替代 `gold`。
2. 动态链接器 dynamic linker 负责在执行期间载入 shared library，linux 上使用 `ld.so`。

为了称呼简便，下文将 static linker 称为 linker，将 dynamic linker 称为 loader。

## 搜索路径
1. 对于 linker，使用 `-L` option，例如 `ld -o main main.c -L. -lhello`
2. 对于 loader，使用 `-rpath`，例如 `gcc -o main main.c -Wl,-rpath=. -L. -lhello`

动态库搜索路径的设置方法有以下几种：

1. 如果当前目录 `.` 下 hello 库为静态库 `libhello.a`，那么 使用第 `1` 条命令即可，如果是 动态库 `libhello.so`，那么仅指定 `-L. -lhello` 还不够，还需要指定 `-Wl,-rpath=.`，其他 `-Wl` 表示逗号之后的 option 均传给链接器 linker，`-rpath` 表示将动态库的目录嵌入到可执行体，这样运行时 loader 才能找到动态库。

2. 设置 `LD_LIBRARY_PATH` 这个环境变量，使得 loader 在运行时尝试从这个环境变量指定的目录中寻找动态库。

3. 将动态库目录添加至配置文件。配置文件通常为 `/etc/ld.so.conf`，在此文件中添加搜索路径条目，然后再执行 `ldconfig`。如果将动态库安装到 `/lib` 或 `/usr/lib` ，则不需要修改配置文件，只需要执行 `ldconfig`即可。

注意：`LD_LIBRARY_PATH` 有副作用，不是一个好的方法，应尽量避免使用。

给 `-rpath` 选项赋值，有时候会使用 `$ORIGIN`，这个路径表示可执行体所在的目录，而 `.` 则表示当前的工作目录（编译链接命令执行的路径）。如果是在 Makefile 文件中，则需要写成 `-Wl,-rpath='$$ORIGIN'`，在命令窗口中写成 `-Wl,-rpath='$ORIGIN'`。