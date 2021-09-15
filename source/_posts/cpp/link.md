---
title: 链接
date: 2021-06-08 11:06:44
tags: c++
p: cpp/link
---
## 动态库目录
1. 加入 `/lib`, `/usr/lib` 等默认搜索路径
2. 设置 `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<YOUR LIB PATH>`
3. 修改配置文件 `/etc/ld.so.conf`，然后执行 `ldconfig`
4. `gcc` 添加选项 `-Wl,-rpath=<YOUR LIB PATH>`。这个路径会保存到程序中。
以上是动态（运行时）链接器（也可称为加载器）寻找动态库的搜索目录。


### -Wl
这个参数表示后面的参数传递给链接器 `ld`

### -rpath
添加一个目录到运行库搜索路径，可以使用 `$ORIGIN`，它表示执行文件所在的目录，注意在 Makefile 中需要写为 `$$ORIGIN`。

### --as-needed
链接器参数，表示仅链接其 symbol 在 binary 中用到的库。默认是开启这个 flag，若要关闭，使用 `--no-as-needed`，例如

```c++
// main.c
void foo();

int main(void) {
    foo();
    return 0;
}
```

```c++
// lib/foo.c
#include <stdio.h>

void foo() { printf("foo\n"); }
```

```c++
// lib/bar.c
#include <stdio.h>

void bar() { printf("bar\n"); }
```

```makefile
# Makefile
main: main.c lib/libfoo.so lib/libbar.so
    gcc -o main main.c -Wl,--no-as-needed,-rpath='$$ORIGIN/lib' -L. -lfoo -lbar

lib/libfoo.so: lib/foo.c
    gcc -fPIC -shared -o lib/libfoo.so lib/foo.c

lib/libbar.so: lib/bar.c
    gcc -fPIC -shared -o lib/libbar.so lib/bar.c
```

查看 `main` 链接的库，
```shell
$ ldd main
```
