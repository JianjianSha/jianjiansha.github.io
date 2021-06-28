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

### -Wl
这个参数表示后面的参数传递给链接器 `ld`

### -rpath
添加一个目录到运行库搜索路径

