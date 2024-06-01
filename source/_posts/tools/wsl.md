---
title: wsl 子系统
date: 2024-05-22 10:35:48
tags: wsl
p: tools/
---


```sh
# 查看可以在线安装的 wsl 系统版本
wsl --list --online
# 安装一个 wsl 发行版
wsl --install Ubuntu-20.04
# 登录子系统
wsl -d Ubuntu-20.04

# 登录默认子系统，例如 Ubuntu
wsl
```

**# 查看当前机器上已经安装的 wsl 系统**

```sh
wsl -l -v
# 或者使用下面的，都一样
wsl --list --verbose
```

**# 默认系统**

```sh
# 查看默认子系统
wslconfig /list
# 修改默认子系统
wslconfig /setdefault Ubuntu-20.04
```

# conda 错误总结

```
Error while loading conda entry point: conda-libmamba-solver (libarchive.so.19: cannot open shared object file: No such file or directory)
```

原因是这些包必须来自同一通道。
解决方案：
```
conda install --solver=classic conda-forge::conda-libmamba-solver conda-forge::libmamba conda-forge::libmambapy conda-forge::libarchive
```