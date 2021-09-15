---
title: cmake 查找
date: 2021-06-08 14:08:47
tags:
p: cpp/cmake_find
---
cmake 的包查找。
<!-- more -->

# find_package

1. CMake 内置模型引入依赖包

为了方便我们在项目中引入外部依赖包，cmake官方为我们预定义了许多寻找依赖包的Module，他们存储在`path_to_your_cmake/share/cmake-<version>/Modules`目录下。每个以`Find<LibaryName>.cmake`命名的文件都可以帮我们找到一个包。我们也可以在官方文档中查看到哪些库官方已经为我们定义好了，我们可以直接使用find_package函数进行引用官方文档：[Find Modules](https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html)。

例如 `CURL`，可直接使用
```
find_package(CURL)
```

每一个模块都会定义以下几个变量 -

```
<LibaryName>_FOUND

<LibaryName>_INCLUDE_DIR or <LibaryName>_INCLUDES <LibaryName>_LIBRARY or <LibaryName>_LIBRARIES
```

2. 引入支持 cmake 编译安装的库

例如项目引用库 `glog`，cmake 的 Module 目录下没有 `FindGlog.cmake`，于是需要自行安装 glog 库，安装过程如下，

```
# clone该项目
git clone https://github.com/google/glog.git 
# 切换到需要的版本 
cd glog
git checkout v0.40  

# 根据官网的指南进行安装
cmake -H. -Bbuild -G "Unix Makefiles"
cmake --build build
cmake --build build --target install
```

此时我们便可以通过与引入curl库一样的方式引入glog库了
```
find_package(GLOG)
add_executable(glogtest glogtest.cc)
if(GLOG_FOUND)
    # 由于glog在连接时将头文件直接链接到了库里面，所以这里不用显示调用target_include_directories
    target_link_libraries(glogtest glog::glog)
else(GLOG_FOUND)
    message(FATAL_ERROR ”GLOG library not found”)
endif(GLOG_FOUND)
```

find_package 内部是如何查找依赖库的呢？有两种模式，

## Module 模式

如 curl 库的引用方式。cmake 首先找到 `Find<LibraryName>.cmake` 文件，这个 `.cmake` 文件可以从 `share/cmake-<version>/Modules` 目录下寻找，还可以从 `CMAKE_MODULE_PATH` 这个变量指定的搜索目录下搜索。

如果 Module 模式没有找到这个 `.cmake` 文件，那么尝试 Config 模式，这就是另外一个模式。

## Config 模式
通过 `<LibraryName>Config.cmake` 或 `<lower-case-package-name>-config.cmake` 这个配置文件来引入依赖库。创建包配置文件可以参考我的另一篇文章 [cmake 导入导出](/2021/06/03/cpp/cmake_im_ex)。


# find_library
简单形式为 
```
find_library (<VAR> name1 [path1 path2 ...])
```
一般形式为
```
find_library (
          <VAR>
          name | NAMES name1 [name2 ...] [NAMES_PER_DIR]
          [HINTS path1 [path2 ... ENV var]]
          [PATHS path1 [path2 ... ENV var]]
          [PATH_SUFFIXES suffix1 [suffix2 ...]]
          [DOC "cache documentation string"]
          [NO_DEFAULT_PATH]
          [NO_CMAKE_ENVIRONMENT_PATH]
          [NO_CMAKE_PATH]
          [NO_SYSTEM_ENVIRONMENT_PATH]
          [NO_CMAKE_SYSTEM_PATH]
          [CMAKE_FIND_ROOT_PATH_BOTH |
           ONLY_CMAKE_FIND_ROOT_PATH |
           NO_CMAKE_FIND_ROOT_PATH]
         )
```
命令用于查找库，结果保存在一个名为 `<VAR>` 的缓存条目里面。如果没有找到，那么结果为 `<VAR>-NOTFOUND`。搜索库的名为 `name1` 等。搜索路径可以通过 `PATHS` 选项指定。`PATH-SUFFIXES` 指定在查找搜索路径之外还查找搜索路径的子目录。如果 `NO_DEFAULT_PATH` 指定，那么不使用默认搜索路径，否则，搜索过程如下：

1. 搜索某些特定 cmake 的缓存变量中的路径，例如在命令行中指定 `-DVAR=value`。如果设置了 `NO_CMAKE_PATH`，那么不考虑这条搜索规则。下面考虑这些特定 cmake 缓存变量
    - 如果设置了 `CMAKE_LIBRARY_ARCHITECTURE`，那么搜索 `<prefix>/lib/<arch>` 路径；以及 `<prefix>/lib` 路径，其中 `<prefix>` 是 `CMAKE_PREFIX_PATH` 指定的路径集合之一。
    - 搜索 `CMAKE_LIBRARY_PATH` 中的路径
    - 搜索 `CMAKE_FRAMEWORD_PATH` 中的路径

2. 搜索某些特征 cmake 的环境变量中的路径。环境变量在用户的 shell 配置中设置，例如 `~/.bashrc`。如果设置了 `NO_CMAKE_ENVIRONMENT_PATH`，那么不考虑这条规则。
    - 如果设置了 `CMAKE_LIBRARY_ARCHITECTURE`，那么搜索 `<prefix>/lib/<arch>` 路径；以及 `<prefix>/lib` 路径，其中 `<prefix>` 是 `CMAKE_PREFIX_PATH` 指定的路径集合之一。
    - 搜索 `CMAKE_LIBRARY_PATH` 中的路径
    - 搜索 `CMAKE_FRAMEWORD_PATH` 中的路径

3. 搜索由 `HINTS` 指定的路径。这个选项不是特别理解。暂不详述。

4. 搜索标准的系统环境变量，如果设置了 `NO_SYSTEM_ENVIRONMENT_PATH`，那么这条规则不启用。
    - 如果设置了 `CMAKE_LIBRARY_ARCHITECTURE`，那么搜索 `<prefix>/lib/<arch>`，此外搜索 `PATH` 环境变量中每个路径条目，如果路径条目形式为 `<prefix>/[s]bin`，那么搜索路径 `<prefix>/lib`，否则就以路径条目为 `<prefix>` 来搜索路径 `<prefix>/lib`已经路径条目自身 `<prefix>`

5. 搜索与当前操作系统相关的 cmake 变量。如果设置了 `NO_CMAKE_SYSTEM_PATH`，那么这条规则不启用。
    - 如果设置了 `CMAKE_LIBRARY_ARCHITECTURE`，那么搜索 `<prefix>/lib/<arch>` 路径；以及 `<prefix>/lib` 路径，其中 `<prefix>` 是 `CMAKE_SYSTEM_PREFIX_PATH` 指定的路径集合之一。
    - 搜索 `CMAKE_SYSTEM_LIBRARY_PATH` 中的路径
    - 搜索 `CMAKE_SYSTEM_FRAMEWORD_PATH` 中的路径

