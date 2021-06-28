---
title: cmake 常用命令（一）
date: 2021-06-04 18:50:48
tags:
---
## find_program
```
find_program (
          <VAR>
          name | NAMES name1 [name2 ...] [NAMES_PER_DIR]
          [HINTS path1 [path2 ... ENV var]]
          [PATHS path1 [path2 ... ENV var]]
          [PATH_SUFFIXES suffix1 [suffix2 ...]]
          [DOC "cache documentation string"]
          [REQUIRED]
          [NO_DEFAULT_PATH]
          [NO_PACKAGE_ROOT_PATH]
          [NO_CMAKE_PATH]
          [NO_CMAKE_ENVIRONMENT_PATH]
          [NO_SYSTEM_ENVIRONMENT_PATH]
          [NO_CMAKE_SYSTEM_PATH]
          [CMAKE_FIND_ROOT_PATH_BOTH |
           ONLY_CMAKE_FIND_ROOT_PATH |
           NO_CMAKE_FIND_ROOT_PATH]
         )
```
简单形式为 `find_program (<VAR> name1 [path1 path2 ...])`。

查找程序。一个缓存条目 `VAR` 存储命令结果，如果没有找到，结果为 `<VAR>-NOTFOUND`。
- NAMES

指定一个或多个程序名称

- HINTS, PATHS

指定在默认目录之外的搜索路径。`ENV var` 指定环境变量。

- PATH-SUFFIXES

指定在每个搜索路径下要检查的子路径。

如果 `NO_DEFAULT_PATH` 指定，那么默认搜索路径不再考虑，否则搜索过程如下：

...

## file
`file(WRITE filename "message to write" ...)`
将信息写如文件 `filename` 中。

`file(APPEND filename "message to write" ...)`
将信息追加到文件末尾

`file(READ filename variable [LIMIT numBytes] [OFFSET offset] [HEX])`
读取文件内容并存储到 `variable` 中。`HEX` 表示二进制数据转为为十进制。

`file({GLOB|GLOB_RECURESE} <variable> ... [<globbing-expressions>...])`
根据表达式 `<globbing-expressions>` 匹配文件，并将文件列表保存到 `<variable>` 中。

`GLOB_RECURSE` 将会遍历匹配的目录的子目录，从而匹配文件，例：
```
/dir/*.py  - match all python files in /dir and subdirectories
```

## configure_file
`configure_file(<input> <output>
               [NO_SOURCE_PERMISSIONS | USE_SOURCE_PERMISSIONS |
                FILE_PERMISSIONS <permissions>...]
               [COPYONLY] [ESCAPE_QUOTES] [@ONLY]
               [NEWLINE_STYLE [UNIX|DOS|WIN32|LF|CRLF] ])`
将 `input` 文件内容复制到 `output` 文件中。根据参数规则，替换 `@VAR@` 或 `${VAR}` 变量。

`<input>` 文件中 `#cmakedefine VAR` 会被替换为：
1. 如果 `VAR` 设置为 `ON`，那么替换为 `#define VAR`
2. 如果 `VAR` 设置为 `OFF`，那么替换为 `/* #undef VAR */`

同理，`#cmakedefone01 VAR` 则会被替换为 `#define VAR 1` 或 `#define VAR 0`

注：这个命令的 `IMMEDIATE` 选项已经被弃用，因为文件复制已经是立即执行。


## 内置模块
cmake 提供了一些内置模块，可以直接 include 然后使用，参见 [这里](https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html)




