---
title: cmake TARGET
date: 2021-06-08 17:15:50
tags:
p: cpp/target
---

# target_compile_definitions
```
target_compile_definitions(<target>
  <INTERFACE|PUBLIC|PRIVATE> [items1...]
  [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
```

指定编译给定目标 `<target>` 时的编译定义。目标 `<target>` 由 `add_executable()` 或者 `add_library()` 创建。

- PRIVATE, PUBLIC

    这两个选项指定给 `<target>` 的 `COMPILE_DEFINITIONS` 属性赋值（append）

- PUBLIC, INTERFACE

    指定给 `<target>` 的 `INTERFACE_COMPILE_DEFINITIONS` 属性赋值（append）

编译定义中的前导 `-D` 会被移除，空定义项被忽略。以下各行等价，
```
target_compile_definitions(foo PUBLIC FOO)
target_compile_definitions(foo PUBLIC -DFOO)  # -D removed
target_compile_definitions(foo PUBLIC "" FOO) # "" ignored
target_compile_definitions(foo PUBLIC -D FOO) # -D becomes "", then ignored
```

# set_target_properties
```
set_target_properties(target1 target2 ...
                      PROPERTIES prop1 value1
                      prop2 value2 ...)
```
设置目标的属性。

# target_include_directories
```
target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
  <INTERFACE|PUBLIC|PRIVATE> [items1...]
  [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
```
为目标添加头文件路径。

- PRIVATE, PUBLIC

    给目标的 `INCLUDE_DIRECTORIES` 属性添加值

- PUBLIC, INTERFACE

    给目标的 `INTERFACE_INCLUDE_DIRECTORIES` 属性添加值

指定的包含目录可以是绝对或相对路径，对同一个目标调用多次这个命令，则会按顺序附加包含目录。

`target_include_directories` 命令参数可能会使用 “生成器表达式”，语法为 `$<...>`。关于生成器表达式可参考[这里](https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html#manual:cmake-generator-expressions(7))。 例如，

```
target_include_directories(mylib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/mylib>
  $<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
)
```
给目标添加包含目录。如果使用 `export()` 导出这个目标的包含目录属性，那么使用 `${CMAKE_CURRENT_SOURCE_DIR}/include/mylib>`，如果使用 `install(EXPORT)` 导出目标的包含目录属性，那么使用 `include/mylib>`。



