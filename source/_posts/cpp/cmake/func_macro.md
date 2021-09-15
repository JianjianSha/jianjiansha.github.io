---
title: function vs. macro
date: 2021-08-10 14:09:28
tags: cmake, c++
p: cpp/cmake/func_macro
---

# macro
宏定义语法，
```cmake
macro(<name> [<arg1> ...])
  <commands>
endmacro()
```
宏名称 `<name>` 后跟参数 `<arg1>,...`。
