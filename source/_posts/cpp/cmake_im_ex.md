---
title: cmake import export
date: 2021-06-03 13:59:06
tags:
---

cmake 的导入导出机制。
<!-- more -->

# 导入目标
被导入的目标位于当前 cmake 项目的外部。要创建一个被导入目标，可在 `add_executable()` 和 `add_library()` 中加入 `IMPORTED` 选项，`IMPORTED` 选项是的这两个命令不会生成真正的目标文件（即，没有物理文件生成，而是将外部的目标文件作为当前cmake 项目的逻辑目标）。使用这两个命令导入后，被导入目标可以像其他目标一样被引用并使用。被导入目标的默认 scope 为当前目录以及子目录，可以使用 `GLOBAL` 使得被导入目标在 cmake 生成系统全局可见，
```
add_executable(<name> IMPORTED [GLOBAL])
```
## 导入可执行体
以一个例子说明，完整代码位于 cmake 官方代码库的 Help/guide/importing-exporting 目录下。

操作命令如下，
```shell
$ cd Help/guide/importing-exporting/MyExe
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
$ cmake --install . --prefix <install location>
$ <install location>/myexe
$ ls
[...] main.cc [...]
```
为了方便，也给出了 CMakeLists.txt 文件内容，
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyExe)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add executable
add_executable(myexe main.cxx)

# install executable
install(TARGETS myexe)
```
整个文件目录为，
```
MyExe/
    CMakeLists.txt
    main.cxx
```
main.cxx 文件 main 函数执行后会创建一个 main.cc 的文件。

现在我们将这个生成的 `myexe` 可执行体导入到另一个项目中。另一个项目源码位于 Help/guide/importing-exporting/Importing，其中 CMakeLists.txt 文件内容为，
```cmake
cmake_minimum_required(VERSION 3.15)
project(Importing)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add executable
add_executable(myexe IMPORTED)

# Set imported location
set_property(TARGET myexe PROPERTY
             IMPORTED_LOCATION "../InstallMyExe/bin/myexe")

# Add custom command to create source file
add_custom_command(OUTPUT main.cc COMMAND myexe)

# Use source file
add_executable(mynewexe main.cc)
```

以上，`myexe` 使用 `IMPORTED` 告诉 CMAKE 这是一个外部引用，并设置其属性 `IMPORTED_LOCATION`，这样就知道外部目标文件的位置。

```
add_custom_command(OUTPUT main.cc COMMAND myexe)
```
上面这句命令指定构建时执行的命令为 `myexe`，生成的输出文件为 `main.cc` （这是一个相对于当前源目录的文件路径），这句指令本身不会让 cmake 构建，而是下一句，
```
add_executable(mynewexe main.cc)
```
这句构建一个可执行目标，该目标构建依赖于 `main.cc`。

## 导入库
与可执行目标导入类似，库文件也可以被导入。
```
add_library(foo STATIC IMPORTED)
set_property(TARGET foo PROPERTY
             IMPORTED_LOCATION "path/to/libfoo.a")
```
添加一个导入静态库，并设置其路径属性。

使用这个导入库如下，
```
add_executable(myexe src1.c src2.c)
target_link_libraries(myexe PRIVATE foo)
```

# 导出目标
导入库有用，但是需要知道被导入库的文件路径。被导入目标的真正强大之处在于，当 cmake 项目提供目标文件时，cmake 项目同时提供一个 CMake 文件 .cmake，使得在其他地方可以非常方便的导入这些目标。

首先定位到 cmake 官方代码库的 Help/guide/importing-exporting/MathFunctions 目录，其中头文件 `MathFunctions.h` 的内容为，
```cpp
#pragma once
namespace MathFunctions {
double sqrt(double x);
}
```
源文件 `MathFunctions.cxx` 为，
```cpp
#include "MathFunctions.h"
#include <cmath>
namespace MathFunctions {
double sqrt(double x) {
    return std::sqrt(x);
}
}
```
CMakeLists.txt 文件内容较多，
```cmake
cmake_minimum_required(VERSION 3.15)
project(MathFunctions)

# make cache variables for install destinations
include(GNUInstallDirs)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# create library
add_library(MathFunctions STATIC MathFunctions.cxx)

# add include directories
target_include_directories(MathFunctions
                           PUBLIC
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
                           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# install the target and create export-set
install(TARGETS MathFunctions
        EXPORT MathFunctionsTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install header file
install(FILES MathFunctions.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# generate and install export file
install(EXPORT MathFunctionsTargets
        FILE MathFunctionsTargets.cmake
        NAMESPACE MathFunctions::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MathFunctions
)

# include CMakePackageConfigHelpers macro
include(CMakePackageConfigHelpers)

# set version
set(version 3.4.1)

set_property(TARGET MathFunctions PROPERTY VERSION ${version})
set_property(TARGET MathFunctions PROPERTY SOVERSION 3)
set_property(TARGET MathFunctions PROPERTY
  INTERFACE_MathFunctions_MAJOR_VERSION 3)
set_property(TARGET MathFunctions APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING MathFunctions_MAJOR_VERSION
)

# generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfigVersion.cmake"
  VERSION "${version}"
  COMPATIBILITY AnyNewerVersion
)

# create config file
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MathFunctions
)

# install config files
install(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MathFunctions
)

# generate the export targets for the build tree
export(EXPORT MathFunctionsTargets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsTargets.cmake"
       NAMESPACE MathFunctions::
)
```

构建库 `MathFunctions` 之后需要为其指定头文件目录，根据是生成库还是从已安装路径使用库，分别指定不同的头文件目录，如果对头文件目录不加以区分，那么 CMake 在创建导出信息时，将会导出依赖于当前生成目录的一个路径，这个路径显然在其他项目中无效。

`install(TARGETS)` 和 `install(EXPORT)` 安装库目标和 .cmake 文件，这里 .cmake 文件则方便其他 CMake 项目导入这个库目标。生成的导出文件（.cmake文件）中包含了创建导入库的代码，
```cmake
# Create imported target MathFunctions::MathFunctions
add_library(MathFunctions::MathFunctions STATIC IMPORTED)

set_target_properties(MathFunctions::MathFunctions PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
)
```
这段代码与上面我们手动导入库的 cmake 代码很相似。外部其他项目可以 include 这个 .cmake 文件，从而引用导入库 `MathFunctions`，
```cmake
include(${INSTALL_PREFIX}/lib/cmake/MathFunctionTargets.cmake)
add_executable(myexe src1.c src2.c )
target_link_libraries(myexe PRIVATE MathFunctions::MathFunctions)
```
注：这段代码来自官方文档，但是个人觉得这里路径错了，应该是 
`include(${INSTALL_PREFIX}/lib/cmake/MathFunctions/MathFunctionTargets.cmake)`。



任意数量的目标都可以关联到相同的导出名称，且 `install(EXPORT)` 只需要调用一次。__导出名称是全局 scope 的，所以任何目录都可以使用__ 。例如以下的导出名称 `myproj-targets`，
```cmake
# A/CMakeLists.txt
add_executable(myexe src1.c)
install(TARGETS myexe DESTINATION lib/myproj
        EXPORT myproj-targets)

# B/CMakeLists.txt
add_library(foo STATIC foo1.c)
install(TARGETS foo DESTINATION lib EXPORTS myproj-targets)

# Top CMakeLists.txt
add_subdirectory (A)
add_subdirectory (B)
install(EXPORT myproj-targets DESTINATION lib/myproj)
```

## 创建包
我们还可以生成一个配置文件，以便 `find_package()` 可以发现目标。步骤如下，

1. include `CMakePackageConfigHelpers` 模块，获得创建配置文件的函数。

### 创建包配置文件
使用 `CMakePackageConfigHelpers` 模块中的 `configure_package_config_file()` 命令生成包配置文件，
```
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MathFunctions
)
```
`INSTALL_DESTINATION` 的路径值为 `MathFunctionsConfig.cmake` 安装路径。

`configure_package_config_file` 命令用于创建一个配置文件 `<PackageName>Config.cmake` 或者 `<PackageName>-Config.cmake`，
```
configure_package_config_file(<input> <output>
  INSTALL_DESTINATION <path>
  [PATH_VARS <var1> <var2> ... <varN>]
  [NO_SET_AND_CHECK_MACRO]
  [NO_CHECK_REQUIRED_COMPONENTS_MACRO]
  [INSTALL_PREFIX <path>]
  )
```
`INSTALL_DESTINATION` 可以是绝对路径，或者是相对 `INSTALL_PREFIX` 的路径。这个命令根据输入文件替换变量（@@包围的变量）的值得到输出文件。然后再安装到指定路径，这个配置文件中设置了 `MathFunctionsTargets.cmake` 的路径。

## 创建包版本文件

使用 `write_basic_package_version_file()` 创建包版本文件，当 CMAKE 使用 `find_package` 时，这个包版本文件将被 CMAKE 读取以决定版本是否兼容。


## 从生成树中导出目标

通常，一个项目都是在被外部其他项目使用之前就生成并安装完成，但是有些情况下，我们想在生成项目后直接导出目标，跳过安装过程，这时可以使用 `export()` 达成这一目的，如上文那一大段 CMakeLists.txt 内容的最后一个命令调用，在这个调用中，我们在生成目录创建文件 `MathFunctionsTargets.cmake`，但是需要注意，这个文件与 `lib/cmake/MathFunctions/MathFunctionsTargets.cmake` 不同，不具有路径重定向功能，因为其中 `MathFunctions` 目标的几个路径属性值全部是 hardcode 的，而非使用 `${_IMPORT_PREFIX}`。