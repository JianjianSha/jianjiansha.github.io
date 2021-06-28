---
title: cmake tutorial
date: 2021-06-02 10:35:18
tags: cmake
---


记录 cmake 的各种用法。从一个简单的例子开始入手。

# 一个简单的例子

__main.cpp__

```cpp
#include <iostream>

int main() {
    std::cout << "Hello World!\n";
    return 0;
}
```
__CMakeLists.txt__
```cmake
cmake_minimum_required(VERSION 3.15)
project(hello_world)
add_executable(app main.cpp)
```
以上两个文件在同一目录 `demo` 下，
```
demo
| -- main.cpp
| -- CMakeLists.txt
```
在 `demo` 目录下，使用以下两个命令生成，
```shell
> cmake .
> cmake --build .
```
第一个命令是 生成 Makefile，第二个命令是生成 可执行文件。这种生成方式会生成一些文件，扰乱源码，于是可以在 `demo` 下创建一个 `build` 目录，所有生成的文件均放在 `build` 目录下，这样不污染源码文件，命令如下，
```sh
> mkdir build
> cd build
> cmake ..
> cmake --build .
```

# 多个源文件的生成
文件目录为
```
demo
| -- build
| -- main.cpp
| -- foo.h
| -- foo.cpp
| -- CMakeLists.txt
```

各文件内容为，

__main.cpp__
```cpp
#include "foo.h"

int main() {
    foo();
    return 0;
}
```

__foo.h__
```cpp
void foo();
```

__foo.cpp__
```cpp
#include <iostream>
#include "foo.h"

void foo() {
    std::cout << "Hello World!\n";
}
```

__CMakeLists.txt__
```cmake
cmake_minimum_required(VERSION 3.15)
project(hello_world)

add_executable(app main.cpp foo.cpp)
```

如果将头文件统一放入 `includes` 目标中，即
```
demo
| -- build
| -- inlcude
    |-- foo.h
| -- main.cpp
| -- foo.cpp
| -- CMakeLists.txt
```
那么还需要指定头文件搜索路径，否则找不到头文件，指定头文件搜索路径可使用
```cmake
...
inlcude_directories("${PROJECT_SOURCE_DIR}/includes")
add_executable(app main.cpp foo.cpp)
```
其中 路径的引号可以去掉。

# 生成库和链接库
修改 CMakeLists.txt 文件，其他不变，
```cmake
cmake_minimum_required(VERSION 3.15)
project (hello_world)

include_directories("${PROJECT_SOURCE_DIR}/includes")
add_library(foo foo.cpp)
add_executable(app main.cpp)
target_link_libraries(app foo)
```

# 添加头文件搜索路径
语法：
- `include_directories([AFTER|BEFORE] [SYSTEM] dir1 [dir2 ...])`

|参数|描述|
|--|--|
|dirN| 一个或多个相对路径或绝对路径|
|AFTER, BEFORE| 搜索路径是添加到当前搜索路径列表的后面还是前面。默认行为由 CMAKE_INCLUDE_DIRECTORIES_BEFORE 指定|
|SYSTEM|添加的路径是否视作系统头文件路径|

由于可以是相对路径，故前面添加头文件路径也可以写作
```
include_directories(include)
```
添加的头文件搜索路径，对当前 directory 中所有 targets 以及所有 subdirectories （由 add_subdirectory() 给定） 均有效。

# 生成目标
语法：
- `add_executable(target_name [EXCLUDE_FROM_ALL] source1 [source2...])`
- `add_library(lib_name [STATIC|SHARED|MODULE][EXCLUDE_FROM_ALL] source1 [source2...])`

例如，
```
add_executable(my_ext main.cpp util.cpp)
```

这会生成 `my_exe` 目标（例如 linux 上使用 `make my_exe`），默认情况下，所有的可执行目标均添加到 `all` 目标下，如果要从 `all` 下排除某个目标，可使用 `EXCLUDE_FROM_ALL` 参数，
```
add_executable(my_exe EXCLUDE_FROM_ALL main.cpp)
```

`add_library` 用于生成库，`BUILD_SHARED_LIBS` BOOL 型变量控制生成一个 static 库还是 shared 库，例如 `cmake .. -DBUILD_SHARED_LIBS=ON`，也可以直接指定，
```
add_library(my_lib SHARED lib.cpp)
```
`MODULE` 指定这个库在 runtime 时使用 `dlopen` 之类的函数进行动态加载。

# MACROS
宏和函数的区别是，函数本身是一个新的 scope，而宏则在当前 context 中执行，因此，函数中定义的变量在函数结束后变得未知，而宏中的变量在宏结束后继续保持定义。
例子：
```cmake
macro(set_my_variable _INPUT)
  if("${_INPUT}" STREQUAL "Foo")
    set(my_output_variable "foo")
  else()
    set(my_output_variable "bar")
  endif()
endmacro(set_my_variable)
```
使用宏，
```
set_my_variable("Foo")
message(STATUS ${my_output_variable})
```

# 多层级项目
项目目录
```
CMakeLists.txt
editor/
    CMakeLists.txt
    src/
        editor.cpp
highlight/
    CMakeLists.txt
    include/
        highlight.h
    src/
        highlight.cpp
```

文件内容为，
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project (example)

add_subdirectory(highlight)
add_subdirectory(editor)
```

highlight 库，
```cmake
# highlight/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project (highlight)

add_library(${PROJECT_NAME} src/highlight.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)
```
使用 `target_include_directories()` 代替 `include_directories()`，那么头文件搜索路径可以传递到这个库的使用者那里。

可执行程序，
```cmake
# editor/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project (editor)

add_executable(${PROJECT_NAME} src/editor.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC highlight)
```
cmake 自动处理 highlight 的库文件路径和头文件路径。


# 安装
指定安装时的所作的事情，即 `make install`，或者使用 `cmake --install .`。

## 安装目标文件

`install(TARGETS <target>... [...])`

target 指定被安装的目标，可以是多个，`[...]` 中指定安装选项，常见选项有，
- DESTINATION

指定目标安装的路径，可以是绝对路径或者相对路径，如果是相对路径，那么路径是相对于 `CMAKE_INSTALL_PREFIX`，此值默认为 `/usr/local`（linux），可以在命令选项中更改 `cmake -DCMAKE_INSTALL_PREFIX=/my/path ..`

- PERMISSIONS

指定安装文件的权限，值可以是 `OWNER_READ, OWNER_WRITE, OWNER_EXECUTE, GROUP_READ, GROUP_WRITE, GROUP_EXECUTE, WORLD_READ, WORLD_WRITE, WORLD_EXECUTE, SETUID, SETGID` 

多个目标可以是可执行文件，动态库，静态库，以及头文件等，可以通过选项 `RUNTIME, LIBRARY, ARCHIVE, PUBLIC_HEADER, PRIVATE_HEADER` 等分别指定，例如，
```cmake
INSTALL(TARGETS myapp, mylib, mystaticlib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
```

- CONFIGURATIONS

指定安装规则所应用的生成配置，例如 Debug，Release，这仅适用于 `CONFIGURATIONS` 之后的选项上，例如，对于 Debug 和 Release 配置，分别指定不用的安装路径，
```cmake
install(TARGETS target
    CONFIGURATIONS Debug
    RUNTIME DESTINATION Debug/bin)
install(TARGETS target
    CONFIGURATIONS Release
    RUNTIME DESTINATION Release/bin)
```
注：可通过 `-DCMAKE_BUILD_TYPE=Debug` 指定生成配置。

- RENAME

重命名被安装的目标文件。这个选项仅在命令中只有一个文件被安装的时候可以用。

- OPTIONAL

如果被安装的文件不存在，那么不会抛出错误。

- EXCLUDE_FROM_ALL

从默认安装中排除此文件的安装。

- COMPONENT

给安装规则指定一个组件名，于是可以安装指定的组件，而其他组件则不被安装。在全安装（不知道组件名）时，所有除了 EXCLUDE_FROM_ALL 的安装规则都将被执行。默认情况安装规则的组件名为 `Unspecified`。

完整的安装目标的命令如下，
```cmake
install(TARGETS targets... [EXPORT <export-name>]
        [[ARCHIVE|LIBRARY|RUNTIME|OBJECTS|FRAMEWORK|BUNDLE|
          PRIVATE_HEADER|PUBLIC_HEADER|RESOURCE]
         [DESTINATION <dir>]
         [PERMISSIONS permissions...]
         [CONFIGURATIONS [Debug|Release|...]]
         [COMPONENT <component>]
         [NAMELINK_COMPONENT <component>]
         [OPTIONAL] [EXCLUDE_FROM_ALL]
         [NAMELINK_ONLY|NAMELINK_SKIP]
        ] [...]
        [INCLUDES DESTINATION [<dir> ...]]
        )
```

- EXPORT

给安装的目标文件关联到一个导出上，导出名字为 `export-name`。EXPORT 必须出现在其他选项之前。

## 安装普通文件
命令如下，
```cmake
install(<FILES|PROGRAMS> files...
        TYPE <type> | DESTINATION <dir>
        [PERMISSIONS permissions...]
        [CONFIGURATIONS [Debug|Release|...]]
        [COMPONENT <component>]
        [RENAME <name>] [OPTIONAL] [EXCLUDE_FROM_ALL])
```
如果给出的是文件相对路径，那么是相对于当前源目录 `CMAKE_SOURCE_DIR`。
FILES 表示普通文件，PROGRAMS 表示非目标文件的可执行程序，如脚本。

- TYPE

不同的 TYPE 值，默认安装路径也不同。

## 安装目录
```cmake
install(DIRECTORY dirs...
        TYPE <type> | DESTINATION <dir>
        [FILE_PERMISSIONS permissions...]
        [DIRECTORY_PERMISSIONS permissions...]
        [USE_SOURCE_PERMISSIONS] [OPTIONAL] [MESSAGE_NEVER]
        [CONFIGURATIONS [Debug|Release|...]]
        [COMPONENT <component>] [EXCLUDE_FROM_ALL]
        [FILES_MATCHING]
        [[PATTERN <pattern> | REGEX <regex>]
         [EXCLUDE] [PERMISSIONS permissions...]] [...])
```

- DIRECTORY

此选项后面跟一个或多个目录，用于被安装到指定 `DESTINATION` 下。dirs 这个目录如果末尾没有 `/`，那么这个目录内容连同目录自身都将被安装，否则只安装目录的内容。

- USE_SOURCE_PERMISSIONS

`FILE_PERMISSIONS` 和 `DIRECTORY_PERMISSIONS` 用于指定目录中文件和目录的权限。如果指定了 `USE_SOURCE_PERMISSIONS` 且未指定 `FILE_PERMISSIONS`，那么从源目录结构中复制文件权限。

- PATTERN REGEX

精细粒度的控制目录安装。PATTERN 匹配完整的文件名，REGEX 使用正则匹配。例子，
```cmake
install(DIRECTORY icons scripts/ DESTINATION share/myproj
        PATTERN "CVS" EXCLUDE
        PATTERN "scripts/*"
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ)
```
将 `icons` 目录和 `scripts` 目录内容 安装到 `share/myproj`， 其中 `CVS` 子目录或文件不被安装。对于 `scripts/*` 中的文件，指定权限）。

在 `PATTERN "CVS" EXCLUDE` 中，如果去掉 `EXCLUDE`，那么 "CVS" 子目录或文件依然会被安装。

- FILES_MATCHING

默认情况下，无论文件是否匹配中，都会被安装。如果在所有匹配模式前面增加 `FILES_MATCHING` 选项，那么那些未被任何模式匹配中的文件或目录则不会被安装。例如，
```cmake
install(DIRECTORY src/ DESTINATION include/myproj
        FILES_MATCHING PATTERN "*.h")
```
仅安装源目录中的头文件。

## 安装时脚本运行
```cmake
install([[SCRIPT <file>] [CODE <code>]]
        [COMPONENT <component>] [EXCLUDE_FROM_ALL] [...])
```
SCRIPT 指定安装时需要执行的 CMAKE 脚本，如果脚本文件是相对路径，那么是相对于当前源路径。

CODE 指定安装时需要执行的 CMAKE 代码，例如，
```cmake
install(CODE "MESSAGE(\"Sample install message.\")")
```

## 安装导出
```
install(EXPORT <export-name> DESTINATION <dir>
        [NAMESPACE <namespace>] [[FILE <name>.cmake]|
        [PERMISSIONS permissions...]
        [CONFIGURATIONS [Debug|Release|...]]
        [EXPORT_LINK_INTERFACE_LIBRARIES]
        [COMPONENT <component>]
        [EXCLUDE_FROM_ALL])
install(EXPORT_ANDROID_MK <export-name> DESTINATION <dir> [...])
```
安装导出。导出目标在 `install(TARGETS)` 中的 `EXPORT` 选项指定。`NAMESPACE` 指定导出目标名称的命令空间（相当于前缀），默认情况下安装的导出文件名为 `<export-name>.cmake`，但可以通过 `FILE` 进行重命名。`DESTINATION` 指定这个 .cmake 文件安装的路径。
