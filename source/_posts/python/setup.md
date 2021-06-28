---
title: python setup
date: 2021-06-12 17:03:32
tags:
p: python/setup
---


研究 python setup.py 脚本中的 setup 方法使用。

一个简单的例子
```
from distutils.core import setup

setup(name='Distutils',
      version='1.0',
      description='Python Distribution Utilities',
      author='Greg Ward',
      author_email='gward@python.net',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=['distutils', 'distutils.command'],
     )
```

下面分别对各参数进行解释说明

## packages
packages 列举了需要处理（生成、分发以及安装等）的纯 python 包。这里需要注意 包名称与文件路径之间的映射关系。例如，`distutils` 包应该对应 root 目录下的 `distutils` 文件夹，root 目录即 `setup.py` 文件所在目录。如果指定 `packages=['foo']`，那么 root 目录下应该有 `foo/__init__.py` 文件。

当然，以上是默认约定规则，也可以手动建议映射关系：使用 `package_dir` 参数。

## package_dir
例如所有的 python 源文件均位于 root 目录下的 `lib` 文件夹中，也就是说 "root package" 实际上对应 `lib` 文件夹，例如 `foo` 包则对应 `lib/foo`文件夹，那么设置
```
package_dir = {'': 'lib'}
```
这是一个字典，key 表示包名称，empty string 表示 "root package"，value 表示文件目录（相对于 setup.py 所在目录），故如果此时设置 `packages=['foo']`，这表示 `lib/foo/__init__.py` 一定存在。

如果 `package_dir = {'foo': 'lib'}`，这表示只有 `foo` 包不对应 root 目录下的 `foo` 文件夹，而直接对应 `lib` 文件夹，即 `lib/__init__.py` 一定存在。package_dir 的规则将（递归）应用到某个包内的所有包上，所以 `foo.bar` 包对应 `lib/bar`，即 `lib/bar/__init__.py` 一定存在。

注意：`packages` 不会递归应用到某个包的所有子包上，所以如果要处理子包，需要显式的列出来。

## py_modules
对于小的模块分发，可能想直接列出模块，而不是包，那么使用这个参数，例如
```
py_modules = ['mod1', 'pkg.mod2']
```
记住，模块以根目录为相对起点，所以上面例子中 `pkg` 必须是一个包，即 `pkg/__init__.py` 必须要存在。

当然也可以通过设置 `package_dir` 来手动定义 包 - 目录 的映射关系。

## ext_modules

写 python 扩展模块比写 纯 python 模块复杂一些，同样，描述如何处理这些 模块模块 也比 描述如何处理纯 python 模块要复杂，需要指定扩展模块名称，源文件，编译链接需求（头文件包含路径，链接库，flags 等）

ext_modules 是 `Extension` 的列表， `Extension` 描述扩展模块。一个简单的例子，
```
Extension('foo', ['foo.c'])
```
表示扩展模块名称为 `foo`，相关的源文件为 `foo.c`。

### 扩展名和包
`Extension` 构造器的第一个参数为扩展模块的名称，也可以是包名称，
```
Extension('foo', ['src/foo1.c', 'src/foo2.c'])
```
指定了一个名为 `foo` 且位于 root package 下的扩展模块，而
```
Extension('pkg.foo', ['src/foo1.c', 'src/foo2.c'])
```
制定了一个相同的扩展模块，但是位于 `pkg` 包内。

### 扩展源文件
`Extension` 构造器的第二个参数为扩展源文件，目前支持 C/C++/Objective-C，也可以是 SWIG 接口文件 (`.i` 后缀)。

### 预处理器选项
`Extension` 有三个可选参数，1. `include_dirs`，2. `define_macros`，3. `undef_macros`，分别指定头文件包含路径，定义宏，取消定义宏。

例如，指定相对于项目 root 路径的 `include` 文件夹为头文件包含路径，
```
Extension('foo', ['foo.c'], include_dirs=['include'])
```

当然也可以使用绝对路径，但是尽量避免使用绝对路径，这对分发不友好。

生成 python 扩展库时，Python 包含目录会自动被搜索，例如我的机器上 python 包含目录为 
```
~/tool/miniconda3/include/python3.8
```
所以这个头文件目录不需要手动添加到 `include_dirs` 中。

这个路径可以使用 sysconfig 模块中的方法获得。

`define_macros` 用于定义宏，它是一个 `(name, value)` 元组的列表，其中 `name` 为宏名称，`value` 为宏值，是字符串类型或者 `None` 类型，`value` 等于 `None` 时，相当于 C 中定义宏 `#define FOO` ，这在一些编译器中，`FOO` 值为 `1` 。

`undef_macros` 则是取消定义宏的列表。例如，
```
Extension(...,
          define_macros=[('NDEBUG', '1'),
                         ('HAVE_STRFTIME', None)],
          undef_macros=['HAVE_FOO', 'HAVE_BAR'])
```

等价于 C 源码
```
#define NDEBUG 1
#define HAVE_STRFTIME
#undef HAVE_FOO
#undef HAVE_BAR
```

### 库选项
`Extension` 构造器中，可以指定链接库： `libraries` 参数，链接库的链接时搜索目录：`library_dirs` 参数，链接库运行时的搜索目录（动态库加载时搜索目录）：`runtime_library_dirs`。

### 其他选项

`Extension` 构造器还有一些其他选项参数。

1. `optional` bool 类型，如为 true，那么扩展库生成失败时不会导致整个 生成过程退出。

2. `extra_objects` 是目标文件的列表，这些目标文件提供给连接器进行链接。

3. `extra_compile_args` 指定额外的命令行选项供编译器使用，`extra_link_args` 指定命令行选项供链接器使用。

4. `export_symbols` Windows 系统上使用，这里略。

5. `depends` 是文件列表，指定扩展库所依赖的文件，例如头文件，那么当依赖文件有所改变时，生成命令将调用编译器重新编译。

以上是 `Extension` 扩展的参数介绍。

## 分发和包之间的联系
分发可以 依赖/提供/废除 包或者模块，这在 `distutils.core.setup()` 中实现。

对其他 python 模块/包 的依赖可以通过 `setup()` 中的 `requires` 参数指定，这个参数值是字符串列表，其他每个字符串指示一个包，并且可选择是否附加包的 version。例如指定任意 version 模块 `mymodule` 或者 `xml.parsers.expat`，如果需要指定版本，那么在括号中指定版本修饰，可以有多个版本修饰，每个修饰之间使用 `,` 逗号分隔，修饰可以包含一个比较符，
```
<   >   ==
<=  >=  !=
```
例如，
|依赖库版本| 解释|
|--|--|
|==1.0| 仅 1.0 版本兼容|
|>1.0, !=1.5.1, <2.0|在 1.0 以后 2.0 以前的版本兼容，其中 1.5.1 除外|

上面指定了所依赖的版本，我们也可以提供当前项目包/模块的版本，供其他项目依赖，通过`setup()` 中的 `provides` 参数指定，参数值是字符串列表，每个字符串指示 python 的模块或包名称，且可选地提供其版本，如果未提供版本，那么认为与分发版本一致。例如，
|提供库表达式|解释|
|--|--|
|mypkg| 提供库 `mypkg`，使用分发版本|
|mypkg (1.1)| 提供库 `mypkg`，版本为 1.1|

通过 `obsoletes` 参数指定废除一些包/模块，与上面的 `requires` 值类似，是字符串列表，其他每个字符串指定 包/模块 地名称，后面可跟一个或多个版本修饰，版本修饰至于 `()` 中。

## 安装脚本

上面介绍的内容，处理了 python 的包和模块，这些包和模块自己不会运行，而是在脚本中被导入使用。

脚本中包含 python 源码，且可以在命令行中启动执行。`scripts` 参数指定了脚本文件列表，这样，分发安装后，脚本文件就被复制到 `PATH` 下。例如，
```
setup(...,
      scripts=['scripts/xmlproc_parse', 'scripts/xmlproc_val']
      )
```
文件路径是相对于分发 root 路径，安装后，脚本文件被拷贝到`PATH` 下，于是就可以直接在命令行中，
```
$ xmlproc_parse
...

$ xmlproc_val
...
```

## 安装包数据
有时，其他一些文件也需要被安装，例如一些数据文件，或者包含文档的文本文件。这些文件统称为 包数据。

使用 `package_data` 参数指定包数据，参数值是一个映射（字典类型），从包名称到相对路径列表的映射，相对路径指示数据文件，这些文件应该被拷贝到对应的包。相对路径是相对于 包 对应的目录（注意，可能由 `package_dir` 修改过，而非默认目录）。

例如，源码目录如下，
```
setup.py
src/
    mypkg/
        __init__.py
        module.py
        data/
            tables.dat
            spoons.dat
            forks.dat
```

`setup()` 函数调用为
```
 setup(...,
      packages=['mypkg'],
      package_dir={'mypkg': 'src/mypkg'},
      package_data={'mypkg': ['data/*.dat']},
      )
```

## 安装其他文件
安装分发所需的其他文件，可以使用 `data_files` 参数，参数值是 `(directory, files)` 元组的列表，例如
```
setup(...,
      data_files=[('bitmaps', ['bm/b1.gif', 'bm/b2.gif']),
                  ('config', ['cfg/data.cfg'])],
     )
```

`files` 中每个文件均相对于 `setup.py` 所在目录。可以重定义文件被安装的目录，但不能改变文件名。

`directory` 相对于安装 prefix，系统级安装则为 `sys.prefix`，用户级安装则为 `site.USER_BASE`。 `directory` 也可以为绝对路径，但是通常不建议，会导致与 wheel 包格式的不兼容。