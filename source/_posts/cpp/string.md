---
title: C++ 边学边忘——字符（串）
date: 2021-06-25 12:02:49
tags:
---

# 编码
我们写的源码（source code）被保存在文件中，我们知道文件也是有编码格式的，例如打开微软的 VS 2019，新建一个C++ Console App，会自动生成一个 `ConsoleApplication1.cpp` 文件，这个文件的编码可以通过 VS 2019 的 “文件”菜单下的 “高级保存选项”命令进行设置（如果没有说明被 VS 隐藏了，请百度如何显示这个命令），默认是`GB2312`，可以把它改为 `utf-8` 等。vscode 中也可以更改文件编码，通过 `ctrl+shift+p` 打开命令搜索框，然后输入 `encode` 关键词，就有 `Change File Encoding` 命令出来，然后 vscode 右下角 也出现 `UTF-8`、`CRLF` 等按钮，可以直接点击进行修改。

这里强调一下，unicode 与 UTF-8 的关系，准确的讲，unicode 是字符集（也有编码，分 UCS2 和 UCS4 两种），UTF-8 是 unicode 的一种编码，是一种变长编码，但所覆盖的字符集范围与 unicode 是相同的，而 unicode 是定长编码，UTF-8 是为了便于传输（省流量）以及与 ASCII 兼容。编码 gb2312 与 UTF-8 两者对应的字符集不兼容。编码 GBK 兼容 gb2312，GBK 字符集是 gb2312 字符集的超集。编码 UTF-8 、GBK 和 gb2312 兼容 ASCII，而 unicode 不兼容 ASCII。关于字符集和编码的更多知识请百度。

对于一个源文件，我们可以在 Windows 上使用 ultraedit 或 UEStudio 来查看文件的十六进制数据，在 Linux 上使用 vim 打开源文件后，使用命令 `:%!xxd`，切换到十六进制，也可以使用命令 `hexdump <filename> -C` 查看文件的 16 进制数据。
> 使用 vim `:%!xxd` 查看文件 16 进制时，发现 gb2312 的文件总是会先被转换为 utf-8 编码，导致查看的都是 utf-8 编码的 16 进制，由于对 vim 不是特别熟悉，不知道是什么原因。

现在我们需要搞清楚的是：编译器读取源文件，将源文件中的字符映射得到编译时的字符，这个编译时字符集称为 `源字符集`，经过映射的字符作为预处理阶段的输入，经过预处理后，字符串和字符常量会再转换为 `执行字符集`，保存在可执行文件中。这里的映射均由编译器实现定义。

## 源字符集
Windows 上编译 C++ 使用 `cl.exe` 工具。默认使用当前活动页对应的编码作为 `源字符集`，可以使用 `chcp` 查看当前活动页，
```shell
> chcp
Active code page: 936
```
936 对应编码 `gb2312`。我们举一个例子说明，例如代码
```c++
// main.cpp
int main(int argc, char** argv) {
    wchar_t c = L'好';
    return 0;
}
```
使用 vs 2019 创建 Console App 项目后，直接将上面代码替换掉自动生成的代码，这里默认的源文件编码方式为 `gb2312`，在系统开始菜单中找到 Visual Studio 2019 文件夹，然后打开 `Developer Command Prompt for VS 2019`，输入以下命令进行预处理，
```shell
> cd myproj
> cl main.cpp /E > main-gb2312.i
```
得到预处理后的文件 `main-gb2312.i` 编码为 `gb2312`，可以检测其中的中文 `好` 被编码为 `BAC3`。

当然我们还可以修改源文件的编码方式，在 VS 2019 文件菜单下的 “高级保存选项” 窗口，设置为 `UTF-8 with signature`，保存好后，再次执行预处理，
```shell
> cl main.cpp /E > main-utf8-sig.i
```
得到预处理后的文件 `main-utf8-sig.i` 编码为 `gb2312`，可以检测其中的中文 `好` 被编码为 `BAC3`。

but，如果将源文件编码改为 `UTF-8 without signature`，那么源文件将是 `UTF-8` 编码，且 `cl.exe` 无法识别，此时采用默认的 `gb2312` 进行解码，那么将会导致编译出现意想不到的结果，甚至无法通过编译。在以上这个例子中，`好` 的 UTF-8 编码为 `E5A5BD`，后面一个字符单引号 `'` 的UTF-8 编码为 `27`，而 `E5A5` 被 gb2312 错误的识别为 `濂`， 剩余的一个字节 `BD` 将会与后面的字节 `27` 连起来，但是这不是一个有效的 gb2312 字符编码，所以无法识别，预处理阶段，将它替换为一个问好 `?` 的编码 `3F`，这导致丢失了单引号字符 `'`。我们使用 VS 直接编译这个源文件，报错如下，

![](/images/cpp/string1.png)

有的时候，如果源码中出现的中文 utf-8 的编码全部处于 gb2312 编码范围内，那就不会报错，但却被编译器错误的识别为其他字符，导致程序能生成，但是执行结果不对。

那么，对于这个 `UTF-8 without signature` 的源文件，我们就无法处理了吗？

显然不是，一个是修改系统的 CODE PAGE 为 65001，这样，`源字符集` 默认就改成 UTF-8，肯定是可以的，预处理后的文件也是 `UTF-8` 编码的，但是这种方法牵一发而动全身，我们选用另一个方法， 即，通过命令行选项 `\source-charset:utf-8` 指定，
```shell
> cl /source-charset:utf-8 main.cpp /E > main-gb2312.i
```

Linux 上，我们以 GCC 为例，默认的 `源字符集` 为 UTF-8，预处理命令为
```shell
$ gcc -E main.cpp -o main-utf8.i
```
查看 `main-utf8.i` ，发现中文 `好` 编码为 `E5A5BD`，这说明确实是 UTF-8 编码。

使用 vscode 将源文件编码改为 `gb2312` ，继续执行命令 `gcc -E main.cpp -o main-gb2312.i`，查看 `main-gb2312.i`，发现中文 `好` 编码为 `BAC3`，这是 gb2312 编码，加入其他中文，发现全部都保留了 gb2312 编码，事实上，这是因为 gcc 默认按照 UTF-8 解码，在第一个单引号 `'` 之后，遇到 `好'`，其 gb2312 编码为 `BAC327`，但是 gcc 使用 UTF-8 解码时，被解码成一个`ڃ'`，这是一个乱码后跟一个单引号，也就是说，`BAC3` 在 UTF-8 中是有对应字符的，所以本质上<font color="red">不是 gb2312 编码，而仍然是 UTF-8 编码</font>，将中文的 gb2312 编码按 UTF-8 解码势必出错，所以我们需要通过 gcc 的 `-finput-charset` 命令选项来指出源文件的编码，
```shell
$ gcc -finput-charset=gb2312 -E main.cpp -o main-gb2312-ic.i
```
注意这个选项仅用于指出源文件编码，而生成的预处理文件 `main-gb2312-ic.i` 仍然是 UTF-8 编码，可以发现，中文 `好` 的编码为 `E5A5BD`。GCC 中 `源字符集` 为 UTF-8，本人暂时没有找到可以更改 gcc `源字符集` 的方法。

> 查询 UTF-8 编码可使用 https://www.branah.com/unicode-converter ， 查询 gb2312 编码可使用 https://www.qqxiuzi.cn/bianma/zifuji.php 。


## 执行字符集

预处理之后进行编译，字符串和字符常量将会被转换为 `执行字符集`，根据[微软官方说明](https://docs.microsoft.com/en-us/cpp/build/reference/execution-charset-set-execution-character-set?view=msvc-160)，`执行字符集` 是对文本进行编码然后作为在预处理之后的后续编译阶段的输入。

在 VS 上，可以使用 `/execution-charset:<charset>` 命令选项进行设置，在 GCC 上可以设置命令选项 `-fexec-charset=<charset>` ，对于宽字符，则需要设置 `-fwide-exec-charset=<charset>`。

以 GCC 为例（Ubuntu 上），首先修改源码如下，
```c++
// main.cpp
int main(int argc, char** argv) {
    char c = '好';
    return 0;
}
```
这里变量 `c` 的类型改为 char 类型，然后编译，
```shell
$ gcc -c main.cpp -o main.o
$ objdump -Sr main.o

f:  c6 45 ff bd    movb    $0xbd,-0x1(%rbp)
```
注意到 `0xbd` 就是 `好` 的 utf-8 编码 `E5A5BD` 的低位的第一个字节，因为 char 类型只能存储一个字节，其他字节被截掉。现在加上 `执行字符集` 命令选项，
```shell
$ gcc -c main.cpp -o main.o -fexec-charset=gb2312
$ objdump -Sr main.o

f:  c6 45 ff c3     movb    $0xc3,-0x1(%rbp)
```
这里 `0xc3` 是 `好` 的 gb2312 编码 `BAC3` 的低位字节。现在将源码改为，
```c++
// main.cpp
int main(int argc, char** argv) {
    wchar_t c = '好';
    return 0;
}
```
注意这里字符字面量没有前缀 `L`，编译和反编译如下，
```shell
$ gcc -c main.cpp -o main.o
$ objdump -Sr main.o

f:  c7 45 fc bd a5 e5 00    movl    $0xe5a5bd,-0x4(%rbp)
```
可见 `好` 编码为 UTF-8 的 `E5A5BD`。加上 `执行字符集` 命令选项，那么有
```shell
$ gcc -c main.cpp -o main.o -fexe-charset=gb2312
$ objdump -Sr main.o

f:  c7 45 fc c3 ba 00 00    movl    $0xbac3,-0x4(%rbp)
```

这与预料的一样，`好` 编码为 gb2312 的 `BAC3`。现在我们继续修改源码，
```c++
// main.cpp
int main(int argc, char** argv) {
    wchar_t c = L'好';
    return 0;
}
```
不使用 `源字符集` 命令选项，结果为，
```shell
$ gcc -c main.cpp -o main.o
$ objdump -Sr main.o

f:  c7 45 fc 7d 59 00 00    movl    $0x597d,-0x4(%rbp)
```
这个 `0x597d` 为 `好` 的 UTF-32 编码。不难想象，如果源码改为 `char = L'好'`，那么目标文件中只有 `好` 的 UTF-32 编码的低位字节，即 `7D`，这个可以自己试一下。

这是因为带 `L` 前缀的字符（串）字面量的默认 `执行字符集` 为 `UTF-32` （我这里的 wchar_t 为 4 字节，如果是 2 字节，那么对应 `执行字符集` 为 `UTF-16`）。我们现在增加 `-fwide-exec-charset` 命令选项修改 __宽__ 字符（串）字面量的 `执行字符集`，
```shell
$ gcc -c main.cpp -o main.o -fwide-exec-charset=gb2312
$ objdump -Sr main.o

f:  c7 45 fc 00 00 ba c3   movl    $0xc3ba0000,-0x4(%rbp)
```
可见确实将宽字符编码为 `gb2312`。

总结：

1. `-fexec-charset` 改变窄字符（例如 `'好'`）的字符集
2. `-fwide-exec-char` 改变宽字符（带前缀 `L`）的字符集
3. GCC 默认，窄字符的 `执行字符集` 为 `UTF-8`，宽字符的 `执行字符集` 为 `UTF-32`


可以使用如下源码试一试，
```c++
// main.cpp
int main(int argc, char** argv) {
    wchar_t c = L'好';      // -fwide-exec-charset=gb2312
    wchar_t d = '好';       // -fexec-charset=gb2312
    char32_t e = U'好';     // 宽字符，对应 -fwide-exec-charset 命令选项
    return 0;
}
```

# 输出
本节内容以 GCC 作为编译器进行讨论说明。现在我们来看字符（串）的输出。给出测试代码如下，
```c++
// main.cpp
#include <stdio.h>
int main(int argc, char** argv) {
    wchar_t a = L'好';
    char32_t b = U'好';
    char c = '好';
    int d = 0x597D;
    wint_t e = 0x597D;

    printf("a->%c, b->%c, c->%c, d->%c, e->%c\n", a, b, c, d, e);
}
```
生成命令如下，
```shell
$ g++ main.cpp -o main
$ ./main

a->}, b->}, c->?, d->}, e->}
```
`c` 打印为乱码（这里乱码使用 问号 `?` 表示，这里乱码仅指无法识别为 ASCII 和 中文的意思），`a,b,d,e` 打印为右大括号 `}`，这很好理解，参考 [printf 函数格式串说明](http://www.cplusplus.com/reference/cstdio/printf/)，`a,b` 在执行文件中才有 UTF-32 编码 `597D`，与 `d,e` 相同，在格式化过程时先被转换为了 char 类型，即 `7D`，这是 `}` 的 ASCII。`c` 保存了 `好` 的 UTF-8 编码的低位字节（GCC 中窄字符采用 UTF-8 编码）， 为 `BD`（等效于设置 `char c = 0xBD`），超出了 `%c` 的有效范围（127），在 ASCII 的扩展字符集（即 128~255）中才有，我们应该明确避免这种打印超过范围字符。可以改为 `wchar_t c=0xBD`，并使用 `%lc` 来打印这个 ASCII 扩展字符，同样的 `好` 的编码值也超出了 `char` 类型的有效范围，故上面使用 `wchar_t` 类型变量保存，同时也应该改为使用 `%lc` 打印。这里需要说明，C 程序执行时默认使用标准的 `C` locale，这个 locale 使得在终端上无法正常显示宽字符，而显示乱码，可以增加 locale 设置语句，可以使用指令 `locale` 查看系统相关配置，打印代码如下，
```c++
#include <wchar.h>
int main(int argc, char** argv) {
    setlocale(LC_CTYPE, "");
    wchar_t a = L'好';
    char32_t b = U'好';
    wchar_t c = 0xBD;   // ASCII 扩展字符
    int d = 0x597D;
    wint_t e = 0x597D;
    printf("a->%lc, b->%lc, c->%c, d->%lc, e->%lc\n", a, b, c);
}
```

结果为 
```
a->好, b->好, c->½, d->好, e->好
```

注意这里打印 `c` 变量使用 `%lc`，以宽字符形式打印，打印结果为 `½`（分数 1/2），可以查看 [扩展 ASCII 表](https://www.w3school.com.cn/charsets/ref_html_8859.asp)。另外注意到 `好` 使用 unicode 编码值 `597D`，所以如果 `wchar_t c=0xE5A5BD`（ UTF-8 编码值），那么将无法在终端正确打印 `好`。可能有人会好奇，前面说 GCC 中 `exec-charset` 默认为 `UTF-8`，为什么 `UTF-8` 编码值就无法用来打印字符呢？


因为 `exec-charset` 是对应窄字符的，所以不能使用 `%lc` 来打印，但是我们可以使用 `%s` 来打印，即打印字符串，
```c++
#include <locale.h>
#include <wchar.h>
#include <stdio.h>
int main(int argc, char** argv) {
    setlocale(LC_CTYPE, "");    // 设置系统当前 locale
    const wchar_t* a = L"好";
    const char32_t* b = U"好";
    const char* c = "好";
    const char* d = "\xE5\xA5\xBD";
    unsigned char e[4] = {0xE5, 0xA5, 0xBD};
    const char f[4] = {'\xE5', '\xA5', '\xBD'};
    printf("a->%ls, b->%ls, c->%s, d->%s, e->%s, f->%s\n", a, b, c, d, e, f);
}
```
其中，`c` 对应 `%s`，用于打印窄字符，而 `a,b` 均需要用 `%ls` 来打印宽字符（否则无法正确打印），我们不能搞混，否则可能无法正常打印。

结果为 
```
a->好, b->好, c->好, d->好, e->好, f->好
```

上面，`d,f` 是字符串 `"好"` 的 16 进制表示（UTF-8 编码），`e` 存储了 `好` 和 `\0` 两个字符，其中 `好` 依然是 UTF-8 编码。`d,e,f` 变量的字符串打印使用 `%s`，如上结果所示，可正常打印。此外，`d,f` 可以直接打印，即 `printf(d);printf(f);`。

> 1. 可以使用 printf("%x\n", a); 打印一个字符变量的 16 进制值。

> 2. GCC 中不要混用 `printf` 和 `wprintf`，相关知识点可搜索 `fwide` 函数.
