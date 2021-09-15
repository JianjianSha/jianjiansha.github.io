---
title: C++ 边学边忘——类型声明
date: 2021-07-05 17:30:28
tags:
p: cpp/declare
---
# const 与 constexpr
`const` 常量，一定定义后无法更改其值。

`constexpr` 常量表达式，除了定义后无法更改其值，还必须是编译期可知的常量，或者说被 `constexpr` 修饰的变量需要有一个常量表达式的初始化器。

可被 `constexpr` 修饰的类型为 字面量类型，包括 数值型，引用，和指针类型（我们自定义的class类型则不属于字面量类型）。由于有编译期可知这一限制条件，`constexpr` 指针可以初始化为 `nullptr` 或者 `0`，以及具有固定地址的变量，对这些变量取址可对 `constexpr` 指针初始化。定义在函数外部的变量以及函数内部 `static` 修饰的变量具有固定地址。

`constexpr` 指针：`constexprt` 关键字修饰指针自身，而非指针所指向对象，例：
```c++
// 函数外部
const int i = 1;
int j = 2;
constexpr int *q = &j;          // OK, q 是常量指针，其存储的地址不可改变，但是可以修改其指向对象的值
constexpr int *p = &i;          // error，p 所执对象类型为 int，不能用 const int 来初始化
constexpr const int *cp = &i;   // OK
```

# 类型别名
直接看代码示例，
```c++
typedef char *pstring;          // pstring 为 char *
const pstring cstr = 0;         // cstr 是常量指针，指向 char 类型对象
const pstring *ps;              // ps 是一个普通指针，指向 char * const 类型，即指向一个常量指针，这个常量指针指向 char 类型对象
```
注意：不可直接用 `char *` 替换！！！ 

`const` 修饰的是 `pstring`，由于 `pstring` 是指针，所以 `const pstring` 是常量指针，实际效果是 `char * const`。那你要问，既然 `const pstring cstr` 不是指向一个 const 对象，那么指向一个 const 对象 正确写法是什么？当然是 `typedef const char *cpstring`。

# auto 
由编译器根据初始化器来确定变量类型。简单场景这里不再讨论，需要注意一条声明仅包含一个基本类型，多个变量的初始化器类型不同将会报错。

## 复合类型，const 和 auto
1. 使用引用类型作为初始化器时，实际使用的是被引用的对象，所以 `auto` 应该为被引用的对象类型

2. `auto` 忽略 top-level 的 `const`，保留 low=level 的 `const`，如需保留 top-level `const`，显示声明 `const auto`，例
```c++
const int i = 123;
const int *const p = &i;
auto x = pi;                // x 是普通指针，指向 const int
const auto cx = pi;         // cx 是常量指针，并指向 const int
```

3. 指定 `auto` 的引用类型，此时 top-level `const` 不能忽略，这很好理解，因为 “引用” 指明不拷贝创建新对象，而绑定已有对象，如果忽略 top-level `const`，那么将导致通过引用来修改 `const` 对象。
```c++
const int i = 123;  // top-level const
auto &r = i;    // r 是 i 的引用，由于不能修改 i，所以 r 应该是 const int & 类型，而不能是 int & 类型。
```

4. 引用 `&` 和指针 `*` 不是基础类型的一部分，而 `const` 是基础类型一部分，即 `int` 和 `const int` 基础类型不同，故不能声明
```c++
int i = 123;
const int j = 234;
auto &r = i, *p = &j    // error. r 是 int & 类型，p 是 const int * 类型，基础类型不一致，不能出现在同一声明中
```

# decltype

编译器从表达式推断出类型，而不用实际计算表达式。

1. decltype 应用到 变量 上，结果为 变量 的类型，包括 top-level `const` 和 引用符号
```c++
const int ci = 123, &cj = ci;
decltype(ci) x = 0;             // x 是 const int
decltype(cj) y = x;             // y 是 const int &
```

2. decltype 应用到 表达式 上，结果为 表达式 的类型。如果表达式生成可位于赋值左侧的对象，那么 `decltype(expr)` 结果为一引用类型

```c++
int i = 123, *p = &i, &r = i;
decltype(r+1) b;                // OK b 是 int 类型，由于 r+1 不是左值，所以 b 类型不是引用类型
decltype(*p) c;                 // error, c 是 int & 类型，因为 *p 可用于左值（可被赋值）, c 未被初始化，所以报错
```

3. 如果变量被括号包围，例如 `(x)`，那么就是一个表达式了，此时 `decltype((x))` 结果为一引用类型


