---
title: gcc-src
date: 2019-07-02 13:49:38
tags: c++
---
多阅读 c++ 标准库源码，才能更好的理解 c++ 标准库。
<!-- more -->
以 ubuntu 为例，gcc 版本为 7.3.0，目录 /usr/include/c++/7/ 包含了大多数标准库（头）文件，标准库的大多数的实现逻辑也在这些头文件中，如要获取完整的源码，则可以去
1. [gcc-mirror/gcc](https://github.com/gcc-mirror/gcc) clone 这个位于 github 的远程仓库
2. [GNU Mirror List](http://www.gnu.org/prep/ftp.html) 选择一个镜像网址，直接下载 gcc 源码

C++ 标准模板库包含容器，以及容器相关的算法，涉及到的概念包括容器，算法，迭代器以及分配器等，各自功能从名称可窥见一二。

我们采用自底向上的方式来分析各类的实现，虽然自顶向下才是阅读这些源码的最自然的方式，不过阅读方式并不影响什么，待熟悉了这些类的各自功能后，反过来梳理一遍正好可以加深印象。

# Allocator
我们以 __allocator_traits_base 为例开始分析，此类位于 libstdc++-v3/include/bits/alloc_traits.h 头文件中，
```c++
struct __allocator_traits_base
{
    template<typename _Tp, typename _Up, typename = void>
    struct __rebind : __replace_first_arg<_Tp, _Up> { };
    template<typename _Tp, typename _Up>
    struct __rebind<_Tp, _Up, __void_t<typename _Tp::template rebind<_Up>::other>>
    { using type = typename _Tp::template rebind<_Up>::other; };

    // 定义一系列 _Tp 内部类型的别名
protected:
    template<typename _Tp>
    using __pointer = typename _Tp::pointer;
    ...
};
```
类内部定义了模板类 __rebind，对于类 __rebind，分以下几种情况讨论：
1. 提供三个模板参数，并且第三个模板参数不为 void，此时匹配最泛型模板类，即第一个 __rebind 定义 
2. 提供三个模板参数并且第三个模板参数为 void，或者仅提供两个模板参数，此时再分两种情况：
    - 前两个模板参数满足 _Tp::template rebind<_Up>::other 为有效定义，那么匹配第二个 __rebind 定义
    - 否则，匹配第一个 __rebind 定义

接着此文件定义了
```c++
template<typename _Alloc, typename _Up>
using __alloc_rebind = typename __allocator_traits_base::template __rebind<_Alloc, _Up>::type;
```
根据前面的分析，只有 _Alloc::template rebind<_Up>::other 这个类型存在时，这个别名才存在，并且就是这个类型的别名，否则的话，根据第一个 __rebind 模板定义，当 _Alloc 是模板类时，__alloc_rebind 为 _Alloc<_Up, ...>::type。

然后就是 allocator_traits 类，这个特性用于萃取分配器相关的类型，定义如下，
```c++
template<typename _Alloc>
struct allocator_traits: _allocator_traits_base
{
    typedef _Alloc allocator_type;
    typedef type _Alloc::value_type value_type;

    using pointer = __detected_or_t<value_type*, __pointer, _Alloc>;
    ...
};
```
类中 pointer 这个别名指的是 _Alloc::pointer 类型，如果这个类型存在的话，否则就是类型 value_type*。当然通常情况下，_Alloc::pointer 其实也就是 value_type* 类型。__pointer 来自基类成员类型，是一个模板类。  
在 std/type_traits 文件中可查看 __detected_or_t 定义，与前文 __rebind 匹配规则类似，不再赘述。我们再继续看 allocator_traits 其他内部类，
```c++
template<template<typename> class _Func, typename _Tp, typename = void>
struct _Ptr
{
    using type = typename pointer_traits<pointer>::template rebind<_Tp>;
};
template<template<typename> class _Func, typename _Tp>
struct _Ptr<_Func, _Tp, __void_t<_Func<_Alloc>>>
{
    using type = _Func<_Alloc>;
};
```
定义了模板类 _Ptr，具有内部类型 type 为 _Func<_Alloc>，当这个类型存在时，也就是说 _Func 是模板类型，否则 type 就是 `pointer_traits<pointer>::template rebind<_Tp>`，此时，假设 pointer 为 value_type* 类型（参见上文介绍），根据 bits/ptr_traits.h 中的 pointer_traits 定义，
```
template<typename _Tp>
struct pointer_traits<_Tp*>
{
    ...
    template<typename _Up>
    using rebind = _Up*;
    ...
};
```
可知此时 _Ptr::type 为 _Tp* 类型。allocator_traits 内部还定义了很多模板类，比如 _Diff，其类型成员 type 表示指针位移类型（一般是有符号长整型），_Size 的类型成员 type 表示数量类型（一般是无符号长整型），对 _Ptr::type, _Diff::type 和 _Size::type 设置类型别名，
```c++
// 如果 _Alloc::const_pointer 存在，则为 _Alloc::const_pointer，否则为 const value_type*
using const_pointer = typename _Ptr<__c_pointer, const value_type>::type;
// 为 _Alloc::void_pointer 类型，如果这个类型存在的话，否则为 void*
using void_pointer = typename _Ptr<__v_pointer, void>::type;
// 为 _Alloc::difference_type 类型，如果它存在的话，否则为 pointer_traits<pointer>::difference_type，此时一般为（有符号长整型）
using difference_type = typename _Diff<_Alloc, pointer>::type;
// 为 _Alloc::size_type 类型，如果它存在的话，否则为 difference_type 的无符号版本类型
using size_type = typename _Size<_Alloc, difference_type>::type;
```
篇幅有限，不一一介绍，后面的讨论中如果遇到，则根据需要再进行展开讨论。

我们再看一个类 __alloc_traits，位于 ext/alloc_traits.h 中，看看它提供了哪些类型萃取，
```c++
template<typename _Alloc, typename = typename _Alloc::value_type>
struct __alloc_traits
    : std::allocator_traits<_Alloc>     // 假设 __cplusplus >= 201103L，其他情况这里不考虑
{
    typedef _Alloc allocator_type;
    // std::allocator_traits 就是上面刚讨论的那个特性类
    typedef std::allocator_traits<_Alloc>               _Base_type;
    typedef typename _Base_type::value_type             value_type;
    // _Alloc::pointer or value_type*
    typedef typename _Base_type::pointer                pointer;
    typedef typename _Base_type::const_pointer          const_pointer;
    typedef typename _Base_type::size_type              size_type;
    typedef typename _Base_type::difference_type        difference_type;
    typedef value_type&                                 reference;
    typedef const value_type&                           const_reference;
    // 以上各类型含义已经非常明显易懂了，不再赘述。以下引入一组方法到当前域
    using _Base_type::allocate;
    using _Base_type::deallocate;
    using _Base_type::construct;
    using _Base_type::destroy;
    using _Base_type::max_size;
    ...
}
```
于是回到 std::allocator_traits 中查看例如 allocate 的定义，
```c++
_GLIBCXX_NODISCARD static pointer       // _GLIBCXX_NODISCARD 指示编译器，如果返回结果被抛弃，则编译器发出警告。显然这么做是应该的，否则动态申请的内存，将无法被释放，造成内存泄漏
allocate(_Alloc& __a, size_type __n)
{ return __a.allocate(__n); }

_GLIBCXX_NODISCARD static pointer
allocate(_Alloc& __a, size_type __n, const_void_pointer __hint) 
{ return _S_allocate(__a, __n, __hint, 0); }
```
上面代码片段中，_Alloc 表示分配器类型，第一个 allocate 模板直接调用分配器 __a 分配 __n 个元素的内存，第二个 allocate 模板增加了一个参数 __hint 指向临近内存位置的指针，分配器会尝试尽可能分配靠近 __hint 的内存块。易知，分配器特性类的 allocate 方法实际上依赖具体分配器的 allocate 方法实现。实际上，不光是 allocate，deallocate, construct, destroy, max_size 也可能依赖于分配器的同名方法实现（当然，如果目标类型 _Tp 有相应方法实现，则依赖于 _Tp 的同名方法实现）。

由于 std::allocator_traits::construct 的方法参数为，
```c++
template<typename _Tp, typename... _Args>
static auto construct(_Alloc& __a, _Tp* __p, _Args&&... __args)
...
```
发现参数类型为 _Tp*，这是 _Tp 类型的标准内存指针，在 __gnu_cxx::__alloc_traits 中还实现了使用自定义指针类型作为参数的 construct 方法，
```c++
// 是否是自定义指针的判断
template<typename _Ptr>
using __is_custom_pointer
// 如果 _Ptr 与 pointer 相同，那么 _Ptr 不是指针时 => __is_custom_pointer 为真
// 如果 _Ptr 与 pointer 不同，那么 __is_custom_pointer 为假
= std::__and_<std::is_same<pointer, _Ptr>,
        std::__not_<std::is_pointer<_Ptr>>>;

// 重载非标准指针类型的 构造函数
template<typename _Ptr, typename... _Args>
// 条件判断，当 __is_custom_pointer<_Ptr> 为真时，enable_if<xx>::type 才存在
static typename std::enable_if<__is_custom_pointer<_Ptr>::value>::type
construct(_Alloc& __a, _Ptr __p, _Args&&... __args)
...
```
我们阶段性地总结一下以上分配器特性类，主要提供了与分配器有关的类型萃取，如分配器类型，分配的目标对象类型（包含值类型，const 型，指针型），分配的目标对象数量类型（因为是用于容器/序列的分配，涉及到元素对象的数量），位移类型（常用于序列的迭代器）等。另外还提供了一些方法，比如 rebind，由于模板参数 _Alloc 可由调用者传入，假如传入的 _Alloc 其用于分配的模板对象类型 value_type 与当前分配器特性类 allocator_traits 的 value_type 不一致，那么 rebind 将重新绑定得到与 allocator_traits::value_type 一致的分配器类型 Allocator。

接下来则是分配器类，注意与分配器特性类区别开来，后者更注重与分配器有关的类型萃取，前者更注重完成分配器的如分配，反分配，对象构造/析构等实际工作。可能是故意分开成两个类，这种设计能提高自由度，当然，这是我个人理解。

std::allocator 类位于文件 bits/allocator.h 中，包含了分配器模板类定义和偏特化模板类定义，现在理解这些代码应该比较容易了，其中最泛化的模板类继承了 __allocator_base，这个类为 __gnu_cxx::new_allocator 的类型别名，在 new_allocator 中我们可以看到 allocate, deallocate, max_size, construct, destroy 等方法实现。读者可仔细阅读这些源码，这里不再一一分析。

# Iterator
开门见山不绕弯子，位于 bit/stl_iterator_base_types.h 中，
```c++
// 定义一组 Iterator 标记，它们都是空类型，仅仅用于区分不同的迭代器
// 迭代器底层的算法会根据迭代器本身的类型标记来选择最优算法
input_iterator_tag
output_iterator_tag
forward_iterator_tag
bidirectional_iterator_tag
random_access_iterator_tag
```
对于一个迭代器，需要指定迭代器自身的类型（上述某 iterator_tag 之一），迭代目标对象的值/指针/引用类型，迭代位移类型等，这正是 std::iterator 的定义，然后还需要一个相关的特性模板 iterator_traits 用于萃取其相关的类型。

在 bits/stl_iterator.h 中还提供了几个迭代器适配器，其本质也是一个迭代器，只不过是提供某些专有功能的迭代器。我们先阐述上面五种迭代器类型，然后再结合迭代器适配器理解更有心得，

1. input 迭代器。看到 input 可以将容器类比标准输入，比如从屏幕读取输入，这里 input 迭代器类似，从容器读取元素，迭代器迭代器，说明是依次向前读取容器内的元素。  
   支持的操作：自增（向前），解引用（右值，取值），判断两个迭代器是否相等（是否迭代到头）
2. output 迭代器。与 input 迭代器相反，依次向容器写入元素。  
   支持的操作：自增（向前），解引用（左值，赋值）。
3. forward 迭代器。结合了 input 和 output 迭代器，解引用，既可作左值也可作右值。自增指向下一个元素。与 input/output 迭代器不同的是，forward 迭代器支持 multipass 算法。
4. bidirectional 迭代器。在 forword 迭代器的基础上增加了自减操作，指向上一个元素。
5. random-access 迭代器。在 bidirectional 迭代器基础上增加关系比较，算术运算等。

也许上面的总结还不能完全理解，没关系，现在结合迭代器适配器的代码来综合理解。
## reverse_iterator
```c++
template<typename _Iterator>
class reverse_iterator      // 首先是一个迭代器，其次是提供某些特殊功能的迭代器
: public iterator<typename iterator_traits<_Iterator>::iterator_category,
                typename iterator_traits<_Iterator>::value_type,
                typename iterator_traits<_Iterator>::difference_type,
                typename iterator_traits<_Iterator>::pointer,
                typename iterator_traits<_Iterator>::reference>
{
protected:
    _Iterator current;  // 声明所用迭代器类型的一个变量，反向迭代器正是在此迭代器之上进行构造得到
public:
    // 构造函数略
    ...
    _GLIBCXX17_CONSTEXPR reference
    operator*() const {
        _Iterator __tmp = current;
        return *--__tmp;// 先自减，然后解引用，返回的是前一个元素值的引用，返回值只能用作右值
                        // 由于是在临时变量临时变量上自减，故当前迭代器所指位置不变
    }

    _GLIBCXX17_CONSTEXPR reverse_iterator&
    operator++() {
        --current;      // 反向迭代器表示从右往左，故自增表示正常迭代器的自减
        return *this;
    }
}
```
记反向迭代器为 r，其内部迭代器为 i，那么 r 的所有操作均转换为 i 上的操作，并由 i 完成，如
1. 解引用: *r = *(i-1)
2. ++r = --i, --r = ++i
3. r+n = i-n, r-n = i+n

还有其他一些操作如关系比较，基本上，r 的操作与 i 的操作相反（除了等于，不等于操作）

## back_insert_iterator
```c++
template<typename _Container>
class back_insert_iterator
:public iterator<output_iterator_tag, void, void, void, void> // 指定迭代器标签，其他相关类型则由容器决定
{
protected:
    _Container* container;  // 构造此迭代器时，需要传入容器变量，此迭代器用于向这个容器末端插入元素
public:
    back_insert_iterator&   // 给此迭代器赋值就等于向容器末端插入元素
    operator=(const typename _Container::value_type& __value)
    {
        container->push_back(__value);
        return *this;
    }
    ...

    back_insert_iterator&
    operator*() { return *this; }       // 解引用不是取所指元素的值，因为是 output 迭代器
    back_insert_iterator&
    operator++() {return *this; }       // 自增也不是指向下一个元素，因为只能向容器末端插入值
    back_insert_iterator&
}
```
与此类似还有 front_insert_iterator, insert_iterator 分别表示像容器首端插入值和像容器插入值，插入操作的实现均依赖于容器自身的插入操作，你所能做的，就是给这些迭代赋值，除了赋值还是赋值。。。

## __normal_iterator
这是一个正常的迭代器模板，有两个模板参数 _Iterator, _Container，内部维持了一个迭代器对象，用于迭代操作，_Container 作用仅仅是用于生成不同的 __normal_iterator 类型，
```c++
template<typename _Iterator, typename _Container>
class __normal_iterator
{
protected:
    _Iterator _M_current;   // _normal_iterator 的迭代操作实际上由 _M_current 完成
    ...
public:
    // 构造函数略
    ...
    // 迭代器的解引用，自增，自减，指针访问成员，位移等均由 _M_current 完成
    // 两个 __normal_iterator 的关系比较也由对应的两个 _M_current 的关系比较完成
    // 确实是再 normal 不过了
}
```

## move_iterator
顾名思义就是提供 move 操作，其内部也有一个迭代器，move_iterator 的解引用就是将其内部解引用得到的值进行 move 从而转为右值引用，这用于某些泛型方法中，move 代替了 copy，提高了效率。
```c++
template<typename _Iterator>
class move_iterator
{
protected:
    _Iterator _M_current;
    typedef iterator_traits<_Iterator>          __traits_type;  // _Iterator 的特性类
    typedef typename __traits_type::reference   __base_ref;     // 萃取 _Iterator 相关的元素引用类型
public:
    ...
    // 如果__base_ref 是引用类型，将其转为右值引用，否则保持不变。通常来讲，__base_ref 都是引用类型
    typedef typename conditional<is_reference<__base_ref>::value,
        typename remove_reference<__base_ref>::type&&,
        __base_ref>::type               reference;
    
    _GLIBCXX17_CONSTEXPR reference
    operator*() const
    { return static_cast<reference>(*_M_current); } // 将内部迭代器解引用得到的值转为右值引用

    _GLIBCXX17_CONSTEXPR reference
    operator[](difference_type __n) const
    { return std::move(_M_current[__n]); }  // 随机访问取值，也转为右值引用
}
```


# Container
## Vector
以 vector 为例，代码位于 bits/stl_vector.h 中，首先是基类 _Vector_base，
```c++
template<typename _Tp, typename _Alloc>
struct _Vector_base
{
    typedef typename __gnu_cxx::__alloc_traits<_Alloc>::template rebind<_Tp>::other _Tp_alloc_type;
    typedef typename __gnu_cxx::__alloc_traits<_Tp_alloc_type>::pointer pointer;
    ...
}
```

模板参数 _Alloc 的 value_type 不一定是 _Tp，所以通过 rebind 得到 value_type 为 _Tp 的 alloctor（即 alloctor<_Tp>），设置其别名为 _Tp_alloc_type，然后设置其关联的 pointer，即 _Tp_alloc_type::pointer，如果它存在的话，否则为 _Tp*，然后根据 std::allocator 模板定义不难知道 _Tp_alloc_type::pointer 其实就是 _Tp*，所以 _Vector_base::pointer 就是 _Tp*。

接着 _Vector_base 中又定义了几个内部结构
```c++
struct _Vector_impl_data
{
    pointer _M_start;   // 指向 vector 中第一个元素的内存位置
    pointer _M_finish;  // 指向 vector 中 past-the-last-element 的内存位置
    pointer _M_end_of_storage;  // vector 分配 past-the-max-element 内存位置

    // 构造函数，拷贝函数，交换数据函数。比较简单，略
    ...
}

struct _Vector_impl : public _Tp_alloc_type, public _Vector_impl_data
{
    // 构造函数，略
    // vector 内存 overflow 检测，需要指定 _GLIBCXX_SANITIZE_VECTOR。参考 AddressSanitizer
}
```
回到 _Vector_base 中来，_Vector_base 定义了类型别名和一个变量，
```c++
template<typename _Tp, typename _Alloc>
struct _Vector_base
{
    ...
public:
    typedef _Alloc allocator_type;
    _Vector_impl _M_impl;           // 分配器变量
    ...
    // 构造/析构 函数

    pointer _M_allocator(size_t __n) {      // 分配 n 个元素的内存，起始位置保存到 pointer 类型变量中
        typedef __gnu_cxx::__alloc_traits<_Tp_alloc_type> _Tr;
        // 如 __n=0，则返回 nullptr，否则使用分配器 _M_impl 分配内存
        return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
    }

protected:
    void _M_create_storage(size_t __n) {    // 分配内存，并保存内存起始位置和截止位置
        this->_M_impl._M_start = this->_M_allocate(__n);
        this->_M_impl._M_finish = this->_M_impl._M_start;
        this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
    }
}
```
基类 _Vector_base 中仅仅做了内存分配和记录内存块位置的事情，其他 vector 相关的操作则放在 vector 类中，
```c++
// vector 模板参数指明了 vector 关联的元素类型 _Tp，以及 vector 的内存分配器类型 _Alloc，
//  默认 _Alloc 为 std::allocator<_Tp>，显然是于 _Tp 匹配的，
//  若提供的模板参数 _Alloc 与 _Tp 不匹配，那么也由 _Alloc::rebind 获取匹配的 alloctor
template<typename _Tp, typename _Alloc = std::allocator<_Tp>>
class vector : protected _Vector_base<_Tp, _Alloc>
{
    typedef _Vector_base<_Tp, _Alloc>               _Base;
    typedef typename _Base::_Tp_alloc_type          _Tp_alloc_type;
    typedef __gnu_cxx::__alloc_traits<_Tp_alloc_type>   _Alloc_traits;
public:
    typedef _Tp                             value_type;
    typedef typename _Base::pointer         pointer;
    typedef __gnu_cxx::__normal_iterator<pointer, vector>   iterator;
    typedef std::reverse_iterator<iterator>                 reverse_iterator;
    ...
}
```
事实上，指针包含解引用，自增和自减等操作，也可看作是一种特殊的迭代器，所以这里 vector 内部类型 iterator，使用 pointer 作为 __gnu::cxx::__normal_iterator 的模板参数 _Iterator。  
然后是 vector 的各种构造函数，需要注意到 vector 在实际分配内存后，都会更新 _M_impl._M_finish 使其指向 past-the-last-element 的位置。我们来看一下 vector 获取迭代器的函数，
```c++
iterator
begin() _GLIBCXX_NOEXCEPT
{ return iterator(this->_M_impl._M_start); }

iterator
end() _GLIBCXX_NOEXCEPT
{ return iterator(this->_M_impl._M_finish); }

reverse_iterator
rbegin() _GLIBCXX_NOEXCEPT
{ return reverse_iterator(end()); }

reverse_iterator
rend() _GLIBCXX_NOEXCEPT
{ return reverse_iterator(begin()); }
```
可见迭代器的自增自减解引用均转为指针的自增自减解引用操作。

其他的 vector 操作，resize 表示重置 vector 中有效元素的数量，重置后 new_size > old_size，那么末尾多出来的元素使用默认值填充（如果 resize 提供了指定值，那么使用指定值填充），如果 new_size<=old_size，则重置 _M_impl._M_finish 所指位置（[_M_start, _M_finish) 范围内的元素有效），_M_finish 之后的元素则根据元素类型决定是调用元素的析构函数还是放任不理，注意这一过程中内存占用没有改变。

来看 vector 的 push_back 函数实现，
```c++
void
push_back(const value_type& __x)
{
    if(this->_M_impl._M_finish != this->_M_impl._M_end_of_storage) {
        // 当前分配的内存空间还足以存储新的元素 __x
        ...
    } else
        _M_realloc_insert(end(), __x);  // 重新分配内存，并在 _M_finish 位置插入元素 __x，然后
                                        //  将 _M_finish 所指位置向前移动一个元素单位
}
```
在 bits/vector.tcc 中找到 _M_realloc_insert 的实现，
```c++
template<typename _Tp, typename _Alloc>
template<typename ..._Arg>
void
vector<_Tp, _Alloc>::_M_realloc_insert(iterator __position, _Args&&... _args)
{
    // 计算即将重新分配元素数量，这里重新分配的元素数量是原来元素数量的 2 倍，参见 _M_check_len
    const size_type __len = _M_check_len(size_type(1), "vector::_M_realloc_insert");
    pointer __old_start = this->_M_impl._M_start;
    pointer __old_finish = this->_M_impl._M_finish; // 原来的起始元素指针和 past-the-last 元素指针
    const size_type __elems_before = __position - begin();// 插入位置之前的元素数量
    pointer __new_start(this->_M_allocate(__len));   // 重新分配内存，使得能容纳 __len 个元素
    pointer __new_finish(__new_start);  // 由于尚未填充元素，故此时 past-the-last 指针与起始指针相等
    __try
    {
        // 在指定位置处插入目标对象
        _Alloc_traits::construct(this->_M_impl,                     // 使用此分配器
                                 __new_start + __elems_before,      // 在指定位置处
                                 std::forward<_Args>(__args)...);   // 根据此参数构造对象
        // 此时 vector 中有了元素，将 __new_finish 先置为 nullptr，等元素全部填充完毕，再更新其值
        __new_finish = pointer();

        if _GLIBCXX17_CONSTEXPR (_S_use_relocate()) {   // 如果元素类型支持移动插入
            // 将原来起始位置到插入位置截止，之间的元素重定位到新的起始位置
            __new_finish = _S_relocate(__old_start, __position.base()
                __new_start, _M_get_Tp_allocator());
            // 此时 __new_finish 所指位置就是新插入的元素，自增 1，移动新插入元素之后，将原来剩余的元素重定位到此位置处
            ++__new_finish;
            __new_finish = _S_relocate(__position.base(), __old_finish,
                __new_finish, _M_get_Tp_allocator());   // 此时 __new_finish 就是新的 past-of-last 元素位置了
        }
        ...
        // 失败处理，略
        // 析构原先内存上的对象，并释放内存，略
        // 更新元素起始和截止位置等，略
    }
}
```
vector 类中还有很多其他方法，但是到了这一步，相信这些方法的代码实现不难理解了，由于篇幅有限，不对这些方法进行分析。

本文结束