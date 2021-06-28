---
title: python C/C++ Extension Type
date: 2021-06-16 18:23:47
tags:
p: python/ext3
---

每个 Python 对象均是 `PyObject*` 的变体，`PyObject` 仅包含 引用计数 以及 类型对象的指针。因为 Python 是动态类型语言，每个对象自身包含了其类型，这个 类型对象决定了可以用 Python 解释器 对这个对象调用哪些函数，例如获取对象的属性，调用对象方法等。要定义一个新对象，需要创建一个新的类型对象。例子，

```cpp
#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
} CustomObject;

static PyTypeObject CustomType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "custom.Custom",
    .tp_doc = "Custom objects",
    .tp_basicsize = sizeof(CustomObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};

static PyModuleDef custommodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "custom",
    .m_doc = "Example module that creates an extension type.",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_custom(void)
{
    PyObject *m;
    if (PyType_Ready(&CustomType) < 0)
        return NULL;

    m = PyModule_Create(&custommodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&CustomType);
    if (PyModule_AddObject(m, "Custom", (PyObject *) &CustomType) < 0) {
        Py_DECREF(&CustomType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
```
上述代码，首先定义了一个结构体，表示自定义对象，
`CustomObject` 结构体中，`PyObject_HEAD` 是强制必须有的，且在结构体第一个位置，这个宏定义了 `ob_base` 字段，类型为 `PyObject`，这个字段中包含一个类型对象 `ob_type` 和一个引用计数 `ob_refcnt`，可以分别使用 `Py_TYPE` 和 `Py_REFCNT` 进行访问。`PyObject_HEAD` 之后可以列出类型的其他字段，例如
```cpp
typedef struct {
    PyObject_HEAD
    double ob_fval;
} PyFloatObject;
```

然后是对象的类型定义（这个类型本身也是一个对象），
```cpp
static PyTypeObject CustomType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "custom.Custom",
    .tp_doc = "Custom objects",
    .tp_basicsize = sizeof(CustomObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};
```
这里使用的是 C99 的初始化风格，这样不用列出所有的字段，且不用考虑字段的顺序，实际上 `PyTypeObject` 有很多的字段，上面没有列出来的字段，均由编译器初始化为 `0`。如果不使用 C99 初始化风格，那么将会是，
```cpp
static PyTypeObject CustomType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "custom.Custom",
    "Custom objects",
    sizeof(CustomObject),
    0,
    ...
    Py_TPFLAGS_DEFAULT,
    PyType_GenericNew,
    ...
};
```
来依次分析，
```
PyVarObject_HEAD_INIT(NULL, 0)
```
这句是强制的，用于初始化 `ob_base`，这个宏是初始化可变对象的头部，宏定义为
```
#define PyVarObject_HEAD_INIT(type, size) 1, type, size,
```

```
.tp_name = "custom.Custom", # 类型名
.tp_basicsize = sizeof(CustomObject),   # 指示如何分配内存
.tp_itemsize = 0,   # 可变大小的对象用到，不可变大小例如 bool，int，则为0
```

```
.tp_new = PyType_GenericNew,
```
提供一个 `tp_new` 句柄，等价于 `__new__()`，用于创建对象，`PyType_GenericNew` 是创建对象的默认实现。

## 添加数据成员和方法
参考[官方文档](https://docs.python.org/3.9/extending/newtypes_tutorial.html)


# 循环垃圾回收
循环垃圾回收机制使得 Python 可以识别 引用计数不为 0 但是已经不再需要的对象，并将其回收，例如
```python
>>> l = []
>>> l.append(l)
>>> del l
```
当删除 `l` 这个列表时，它仍然有引用，引用计数不为 0，但是 Python 的循环垃圾回收器可以识别并释放这个对象。

