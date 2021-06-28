---
title: python C/C++ Extension（二）
date: 2021-06-16 14:09:08
tags:
p: python/ext2
---

# 引用计数
C/C++ 动态申请的内存，需要手动释放，否则出现内存泄漏。同时已经释放掉的内存块，不可以再次使用。Python 中采取的策略是引用计数，原理：每个对象包含一个计数器，当对象的一个引用被存储，那么增加一次计数，当对象的一个引用被删除，则减小一次计数，当计数归 0，表示对象的最后一个引用被删除，此时是否对象所占内存。

另一种策略是自动垃圾回收，这种策略的优点是使用者无需显式调用 `free()` 释放内存，缺点是 C 中没有一个真正的轻便的自动垃圾回收器，而引用计数则可以很方便的实现。

## Python 中的引用计数
`Py_INCREF(x)` 和 `Py_DECREF(x)` 这两个宏，用于增加和减小计数。当计数将为 0 时，`Py_DECREF(x)` 会释放对象。如何使用这两个宏？

为此我们需要弄清楚一些概念。我们不直接拥有对象，而是拥有对象的一个引用，对象的引用计数就是拥有引用的数量。当引用不再被需要时，引用的拥有者负责调用 `Py_DECREF()` 。引用的拥有关系可以被转移。有三种方式处置所拥有的引用：1. 将引用转移；2. 存储引用；3. 调用 `Py_DECREF()`。不处理引用将导致内存泄漏。

可以借用一个对象的引用，但是借方不能比这个引用的拥有者存活更久。通过调用 `Py_INCREF()`，这个出借的引用可以变成借方拥有的引用，这不影响原先拥有者的状态。

## 拥有关系规则
大部分返回对象引用的函数，都是转移引用的拥有关系。具体而言，所有用于创建一个新对象的函数，例如 `PyLong_FromLong()` 和 `Py_BuildValue()`，将拥有关系转移给接收者。

当你将一个对象引用传递给一个函数时，通常，函数是向你借用引用，如果函数需要存储这个引用，那么它将使用 `Py_INCREF()`，从而成为一个独立的引用拥有者。

python 中调用一个 C 函数时，C 函数从调用者那里借用对象引用。调用者拥有引用，在 C 函数中，引用的生命周期可以得到保证。

# 为扩展模块提供 C API
大多数时候扩展模块的函数都是在 Python 中使用，但是有时候扩展模块的函数可以在另一个扩展模块中使用。例如，一个扩展模块中可以实现一个类似 `list` 的集合类型，但是元素是无序的，这个新集合类型包含一些 C 函数，可以在其他扩展模块中直接使用。

乍一看好像很简单，C 函数不再声明 static 即可。这在扩展模块静态链接至 Python 解释器时有效，如果扩展模块是动态链接库，那么一个模块中的符号在另一个模块中将不可见。

所以我们不应该对符号可见性有任何预先设定，所以除了模块初始化函数，其他符号都应该声明为 `static`，以避免名称冲突。Python 提供一个特殊机制以实现 C level 的信息传输————从一个扩展模块到另一个扩展模块————胶囊。胶囊是一个Python 数据类型，存储了一个 `void *` 类型指针，胶囊仅在它的 C API 中被创建和访问，无法传递到其他 Python 对象。每个胶囊在扩展模块的命名空间里有自己的名称，其他扩展模块可以导入这个扩展模块，然后得到胶囊的名称，从而获取胶囊的指针。

用于导出 C API 的胶囊应该遵循以下命名规则：
```
modulename.attributename
```

以一个例子说明，
```CPP
static int
PySpam_System(const char *command)
{
    return system(command);
}

static PyObject *
spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = PySpam_System(command);
    return PyLong_FromLong(sts);
}
```

紧接着 `#include <Python.h>` 之后添加 
```
#define SPAM_MODULE
#include "spammodule.h"
```
然后定义模块初始化函数，
```CPP
PyMODINIT_FUNC
PyInit_spam(void)
{
    PyObject *m;
    static void *PySpam_API[PySpam_API_pointers];
    PyObject *c_api_object;

    m = PyModule_Create(&spammodule);
    if (m == NULL)
        return NULL;

    /* Initialize the C API pointer array */
    PySpam_API[PySpam_System_NUM] = (void *)PySpam_System;

    /* Create a Capsule containing the API pointer array's address */
    c_api_object = PyCapsule_New((void *)PySpam_API, "spam._C_API", NULL);

    if (PyModule_AddObject(m, "_C_API", c_api_object) < 0) {
        Py_XDECREF(c_api_object);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
```

`spammodule.h` 头文件内容如下，
```cpp
#ifndef Py_SPAMMODULE_H
#define Py_SPAMMODULE_H
#ifdef __cplusplus
extern "C" {
#endif

/* Header file for spammodule */

/* C API functions */
#define PySpam_System_NUM 0
#define PySpam_System_RETURN int
#define PySpam_System_PROTO (const char *command)

/* Total number of C API pointers */
#define PySpam_API_pointers 1


#ifdef SPAM_MODULE
/* This section is used when compiling spammodule.c */

static PySpam_System_RETURN PySpam_System PySpam_System_PROTO;

#else
/* This section is used in modules that use spammodule's API */

static void **PySpam_API;

#define PySpam_System \
 (*(PySpam_System_RETURN (*)PySpam_System_PROTO) PySpam_API[PySpam_System_NUM])

/* Return -1 on error, 0 on success.
 * PyCapsule_Import will set an exception if there's an error.
 */
static int
import_spam(void)
{
    PySpam_API = (void **)PyCapsule_Import("spam._C_API", 0);
    return (PySpam_API != NULL) ? 0 : -1;
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* !defined(Py_SPAMMODULE_H) */
```

客户端模块内容如下，
```CPP
PyMODINIT_FUNC
PyInit_client(void)
{
    PyObject *m;

    m = PyModule_Create(&clientmodule);
    if (m == NULL)
        return NULL;
    if (import_spam() < 0)
        return NULL;
    /* additional initialization can happen here */
    return m;
}
```