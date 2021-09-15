---
title: python C/C++ Extensions（一）
date: 2021-06-15 14:06:41
tags:
p: python/ext1
---
python 的 C/C++ 扩展使用说明（一）。
<!-- more -->
本文假设已经熟悉了 Python 的基本知识。 对于 Python 的更多介绍，可参考[ The Python Tutorial](https://docs.python.org/3/tutorial/index.html#tutorial-index)。 [The Python Language Reference](https://docs.python.org/3/reference/index.html#reference-index) 提供了更多关于 Python 语言的介绍。[The Python Standard Library](https://docs.python.org/3/library/index.html#library-index) 则归档了 Python 对象类型，函数以及模块。 

如要获取更全面的 Python/C API, 请参考 [Python/C API Reference Manual](https://docs.python.org/3/c-api/index.html#c-api-index)。


有很多第三方工具可用来创建 python 扩展，例如 Cython， cffi， SWIG 以及 Numba，但这里不借助这些第三方工具。

# 使用 C/C++ 扩展 Python

Python API 定义了一系列的 函数，宏 以及变量用以访问 Python 运行时系统，方便扩展。Python API 包含在头文件 `Python.h` 中。

举一个例子，创建 `spam` 扩展模块，其中提供对应于 C 语言库函数 `system()` 的 python 接口。这个库函数的参数为 null 结尾的字符串，函数返回为一个整数。我们希望 `spam` 模块中这个接口使用形式为，
```python
>>> import spam
>>> status = spam.system("ls -l")
```
首先创建一个文件 `spammodule.c`，这个源文件中实现 `spam` 模块，头两行代码为
```cpp
#define PY_SSIZE_T_CLEAN
#include <Python.h>
```
注意：Python 中可能会包含一些预处理定义，这些定义会影响其他标准头文件，所以通常第一个包含 `<Python.h>`，然后再考虑包含其他头文件。此外，推荐定义 `PY_SSIZE_T_CLEAN` 宏。

下一步定义一个 C 函数，当调用 `spam.system(string)` 时，这个 C 函数将被调用，
```cpp
static PyObject *
spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}
```
这个 C 函数有两个参数，按惯例命名为 `self` 和 `args`，其中 `self` 对模块级别的函数而言表示 模块对象自身，对于 类方法而言表示 类实例自身；`args` 指向 Python 元组对象，这个元组包含了函数参数，元组中每个 item 均为 Python 对象，需要将他们转为 C 类型值，才能调用 C 函数 `system()`，使用 `PyArg_ParseTuple` 完成这种转换，如果元组中每个 item 均转换成功，那么返回 true。

## 错误和异常
当函数调用失败，设置一个异常，并返回一个错误值（通常为 `NULL`），异常保存再一个静态全局变量中，如果这个变量为 `null`，那么说明没有异常发生。第二个全局变量存储了异常的关联值 （raise 中第二个参数：`raise expr from expr` 中后一个 `expr`，表示原始异常对象），第三个变量包含了堆栈的 traceback 信息，这三个变量是 Python 中执行 `sys.exc_info()` 返回结果的 C 等价体。

Python API 中有一系列的函数用于设置异常类型。最常见的是 `PyErr_SetString()`，参数是一个异常对象和一个 C 字符串，其中 异常对象通常是预定义类型对象，例如 `PyExc_ZeroDivisionError`，C 字符串表明错误原因。调用这个函数就完成了异常设置（相当于 python 中抛出异常）。

我们可以使用 `PyErr_Occurred()` 测试是否有异常发生，如有，则返回异常对象，否则返回 `NULL`。

当调用了 函数 `g` 的函数 `f` 检测出 `g` 函数调用失败，`f` 应该返回一个错误值 `NULL` 或者 `-1`，而不需要调用 `PyErr_*()` 函数来设置异常，因为在 `g` 中已经设置过。调用 `f` 的函数也应该返回一个错误值，同样不需要调用 `PyErr_*()`。

通过显示调用 `PyErr_Clear()` 可以忽略异常。调用 `malloc` 或者 `realloc` 失败时，需要设置异常，调用 `PyErr_NoMemory`。所有的创建对象的函数（例如 `PyLong_FromLong()` 已经实现了这个规则，这里说明一下，仅是为了针对那些直接调用  `malloc` 或者 `realloc` 的地方，在调用失败时不要忘记设置 `PyErr_NoMemory`。

注意，除了 `PyArg_ParseTuple()` 以及其他类似的函数之外，其他返回一个整型状态值的函数都在执行成功时返回一个非负值，在执行失败时，返回 `-1`，这与 Unix 系统类似。

最后需要注意，当返回一个错误值时，需要对我们自己创建的对象清除和垃圾回收（调用 `Py_XDECREF()` 或者 `Py_DECREF()`）。

有很多预定义的 异常类型，当然也可以自定义异常，例如要定义对当前模块唯一的异常，为此，在模块实现文件的开始处定义一个静态对象变量，
```cpp
static PyObject *SpamError;
```
然后在模块初始化函数 `PyInit_spam()` 中进行初始化，
```cpp
PyMODINIT_FUNC
PyInit_spam(void)
{
    PyObject *m;

    m = PyModule_Create(&spammodule);
    if (m == NULL)
        return NULL;

    SpamError = PyErr_NewException("spam.error", NULL, NULL);
    Py_XINCREF(SpamError);
    if (PyModule_AddObject(m, "error", SpamError) < 0) {
        Py_XDECREF(SpamError);
        Py_CLEAR(SpamError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
```
`PyErr_NewException()` 函数将创建一个 Exception 类型的子类，对应的 python 类型为 `spam.error`。现在我们在`system()` 调用失败时设置异常，代码如下，
```cpp
static PyObject *
spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    if (sts < 0) {
        PyErr_SetString(SpamError, "System command failed");
        return NULL;
    }
    return PyLong_FromLong(sts);
}
```

如果调用的是一个返回 void 的 C 函数，那么对应的 Python 函数则应该返回 None，所以使用如下代码实现，
```cpp
Py_INCREF(Py_None);
return Py_None;
```
或者使用宏 `Py_RETURN_NONE` 更简洁。

## 模块的方法表以及初始化
以下代码显示了如何从 Python 程序中调用 `spam_system()`，
```cpp
static PyMethodDef SpamMethods[] = {
    ...
    {"system",  spam_system, METH_VARARGS,
     "Execute a shell command."},
    ...
    {NULL, NULL, 0, NULL}        /* Sentinel */
};
```
这个数组中每一项表示一个模块方法（python 到 C 方法映射）。数组的每个条目中，第一个为字符串，表示 python 方法明，第二个为 C 方法，第三个参数可以是 `METH_VARARGS` 或者 `METH_VARARGS | METH_KEYWORDS`，对于`METH_VARARGS`，表示在 python 侧，参数以元组形式传递进来，然后使用 `PyArg_ParseTuple()` 解析成 C 类型变量。对于 `METH_KEYWORDS`，表示传递关键字参数（参数有默认值），这种情况下，C 侧函数还有第三个参数 `PyObject *` 类型，使用 `PyArg_ParseTupleAndKeywords()` 解析。

整个模块定义为，
```cpp
static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam",   /* name of module */
    spam_doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpamMethods
};
```

这个结构体需要传给 python 解释器的模块初始化函数，初始化函数名为 `PyInit_<modulename>()`，其中 `<modulename>` 表示 python 模块名，在模块定义文件中，初始化函数是唯一非静态修饰的。
```cpp
PyMODULEINIT_FUNC
PyInit_spam(void) {
    return PyModule_Create(&spammodule);
}
```

当在 python 程序中首次 import  `spam` 模块时，`PyInit_spam()` 方法被调用，其中调用 `PyModule_Create()`，返回一个模块对象指针。

在 C 代码中嵌入 Python 时，`PyInit_spam()` 不会自动调用，除非 `PyImport_Inittab` 中插入相应的一项。如下代码所示，
```cpp
int
main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    if (PyImport_AppendInittab("spam", PyInit_spam) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required.
       If this step fails, it will be a fatal error. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */
    PyObject *pmodule = PyImport_ImportModule("spam");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'spam'\n");
    }

    ...

    PyMem_RawFree(program);
    return 0;
}
```
注：所谓嵌入 python，是指将 CPython 运行时嵌入到一个更大的程序中，而不仅仅局限在实现 Python 的 C 扩展并在 Python 解释器中执行。

## 编译和链接
实现 C 扩展代码后，还需要进行编译和链接。后面会专门讨论如何实现编译链接成动态库，这里简单介绍如何将实现的 C 扩展模块作为 python 解释器的一部分，即内置模块。

将 `spammodule.c` 文件至于 python 源码的 `Modules/` 目录下，然后再 `Modules/Setup.local` 中添加一行：
```
spam spammodule.o
```

然后在 top-level 目录下运行 `make` 以重新生成 python 解释器。

如果我们自己实现的C扩展模块需要额外的链接库，也可以在配置文件 `Modules/Setup.local` 中列出，例如，
```
spam spammodule.o -lX11
```

这种将自定义模块作为解释器一部分的思路并不常见，所以不过多介绍，重点还是后面即将介绍的动态库生成。

## 从 C 中调用 Python 函数
前面介绍了如何从 Python 中调用 C 函数，现在反过来，从 C 中如何调用 python？这在支持回调的函数中尤其有用，Python 侧调用 C 扩展时，需要提供一个 回调。

还以上面那个 `spammodule.c` 文件为例，我们现在需要提供一个函数，用于接收 Python 侧提供的回调，并将回调函数对象保存到一个全局变量中，代码如下，
```cpp
static PyObject *my_callback = NULL;

static PyObject *
my_set_callback(PyObject *dummy, PyObject *args)
{
    PyObject *result = NULL;
    PyObject *temp;

    if (PyArg_ParseTuple(args, "O:set_callback", &temp)) {
        if (!PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError, "parameter must be callable");
            return NULL;
        }
        Py_XINCREF(temp);         /* Add a reference to new callback */
        Py_XDECREF(my_callback);  /* Dispose of previous callback */
        my_callback = temp;       /* Remember new callback */
        /* Boilerplate to return "None" */
        Py_INCREF(Py_None);
        result = Py_None;
    }
    return result;
}
```
同样的，这个函数，需要注册到 `spam` 模块中，与上面 `spam_system()` 类似，例如 
```
static PyMethodDef SpamMethods[] = {
    ...
    {"set_cb",  my_set_callback, METH_VARARGS,
     "Set a callback function"},
    ...
    {NULL, NULL, 0, NULL}        /* Sentinel */
};
```
在 Python 侧调用 `spam.set_cb()` 就可以设置回调函数了，之后可以在 C 代码中任意其他地方调用这个回调， 例如另一个 C 函数 `use_cb()` 中，
```cpp
int arg;
PyObject *arglist;
PyObject *result;
...
arg = 123;
...
/* Time to call the callback */
arglist = Py_BuildValue("(i)", arg);
result = PyObject_CallObject(my_callback, arglist);
Py_DECREF(arglist);
```
使用 `PyObject_CallObject()` 调用回调，有两个参数，第一个是回调对象，第二个是回调函数的参数列表，这个参数列表是一个 tuple 对象，如果回调函数无参数，那么这个参数列表可以是 `NULL`，或者一个 empty tuple。不能使用 C 类型参数，而应该使用 `Py_BuildValue()` 转换为 Python相关的类型。

`PyObject_CallObject()` 对于其参数而言，是“引用计数中立”的，所以在调用 `PyObject_CallObject()` 之后，需要立即将参数 `Py_DECREF()`。

`PyObject_CallObject()` 的返回值也需要 `Py_DECREF()`，除非将返回值保存至一个全局变量中（这个变量已经增加其引用计数）。当然在降低引用计数之前需要检查返回值是否为 `NULL`，

```
PyObject *arglist;
...
arglist = Py_BuildValue("(l)", eventcode);
result = PyObject_CallObject(my_callback, arglist);
Py_DECREF(arglist);
if (result == NULL)
    return NULL; /* Pass error back */
/* Here maybe use the result */
Py_DECREF(result);
```

也可以使用 `PyObject_Call()` 来调用有关键字参数的函数，例如，
```
PyObject *dict;
...
dict = Py_BuildValue("{s:i}", "name", val);
result = PyObject_Call(my_callback, NULL, dict);
Py_DECREF(dict);
if (result == NULL)
    return NULL; /* Pass error back */
/* Here maybe use the result */
Py_DECREF(result);
```