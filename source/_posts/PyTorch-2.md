---
title: PyTorch-2
date: 2019-06-13 10:19:52
tags: PyTorch
category: DL Framework
---
# torch installization
依然采取自顶向下的原则剖析，借助PyTorch的python接口。我们知道使用PyTorch第一步都是
```
import torch
```
于是阅读torch/__init__.py，发现需要加载torch._C这个库，但是需要以（RTLD_GLOBAL|RTLD_LAZY）这个模式动态加载，于是先将动态加载模式设置到（RTLD_GLOBAL|RTLD_LAZY）之后加载torch._C然后再恢复动态加载模式，
```
old_flags=sys.getdlopenflags()
sys.setdlopenflags(_dl_flags.RTDL_GLOBAL | _dl_flags.RTLD_LAZY)
from torch._C import *
__all__ += [name for name in dir(_C)
            if name[0] != '_' and
            not name.endswith('Base')]
sys.setdlopenflags(old_flags)
```
<b>将torch._C中（不包括_开头和Base结尾）的属性导出到当前域。</b>

__init__.py除了import torch._C，还import了同目录下其他module，以及同目录下的package。首先看torch._C导入时做了什么， torch._C的源文件只有torch/csrc/stub.cpp，链接库为shm和torch_python，stub.cpp中仅仅是初始化模块，
```
extern PyObject* initModule();
PyMODINIT_FUNC PyInit__C()   // 在python脚本中，import _C 时调用
{
  return initModule();
}
```
根据python3扩展库的规则可知，`import torch._C` ，调用PyInit__C函数（调用名为PyInit_&lt;package>的函数），这个函数内部调用initModule，也就是说，具体的模块定义由initModule实现。看到extern知道initModule方法定义在外部，所以只能从shm和torch_python对应的源文件中寻找方法定义。

shm库实现Domain Socket通信获得共享内存的句柄，解决多进程的内存分配问题，查看torch/CMakeLists.txt，发现生成shm相关语句为，
```
set(LIBSHM_SUBDIR libshm)
set(LIBSHM_SRCDIR ${LIBSHM_SRC_DIR}/lib/${LIBSHM_SUBDIR})
add_subdirectory(${LIBSHM_SRCDIR})
```
从上面语句得知shm库的源码位于torch/lib/libshm目录下，这个跟torch._C模块定义没有关系，暂且不细展开，继续查看torch_python的源码以寻求initModule方法定义。在torch/CMakeLists.txt中发现
```
add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})
```
TORCH_PYTHON_SRCS是一个列表，存储了torch_python库的源文件，生成torch_python库所需要的源文件以及依赖库直接查看torch/CMakeLists.txt，这里不再展开一一说明。

initModule方法定义在torch/csrc/Module.cpp，
```
#ifdef USE_CUDA
namespace torch { namespace cuda {
void initModule(PyObject* module);       // 模块中有关cuda部分的初始化函数声明
}}
#endif

static std::vector<PyMethodDef> methods;

PyObject* module;
PyObject* initModule() {                 // 声明并定义模块初始化函数
  // 向methods中添加方法定义
  THPUtils_addPyMethodDefs(methods, TorchMethods);
  THPUtils_addPyMethodDefs(methods, DataLoaderMethods);
  ...
  // 真正的扩展模块定义
  static struct PyModuleDef torchmodule = {
    PyModuleDef_HEAD_INIT,
    "torch._C",                          // 扩展模块名
    nullptr,                           
    -1,
    methods.data()                       // 模块中的方法定义
  };
  ASSERT_TRUE(module = PyModule_Create(&torchmodule)); // 创建模块并确保创建成功
  // 对模块进行各种初始化
#ifdef USE_CUDA
  torch::cuda::initModule(module);       // 执行cuda相关的初始化
#endif
  ...
  // 定义模块的属性设置函数，setter
  // 属性名为name，值为v，incref表示是否对值对象增加引用计数
  // 设置成功返回1，否则返回0
  auto set_module_attr = [&](const char* name, PyObject* v, bool incref = true) 
  {
    if(incref) {
      Py_INCREF(v);
    }
    return PyModule_AddObject(module, name, v) == 0;
  }
  // 设置模块属性
  ...
  ASSERT_TRUE(set_module_attr("has_cudnn", has_cudnn));
  // 向模块添加方法
  auto py_module = py::reinterpret_borrow<py::module>(module);
  py_module.def("_demangle", &c10::demangle);
  py_module.def("_log_api_usage_once", &LogAPIUsageOnceFromPython);
  ...    // 设置模块其他属性
  ASSERT_TRUE(set_module_attr("default_generator", 
        (PyObject*)THPDefaultGenerator, false));
  torch::nn::init__THNN(module);  // 增加 _THNN 属性
#ifdef USE_CUDA
  torch::nn::init_THCUDD(module);
#endif
  return module;
  ...
}
```
从上面的代码中可见，定义并生成名为torch._C的模块，然后对这个模块设置attr，添加方法，添加子模块等。
# methods/members in torch._C
- 使用 THPUtils_addPyMethodDefs 向torch._C 添加模块方法。包括
```
# TorchMethods 
_initExtension
_autograd_init
...
# DataLoaderMethods 
_set_worker_signal_handlers
_set_worker_pids
...
# torch::autograd::python_functions(), torch/csrc/autograd/init.cpp
set_grad_enabled
is_grad_enabled
set_anomaly_enabled
is_anomaly_enabled
# torch::multiprocessing::python_functions(), torch/csrc/multiprocessing/init.cpp
_multiprocessing_init
# torch::distributed::c10d::python_functions()  同上类似
...
# THCPModule_method(), torch/csrc/cuda/Module.cpp
_cuda_init
_cuda_setDevice
...
_nccl_version
...
# THCUDNN_method()
_cudnn_version
# THDPModule_methods(), torch/csrc/distributed/Module.cpp
_dist_init_extension
_dist_init_process_group
...
```
- 生成模块torch._C 后再向其添加如下成员：

    - 向torch._C添加类型_PtrWrapper，Generator，FatalError，Size，dtype，iinfo，layout，memory_format，device，_LegacyVariableBase，_TensorBase，_VariableFunctions，_FunctionBase，_EngineBase，JITException，IODescriptor，_THNN，_THCUNN。

        torch._C._TensorBase这个类型具有属性
        ```
        _cdata
        _version
        grad_fn
        _grad_fn
        is_leaf
        data
        _grad
        grad
        ...
        device
        ndim
        ```
        并且具有以下方法
        ```
        # variable_methods, torch/csrc/autograd/generated/python_variable_methods.cpp
        __add__
        __radd__
        ...
        apply_
        byte
        char
        contiguous
        ...
        where
        zero_
        # extra_method
        _make_subclass
        ```
        类型torch._C._FunctionBase， 这个类型具有方法和属性为
        ```
        # method
        apply
        _do_forward
        _do_backward
        _register_hook_dict
        register_hook
        # property
        saved_tensors
        saved_variables
        ...
        requires_grad
        metadata
        ```
        不难知道_TensorBase是Tensor的基类，包含了Tensor的各种操作，_FunctionBase则包括了前后向传播方法，从这里能将深度学习中的一些概念与代码实现建立一点点联系了。

    - 向torch._C中添加函数 _wrap_tensor_impl，_tensor_impl_raw_handle，_demangle，_log_api_usage_once，以_jit开头的一系列函数。

    - 向torch._C添加模块， _nn，cpp，_onnx。

    - 向torch._C添加属性 has_cudnn，has_openmp，has_mkl，has_lapack，has_cuda，has_mkldnn，_GLIBCXX_USE_CXX11_API，default_generator。

# some installization w.r.t. torch._C
### THPxxxStorage_init
torch._C模块中各种Tensor的定义通过 THPxxxStorage_init 和 THCPxxxStorage_init 完成，在项目中是无法直接搜索到这两种函数定义的，下面讲解这两个函数的定义。

注意到从Module.cpp文件中头文件引用：
```
#include <TH/TH.h>               // TH=TorcH
#include <c10/util/Logging.h>
#include <ATen/ATen.h>
...
#include <torch/csrc/THP.h>      // THP=TorcH Python
...
```
可以看出先引用ATen和c10库的头文件，然后再引用torch中的头文件，这是因为ATen [A Tensor Library的缩写] 实现了Tensor的运算等，c10 [表示caffe2和ATen] 实现了Tensor存储等，这两个库作为基础。

一方面，头文件 TH/TH.h 中引用了#include <TH/THGeneral.h>，在aten/src/TH目录下的CMakeLists.txt中有这么一行
```
CONFIGURE_FILE(THGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h")
```
在THGeneral.h中有如下宏定义
```
#define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
#define TH_CONCAT_4(x,y,z,w) TH_CONCAT_4_EXPAND
```
另一方面，torch/csrc/THP.h 中引用了#include <torch/src/Storage.h>，在这个Storage.h中有如下语句
```
#define THPStorage_(NAME) TH_CONCAT_4(THP, Real, Storage_, NAME)
...
#include <torch/csrc/generic/Storage.h>
#include <TH/THGenerateAllType.h>

#include <torch/csrc/generic/Storage.h>
#include <TH/THGenerateHalfType.h>

#include <torch/csrc/generic/Storage.h>
#include <TH/THGenerateBoolType.h>

#include <torch/csrc/generic/Storage.h>
#include <TH/THGenerateQTypes.h>
```
上面是4组include操作（根据不同类型生成对应的方法声明/定义，这种策略，后面还会用到很多次），可以看到每组include一次 torch/csrc/generic/Storage.h，这是为什么呢？查看文件torch/csrc/generic/Storage.h 发现其包含语句
```
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch/csrc/generic/Storage.h"         // (0)
#else
...
bool THPStorage_(init)(PyObject *module);                      // (1)
...
#endif
```
而文件TH/THGenerateAllType.h则包含语句
```
#include <TH/THGenerateFloatTypes.h>
#include <TH/THGenerateIntTypes.h>
...
#undef TH_GENERIC_FILE
```
4组include操作中，每组的第二个被include的文件均包含#undef TH_GENERIC_FILE，这使得每组include操作中，include torch/csrc/generic/Storage.h时均执行语句 (0)，而非语句 (1)，继续进一步查看TH/THGenerateFloatTypes.h，发现有
```
// 此时 TH_GENERIC_FILE是已定义的
#include <TH/THGenerateFloatType.h>
#include <TH/THGenerateDoubleType.h>
#undef TH_GENERIC_FILE     // 这里将TH_GENERIC_FILE 设为未定义
```
以TH/THGenerateFloatType.h为例说明，此文件中有语句
```
#define Real Float
...
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE         // (2)
...
#undef Real
```
注意语句 (2) 是include torch/csrc/generic/Storate.h，而此时TH_GENERIC_FILE是已定义的，所以执行 语句 (1)， 于是按如下过程进行宏替换
```
bool THPStorage_(init)(PyObject *module);  ->
bool TH_CONCAT_4(THP, Real, Storage_, init)(PyObject *module);    ->
bool TH_CONCAT_4(THP, Float, Storage_, init)(PyObject *module);   ->
bool TH_CONCAT_4_EXPAND(THP, Float, Storage_, init)(PyObject *module); ->
bool THPFloatStorage_init(PyObject *module);
```
类似地，#include <TH/THGenerateDoubleType.h>，则得到THPDoubleStorage_init，

#include <TH/THGenerateIntTypes.h> 得到
```
THPByteStorage_init
THPCharStorage_init
THPShortStorage_init
THPIntStorage_init
THPLongStorage_init
```
对4组include中的其他三组，则得到
```
THPHalfStorage_init
THPBoolStorage_init
THPQUInt8Storage_init
THPQInt8Storage_init
THPQInt32Storage_init
```
以上仅得到函数的声明，我们还需要弄清楚其定义，定义部分的构造与声明类似，首先查看torch/csrc/Storage.cpp，其中包含
```
#include <TH/THStorageFunctions.hpp>
#include <torch/csrc/THP.h>                   // include THPxxxStorage_init 函数声明
...
#include <torch/csrc/generic/Storage.cpp>
#include <TH/THGenerateAllTypes.h>

#include <torch/csrc/generic/Storage.cpp>
#include <TH/THGenerateHalfType.h>

#include <torch/csrc/generic/Storage.cpp>
#include <TH/THGenerateBoolType.h>

#include <torch/csrc/generic/Storage.cpp>
#include <TH/THGenerateQTypes.h>
```
又是4组include 操作，还是熟悉的配方，torch/csrc/generic/Storage.cpp中，
```
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch/csrc/generic/Storage.cpp"              // (11)
#else
...                                                                   // (12)
bool THPStorage_(init)(PyObject *module)
{
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, THPStorage_(methods));
#ifndef THD_GENERIC_FILE
  THPUtils_addPyMethodDefs(methods, THPStorage_(sharingMethods);
#endif
  
  THPStorageType.tp_methods = methods.data();
  THPStorageType.tp_members = THPStorage_(members);
  THPStorageType.tp_getset = THPStorage_(properties);
  if (PyType_Ready(&THPStorageType) < 0)
    return false;
  Py_INCREF(&THPStorageType);
  PyModule_AddObject(module, THPStorageBaseStr, (PyObject*)&THPStorageType);
  THPStorage_(initCopyMethods)();
  return true;
}
```
上述代码容易看出是向模块module添加字段THPStorageBaseStr， 在torch/csrc/Storage.h中有宏
```
#define THPStorageBaseStr TH_CONCAT_STRING_2(Real, StorageBase)
```
在TH/THGeneral.h中存在宏定义
```
#define TH_CONCAT_STRING_2(x,y) TH_CONCAT_STRING_2_EXPAND(x,y)
#define TH_CONCAT_STRING_2_EXPAND(x,y) #x #y
```
由于StorageBase没有宏定义，Real则可以是 Int, Float, Double, Short, Char等（见前面THPxxxStorage_init的声明分析部分），以Real=Float为例，THPStorageBaseStr此时变为"FloatStorageBase"，所以实际上是向torch._C添加字段 FloatStorageBase， 此字段类型为python class torch._C.FloatStorageBase。

以4组include操作的第一组为例说明，首次include torch/csrc/generic/Storage.cpp时，TH_GENERIC_FILE未定义，所以执行 (11)，然后include TH/THGenerateAllTypes.h，同样的，在TH/THGenerateFloatType.h中根据
```
#define Real Float
...
#include TH_GENERIC_FILE
```
即，再一次include torch/csrc/generic/Storage.cpp，此时TH_GENERIC_FILE已定义，所以从 (12) 处开始执行，得到THPFloatStorage_init的函数定义，前面已经分析过，此函数用于向torch._C 模块添加类 FloatStorageBase。

其他如Int，Char，Byte，Double，Half，QUInt8等类似处理。

torch/csrc/Module.cpp中模块初始化initModule函数中还有一些 THCPxxxStorage_init 的函数，这些函数的声明和定义与 THPxxxStorage_init 的声明和定义 的生成方式一样，不再展开细讲，直接阅读torch/csrc/cuda/Storage.h 和 torch/csrc/cuda/Storage.cpp 两个文件。

现在我们来看一下上面所述的torch._C模块中新增类到底是什么。以FloatStorageBase为例，查看torch/csrc/generic/Storage.cpp中 THPStorageType的定义，
```
PyTypeObject THPStorageType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C." THPStorageBaseStr,               /* tp_name */
  sizeof(THPStorage),                          /* tp_basicsize */
  ...
  THPStorage_(pynew),                          /* tp_new */
}
```
可见python中的类型FloatStorageBase对应在C++中的类型为THPStorage，在 torch/csrc/StorageDef.h中查看THPStorage定义
```
struct THPStorage {
  PyObject_HEAD
  THWStorage *cdata;
};
```
（插播一下，torch/csrc/generic/Storage.cpp 这里如何找到 THPStorage的定义？首先，torch/csrc/Storage.cpp中include了文件 torch/csrc/THP.h，torch/csrc/generic/Storage.cpp，然后 torch/csrc/THP.h 中include 了文件torch/csrc/Storage.h，torch/csrc/Storage.h又include了torch/csrc/generic/Storage.h，最后在这个generic/Storage.h中include了 torch/csrc/StorageDef.h）

然后查看类创建 THPStorage_(pynew) 的定义
```
static PyObject* THPStorage_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;   // 可变长度参数的个数

  THPStoragePtr self((THPStorage *)type->tp_alloc(type, 0); // 分配内存，让self指向这个内存块
  ...
  c10::Allocator * allocator = nullptr;

  if (kwargs != nullptr) {                               // named arguments
    PyObject *allocator_ptr = PyDict_GetItemString(kwargs, "allocator"); // 获取参数allocator的值
    if (allocator_ptr) {
      THPUtils_assert(THPUtils_checkLong(allocator_ptr), "invalid allocator");
      // 转为 c10::Allocator 指针
      allocator = static_cast<c10::Allocator*>(PyLong_AsVoidPtr(allocator_ptr));
      PyDict_DelItemString(kwargs, "allocator");
    }
    Py_ssize_t num_kwargs = PyDict_Size(kwargs);
    if (num_args == 0) {
      PyObject *cdata_ptr = PyDict_GetItemString(kwargs, "cdata");
      if (num_kwargs==1 && cdata_ptr && THPUtils_checkLong(cdata_ptr)) {   // 提供了cdata值
        THWStorage *ptr = (THWStorage*)PyLong_AsVoidPtr(cdata_ptr);
        self->cdata = ptr;
        return (PyObject*)self.release();       // 返回THPStorage指针
      }
    }
    THPUtils_assert(num_kwargs == 0, THPStoragePtr "(): invalid keyword arguments");
  }

  if (num_args == 0) {
    if (allocator) {                            // 未提供cdata值，则需要创建THWStorage类型实例
      self->cdata = THPStorage_(newWithAllocator)(0, allocator);
    } else {
      self->cdata = THWStorage_(new)(LIBRARY_STATE_NOARGS);
    }
    return (PyObject*)self.release();
  }
  ...     // 使用其他方法设置 self->cdata
}   
```
从上面的代码中可见，创建FloatStorageBase实例时，核心是设置 THPStorage.cdata的值，其指向一个THWStorage类型对象，在torch/csrc/THP.h中有宏定义
```
#define THWStorage THStorage
```
转而去寻找 THStorage 的定义，我们从torch/csrc/Storage.cpp出发，逐级查看被include的文件，
```
Storage.cpp                 ->
#include <TH/TH.h>          ->
#include <TH/THStorageFunction.h>   ->
#include <TH/generic/THStorage.h>   ->
#include <c10/core/StorageImpl.h>
```
在 TH/generic/THStorage.h 中找到宏定义
```
#define THStorage at::StorageImpl
```
在 c10/core/StorageImpl.h 中找到结构定义
```
namespace c10 {
struct C10_API StorageImpl final : public c10::intrusive_ptr_target {
...
private:
  caffe2::TypeMeta  data_type_;  // 数据类型
  DataPtr data_ptr_;             // 数据指针
  int64_t numel_;                // 数据数量
  bool resizable_;
  bool received_cuda_;
  Allocator* allocator_;         // 数据的内存分配器
};
}
```
所以，THWStorage实际上是类型 at::StorageImpl，这个结构是数据存储实现，我们先不去深挖这个结构，转而继续 THPStorage_(pynew) 的定义，当未提供 cdata变量值时，需要创建 THWStorage 类型实例，使用THWStorage_(NAME)函数，NAME可能的值为
```
new                // 新建THStorage，未指定 size，即size=0，使用默认Allocator
free
size
get
set
data
newWithSize        // 新建THStorage，指定 size，使用默认Allocator
newWithAllocator   // 新建THStorage，指定 size 和 Allocator
copy_functions
copyByte
...
copyCudaByte
...
```
此外有宏定义
```
#define THWStorage_(NAME) THStorage_(NAME)     // torch/csrc/THP.h
#define THStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)   // TH/THStorageFunctions.h
```
函数THStorage_(NAME) 声明分布在文件 TH/generic/THStorage.h，TH/generic/THStorageCopy.h，实现部分则位于相应的 cpp文件。

（插播：在使用cuda的情况下，#define THWStorage_(NAME) THCStorage_(NAME)，后者的声明则分布在THC/generic/THCStorage.h，THC/generic/THCStorageCopy.h）

以 THStorage_(newWithSize)函数为例说明，查看 TH/generic/THStorage.cpp，有定义
```
THStorage* THStorage_(newWithSize)(ptrdiff_t size)
{
  THStorage* storage = c10::make_instrusive<at::StorageImpl>(
#ifdef THQUANTIZED
    caffe2::TypeMeta::Make<quantized_t>(),
#else
    caffe2::TypeMeta::Make<scalar_t>(),        // 新建scalar_t 类型
#endif
    size,
    getTHDefaultAllocator(),
    true).release();
  return storage;
}
```
从这段代码中不难看出，创建StorageImpl对象，以及指向其的一个intrusive_ptr类型的指针，返回一个新的普通指针，指向这个StorageImpl，并销毁intrusive_ptr 内部指针，上文讲过有宏定义 THStorage 就是 at::StorageImpl，所以这个方法就是新建一个StorageImpl对象，并返回指向它的指针。根据c10::make_instrusive的函数定义，实际上是调用StorageImpl的构造函数完成这项工作，此构造函数为，
```
StorageImpl(
    caffe2::TypeMeta data_type,
    int64_4 numel,
    at::Allocator* allocator,
    bool resizable)
...
```
我们看上上个代码片段中StorageImpl构造函数的实参，

首先回顾一下我们是从FloatStorageBase出发走到现在这里，所以在TH/THGenerateFloatType.h 文件中找到（如果理解上文所说的 4组include操作，就能理解为什么是在这个文件中）
```
#define scalar_t float
```
于是，
```
caffe2::TypeMeta::Make<scalar_t>()    // 假设 THQUANTIZED 未定义
```
caffe2::TypeMeta::Make 这个方法是创建caffe2::TypeMeta 对象，其内部维护一个detail::TypeMetaData* 变量data_，如何new 一个TypeMetaData对象暂且不表，我们先看一组宏，
```
#define _CAFFE_KNOWN_TYPE_DEFINE_TYPEMETADATA_INSTANCE(T, Counter)         \
  namespace detail {                                                       \
  const TypeMetaData C10_CONCATENATE(_typeMetaDataInstance_, Counter) =    \
    _makeTypeMetaDataInstance<T>(_typeName<T>(#T));                        \
  }                                                                        \
  template<>                                                               \
  EXPORT_IF_NOT_GCC const detail::TypeMetaData*                            \
  TypeMeta::_typeMetaDataInstance<T>() noexcept {                          \
    return &C10_CONCATENATE(detail::_typeMetaDataInstance_, Counter);      \
  }
  _CAFFE_KNOWN_TYPE_DEFINE_TYPEMETADATA_INSTANCE(T, __COUNTER__)

#define C10_CONCATENATE_IMPL(s1,s2) s1##s2
#define C10_CONCATENATE(s1, s2) C10_CONCATENATE_IMPL(s1, s2)
```
经过宏替换，得到 _typeMetaDataInstance的模板函数定义
```
template<>
const detail::TypeMetaData*
TypeMeta::_typeMetaDataInstance<T>() noexcept {
  return &detail::_makeTypeMetaDataInstance<T>(_typeName<T>(#T));
}
```
还有一组宏，用于生成模板特例化，
```
#define CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(PreallocatedId, T)       \
  template<>                                                           \
  inline C10_EXPORT TypeIdentifier TypeIdentifier::Get<T>() {          \
    return TypeIdentifier(PreallocatedId);                             \
  }                                                                    \
  namespace detail {                                                   \
  C10_EXPORT extern const TypeMetaData C10_CONCATENATE(                \
    _typeMetaDataInstance_preallocated_,                               \
    PreallocatedId);                                                   \
  }                                                                    \
  template<>                                                           \
  inline const detail::TypeMetaData*                                   \
  TypeMeta::_typeMetaDataInstance<T>() noexcept {                      \
    return &C10_CONCATENATE(                                           \
      detail::_typeMetaDataInstance_preallocated_, PreallocatedId);    \
  }                                                                    \
#define CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(PreallocatedId, T)      \
  namespace detail {                                                 \
  const TypeMetaData C10_CONCATENATE(                                \
    _typeMetaDataInstance_preallocated_,                             \
    PreallocatedId) = _makeTypeMetaDataInstance<T>(_typeName<T>(#T));\
  }                                                                  
// 调用
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(0, uint8_t)
```
对于系统内部变量如 float，得到函数模板特例化的定义
```
// 函数声明
namespace detail {
__attrubyte((__visibility("default"))) extern const TypeMetaData
_typeMetaDataInstance_preallocated_Preallocated;
}

template<>
inline const detail::TypeMetaData*
TypeMeta::_typeMetaDataInstance<float>() noexcept {
  return &detail::_typeMetaDataInstance_preallocated_Preallocated;
}
```
另外，在c10/util/typeid.cpp中有如下调用
```
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(0, float)
```
经过宏替换得到
```
namespace detail {                                                 
  const TypeMetaData _typeMetaDataInstance_preallocated_PreallocatedId
    = _makeTypeMetaDataInstance<float>(_typeName<float>("float"));
}   
```
于是函数模板特例化最终形式为，
```
template<>
inline const detail::TypeMetaData*
TypeMeta::_typeMetaDataInstance<float>() noexcept {
  return &detail::_makeTypeMetaDataInstance<float>(_typeName<float>("float"));
}
```
detail::_makeTypeMetaDataInstance是一个模板函数，根据模板参数提供的类型创建相应类型的TypeMetaData实例，TypeMetaData是类型元数据，指定了类型在内存占多少字节空间（比如 float四个字节），类型名称，类型的构造函数、析构函数和拷贝函数等，以及类型的全局id，
```
struct TypeMetaData final {
// 函数类型的别名
using New = void*();                            // new
using PlacementNew = void(void*, size_t);       // 占位new
using Copy = void(const void*, void*, size_t);  // 类型数组拷贝
using PlacementDelete = void(void*, size_t);
using Delete = void(void*);
... //构造函数

size_t itemsize_;  // 类型占多少字节
New* new_;
PlacementNew* placementNew_;   // 定位放置 new
Copy* copy_;        // 类型拷贝
Delete* delete_;    // 类型析构
TypeIdentifier id_; // 类型全局唯一id
const char* name_;  // 类型名称
};
```
我们还以float为例，看看如何构造这个类型元数据的实例，根据以上分析查看detail::_makeTypeMetaDataInstance 模板函数的定义
```
template <class T>
inline TypeMetaData _makeTypeMetaDataInstance(const char* typeName) {
  return {sizeof(T),                 // 类型T占多少字节
          _PickNew<T>(),             // 通过 new T
          _PickPlacementNew<T>(),
          _PickCopy<T>(),      
          _PickPlacementDelete<T>(),
          _PickDelete<T>(),
          TypeIdentifier::Get<T>(),  // 获取类型的全局唯一id，
          typeName};                 // 类型名称，例如float的名称为"float"
```
构造struct结构实例，按照struct内字段顺序传入字段的值直接{}构造，类型的全局唯一id的获取使用
```
TypeIdentifier::Get<T>()
```
在上述宏定义CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE中给出这个函数（模板特例化）定义 ，其是通过调用TypeIdentifer(PreallocatedId)获取，对于float，PreallocatedId的实参值为6。

对于其他类型如 int，double，int64_t等类似处理。

PyTorch源码中给定了一些预定义好的类型及其全局唯一id值，如果是自定义变量，那么其全局唯一id则通过宏_CAFFE_KNOWN_TYPE_DEFINE_TYPEMETADATA_INSTANCE得到，具体而言是通过TypeIdentifier::createTypeId()得到，这个函数从PyTorch中预定义好的类型全局唯一id最大值（为32，对应类型为虚构的一个类型_CaffeHighestPreallocatedTypeId）开始，每次对一个自定义类型，id值增1。

至此完成TypeMetaData实例的创建，从而完成TypeMeta（其内部维护TypeMetaData指针）创建，得到构造StorageImpl的第一个实参，回到前面的THStorage_(newWithSize)(ptrdiff_t size)的函数体部分，构造StorageImpl后面的实参分别为
```
size,             // 被构造的StorageImpl包含多少类型变量（类型在TypeMeta中指定，例如float）
getTHDefaultAllocator(),  // 使用默认内存分配器，最终是使用posix_memalign函数实现内存分配
true                      // 被构造的StorageImpl可以resize
```
创建了StorageImpl实例后，就完成了THPStorage实例构造（其内部维护StorageImpl的指针），而THPStorage就对应 torch._C 模块中新增的类型FloatStorageBase

记住，这里仅以float为例说明，THPStorage还可以对应其他类型如IntStorageBase等。

FloatStorageBase的methods, members, properties 参考generic/Storage.cpp中THPStorage_(int)(PyObject* module)函数定义。

类型 _THNN 和 _THCUNN 分别通过如下函数调用添加到模型 torch._C中，
```
  torch::nn::init_THNN(module);
#ifdef USE_CUDA
  torch::nn::init_THCUNN(module);
#endif
```
函数定义位于文件torch/csrc/nn目录下的THNN.cpp和THCUNN.cpp文件中，这两个文件是生成 torch_python 这个TARGET时使用 tools/setup_helpers/generate_code.py这个脚本生成的，具体参见 torch/CMakeLists.txt。

`torch._C`模块初始化过程到这里就完成了。回到 `torch/__init__.py`，继续看看 import torch时接下来做了哪些事情：

1. 定义了模块函数 typename，is_tensor，is_storage等
2. 导入torch下其他子模块
3. 调用_C._init_name，这个函数在文件torch/csrc/Module.cpp 中实现，用于将torch模块中的DoubleStorage名称改为 torch.DoubleStorage，其他类型如FloatStorage，HalfStorage则同样这么处理
4. 调用_C._initExtension，这个函数同样在文件torch/csrc/Module.cpp 中实现，（阅读源码其实不难理解）所做的事情如下：
    - 初始化布局layout，向torch模块添加strided、sparse_coo和_mkldnn布局；
    - 初始化内存格式，向torch模块添加any_format、preserve_format、contiguous_format和channels_last内存格式；
    - 初始化类型，向torch模块添加uint8、int8、float64、float32、int32、int64、int16、float16、complex32、complex64、complex128、bool、qint8、quint8、qint32等类型，其中部分类型有旧名称，所以将旧名称类型也添加进torch模块；
    - 初始化python绑定：1）初始化PyTensorType 类型实例，每个PyTensorType实例对应一组Backend和ScalarType；2）初始化torch.tensortype类型，表示torch.FloatTensor等Tensor的metaclass；3）初始化python的各个Tensor类，如torch.FloatTensor等；4）将各个Tensor类添加到模块 torch 中；5）设置FloatTensor为默认Tensor
    - 共享内存管理初始化，设置文件路径；
    - 执行 THPxxxStorage_postInit(module)，其中xxx是类型名称，这些函数的定义可与THPxxxStorage_Init 类似地得到，其中module是torch（而非torch._C），调用这个函数注册类型相关的Python storage类（比如Float对应torch.FloatStorage），
        ```
        torch::registerStoragePyTypeObject((PyTypeObject*)THPStorageClass, backend, 
        TH_CONCAT_2(at::k, Real));
        ```
        其中 TH_CONCAT_2(at::k, Real)，即at::kReal由以下宏展开得到，是一个常量，当Real=Float时，其值为at::ScalarType::Float，
        ```
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CONSTANT)`
        ```
        这个注册调用其实就是添加THPStorageClass与back+at::kReal之间的映射。

到这里，import torch 的工作全部完成。

# 后记：
初次阅读PyTorch源码，语言组织可能比较乱，加上鄙人还有很多东西没看懂，看懂的部分仅仅是零散分布的点，不一定能连成线，更加没有形成（知识）面，所以如果有错误，请直接指正，多谢。