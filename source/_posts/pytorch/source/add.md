---
title: Tensor add 方法源码分析
date: 2023-03-16 10:49:04
tags: pytorch source code
---

# 1. add

## 1.1 方法定义
`torch.Tensor` 类继承 `torch._C._TensorBase`，这个 _TensorBase 的 add 方法定义为

```c++
// 位于 python_variable_methods.cpp
// e.g.
// >>> a = torch.Tensor([1,2], requires_grad=True)
// >>> b = a + 1    # __add__
// >>> c = 1 + a    # __radd__
// >>> a += 1       # __iadd__
PyMethodDef variable_methods[] = {
  {"__add__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__radd__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iadd__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add_>), METH_VARARGS | METH_KEYWORDS, NULL},
  ...
}
```

其中 `castPyCFunctionWithKeywords` 用于将 PyCFunctionWithKeywords 类型的函数转换为 PyCFunction，原因见以下说明。

说明：

`PyCFunction` 类型的函数用于在C中实现大多数python可调用对象

```c++
PyObject *PyCFunction(PyObject *self,
                      PyObject *args);  // 参数为类实例 self 和 position 参数 args
```

如果函数有 keyword 参数，那么需要使用如下类型，

`PyCFunctionWithKeywords` 类型的函数用于在 C 中实现可调用 python 对象，这个可调用 python 对象的签名为 METH_VARARGS | METH_KEYWORDS。

```c++
PyObject *PyCFunctionWithKeywords(PyObject *self,
                                  PyObject *args,
                                  PyObject *kwargs);
```

但是 C 不支持重载，使用 PyMethodDef 定义 struct 的方法时，PyMethodDef 期望的是 PyCFunction，所以需要将 PyCFunctionWithKeywords 转为 PyCFunction，这样可避免 compiler complaining，但是同时需要传入 METH_KEYWORDS 这个 flag，这样 Python 就知道函数类型实际上是 PyCFunctionWithKeywords，而非 PyCFunction。


`TypeError_to_NotImplemented_` 为一个模板函数，其模板参数也是一个函数 func，TypeError_to_NotImplemented_ 是 func 的一个 wrapper，如果 func 执行失败，那么抛出异常 `Py_NotImplemented` 。

### 1.1.1 `__add__`

本小节以 `__add__` 为例进行分析。分析用例如下，

```python
a = torch.Tensor([1,2])
b = a + 1
```

然后给出 `THPVariable_add` 的源码，

```c++
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);   // _TensorBase 中取 cdata 字段，这是 at::Tensor 类型
  static PythonArgParser parser({
    "add(Scalar alpha, Tensor other)|deprecated",
    "add(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::add(Tensor self, Scalar alpha, Tensor other) -> Tensor
      
      auto dispatch_add = [](const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      
      auto dispatch_add = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
```

例子匹配第二个函数 schema 即 `add(Tensor other, *, Scalar alpha=1)`，例子中 `1` 是 scalar 类型，但是对于 add 函数，允许 scalar 类型参数转为 Tensor 类型（其他允许这种转变的函数请参见 python_arg_parser.cpp 文件中的 should_allow_numbers_as_tensors 列表），并且这种类型转换在 `_r.tensor(0)` 中进行，参考下方源码。


<details>
<summary>number 转为 tensor 的代码</summary>

```c++
// 位于 torch/csrc/utils/python_arg_parser.h
inline at::Tensor PythonArgs::tensor(int i) {
  // 检查第 i 个参数是否是 _TensorBase 类型，如是，则提取其 cdata 字段（at::Tensor 类型）
  if (args[i] && THPVariable_CheckExact(args[i])) {
    return THPVariable_Unpack(args[i]);
  }
  // 否则，将第 i 个参数转换为 tensor
  return tensor_slow(i);
}
```

在 tensor_slow 方法中，先将参数转为 Scalar 类型，然后再通过 scalar_to_tensor 方法转为 at::Tensor，最后设置 is_wrapped_number_ = true，表示这个 at::Tensor 是经过 wrap 而来，相关代码如下
```c++
// 位于 torch/csrc/utils/python_arg_parser.cpp
at::Tensor tensor = scalar_to_tensor(scalar);
tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
```

</details>

然后就是调用 `dispatch_add` 执行加法操作，这个方法中首先释放 GIL，这是因为计算密集的工作通常都是让 c++ 完成，而当前线程（python 调用 c++ 进行 add 操作）主动释放 GIL 以便其他 python 线程可以获取 GIL ，而当前线程执行 c++ 的 add 方法完成后，则重新获取 GIL，然后回到 python 中。这里 `pybind11::gil_scoped_release` 是 RAII 。

下面就是 `self.add(other, alpha)` 这个调用语句了，实现类似 self += alpha * other 的功能，at::Tensor 类方法 `add` 方法定义如下，

```c++
// 位于 TensorBody.h
inline at::Tensor Tensor::add(const at::Tensor & other, const at::Scalar & alpha) const {
    return at::_ops::add_Tensor::call(const_cast<Tensor&>(*this), other, alpha);
}
```

这个方法内部调用了 `at::_ops::add_Tensor::call` 方法完成加法运算，这种格式的命名空间内的 call 方法，与 [torch_library](/2023/03/01/pytorch/source/torch_library) 一文中 empty （即 at::_ops::empty_memory_format::call）方法相同，也就是说，定义一个名为 `add_Tensor` 的 struct，其内部有一个 `call` 和一个 `redispatch` 的 static 方法，这个 struct 定义位于 ATen/ops/add_ops.h 文件中，这里就不在贴出来了，`add_Tensor` 这个 struct 的 static 方法实现与 `empty_memory_format` 相同，均在文件 build/aten/src/ATen/Operators_2.cpp 中，

```c++
// 位于 build/aten/src/ATen/Operators_2.cpp
// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<add_Tensor::schema> create_add_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add_Tensor::name, add_Tensor::overload_name)
      .typed<add_Tensor::schema>();
}

// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor add_Tensor::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    
    static auto op = create_add_Tensor_typed_handle();
    return op.call(self, other, alpha);
}
```

1. 获取 TypedOperatorHandle<add_Tensor::schema>，用于处理 add_Tensor，这是一个 operator，且已经分别通过 TORCH_LIBRARY 和 TORCH_LIBRARY_IMPL 注册 operator 的 Def 和 Impl

    - 使用 TORCH_LIBRARY 注册 Def 位于 build/aten/src/ATen/RegisterSchema.cpp
    - 使用 TORCH_LIBRARY_IMPL 注册 CPU Impl 位于 build/aten/src/ATen/RegisterCPU.cpp
    - 使用 TORCH_LIBRARY_IMPL 注册 CUDA Impl 位于 build/aten/src/ATen/RegisterCUDA.cpp
    - ...
    
2. 调用 operator 相应的 handle，在这个调用里，会根据实参分析出 DispatchKeySet，然后 dispatch 到相应的 kernel 进行处理

    注意，实际上是按先后顺序分发到 DispatchKey::AutogradCPU，DispatchKey::ADInplaceOrView，DispatchKey::CPU，但是我们这里主要讲 add 的计算过程，所以直接看 DispatchKey::CPU 对应的 kernel。

以 CPU 为例，原生的 c++ 加法实现函数为 `wrapper_add_Tensor`，这里我们不再去分析 TORCH_LIBRARY_IMPL 如何将 `wrapper_add_Tensor` 函数注册为 add_Tensor 这个 operator 的 Impl，以及在 `call` 方法中如何根据实参得到 DispatchKeySet 然后分发到相关 kernel 上（这个 kernel 内部就是 wrapper_add_Tensor 函数），相关的分析说明在  [torch_library](/2023/03/01/pytorch/source/torch_library) 一文中已经详细介绍。

直接看 wrapper_add_Tensor 方法实现代码，

```c++
at::Tensor wrapper_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
structured_ufunc_add_CPU_functional op;
op.meta(self, other, alpha);
op.impl(self, other, alpha, *op.outputs_[0]);
return std::move(op.outputs_[0]).take();
}
```

这里 `structured_ufunc_add_CPU_functional` 这个 struct 比较关键，下一小节专门对其讲解。

### 1.1.2 structured_ufunc_add_CPU_functional

下面给出了类继承关系（以及各类所在的文件帮助快速定位）

```c++
// 类名                                 所在文件                             继承自
structured_ufunc_add_CPU_functional (build/aten/src/ATen/RegisterCPU.cpp)   ->
structured_ufunc_add_CPU            (ATen/ops/add_native.h)                 ->
structured_add_Tensor               (ATen/ops/add_meta.h)                   ->
TensorIteratorBase                  (ATen/TensorIterator.h)                 ->
MetaBase                            (ATen/TensorMeta.h)
```


**# structured_add_Tensor::meta**

`meta` 方法在 structured_add_Tensor 中，`impl` 方法在 structured_ufunc_add_CPU 中，均在 struct 定义中声明。

meta 方法定义借助 `TORCH_META_FUNC2` 这个宏，
```c++
// 位于 ATen/TensorMeta.h
#define TORCH_META_FUNC2(name, overload) \
  void structured_##name##_##overload::meta
```

定义如下，

```c++
// 位于 ATen/native/BinaryOps.cpp
TORCH_META_FUNC2(add, Tensor) ( // 展开为 void structured_add_Tensor::meta
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  // 检测 alpha 类型与 tensor 类型是否匹配。满足以下三点
  // 1. 如果 alpha 是 bool 类型，那么 tensor 必须也是 bool 类型
  // 2. 如果 tensor 是 整型，那么 alpha 不能是浮点型
  // 3. 如果 tensor 不是复数型，那么 alpha 也不能是复数型
  native::alpha_check(dtype(), alpha);
}

// 位于 ATen/TensorIterator.cpp
void TensorIteratorBase::build_borrowing_binary_op(
  const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  build(BINARY_OP_CONFIG()
      .add_output(out)
      .add_input(a)
      .add_input(b));
}
```

meta 函数创建一个 TensorIteratorBase 对象，先添加所有输出 Tensor ，然后添加所有的输入 Tensor ，最后根据所有的输入输出 Tensor，计算出 shape，dtype，stride，这是因为输入输出各个 Tensor 可能在 shape，dtype，stride 上存在差异，所以需要统一好 shape，dtype，stride 等。这里 `build` 方法定义为，

```c++
void TensorIteratorBase::build(TensorIteratorConfig& config) {
  // populate some persistent configuration fields
  is_reduction_ = config.is_reduction_;
  enforce_linear_iteration_ = config.enforce_linear_iteration_;

  // fill in operands_ based on configuration
  populate_operands(config);
  // set is_output and is_read_write flags on appropriate tensors
  mark_outputs();
  // Check that the outputs have no internal overlap
  // and do not share memory with inputs.
  compute_mem_overlaps(config);
  // Check that input dimensions are aligned correctly & compute outnames.
  compute_names(config);
  // compute the broadcasted shape
  compute_shape(config);
  // mark outputs for resizing if necessary
  mark_resize_outputs(config);
  // compute the result dtype and device
  compute_types(config);
  // try fast setup output tensor, if failed, fallback to normal setup
  if (!fast_set_up(config)) {
    // compute each tensor's stride after broadcasting
    compute_strides(config);
    // re-order dimensions to improve coalescing
    reorder_dimensions();
    // allocate the output tensor if it's not provided
    allocate_or_resize_outputs();
    // coalesce adjacent dimensions when possible
    if (!is_meta_) coalesce_dimensions();
  }
  ...
}
```

build 方法中各调用函数的作用：

1. `populate_operands` 将 TensorIteratorConfig 中的 tensor 添加到 TensorIteratorBase，包括所有 output Tensor 和 input Tensor，作为 operands

2. `mark_outputs` 标记 operands 中哪些是 output，哪些既是 output 又是 input

3. `compute_mem_overlaps` 确保 output 是 `is_non_overlapping_and_dense_` 且 output 与 input 的内存不是部分重叠的

4. `compute_names` 如果 operands 存在 NamedTensor，那么 unify 所有 input Tensor

5. `compute_shape` 计算出统一的 shape，例如两个 input，维度补全和广播

6. `mark_resize_outputs` 如果 output Tensor 已经定义（有有效Impl）且与上述计算出的统一 shape 不同，那么标记为需要 resize

7. `compute_types` 计算出 `common_dtype_` 和 `common_device_`

8. `compute_strides` 计算 stride_bytes，如果 Tensor 某个维度上进行了广播，那么这个维度的 stride_bytes 为 0，否则为 `stride[i] * elem_size`

9. `reorder_dimensions` 维度重排。从低维开始观察，如果某个 tensor 的低维 stride_bytes > 高维 stride_bytes，那么交换这两个维度（stride_bytes=0 的情况忽略）

    此方法执行后，通常维度顺序变成倒序。例如 common shape 为 `(2, 3, 4)`，其 stride 为 `(12, 4, 1)`，stride_bytes 为 `(48, 16, 4)`（因为 dtype 为 float32），执行此方法后，common shape 变为 `(4,3,2)`， 各 tensor 的 stride_bytes 变为 `(4, 16, 48)`

10. `allocate_or_resize_outputs` 为 output Tensor 分配内存或者 resize。这里先计算出 tensor_shape，然后调用重载的 set_output_raw_strided 方法（ structured_ufunc_add_CPU_functional 类方法）

    这里 tensor_shape 再次经过反正，所以又变成 `(2,3,4)`。


**# structured_ufunc_add_CPU::impl**

impl 方法定义借助 `DECLARE_DISPATCH`，`DEFINE_DISPATCH` 和 `TORCH_IMPL_FUNC` 这几个宏，
```c++
// 位于 ATen/native/DispatchStub.h
#define DECLARE_DISPATCH(fn, name)         \
  struct name : DispatchStub<fn, name> {   \
    name() = default;                      \
    name(const name&) = delete;            \
    name& operator=(const name&) = delete; \
  };                                       \
  extern TORCH_API struct name name
#define DEFINE_DISPATCH(name) struct name name


// 位于 ATen/TensorMeta.h
#define TORCH_IMPL_FUNC(name) void structured_##name::impl
```

`DECLARE_DISPATCH` 定义了一个 struct，`DEFINE_DISPATCH` 定义了这个 struct 的一个变量，`TORCH_IMPL_FUNC` 的宏调用则应该展开为 `structured_ufunc_add_CPU::impl`，这几个宏调用语句为

```c++
// 位于 build/aten/src/ATen/UfuncCPU_add.cpp 
// （注意，仅这个 add kernel 写在一个单独源文件中，其他方法按分类，同一个分类的方法写在同一个源码文件中，下文会详细说明）
using add_fn = void(*)(TensorIteratorBase&, const at::Scalar &);
DECLARE_DISPATCH(add_fn, add_stub);
DEFINE_DISPATCH(add_stub);

TORCH_IMPL_FUNC(ufunc_add_CPU)(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {
  add_stub(device_type(), *this, alpha);
}
```

所以，先定义了 `add_stub` struct 和同名变量，然后 structured_ufunc_add_CPU::impl 方法内部调用 `add_stub` 这个变量的 operator() 方法，而 `add_stub` 这个类继承了 DispatchStub<add_fn, add_stub>，

```
+----------+    +--------------------------------+
| add_stub |--->| DispatchStub<add_fn, add_stub> |
+----------+    | 1. DispatchStubImpl impl;      |
                | 2. add_fn DEFAULT;             |
                +--------------------------------+
```

DispatchStub 的 operator() 定义为

```c++
// 位于 ATen/native/DispatchStub.h

using FnPtr = rT (*) (Args...);

// rT 来自于 DispatchStub 模板类的第一个模板参数 rT (*)(Args...)
// 这里例子中， FnPtr = void(*)(TensorIteratorBase&, const at::Scalar &)
// rT 是 void
// 
// ArgTypes 是 operator() 模板函数的模板参数，这里例子中是 *this, alpha
template <typename... ArgTypes>
rT operator()(DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
}
```

**总结**：impl 方法调用（`op.impl(self, other, alpha, *op.outputs_[0])`）就是 `add_stub::operator()(device_type(), *this, alpha)`，这里的 `this` 指向 `structured_ufunc_add_CPU_functional` 类实例。

## 1.2 dispatch_stub

single instruction, multiple data (SIMD)，指 CPU 支持向量化处理数据，相较于标量数据处理，向量处理器更加高效。

### 1.2.1 CPUCapability

CPU 支持的 SIMD 指令集如 AVX, AVX2, AVX512 等，pytorch 中使用 enum 类型表示，

```c++
enum class CPUCapability {
  DEFAULT = 0,
#if defined(HAVE_VSX_CPU_DEFINITION)
  VSX = 1,
#elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
  ZVECTOR = 1,
#else
  AVX2 = 1,
  AVX512 = 2,
#endif
  NUM_OPTIONS
};
```

### 1.2.2 DispatchStubImpl

`DispatchStubImpl` 这个 struct 提供了 CPU CUDA HIP MPS 等 Device 的分发方法。

```c++
// 根据设备类型获取分发方法。如果是 CPU 且 cpu_dispatch_ptr 尚未初始化，
// 则调用 choose_cpu_impl 对 cpu_dispatch_ptr 进行初始化
void* get_call_ptr(
    DeviceType device_type
    , void *DEFAULT
    ...
);

// 选择合适的 CPU 分发方法；DEFAULT 是默认分发方法
void* choose_cpu_impl(
    void *DEFAULT,
    ...
);
// CPU 由于存在多种 SIMD 指令集，所以延迟根据指令集类型选择合适的分发方法进行初始化
std::atomic<void*> cpu_dispatch_ptr{nullptr};
void* cuda_dispatch_ptr = nullptr;  // 分发到 cuda Device 的方法（函数指针）
void* hip_dispatch_ptr = nullptr;   // 分发到 hip Device 的方法（函数指针）
void* mps_dispatch_ptr = nullptr;   // 分发到 mps device 的方法（函数指针
```


### 1.2.3 DispatchStub

DispatchStub 类保存了一个 DispatchStubImpl 的实例，且提供了设置 CUDA, HIP, MPS 三种 Device 的分发方法。为每个 operator 声明一个 DispatchStub 的子类，例如 add 方法对应 add_stub 这个类。

DispatchStub 的成员变量和成员函数如下，

```c++
template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  ...
  void set_cuda_dispatch_ptr(FnPtr fn_ptr) {
    impl.cuda_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_hip_dispatch_ptr(FnPtr fn_ptr) {
    impl.hip_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_mps_dispatch_ptr(FnPtr fn_ptr) {
    impl.mps_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

    static TORCH_API FnPtr DEFAULT;
#ifdef HAVE_AVX512_CPU_DEFINITION
  static TORCH_API FnPtr AVX512;
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  static TORCH_API FnPtr AVX2;
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  static TORCH_API FnPtr VSX;
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  static TORCH_API FnPtr ZVECTOR;
#endif
private:
  DispatchStubImpl impl;
}
```

对于 CPU 这个 Device，根据 CPU 可支持的 SIMD 指令集类型，又分为 DEFAULT， AVX512, AVX2, VSX, ZVECTOR，通过 `DispatchStubImpl::get_call_ptr` 方法初始化 `impl.cpu_dispatch_ptr` 。

### 1.2.4 REGISTER_DISPATCH

提供了三个辅助类用于设置 cuda，hip ，mps 三种 Device 上的分发方法，

```c++
// 位于 ATen/native/DispatchStub.h
namespace {
template <typename DispatchStub>
struct RegisterCUDADispatch {
  RegisterCUDADispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    stub.set_cuda_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterMPSDispatch {
  RegisterMPSDispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    stub.set_mps_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterHIPDispatch {
  RegisterHIPDispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    // TODO: make this point at hip_dispatch_ptr
    stub.set_cuda_dispatch_ptr(value);
  }
};

} // anonymous namespace
```

这三个辅助类只在当前文件可见，为了能在其他 .cpp 文件中设置分发方法，又提供了一组 MACRO，

```c++
#define REGISTER_CUDA_DISPATCH(name, fn) \
  static RegisterCUDADispatch<struct name> name ## __register(name, fn);

#define REGISTER_HIP_DISPATCH(name, fn) \
  static RegisterHIPDispatch<struct name> name ## __register(name, fn);

#define REGISTER_MPS_DISPATCH(name, fn) \
  static RegisterMPSDispatch<struct name> name ## __register(name, fn);
```

设置 cpu Device 上的分发方法，即设置 DispatchStub::DEFAULT 等字段的 MACRO，

```c++
// arch 可以是 DEFAULT, AVX2, AVX512, VSX, ZVECTOR
#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
  template <> name::FnPtr TORCH_API DispatchStub<name::FnPtr, struct name>::arch = fn;
```

最后借助条件编译将所有的 REGISTER_XXX 宏统一为 `REGISTER_DISPATCH` 宏，

```c++
#if defined(__CUDACC__)
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
#elif defined(__HIPCC__)
// TODO: cut this over to HIP dispatch once we stop pretending that CUDA
// is HIP in the PyTorch HIPify build.
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
// #define REGISTER_DISPATCH(name, fn) REGISTER_HIP_DISPATCH(name, fn)
#elif defined(__OBJC__) && defined(USE_MPS)
// NB: this macro must be used from a 'mm' file in order to dispatch a MPS kernel
#define REGISTER_DISPATCH(name, fn) REGISTER_MPS_DISPATCH(name, fn)
#elif defined(CPU_CAPABILITY)
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#define REGISTER_NO_AVX512_DISPATCH(name)       \
  REGISTER_AVX512_DISPATCH(name, nullptr)
#endif
```

于是，我们只要在源文件 .cpp 中定义一个方法（kernel），然后使用 REGISTER_DISPATCH 注册这个方法，那么调用 `op.impl` 方法时，就能分发到这个 kernel 方法上。

这里以 add 为例，定义了 `add_stub` 这个类，以及这个类的一个同名实例 `add_stub`，`op.impl` 方法内部调用了 `add_stub.operator()` 方法，那么对于 CPU 设备而言，我们在源文件 build/aten/src/ATen/UfuncCPUkernel_add.cpp 中注册了 `add_stub` 类的 CPU 分发方法，

```c++
// 位于 build/aten/src/ATen/UfuncCPUkernel_add.cpp
void add_kernel(TensorIteratorBase& iter, const at::Scalar & alpha) { ... }
using add_fn = void(*)(TensorIteratorBase&, const at::Scalar &);
DECLARE_DISPATCH(add_fn, add_stub); // 定义 add_stub
REGISTER_DISPATCH(add_stub, &add_kernel); // 注册。其实就是给 add_stub::DEFAULT/AVX2 赋值
```

上面这个 REGISTER_DISPATCH 宏调用还需要结合编译条件才知道是注册的是 CPU 哪种 SIMD 指令集类型的分发方法。注意到，除了 UfuncCPUkernel_add.cpp 文件，还有两个文件，

```
UfuncCPUkernel_add.cpp.AVX2.cpp
UfuncCPUkernel_add.cpp.DEFAULT.cpp
```

这两个文件内容与 UfuncCPUkernel_add.cpp 一致，区别就是，这两个文件编译时分别增加了编译条件 `-DCPU_CAPABILITY=AVX2` 和 `-DCPU_CAPABILITY=DEFAULT` 。这样如果 CPU 支持 AVX2 时，优先使用 AVX2 中注册的方法（优先级参见 `DispatchStubImpl::choose_cpu_impl` 方法实现）。


### 1.2.5 溯源编译 FLAGS

上面两个 .cpp 文件的编译 flags 不同，这一点可在 cmake/Codegen.cmake 文件中可以得到验证。

```cmake
# 位于 cmake/Codegen.cmake
# 此文件由 caffe2 的 CMakeLists.txt 引用进来，故 CMAKE_CURRENT_LIST_DIR 就是 caffe2 目录

file(GLOB_RECURSE headers_templates "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/templates/*\.h")
file(GLOB_RECURSE sources_templates "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/templates/*\.cpp")

foreach(gen_type "headers" "sources" "declarations_yaml")
    set("GEN_COMMAND_${gen_type}"
        ${GEN_COMMAND}
        --generate ${gen_type}
        --output-dependencies ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.cmake
    )

    # Dry run to bootstrap the output variables
    # 执行命令，生成文件 build/aten/src/ATen/generated_<headers|sources|declarations_yaml>.cmake
    execute_process(
        COMMAND ${GEN_COMMAND_${gen_type}} --dry-run
        RESULT_VARIABLE RETURN_VALUE
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )
    # 引用这些文件，这些文件里面定义了 cmake 变量
    include("${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.cmake")
    include("${CMAKE_BINARY_DIR}/aten/src/ATen/core_generated_${gen_type}.cmake")
    include("${CMAKE_BINARY_DIR}/aten/src/ATen/cpu_vec_generated_${gen_type}.cmake")
    include("${CMAKE_BINARY_DIR}/aten/src/ATen/cuda_generated_${gen_type}.cmake")
    include("${CMAKE_BINARY_DIR}/aten/src/ATen/ops_generated_${gen_type}.cmake")

    add_custom_command(
      COMMENT "Generating ATen ${gen_type}"
      OUTPUT
        ${generated_${gen_type}}
        ${cuda_generated_${gen_type}}
        ${core_generated_${gen_type}}
        ${cpu_vec_generated_${gen_type}}
        ${ops_generated_${gen_type}}
        ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.cmake
        ${CMAKE_BINARY_DIR}/aten/src/ATen/ops_generated_${gen_type}.cmake
        ${CMAKE_BINARY_DIR}/aten/src/ATen/core_generated_${gen_type}.cmake
        ${CMAKE_BINARY_DIR}/aten/src/ATen/cpu_vec_generated_${gen_type}.cmake
        ${CMAKE_BINARY_DIR}/aten/src/ATen/cuda_generated_${gen_type}.cmake
      COMMAND ${GEN_COMMAND_${gen_type}}
      DEPENDS ${all_python} ${${gen_type}_templates}
        ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml
        ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/tags.yaml
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )
endforeach()
```

例如 `gen_type=sources` 时，`add_custom_command` 生成的文件列表之一 `${generated_sources}`，这个 cmake 列表变量的定义位于 build/aten/src/ATen/generated_sources.cmake，列表中包含了 build/aten/src/ATen/UfuncCPU_add.cpp 这个源文件，而 build/aten/src/ATen/UfuncCPUkernel_add.cpp 源文件则位于 `{cpu_vec_generated_sources}`，这个 cmake 列表变量定义位于 build/aten/src/ATen/cpu_vec_generated_sources.cmake。

> .cpp 文件名中包含 `kernel` 表示使用 REGISTER_DISPATCH 注册一个分发方法到 DispatchStub 的子类中

接着我们看生成 CPU 特定 SIMD 指令集类型的文件，即，为这些源文件添加不同的编译条件，

```cmake
list(APPEND CPU_CAPABILITY_NAMES "DEFAULT")
file(GLOB cpu_kernel_cpp_in "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/cpu/*.cpp" "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/quantized/cpu/kernels/*.cpp")

if(CXX_AVX512_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX512_CPU_DEFINITION")
  list(APPEND CPU_CAPABILITY_NAMES "AVX512")
endif(CXX_AVX512_FOUND)
if(CXX_AVX2_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX2_CPU_DEFINITION")
  list(APPEND CPU_CAPABILITY_NAMES "AVX2")
endif(CXX_AVX2_FOUND)
...
foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
  function(process_vec NAME)  # NAME: 文件名，不包含路径
    list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)    # CPU_CAPABILITY=DEFAULT, AVX2 AVX512 等
    set(NEW_IMPL ${CMAKE_BINARY_DIR}/aten/src/ATen/${NAME}.${CPU_CAPABILITY}.cpp) # 设置 CPU_CAPABILITY 源文件路径
    configure_file("${PROJECT_SOURCE_DIR}/cmake/IncludeSource.cpp.in" ${NEW_IMPL})  # 生成 NEW_IMPL 源文件
    list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
    set(EXTRA_FLAGS "-DCPU_CAPABILITY=${CPU_CAPABILITY} -DCPU_CAPABILITY_${CPU_CAPABILITY}")
    # 为不同的源文件设置编译条件
    set_source_files_properties(${NEW_IMPL} PROPERTIES COMPILE_FLAGS "${FLAGS} ${EXTRA_FLAGS}")
  endfunction()

  foreach(IMPL ${cpu_kernel_cpp_in})
    file(RELATIVE_PATH NAME "${PROJECT_SOURCE_DIR}/aten/src/ATen/" "${IMPL}")
    process_vec("${NAME}")
  endforeach()
  foreach(IMPL ${cpu_vec_generated_sources})
    file(RELATIVE_PATH NAME "${CMAKE_BINARY_DIR}/aten/src/ATen/" "${IMPL}")
    process_vec("${NAME}")
  endforeach()
endforeach()
```

上述代码中，`foreach(IMPL ${cpu_kernel_cpp_in})` 这个循环是匹配两个目录下的所有源文件，例如第一次进入循环，那么 

```cmake
IMPL=${PROJECT_SOURCE_DIR}/aten/src/ATen/native/cpu/Activation.cpp
```

相对路径为 `NAME=Activation.cpp`，那么执行函数 `process_vec(Activate.cpp)`，假设当前 `CPU_CAPABILITY=DEFAULT，那么

```cmake
NEW_IMPL=${CMAKE_BINARY_DIR}/aten/src/ATen/Activation.cpp.DEFAULT.cpp
```

对于 `foreach(IMPL ${cpu_vec_generated_sources})` 这个循环，由于 cmake 列表变量 ${cpu_vec_generated_sources} 只有一个元素，即 build/aten/src/ATen/UfuncCPUkernel_add.cpp，这正是上文所说的为 add_stub 注册分发方法的源文件，这个循环则是为 add_stub 生成不同编译条件（即，不同CPU SIMD 指令集类型）的源文件。

我的电脑 CPU 如下，

![](/images/pytorch/source_add_1.png)

所以为 add_stub 生成了两个源文件，（pytorch 项目中 cmake/Modules 目录下的 FindXXX.cmake 用于帮助确定 CPU 支持的指令集类型）

```shell
UfuncCPUkernel_add.cpp.AVX2.cpp # REGISTER_DISPATCH 用于设置 DispatchStub 的 AVX2 字段
UfuncCPUkernel_add.cpp.DEFAULT.cpp  # REGISTER_DISPATCH 用于设置 DispatchStub 的 DEFAULT 字段
```

相当于 add_stub 这个类的 DEFAULT AVX2 两个字段都被设置，由于获取 CPU capability 时优先考虑 VSX，ZVECTOR，然后再依次考虑 AVX512, AVX2, DEFAULT，所以在我这个电脑上，使用 AVX2 对应的分发方法，不过，add_stub 的 DEFAULT 和 AVX2 的分发方法全部是 `add_kernel`，

```c++
// 位于 build/aten/src/ATen/UfuncCPUkernel_add.cpp
void add_kernel(TensorIteratorBase& iter, const at::Scalar & alpha) { ... }
REGISTER_DISPATCH(add_stub, &add_kernel);
```

### 1.2.6 add_kernel

回顾一下，add operator 实现的是 `c = a + alpha * b`，这样的算术，其中 alpha 默认为 1，加法有两个操作数 `a b` 分别对应 self 和 other，这两个操作数以及输出 out Tensor 全部添加到 TensorIteratorBase 中（参考前面类型的继承关系），所以给 add_kernel 只需要传入 TensorIteratorBase 类实例和 alpha 。

add_kernel 方法内部，根据 TensorIteratorBase 的 common_dtype_（数据类型，ScalarType） 选择不同的处理方法，add 方法支持的数据类型包括：

```
Bool
Byte
Char
Int
Long
Short
Float
Double
ComplexFloat
ComplexDouble
BFloat16
Half
ComplexHalf
```

我们首先以 Int 为例，对应的 handle 为

```c++
AT_DISPATCH_CASE(at::ScalarType::Int,
  [&]() {
    
auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);
  }
)
```

上面代码中，AT_DISPATCH_CASE 有两个参数，一个是数据类型 ScalarType::Int，另一个是 lambda 函数，AT_DISPATCH_CASE 宏定义如下，

```c++
// 位于 aten/src/ATen/Dispatch.h
#define AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...) \
  case enum_type: {                                           \
    AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type);              \
    using HINT C10_UNUSED_DISPATCH_CUDA_WORKAROUND =          \
        c10::impl::ScalarTypeToCPPTypeT<enum_type>;           \
    return __VA_ARGS__();                                     \
  }

#define AT_DISPATCH_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)
```

发现其内部调用 AT_PRIVATE_CASE_TYPE_USING_HINT 这个宏，这里 `__VA_ARGS__` 就是上面那个 lambda 函数，执行过程为

1. 定义 `scalar_t = ScalarTypeToCPPType<c10::ScalarType::Int>` 类型
2. 执行 lambda 函数


**lambda 函数**

定义一个新类型 `ScalarTypeToCPPType<c10::ScalarType::Int>` 是为了限制 ScalarType 只能向特定的 c++ 类型转换，例如 tensor 数据类型是 ScalarType::Int，那么将 alpha（Scalar 类型）转为 c++ int 类型变量 `_s_alpha`，而 `_v_alpha` 则是将 `_s_alpha` 向量化，向量 size 为 `32/sizeof(int)=8`，即 `_v_alpha` 是一个包含 8 个元素的向量，每个元素值均为 `_s_alpha`。


**cpu_kernel_vec**

cpu_kernel_vec 有三个参数，第一个是 TensorIteratorBase 类实例，后面两个参数分别是针对 scalar 的加法实现和 vector 的加法实现，

```c++
// 位于 aten/src/ATen/native/cpu/Loops.h
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  ...
  iter.for_each(make_vectorized_loop2d(op, vop), grain_size); // 将 scalar 和 vector 两种实现方法打包
  iter.cast_outputs();
}

// 位于 aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::for_each(loop2d_t loop, int64_t grain_size) {
  int64_t numel = this->numel();  // 每个 tensor 中数据元素的数量
  if (numel == 0) {
    return;
  } else if (numel < grain_size || at::get_num_threads() == 1) {
    // 如果数据量不够大，或者只有一个线程，那么使用串行计算
    return serial_for_each(loop, {0, numel});
  } else {
    // 使用并行计算
    // 将 [0, numel) 的数据分块并行处理，每个 chunk 上使用 serial_for_each 执行
    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
      serial_for_each(loop, {begin, end});
    });
  }
}
```

上述代码中，make_vectorized_loop2d 函数生成 VectorizedLoop2d 一个实例，其中保存了 `op` 和 `vop` 两个方法，而 TensorIteratorBase::loop2d_t 类型是指

```c++
using loop2d_t = c10::function_ref<
      void(char** data, const int64_t* strides, int64_t size0, int64_t size1)>;
```

而 VectorizedLoop2d 正好有一个 operator() 方法，满足 loop2d_t 类型。

**serial_for_each**

先看串行方法的定义，

```c++
void TensorIteratorBase::serial_for_each(loop2d_t loop, Range range) const {
  if (range.size() == 0) {
    return;
  }

  const auto ntensors = this->ntensors(); // 输入输出 tensor 的数量，这里对于 add 则是 3 个 tensor
  const auto ndim = this->ndim();       // 维度数量，例如 shape (2, 3, 4)，那么 ndim=3

  c10::SmallBuffer<char*, 4> ptrs(ntensors);  // 保存输入输出 tensor 的 data_ptr
  c10::SmallBuffer<int64_t, 8> strides(ntensors * std::max(ndim, 2)); // 保存输入输出 tensor 的各维度 stride

  at::get_base_ptrs(ptrs.data(), operands_);
  at::get_strides(strides.data(), operands_, ndim);
  at::internal::serial_for_each(
      shape_, strides, ptrs.data(), ptrs.size(), loop, range);
}

// 位于 aten/src/ATen/TensorIteratorInternal.h
inline void serial_for_each(
    IntArrayRef shape,
    IntArrayRef strides, // 所有 tensor 的 strides 
    char** base_ptrs,
    size_t ntensors,    // 输入输出 tensor 数量
    typename TensorIteratorBase::loop2d_t loop,
    Range range) {
  const auto ndim = shape.size(); // tensor 的维度数量
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      strides.size() == ntensors * std::max(size_t{2}, ndim));

  if (ndim <= 1) {    // tensor 为 0 维或 1 维
    if (range.begin == 0) { // 范围起始点为 0，不需要偏移 base_ptrs 数据
      loop(base_ptrs, strides.data(), range.size(), 1);
    } else {  // 范围起始点不是 0，需要偏移 base_ptrs 数据
      c10::SmallBuffer<char*, 4> ptrs(ntensors);
      get_data_ptrs(ptrs.data(), {base_ptrs, ntensors}, strides, {range.begin});
      loop(ptrs.data(), strides.data(), range.size(), 1);
    }
  } else {// tensor 维度 >= 2，那么每个维度单独计算
    c10::SmallBuffer<char*, 4> ptrs(ntensors);
    auto counter = DimCounter(shape, range);
    while (!counter.is_done()) {
      get_data_ptrs(
          ptrs.data(), {base_ptrs, ntensors}, strides, counter.values);
      auto step = counter.max_2d_step();
      loop(ptrs.data(), strides.data(), step[0], step[1]);
      counter.increment(step);
    }
  }
}
```

以 `ndim<=1` 且 `range.begin == 0` 分支为例说明，直接调用 loop，这是一个 VectorizedLoop2d 类实例，其 operator() 定义如下，

```c++
// 位于 aten/src/ATen/native/cpu/Loops.h
void operator()(char** base, const int64_t *strides, int64_t size0, int64_t size1) {
  data_t data;
  std::copy_n(base, ntensors, data.data());
  const int64_t *outer_strides = &strides[ntensors];

  if (is_contiguous<traits>(strides)) {
    for (const auto i C10_UNUSED : c10::irange(size1)) {
      vectorized_loop(data.data(), size0, 0, op, vop);
      advance(data, outer_strides);
    }
  } else {
    using Indices = std::make_index_sequence<traits::arity>;
    unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
      if (idx) {
        for (const auto i C10_UNUSED : c10::irange(size1)) {
          vectorized_loop(data.data(), size0, idx, op, vop);
          advance(data, outer_strides);
        }
      } else {
        for (const auto i C10_UNUSED : c10::irange(size1)) {
          basic_loop(data.data(), strides, 0, size0, op);
          advance(data, outer_strides);
        }
      }
    });
  }
}
```

假设 `strides` 连续，那么就直接对着 `{0, numel}` 范围内的数据，vectorized 执行加法操作，这通过 vectorized_loop 方法完成，

```c++
// 位于 aten/src/ATen/native/cpu/Loops.h
template <typename func_t, typename vec_func_t>
static inline void
vectorized_loop(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
  using traits = function_traits<vec_func_t>;
  using scalar_t = typename function_traits<func_t>::result_type;
  using Vec = Vectorized<scalar_t>;
  constexpr int ntensors = traits::arity + 1;

  char* C10_RESTRICT data[ntensors];
  for (const auto arg : c10::irange(ntensors)) {
    data[arg] = data_[arg];
  }

  Vec opt_scalar = Vec(S > 0 ? *(scalar_t*)data[S] : scalar_t(0));
  int64_t i = 0;
  for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
    auto args1 = dereference_vec<traits>(&data[1], opt_scalar, S, i);
    auto args2 = dereference_vec<traits>(&data[1], opt_scalar, S, i + Vec::size());
    auto out1 = c10::guts::apply(std::forward<vec_func_t>(vop), std::move(args1));
    auto out2 = c10::guts::apply(std::forward<vec_func_t>(vop), std::move(args2));
    out1.store(data[0] + i * sizeof(scalar_t));
    out2.store(data[0] + (i + Vec::size()) * sizeof(scalar_t));
  }
  if (i < n) {
    int64_t strides[ntensors];
    for (const auto arg : c10::irange(ntensors)) {
      strides[arg] = (S > 0 && arg == S) ? 0 : sizeof(scalar_t);
    }
    basic_loop(data, strides, i, n, std::forward<func_t>(op));
  }
}
```

上述代码中，

1. `func_t` scalar 的加法（cpu_kernel_vec 的第一个参数，lambda 函数）

2. `vec_func_t` vector 的加法（cpu_kernel_vec 的第二个参数，lambda 函数）

3. data 保存了输出输入 tensor 的 data_ptr，由于数据的最小存储单位是字节，所有都转为 char* 类型，那么对于例子中的 int，一个数据就是 4 字节，vectorize 计算时，一次加载 `32/sizeof(int)=8` 个数据，按这种方法得到加法的两个操作数，即两个 `Vectorized<scalar_t>`，每个分别存有 8 个数据。

4. dereference_vec 方法从 `&data[1]` 开始，依次取 `Vectorized<scalar_t>`，一共取 `function_traits<vec_func_t>` 个，这就是 vop 的输入参数数量，这里例子中是 2 个输入参数，所以得到 2 个 `Vectorized<scalar_t>`，并打包为 tuple，即 `args1` 是一个 tuple，包含了两个操作数，传递给 vop 进行加法计算

5. args2 是第二个取两个操作数，相较于 args1，需要间隔 Vec::size() ，这是因为每次取两个操作数，都是分别取两个操作数的 Vec::size() 个数据，即 `32/sizeof(int)=8` 

6. 计算出两组结果 `out1` 和 `out2`，每组包含 Vec::size() 个结果数据，保存到 `data[0]` 中，这里其实已经表明，只有一个输出

7. 为何循环中，一次计算两组数据 out1 和 out2 呢？不知道，如果将 Vec::size() 扩大为 2 倍，一次计算一组数据，应该也是等效的

8. 不足 2*Vec::size() 的数据量，则采用 scalar 的计算方式，使用 op 计算，一次计算一个数据，而 vop 一次计算一个 Vectorized 的数据

**op/vop**

cpu_kernel_vec 的两个 lambda 函数参数为

```c++
[=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
[=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
```

其中调用的 ufunc::add 的定义为 

```c++
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T add(T self, T other, T alpha) __ubsan_ignore_undefined__ {
  return self + alpha * other;
}
```

在例子中，对于 op 这个 lambda，scalar_t 具体为 int，对于 vop，则用到 `Vectorized<int>` 的 `operator+, operator-, operator*, operator/` 等方法。