---
title: 注册和调用 operator（PyTorch）
date: 2023-03-01 11:20:25
tags: pytorch source code
p: pytorch/source/torch_library
summary: 介绍如何注册 operator 以及从 python 调用时的调用过程
---

准备知识：

1. [Let’s talk about the PyTorch dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)


# 1. 注册一个 dispatched operator

参考 [官方文档](https://pytorch.org/tutorials/advanced/dispatcher.html) 。

创建一个 torch::Library ，并定义其中的 operator，



```c++
TORCH_LIBRARY(myops, m) {
  m.def("myadd(Tensor self, Tensor other) -> Tensor");
}
```


说明：

1. `myops` 是这个 torch::Library 的 namespace，每个 torch::Library 需要将 namespace 注册到 Dispatcher.libraries_ 中，所以 namespace 必须唯一。

2. torch::Library.dispatch_key_ 类型为 optional<DispatchKey>，此时默认为 c10::nullopt，即 has_value() = false 。

3. 参数 `m` 就表示这个创建的 torch::Library，使用 `def` 方法定义 operator 。

注意： Dispatcher 是全局唯一的单实例。

## 1.1 在 Library 中定义一个 operator

这个 torch::Library 初始化过程中定义了一个 operator `"myadd(Tensor self, Tensor other) -> Tensor"` ，从这个字符串中可见，

1. 函数名为 `myadd`

2. 没有命名空间，如要指定命名空间，可写为 `"myops::myadd(..)"`，因为必须要与 torch::Library 的命名空间（`myops`）相同，如未指定，那么使用 torch::Library 的命名空间

3. 函数参数为两个 Tensor，函数返回值为一个 Tensor。

<details>
<summary>相关代码</summary>
位于 aten/src/ATen/core/library.cpp 中的 

```c++
Library& Library::_def(c10::FunctionSchema&& schema, c10::OperatorName* out_name, const std::vector<at::Tag>& tags, _RegisterOrVerify rv)
```

</details>

这个 operator 还需要注册到 Dispatcher 中，其实 Library，operator Def 和 operator Impl 都要注册到 Dispatcher 中，这样用户发起一个函数调用时，Dispather 才能找到相应的 operator 来处理。

**# Dispatcher 的几个字段**，

1. `libraries_` 存储各个 Library 的 namespace
2. `operatorLookupTable_` 存储所有 operator 的 `OperatorName` 到 `OperatorHandle` 的映射
3. `operators_` 存储所有 operator，类似于 `operatorLookupTable_` 中的所有 value 集合

    区别是 value 类型是 OperatorHandle，而 `operators_` 存储的是 OperatorDef 对象。OperatorHandle 与 OperatorDef 是一一对应的，且在 OperatorDef 基础上增加几个方法，以对 operator 进行调用。

    ```
    +-----OperatorHandle----+
    |   1. callBoxed()      |
    |   2. typed()          |       +----OperatorDef----+      +--- OperatorEntry ---+
    |   3. operatorDef_ ----+---->  |   1. op ----------+--->  | 1. registerSchema() |
    |   ...                 |       |   ...             |      | ...                 |
    +-----------------------+       +-------------------+      +---------------------+
    ```

**# 注册 operator Def 的流程**为：

1. 在 `Dispatcher.operatorLookupTable_` 中根据 OperatorName 查找是否有 OperatorHandle

    OperatorName 包含了函数 name 和 overload_name。在这个例子中，name 为 `myops::myadd` （包含了命名空间），overload_name 为空，要指定 overload_name，那么函数名需要是 `[name].[overload_name]` 的形式。

2. 如存在则直接返回找到的 OperatorHandle；如不存在，则根据 OperatorName 创建一个 OperatorHandle，然后加入 `operatorLookupTable_` 映射，返回新建的 OperatorHandle

3. 如上面的类示意图所示，在 OperatorHandle 的 OperatorEntry 上调用 `registerSchema` 完成 operator Def 注册

    要搞清楚 `registerSchema` 做了什么，需要先对 OperatorEntry 这个类了解一下。

### 1.1.1 OperatorEntry

**# OperatorEntry 几个关键字段**：

1. `name_` operator 的名称 OperatorName

2. `dispatchTable_` KernelFunction 的数组，一个 operator 通常由若干个 kernel 实现，最多 `num_runtime_entries` 个 kernel
    
    每个 kernel 对应一个 DispatchKey。

3. `dispatchKeyExtractor_` 根据 operator 的函数参数获取对应的 DispatchKeySet

4. `kernels_` DispatchKey 与 list< AnnotatedKernel > 之间的映射。AnnotatedKernel 是在 KernelFunction 基础上增加了函数 schema 和 debug 信息。

    这里为何一个 DispatchKey 对应多个 Kernel？`dispatchTable_` 中 kernel 与 DispatchKey 不是一一对应的吗？其实，对于每个 DispatchKey， `dispatchTable_` 中的 kernel 是 `kernels_` 中列表的第一个 kernel，列表中其他 kernel 则是 overwrited，即新的 kernel 覆盖了旧的 kernel，而 `dispatchTable_` 中总数存储最新的 kernel。

**# OperatorEntry 的构造函数**

前面讲到，`Dispatcher.operatorLookupTable_` 根据 OperatorName 查找 OperatorHandle，如不存在，则根据 OperatorName 创建 OperatorHandle，在构造 OperatorEntry 对象时，`name_` 就使用这个 OperatorName 初始化，OperatorEntry 的其他字段则全部默认初始化。构造函数中还做了一件事 `updateDispatchTableFull_(..)`，

```c++
OperatorEntry::OperatorEntry(OperatorName&& operator_name) : name_(std::move(operator_name)), ... {
    updateDispatchTableFull_(c10::Dispatcher::singleton());
}
```

`updateDispatchTableFull_` 方法做了哪些事呢？
1. 更新 `dispatchTable_`
2. 记录每个 DispatchKey 是否 fallthrough

对于第 1 点，每个 DispatchKey 均对应 `dispatchTable_` 中的一个下标 index（参见 [DispatchKeySet 一文](2023/02/17/pytorch/source/dispatchkeyset) 中的表 1），有了下标 index，还需要知道每个 DispatchKey 的 kernel，才能更新到 `dispatchTable_` 中去。

> 如何根据 DispatchKey 得到 kernel？

<details>
<summary>相关代码</summary>

```c++
std::pair<const AnnotatedKernel&, const char*> OperatorEntry::computeDispatchTableEntryWithDebug(
    const c10::Dispatcher& dispatcher, 
    DispatchKey dispatch_key)
```

</details>

以下按照优先级从高到低的顺序获取 kernel，

1. 从 `kernels_` 中根据 DispatchKey 获取 kernel（如有，则取已有的 kernel，否则往下执行，下同）

2. 如果 DispatchKey 被某个 alias key 包含，那么从 `kernels_` 中根据这个 alias key 获取 kernel

3. 从 `Dispatcher.backendFallbackKernels_` 中获取 DispatchKey 对应的 kernel

4. 返回默认的 kernel `missingKernel()`

**结论**：在初始化 OperatorEntry 对象时，`dispatchTable_` 中每个 DispatchKey 的 kernel 均为默认 kernel

> `updateDispatchTableFull_` 为哪些 DispatchKey 更新 kernel？

<details>
<summary>updateDispatchTableFull_ 代码</summary>

```c++
void OperatorEntry::updateDispatchTableFull_(const c10::Dispatcher& dispatcher) {
  updateDispatchTable_(dispatcher, DispatchKey::Undefined);
  for (auto k : DispatchKeySet(DispatchKeySet::FULL)) {
    updateDispatchTable_(dispatcher, k);
  }
}
```

</details>

从上面的代码可知，被 updateDispatchTableFull_ 更新的 DispatchKey 包括 `DispatchKey::Undefined`，以及 `DispatchKeySet(Full)` （这是 Dispatchkey 集合）中所有 DispatchKey，共 `41-5+5*14=106` 个，遍历顺序如下：

<details>
<summary>CPU, CUDA, ... , FPGA, ...</summary>

```
CPU
CUDA
...
PrivateUse3

FPGA
ORT
Vulkan
Metal

QuantizedCPU
QuantizedCUDA
...
QuantizedPrivateUse3

CustomRNGKeyId
MkldnnCPU

SparseCPU
SparseCUDA
...
SparsePrivateUse3

SparseCsrCPU
SparseCsrCUDA

NestedTensorCPU
NestedTensorCUDA
...
NestedTensorPrivateUse3

BackendSelect
{其他普通 DispatchKey，这里不在一一列举}
AutogradOther

AutogradCPU
AutogradCUDA
...
AutogradPrivateUse3

AutogradNestedTensor
{其他普通 DispatchKey，这里不在一一列举}
PythonDispatcher
```

</details>

以上 DispatchKey 集合加上 `DispatchKey::Undefined` 一共 107 个，这与 `dispatchTable_` 数组的大小 `c10::num_runtime_entries=107` 相等。


<!-- 一个 OperatorEntry 类实例对应一个 operator，一个 operator 由多个 kernel 实现，每个 kernel 对应一个 DispatchKey，然而经过刚才分析一共有 107 个 DispatchKey，显然为每个 DispatchKey 均实现一个 kernel 太麻烦，实际上也没有必要， -->


**# registerSchema**

前面 **# 注册 operator Def 的流程** 一节讲到使用 OperatorEntry 对象的 registerSchema 完成注册，根据下方代码片段可知，这个方法主要是给 OperatorEntry 的 `schema_` 赋值 。

```c++
void OperatorEntry::registerSchema(FunctionSchema&& schema, std::string&& debug, std::vector<at::Tag> tags) {
  ...
  // 记录 schema 中哪些位置的 param 用于 dispatch。
  // 用于 dispatch 的 param 包括 Tensor，List<Tensor>, Optional<Tensor>, List<Optional<Tensor>>
  dispatchKeyExtractor_.registerSchema(schema); 
  schema_ = AnnotatedSchema(std::move(schema), std::move(debug));
  #ifndef C10_MOBILE
    tags_ = std::move(tags);
  #endif
}
```

## 1.2 注册一个 operator Impl

前面介绍了注册一个 operator Def，但是 operator 的具体实现方法，还未注册，只有注册了 operator Impl，才能调用这个 operator。官方文档的例子如下，

<details>
<summary>一个自定义的加法实现</summary>

```c++
Tensor myadd_cpu(const Tensor& self_, const Tensor& other_) {
  TORCH_CHECK(self_.sizes() == other_.sizes());
  TORCH_INTERNAL_ASSERT(self_.device().type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(other_.device().type() == DeviceType::CPU);
  Tensor self = self_.contiguous();
  Tensor other = other_.contiguous();
  Tensor result = torch::empty(self.sizes(), self.options());
  const float* self_ptr = self.data_ptr<float>();
  const float* other_ptr = other.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = self_ptr[i] + other_ptr[i];
  }
  return result;
}
```

</details>

这个加法基于 CPU 实现，使用 PyTorch 宏注册这个实现，

```c++
TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("myadd", myadd_cpu);
}
```

其中 `"myadd"` 是 operator 的函数名，`myadd_cpu` 是 c++ 实现方法。

查看宏 `TORCH_LIBRARY_IMPL` 的定义发现，这里竟然重新创建了一个 torch::Library 实例，并调用这个实例的 `impl` 方法。

这两个 torch::Library，其中一个是 DEF 类型的，另一个 IMPL 类型的，两者并不冲突，DEF 类型的 torch::Library 用于向 Dispatcher 注册 Library 并注册 operator Def（相当于函数声明），而 IMPL 类型的 torch::Library 用于向 Dispatcher 注册 operator 的 kernel 方法实现，用户在调用一个 operator 时，由于 Def 和 Impl 均注册到 Dispatcher 中，所以 Dispatcher 可以直接 dispatch 到对应的 operator kernel，不需要经过任何 torch::Library 就能到达 operator kernel。


注册 Impl 过程：

1. 将 c++ 实现方法包装成 `CppFunction` 类型的对象

2. 调用 Dispatcher 的 registerImpl 方法完成注册

    - 根据 OperatorName 找到对应的 OperatorHandle （由于之前已经注册 Def，所以 Dispather 中一定有这个 OperatorHandle）
    - 取得 OperatorHandle 内部的 OperatorEntry，然后调用 OperatorEntry 的 registerKernel 方法

下面来仔细看这两个步骤如何完成。

### 1.2.1 包装成 CppFunction

torch::Library 的 `impl` 方法中调用如下语句将 c++ 原生实现方法保证成 CppFunction 实例，

```c++
CppFunction f(std::forward<Func>(raw_f));
```

这里 `Func` 就是 `myadd_cpu` 的类型 `Tensor (*)(const Tensor&, const Tensor&)`，使用的 CppFunction 构造函数如下，

```c++
explicit CppFunction(
      Func* f,
      std::enable_if_t<
          c10::guts::is_function_type<Func>::value,
          std::nullptr_t> = nullptr)
      : func_(c10::KernelFunction::makeFromUnboxedRuntimeFunction(f)),
        cpp_signature_(c10::impl::CppSignature::make<Func>()),
        schema_(
            c10::detail::inferFunctionSchemaFromFunctor<std::decay_t<Func>>()),
        debug_() {}
```

1. 使用函数指针 `f`（实参是 `myadd_cpu`） 构造一个 KernelFunction 实例
2. 根据 `f` 的函数类型构造一个 CppSignature 实例
3. 根据 `f` 的函数类型构造一个 FunctionSchema 实例

CppSignature 内部记录了一个 `std::type_info`，用于封装函数类型相关的信息。FunctionSchema 则记录了函数 OperationName，参数，返回值等信息。重点是第 1 点：构造 KernelFunction 实例，核心调用语句为

```c++
// func 就是 `f` 对象（这里例子的实参是 `myadd_cpu`）
// AllowLegacyTypes 未指定，默认为 false
// FuncType 就是 `f` 的函数类型
return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>(
        guts::make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>(func)
    );
```

（注：decay 为类型T应用从左值到右值（lvalue-to-rvalue）、数组到指针（array-to-pointer）和函数到指针（function-to-pointer）的隐式转换。转换将移除类型T的cv限定符（const和volatile），并定义结果类型为 decay< T >::type）

我们先看模板参数类型 `WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>`，这是一个包装类型，其自身是一个模板类，这个模板类的参数类型是函数指针类型、函数的返回值类型以及函数的参数类型，如下所示，

```c++
template<class FuncType>
  using WrapFunctionIntoRuntimeFunctor = detail::WrapFunctionIntoRuntimeFunctor_<
      FuncType,
      typename guts::infer_function_traits_t<FuncType>::return_type,
      typename guts::infer_function_traits_t<FuncType>::parameter_types
  >;
```

**这个包装类型将任何一个函数指针进行包装**，内部记录了一个函数指针，所有（使用任意类型的函数指针具现化后）的包装类均有一个共同的基类 —— `OperationKernel`，以后，我们可以将 OperationKernel 类型强转为某个具现化的包装类，从而可以获取其内部的函数指针，然后调用函数，不过这个包装类提供了一个 `operation()` 方法，可以直接调用这个包装类的实例，内部实际上正是调用所存储的函数指针。

回到 `makeFromUnboxedFunctor` 调用语句，`make_unique_base` 就是根据参数 `func` （这里例子实参是 `myadd_cpu`）构造包装类 `WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>`，然后再根据这个子类创建父类（即 OperationKernel）的 unique 类型指针 `unique_ptr<OperationKernel>`，其指向的 OperationKernel 实际上是一个 Functor（函子，提供了 operator() 方法的类，可直接当成函数调用），于是，`makeFromUnboxedFunctor` 就是根据一个 Functor 创建 KernelFunction 对象。

（注：暂时先不用关注 `boxed` 和 `unboxed` 两种调用类型的区别）

下面给出 `makeFromUnboxedFunctor` 的实现代码（一目了然，并不复杂），

```c++
template<bool AllowLegacyTypes, class KernelFunctor>
inline KernelFunction KernelFunction::makeFromUnboxedFunctor(std::unique_ptr<OperatorKernel> kernelFunctor) {
    ...
    auto* unboxed_fn = &impl::wrap_kernel_functor_unboxed<KernelFunctor>::call;
    void* void_unboxed_fn = reinterpret_cast<void*>(unboxed_fn);
    bool is_symint = fn_has_symint<decltype(unboxed_fn)>::value;
    return KernelFunction(
        std::move(kernelFunctor),
        &impl::make_boxed_from_unboxed_functor<KernelFunctor, AllowLegacyTypes>::call,
        is_symint ? nullptr : void_unboxed_fn,
        is_symint ? void_unboxed_fn : nullptr
    );
}
```

根据前面 `makeFromUnboxedFunctor` 调用语句，这里的模板参数 `AllowLegacyTypes` 默认为 false，`KernelFunctor` 就是 `WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>` （包装类，函子，内部存储了具体的函数指针），其父类型是 `OperatorKernel`。

简单的说一下上述代码，

1. `wrap_kernel_functor_unboxed` 这是一个模板类，定义如下

    ```c++
    template<class KernelFunctor>
    using wrap_kernel_functor_unboxed = wrap_kernel_functor_unboxed_<KernelFunctor, typename guts::infer_function_traits_t<KernelFunctor>::func_type>;

    template<class KernelFunctor, class ReturnType, class... ParameterTypes>
    struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(ParameterTypes...)> final {
        ...
        static ReturnType call(OperatorKernel* functor, DispatchKeySet, ParameterTypes... args) {
        KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
        return (*functor_)(std::forward<ParameterTypes>(args)...);
        }
    };
    ```

    - `infer_function_traits_t<KernelFunctor>`::func_type  获取函子中 `operator()` 方法类型，这里剥离了 `Class::`，也就是得到的不是一个类方法，而是一个普通的函数，参数和返回值与函子的 `operator()` 完全相同。
    - `wrap_kernel_functor_unboxed_` 对函子 KernelFunctor 又进行了一层包装，提供 `call` 方法，传入一个函子的基类对象（OperatorKernel）对象，以及参数，将基类对象类型强转为具体的函子类型（这里例子中则是 `WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>`），然后将参数传入函子进行调用。

2. `unboxed_fn` 是一个类方法的指针，`ReturnType (wrap_kernel_functor_unboxed_<KernelFunctor>::*)(DispatchKeySet, ParameterTypes...)`

    这里例子中，ReturnType 是一个 Tensor，`ParametersTypes...` 是 `const Tensor&, const Tensor&`

3. 将 `unboxed_fn` 转为一个 `void *` 指针得到 `void_unboxed_fn`

4. `is_symint` unboxed_fn 参数列表中是否含有 `SymInt` 或其相关类（在这个例子中，没有）

5. 创建 KernelFunction 对象
    
    - 第一个参数是函子的基类型指针 `unique_ptr<OperatorKernel>`
    - 第 三/四 个参数是 没有/有 SymInt 相关类型参数时转为 `void*` 的调用方法`::call`
    - 第二个参数是 BoxedKernel::InternalBoxedKernelFunction 类型，

        ```c++
        using InternalBoxedKernelFunction = void(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*);
        ```

        实参是 make_boxed_from_unboxed_functor 的类方法 `call`，从这个类名大概知道是从 unboxed functor 得到 boxed 的调用，类方法 `call` 实现如下（简化后），

        <details>
        <summary>make_boxed_from_unboxed_functor::call 方法实现</summary>

        ```c++
        static void call(OperatorKernel* functor, const OperatorHandle&, DispatchKeySet dispatchKeySet, Stack* stack) {
            using ReturnType = typename guts::infer_function_traits_t<KernelFunctor>::return_type;  // 返回值类型
            using ArgTypes = typename c10::remove_DispatchKeySet_arg_from_func<KernelFunctor>::parameter_types; // 参数类型（去除 DispatchKeySet 参数）
            constexpr bool has_outputs = !std::is_same<void, ReturnType>::value;    // 返回void则表示无输出，否则有输出
            constexpr size_t num_inputs = guts::typelist::size<ArgTypes>::value;    // 参数量
            guts::if_constexpr<has_outputs>([&] (auto delay_check) {    // 编译期 if 语句
                // 如果有返回值，那么调用 call_functor_with_args_from_stack，其内部执行 functor，函数参数存储与 stack 中
                using ReturnType_ = std::decay_t<typename decltype(delay_check)::template type_identity<ReturnType>>;
                ReturnType_ output = call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor, dispatchKeySet, delay_check(stack));
                // 将输入参数从 stack 中移除
                torch::jit::drop(*stack, num_inputs);
                // 将函数输出压入 stack 中
                push_outputs<ReturnType_, AllowDeprecatedTypes>::call(std::move(output), stack);
        
            }, /* else */ [&] {
                // 没有返回值，那么直接调用 functor，参数在 stack 中，调用完毕后将参数从 stack 中移除
                call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor, dispatchKeySet, stack);
                torch::jit::drop(*stack, num_inputs); 
            });

        }
        ```

        </details>


        从这里可见，unboxed 的调用，函数参数是各种类型的，boxed 的调用，参数存储与 Stack 中，且每个参数均包装成 IValue 类型，所以 `make_boxed_from_unboxed_functor<KernelFunctor, AllowLegacyTypes>::call` 是一个 boxed 的调用。


至此，根据函数指针创建 CppFunction 对象完成。由于内容较多，需要将代码完整的看一遍才能彻底搞清楚，以上是将代码中几个关键的部分拎出来讲一下。

**# 总结创建 CppFunction 的过程**

<details>
<summary>示意图</summary>

```c++ 
kernelFunctor ----> +--WrapFunctionIntoRuntimeFunctor<Func*>----+     
                    |  Base: OperatorKernel                     |
                    |  Func*     -------------------------------+-----> |Func* f(e.g. myadd_cpu)|
                    |  operator()                               |
                    +-------------------------------------------+         

KernelFunctor = WrapFunctionIntoRuntimeFunctor<Func*>

                            +--------------wrap_kernel_functor_unboxed<KernelFunctor>----------------+
void_unboxed_fn --(void*)---+--> ReturnType call(OperatorKernel*, DispatchKeySet, ParameterTypes...) |
                            +------------------------------------------------------------------------+



                    +-------make_boxed_from_unboxed_functor<KernelFunctor, AllowLegacyTypes>--------+
boxed_kernel_func --+--> void call(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*)  |
                    +-------------------------------------------------------------------------------+

boxed_kernel_func --->  +----------BoxedKernel-----------+
                        | intrusive_ptr<OperatorKernel> -+--> kernelFunctor
                        | InternalBoxedKernelFunction* --+--> boxed_kernel_func
                        +--------------------------------+

   
func ---->  +---------KernelFunction-----------+     
            | BoxedKernel  --------------------+---> boxed_kernel_func
            | void* unboxed_kernel_func_     \_|__-> void_unboxed_fn   
            | void* sym_unboxed_kernel_func_ / |     
            +----------------------------------+

+----CppFunction----+ 
| KernelFunction  --+--> func
| ...               |
+-------------------+
```

</details>

1. 参数 `Func* f` （这里例子是 `myadd_cpu`） 先是封装到 `unique_ptr<OperatorKernel>` 中（即 kernelFunctor 这个对象），然后封装到 `BoxedKernel` 中，且 `BoxedKernel` 还有一个 boxed 调用，然后 `KernelFunction` 中包含 `BoxedKernel` 以及另外一个 unboxed 调用

2. BoxedKernel 包含 functor 和 boxed call，KernelFunction 在 BoxedKernel 基础上增加一个 unboxed call

3. functor、boxed call 和 unboxed call 的区别大概是
    
    ```c++
    functor(Parameters... args) // 直接调用
    boxed_call(functor, Stack* stack) { functor( (Parameters...)stack ); }
    unboxed_call(functor, Parameters... args) { functor(args); }
    ```

### 1.2.2 registerKernel

这是 OperatorEntry 的类方法（回顾前面 1.2 小节最开始的讲解）。

```c++
OperatorEntry::AnnotatedKernelContainerIterator OperatorEntry::registerKernel(
  const c10::Dispatcher& dispatcher,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
){ ... }
```

参数：

1. `dispatcher` 全局唯一 Dispatcher 单实例

2. `dispatch_key` 这个 kernel 对应的 DispatchKey

    创建 IMPL 的 torch::Library 我们指定了 DispatchKey，而 CppFunction 中未指定 DispatchKey，所以使用 torch::Library 中的 DispatchKey，这里例子中这个参数为 DispatchKey::CPU

3. `kernel` CppFunction 的 `func_` 字段，为 KernelFunction 类型

4. `cpp_signature` CppFunction 中的 `cpp_signature_` 字段

5. `inferred_function_schema` 根据 `Func* f` 原生函数指针类型推断出的 FunctionSchema

    这里例子中则是根据 `myadd_cpu` 推断出的 FunctionSchema

6. `debug` 出错时用于打印的信息

    `debug = c10::str("registered at ", file, ":", line)`，file 和 line 分别为当前 TORCH_LIBRARY_IMPL 调用语句所在文件和行号

在注册 Impl 阶段，从 Dispatcher 中查找的 OperatorEntry 对象与之前注册 Def 阶段中从 Dispatcher 中查找的 OperatorEntry 对象是同一个，在 registerDef 过程中，OperatorEntry 对象已经设置了 `schema_`，所以需要将 `schema_` 与 `infered_function_schema` 比较是否相同，如不同则报错。接着，

1. 从 `kernels_` 中根据 dispatch_key 取相应的列表，这是对应 dispatch_key 的 kernel 列表。

    前面讲过，`kernels_` 的每个 dispatch_key 对应的是 kernel 列表，而不是单个 kernel，这是因为 kernel 可能被 overwrite，而最新的 kernel 则总是列表的第一个元素，这个元素与 `dispatchTable_` 中的 kernel 一致。But，如果 dispatch_key != DispatchKey::Meta，那么 overwrite kernel 会打印一个 WRAN 信息。

2. `kernels_` 是 `flat_hash_map` 类型，使用 `kernels_[*dispatch_key]` 获取列表时，如果 `kernels_` 不存在 dispatch_key，那么会及时的创建一个列表存储到 `kernels_` 中。故在 registerDef 过程中，遍历 `kernels_` 时，`kernels_` 中其实是没有键值对的，而现在 registerImpl/registerKernel 时，对于指定的 DispatchKey，则初始都会创建一个空列表。

3. 将 `kernel` 插入列表第一个位置

4. 将列表第一个位置的元素更新到 `dispatchTable_` 中

5. 设置 operator 对应 dispatch_key 是否是 fallthrough kernel

    如果注册的 kernel （其内部的 boxed call） 是 `fallthrough_kernel` 方法，那么就要相应的设置 flag，表示这个 dispatch_key 有 fallthrough 方法。`fallthrough_kernel` 是位于 kernelfunction.cpp 文件中的一个默认方法。

# 2. 一个真实的例子

## 2.1 注册一个真实的 operator

我们以一个真实存在于 pytorch 中的 operator `empty.memory_format` 为例再梳理一遍注册过程。

**# 注册 Def**

build pytorch 项目之后，可以在文件 build/aten/src/ATen/RegisterSchema.cpp 文件中发现注册了很多 operator，如下代码所示

（这些生成的源码文件都是由 `gen.py` 文件生成）

```c++
TORCH_LIBRARY(aten, m) {
    ...
    m.def("empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor", {});
    ...
}
```

`m.def("xxx", {})` 中的 `{}` 表示 tags 这个参数值，这个参数默认值就是 `{}`，所以前面的 `myadd` 注册时就没有提供这个实参。具体可查看 `Library::def` 方法。

**# 注册 Impl**

注册 operator Impl 的代码则位于 build/aten/src/ATen 目录下的多个源码文件中，

```
RegisterCPU.cpp
RegisterCUDA.cpp
RegisterQuantizedCPU.cpp
RegisterQuantizedCUDA.cpp
RegisterSparseCPU.cpp
RegisterSparseCUDA.cpp
RegisterBackendSelect.cpp
...
```

例如在 RegisterCPU.cpp 中，我们可以找到

```C++
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    ...
    m.impl("empty.memory_format",
    TORCH_FN(wrapper_memory_format_empty));
    ...
}
```

首先需要注意，这里的函数变成了 `TORCH_FN(wrapper_memory_format_empty)`，其中

```c++
at::Tensor wrapper_memory_format_empty(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  return at::native::empty_cpu(c10::asIntArrayRefSlow(size), dtype, layout, device, pin_memory, memory_format);
}
```

`wrapper_memory_format_empty` 的 schema 与前面注册 Def 的保持一致，在应用 `TORCH_FN` 之后就不是一个普通的函数了，而是一个将函数包装成 c++ 类，此时 `m.impl` 方法中创建 CppFunction 对象的构造函数也变成

```c++
// 位于 torch/library.h
template <typename FuncPtr>
explicit CppFunction(
    FuncPtr f,
    std::enable_if_t<
        c10::is_compile_time_function_pointer<FuncPtr>::value,
        std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedFunction(f)),
    cpp_signature_(
        c10::impl::CppSignature::make<typename FuncPtr::FuncType>()),
    schema_(c10::detail::inferFunctionSchemaFromFunctor<
            typename FuncPtr::FuncType>()),
    debug_() {}
```

<details>
<summary>创建一个 empty tensor 过程简介</summary>

回到 `wrapper_memory_format_empty` 函数中来，这个函数调用 `at::native::empty_cpu`，其定义如下

```c++
// 位于
Tensor empty_cpu(IntArrayRef size, c10::optional<ScalarType> dtype_opt, c10::optional<Layout> layout_opt,
                 c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt, c10::optional<c10::MemoryFormat> memory_format_opt) {
  return at::detail::empty_cpu(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}
```

其内部又继续调用 `at::detail::empty_cpu` 定义如下，

```c++
TensorBase empty_cpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto pin_memory = pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return empty_cpu(size, dtype, pin_memory, memory_format_opt);
}

TensorBase empty_cpu(IntArrayRef size, ScalarType dtype, bool pin_memory,
                     c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return empty_generic(size, allocator, cpu_ks, dtype, memory_format_opt);
}
```

第一个 `empty_cpu` 方法中，将各参数由 `optional<T>` 转为 T 类型，如果 `optional<T>` 没有值，那么使用 T 的默认值。第二个 `empty_cpu` 方法实现则是调用 `empty_generic` 方法，

```c++
// 位于 aten/src/ATen/EmptyTensor.cpp
TensorBase empty_generic(
    IntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
    at::detail::check_size_nonnegative(size);
    at::detail::raise_warning_for_complex_half(scalar_type);
    caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
    size_t size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize());
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator->allocate(size_bytes),
        allocator,
        /*resizeable=*/true);

    auto tensor = detail::make_tensor_base<TensorImpl>(
        std::move(storage_impl), ks, dtype);
    // Default TensorImpl has size [0]
    if (size.size() != 1 || size[0] != 0) {
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }

    if (memory_format_opt.has_value()) {
        // Restriding a just-created empty contiguous tensor does nothing.
        if (*memory_format_opt != MemoryFormat::Contiguous) {
        tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
        }
    }
    return tensor;
}
```

在 `empty_generic` 方法中，根据 tensor size 和 dtype 计算所需要的内存大小 `size_bytes`，然后分配内存 `allocator->allocate(size_bytes)`，接着逐步创建一系列对象 
```
StorageImpl -> TensorImpl -> TensorBase
```

函数 `at::native::empty_cpu` 返回时，再根据 `TensorBase` 构造一个 `Tensor` 对象并返回。

</details>



## 2.2 从 python 中调用 operator

`empty.memory_format` 这个 operator 用于创建一个新的 Tensor 对象。现使用如下代码创建 Tensor，

**例 1**

```python
a = torch.Tensor(1,2,3,4)
print(a.shape)
```

打印结果为 `torch.Size([1, 2, 3, 4])`，表示创建了一个 Tensor，其 shape 为 `(1,2,3,4)`。

**例 2**

```python
b = torch.Tensor([1,2,3,4])
print(b)
print(b.shape)
```

打印结果为

```
tensor([1., 2., 3., 4.])
torch.Size([4])
```

**例 3**

```python
c = torch.Tensor(torch.Size([1,2,3,4]))
print(c)
```

输出结果与 例1 中一样


### 2.2.1 代码分析

这里 `torch.Tensor` 是一个 python 类，位于 torch/_tensor.py 文件中，其继承了 `torch._C._TensorBase`，`torch.Tensor` 类没有提供构造函数，故我们查看其父类的构造函数。父类位于 torch/csrc/autograd/python_variable.cpp 文件中，

```c++
PyTypeObject THPVariableType = {
    PyVarObject_HEAD_INIT(
        &THPVariableMetaType,
        0) "torch._C._TensorBase", /* tp_name */
    ...
    THPVariable_pynew, /* tp_new */
}
```

构造函数定义如下，

```c++
PyObject* THPVariable_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      type != &THPVariableType,
      "Cannot directly construct _TensorBase; subclass it and then construct that");
  jit::tracer::warn("torch.Tensor", jit::tracer::WARN_CONSTRUCTOR);
  auto tensor = torch::utils::base_tensor_ctor(args, kwargs);   // <----- 创建 at::Tensor 对象
  // WARNING: tensor is NOT guaranteed to be a fresh tensor; e.g., if it was
  // given a raw pointer that will refcount bump
  return THPVariable_NewWithVar(
      type,
      std::move(tensor),
      c10::impl::PyInterpreterStatus::MAYBE_UNINITIALIZED);
  END_HANDLE_TH_ERRORS
}
```

其中 `base_tensor_ctor` 用于创建 at::Tensor 对象，这是一个 c++ 的 Tensor 类，方法内部调用如下函数，

```c++
Tensor legacy_tensor_generic_ctor_new(
    c10::DispatchKey dispatch_key,  // 实参使用默认值，即 Dispatch::CPU
    at::ScalarType scalar_type,     // 实参使用默认值，即  TypeMeta::Make<float>()
    PyObject* args,
    PyObject* kwargs,
    CtorOrNew ctor_or_new) {
    auto options = dispatchKeyToTensorOptions(dispatch_key);
    static PythonArgParser parser({
        "new(*, Device? device=None)",
        "new(Storage storage)",
        "new(*, int64_t cdata)|hidden",
        // This constructor is no longer legacy, it will also be usable for
        // subclass initialization
        "new(Tensor other)",
        "new(Tensor other, *, Device? device=None)|hidden", // prevent Tensor
                                                            // matching with
                                                            // IntArrayRef,
                                                            // PyObject*
        "new(SymIntArrayRef size, *, Device? device=None)",
        "new(PyObject* data, *, Device? device=None)",
    }); // ? means allow_none, = means set default value

    ParsedArgs<2> parsed_args;
    auto r = parser.parse(args, kwargs, parsed_args);
    if (r.idx == 0) {
    ...
    } else if (r.idx == 5) {
        PyObject* arg = r.pyobject(0);
        auto deviceOptional = r.deviceOptional(1);
        check_legacy_ctor_device(dispatch_key, deviceOptional);
        if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 &&
            arg == PyTuple_GET_ITEM(args, 0)) {
        // new(sequence) binds to this signature but should be treated differently
        // unless the sequences is a torch.Size
        return legacy_new_from_sequence(
            options, scalar_type, deviceOptional, r.pyobject(0));
        }
        return new_with_sizes(
            options, scalar_type, r.deviceOptional(1), r.symintlist(0));
    } 
    ...
    throw std::runtime_error("new(): invalid arguments");
}
```

从上面的方法定义中，可见创建一个 Tensor 对象，有 6 个 schema，我们的 例1 和 例2 均匹配到 `"new(SymIntArrayRef size, *, Device? device=None)"`，也就是代码中的 `r.idx==5` 成立，

1. 例1 `torch.Tensor(1,2,3,4)`，函数输入参数 `args` 为 `(1,2,3,4)`， 解析后的 `arg` 为 `(1,2,3,4)`，于是 `args[0]!=arg`

    执行 `new_with_sizes` 函数，创建一个指定 size 的 Tensor

2. 例2 `torch.Tensor([1,2,3,4])`，函数输入参数 `args` 为 `([1,2,3,4],)`，解析后的 `arg` 为 `[1,2,3,4]`，于是 `args[0]==arg`

    执行 `legacy_new_from_sequence` 函数，创建一个指定数值序列的 Tensor

3. 例3 `c = torch.Tensor(torch.Size([1,2,3,4]))`，函数输入参数 `args` 为 `(torch.Size([1,2,3,4]),)`，解析后的 `arg` 为 `torch.Size([1,2,3,4])`，于是 `args[0]==arg`，但是 `THPSize_Check(arg)=true`

    执行 `new_with_sizes` 函数，创建一个指定 size 的 Tensor


**# new_with_sizes 方法**

```c++
Tensor new_with_sizes(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    const optional<Device>& device,
    c10::SymIntArrayRef sizes) {
  maybe_initialize_cuda(options.device());  // 初始化 cuda（如果device 类型为 cuda）
  pybind11::gil_scoped_release no_gil;      // 释放 GIL
  return at::empty_symint(sizes, build_options(options, scalar_type, device));
}
```

这个函数定义在
torch/include/ATen/ops/empty.h 中，其内部调用 `empty_memory_format::call`，这里 `empty_memory_format` 是一个 struct，定义在 torch/include/ATen/ops/empty_ops.h 中，而类方法 empty_memory_format::call 这个方法定义于 build/aten/src/ATen/Operators_2.cpp 和 build/aten/src/ATen/OperatorsEverything.cpp 两个文件中（源文件使用 torchgen/gen.py 生成，相关函数为 `class ComputeOperators.__call__`），但是 Everything 结尾的源码文件不参与编译，详见 torchgen.utils.py 中 write_sharded() 的代码 

```python
self.filenames.discard(
            f"{self.install_dir}/{base_filename}Everything{extension}"
        )// 丢弃 Everything 结尾的源码文件
```

以及 build/aten/src/ATen/generated_sources.cmake 文件。实际上，OperatorsEverything.cpp文件是几个 Operators_x.cpp 文件的合并。


**# 注册 Impl for BackendSelect**

这里我们突然插入一节内容，介绍注册 DispatchKey::BackendSelect 对应的 kernel 实现，这显得有点突兀，但是因为刚刚介绍了位于 empty_ops.h 文件中的 empty_memory_format 类，这个类除了 `call` 方法，还有一个 `redispatch` 方法，这两个方法从参数上看，后者多了一个 `DispatchKeySet` 类似参数。

通常情况下，Dispatcher 根据 operator 中的 Tensor（或 Generator）类型参数就可以确定 DispatchKey，从而 dispatch 到对应的 kernel 上，然而我们选的 `empty` 函数有些特别，因为 empty 函数参数不包含 Tensor（以及 Generator），这使得 Dispatcher 无法从 Tensor 类型参数中获取有效的 DispatchKey 信息，此时使用 DispatchKey::BackendSelect，所以像 `empty` 这样的 operator，需要注册对应 BackendSelect 的 kernel。这就是我给出这一小节内容的原因。

在文件 RegisterBackendSelect.cpp 中，

```c++
TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
    ...
    m.impl("aten::empty.memory_format", TORCH_FN(empty_memory_format));
    ...
}
```

kernel 所用的 c++ 原生函数定义为

```c++
C10_ALWAYS_INLINE
at::Tensor empty_memory_format(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(dtype, layout, device)); // 计算 DispatchKey
  return at::_ops::empty_memory_format::redispatch(
      _dk, size, dtype, layout, device, pin_memory, memory_format);
}
```

以上代码中，计算 DispatchKey 时，由于 dtype 为 float，layout 为 strided，device 为 cpu，所以得到 DispatchKey::CPU 。 `redispatch` 方法也是位于类 empty_memory_format 中，

```c++
at::Tensor empty_memory_format::redispatch(c10::DispatchKeySet dispatchKeySet, c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    
    static auto op = create_empty_memory_format_typed_handle();
    return op.redispatch(dispatchKeySet, size, dtype, layout, device, pin_memory, memory_format);
}
```

可见，redispatch 与 call 方法一致，都是先获取 TypedOperatorHandle 对象，然后调用 handle 对象的同名方法，然后调用 Dispatcher（全局唯一）的同名方法，

```c++
template<class Return, class... Args>
inline Return Dispatcher::redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKeySet currentDispatchKeySet, Args... args) const {
    detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
    ...
    const KernelFunction& kernel = op.operatorDef_->op.lookup(currentDispatchKeySet);
    return kernel.template call<Return, Args...>(op, currentDispatchKeySet, std::forward<Args>(args)...);
}
```

在 Dispatcher::call 方法中，根据 Tensor 类型参数获取 DispatchKey，而调用 Dispatcher::redispatch 之前，已经事先根据 dtype, layout 和 device 计算出 DispatchKey 。



下面我们看 `empty_memory_format::call` 的方法定义，这部分是关键，涉及到方法 dispatching，

```c++
at::Tensor empty_memory_format::call(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    
    static auto op = create_empty_memory_format_typed_handle();
    return op.call(size, dtype, layout, device, pin_memory, memory_format);
}

static C10_NOINLINE c10::TypedOperatorHandle<empty_memory_format::schema> create_empty_memory_format_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(empty_memory_format::name, empty_memory_format::overload_name)
      .typed<empty_memory_format::schema>();
}
```

`call` 方法第一步是创建 handle 对象，第二部是调用这个 handle 对象创建一个指定 size 的 empty Tensor。

**# 创建 handle 对象**

根据 name 和 overload_name （这俩组成 OperatorName）到 Dispatcher（全局唯一）中查找相应的 OperatorHandle，前面我们特意列出了 `empty.memory_format` 的注册代码语句调用，所以显然 Dispatcher 中存在这样的 OperatorHandle，然后将找到的这个 OperatorHandle 封装为带类型的 handle，即 `TypedOperatorHandle<empty_memory_format::schema>` 类对象，这里模板参数是一个函数类型，位于 `empty_memory_format` 这个类中，

```c++
using schema = at::Tensor (at::IntArrayRef, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>);
```

有了这个模板参数，`TypedOperatorHandle<empty_memory_format::schema>` 可以很容易就知道 operator 的类型（参数类型 Args... ，返回类型 Return），`call` 函数定义如下，

```c++
// TypedOperatorHandle<Return (Args...)> 类方法
C10_ALWAYS_INLINE Return call(Args... args) const {
return c10::Dispatcher::singleton().call<Return, Args...>(*this, std::forward<Args>(args)...);
}

// Dispatcher 类方法
template<class Return, class... Args>
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
    detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
    auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
        .template getDispatchKeySetUnboxed<Args...>(args...);       // 根据参数获取 DispatchKeySet
    ...
    const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);  // OperatorEntry 中寻找对应 DispatchKey 的 kernel
    ...
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
}
```

Dispatcher::call 做了以下三件事：

1. 根据参数获取 DispatchKeySet。

    由于只从 Tensor 和 Generator 相关的类型参数种获取 DispatchKey 信息，我们这里的 例123 的参数均不涉及这两种类型，从参数中没有能获取到 DispatchKey 信息，但是使用 computeDispatchKeySet 方法之后得到默认的 DispatchKeySet({DispatchKey::BackendSelect, DispatchKey::ADInplaceOrView})

2. 从 OperatorEntry（来源路径 OperatorHandle -> OperatorDef -> OperatorEntry)中获取关联的 kernel（for DispatchKey::BackendSelect）

3. 调用这个 kernel（for DispatchKey::BackendSelect）
    
    这个 kernel 执行时，先根据 dtype，layout 和 device 计算好 DispatchKey，本例中为 DispatchKey::CPU，然后 redispatch 到 DispatchKey::CPU 的 kernel（for DispatchKey::CPU），最后调用这个 kernel（for DispatchKey::CPU）

整个创建一个指定 size 的 empty Tensor 过程已经介绍完毕，不过 Tensor 是一个 c++ 类对象，还需要使用  `THPVariable_NewWithVar` 方法将其封装为一个 PyObject 对象，这里是 `torch._C._TensorBase` 对象，这就完成了 torch.Tensor 的基类的构造，从而完成 torch.Tensor 的构造。
