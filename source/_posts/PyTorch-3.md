---
title: PyTorch-3
date: 2019-06-18 16:44:44
tags: PyTorch
categories: DL Framework
---
在 [PyTorch-2](PyTorch-2) 我们已经了解了 torch 包的初始化过程，接下来便可以愉快查看这个 package 包含哪些字段（包含函数和类）了，再参照 PyTorch 的[官方文档](https://pytorch.org/docs/stable/torch.html)，了解其中各个函数的具体实现。
# torch 包
从 `torch/__init__.py` 中可以查看所有的 torch 包的所有字段，包括：
1. 直接在此文件中定义的函数/字段，如 typename, is_tensor, is_storage, _storage_classes 等
2. 从 torch 包的模块中导入的函数/类，如
   ```
   from .random import set_rng_state, get_rng_state, manual_seed, initial_seed
   ...
   ```
3. 从 torch._C 中导入的字段/函数/类
4. 从 torch._C._VariableFunctions 导入的字段/函数
   
PyTorch 官方文档中 torch 包有很多函数。这里举几个例子进行说明。
## torch.empty
这个函数实际上来自于 torch._C._VariableFunctions 这个类。文件 torch/csrc/Module.cpp 中调用函数 THPVariable_initModule，跳转到 torch/csrc/autograd/python_variable.cpp 查看函数定义，其定义体中调用 torch::autograd::initTorchFunctions，而这个函数定义位于 torch/csrc/autograd/generated/python_torch_functions.cpp，这个文件是安装 PyTorch 过程中生成的，按以下步骤查看这个文件的生成过程：
1. caffe2/CMakeLists.txt 中的文件生成语句为
   ```
   set(GENERATED_CXX_PYTHON
     ...
     "${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions.cpp"
     ...)
   ...
   add_custom_command(
       OUTPUT
       ${TORCH_GENERATED_CODE}
       COMMAND
       "${PYTHON_EXECUTABLE}" tools/setup_helpers/generate_code.py
        ...
       DEPENDS
       ...)
   ```
2. 执行 tools/setup_helpers/generate_code.py。在函数 generate_code 中调用了以下四个函数生成文件，
   ```
   generate_nn_wrappers
   gen_autograd_python
   gen_autograd
   gen_jit_dispatch
   ```
这四个函数的实现都是非常繁琐的，这里以生成 torch/csrc/autograd/generated/python_torch_functions.cpp 为例，实际上是将模板文件 tools/autograd/templates/python_torch_functions.cpp 中的 ${py_methods} 和 ${py_method_defs} 分别替换为对应的方法实现和方法签名，这些方法来自于 torch/share/ATen/Declarations.yaml, tools/autograd/deprecated.yaml, tools/autograd/derivatives.yaml，其中第一个文件又需要动态生成，过程为：
1. 在 caffe2/CMakeLists.txt 中有语句
   ```
   include(../cmake/Codegen.cmake)
   ```
2. 在文件 cmake/Codegen.cmake 中调用 `gen.py`
   ```
   SET(GEN_COMMAND
       "${PYTHON_EXECUTABLE}" ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/gen.py
       --source-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen
       --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
       ${GEN_ROCM_FLAG}
       ${cwrap_files})
   ```
   （在 aten/src/ATen/native/native_functions.yaml 找到 `empty` 的函数签名）
3. aten/src/ATen/gen.py 中的 generate_outputs 函数生成 Declarations.yaml 文件
   ```
   file_manager.write("Declarations.yaml", format_yaml(output_declarations))
   ```
4. 根据第 2 点，install_dir 为 build/aten/src/ATen，所以 Declarations.yaml 生成路径此时为 build/aten/src/ATen，根据以下步骤安装此文件
   - CMakeLists.txt 中的 add_subdirectory(caffe2)
   - caffe2/CMakeLists.txt 中的 add_subdirectory(../aten aten)
   - aten/CMakeLists.txt 中的 add_subdirectory(src/ATen)
   - aten/src/ATen/CMakeLists.txt 中有，
     ```
     INSTALL(FILES ${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml
       DESTINATION ${AT_INSTALL_SHARE_DIR}/ATen)
     ```
   事实上，除了这里的 Declarations.yaml，在 aten/src/ATen/CMakeLists.txt 中还安装了很多头文件，其中就包括下文将提到的 build/aten/src/ATen/Functions.h，具体参见 aten/src/ATen/CMakeLists.txt 中其他 INSTALL 指令调用。
   
找到这些函数来源后，通过 tools/autograd/gen_python_functions.py 中的函数 create_python_bindings 生成 ${py_methods} 和 ${py_method_defs} 的内容，
```
PY_VARIABLE_METHOD_VARARGS = CodeTemplate("""\
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
    HANDLE_TH_ERRORS
    static PythonArgsParser parser({
        ${signatures}
    }, /*traceable=*/${traceable});
    ${unpack_self}
    ParserArgs<${max_args}> parsed_args;
    auto r = parser.parse(args, kwargs, parsed_args);
    ${declare_namedtuple_return_types}
    ${dispatch}
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}
""")
...
def create_python_bindings(python_functions, has_self, is_module=False):
    def process_function(name, declarations):
        ...
        env = {
            'name': name,
            'dispatch_name': 'dispatch_{}'.format(name),
            'pycname': 'THPVariable_{}'.format(name),
            'signature': [],
            'max_args': max(len(o['arguments'])+len(o['python_binding_arguments']) for o in declarations),
            'unpack_self': [],
            'dispatch': [],
            'declare_namedtuple_return_types': '',
        }
        ... // 向 env 增加 key-value pair or 更新 env 中已有 key 的 value
        if len(declarations) == 1 and len(declarations[0]['args']) == 1 and has_self:
            ...
        else:
            tmpl = PY_VARIABLE_METHOD_VARARGS
            env['flags'] = 'METH_VARARGS | METH_KEYWORDS'
        if not is_module and not has_self:
            env['flags'] += ' | METH_STATIC'
        
        py_methods.append(tmpl.substitute(env))
        py_methods_defs.append(PY_VARIABLE_METHOD_DEF.substitute(env))
```
通过以上代码片段可知，对于函数定义的生成，使用一个函数定义模板 PY_VARIABLE_METHOD_VARARGS，然后对每个函数，来自于 Declarations.yaml, deprecated.yaml, derivatives.yaml，抽取有关字段的值存储到 env 字典中，然后将 PY_VARIABLE_METHOD_VARARGS 中的占位符使用 env 中相应 key 的值替换，就得到这个函数的定义。

## empty 定义
我们看生成后的 empty 函数定义（位于文件 torch/csrc/autograd/generated/python_torch_function.cpp）
```
static PyObject * THPVariable_empty(PyObject* self_, PyObject* args, PyObject* kwargs)
{
    HANDLE_TH_ERRORS
    static PythonArgParser parser({
        "empty(IntList size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    }, /*tracebalbe*/true); // 大括号初始化器，得到函数签名的vector
    ParseArgs<6> parsed_args;
    auto r = parser.parse(args, kwargs, parseed_args);
    if (r.idx == 0) {       // 函数签名在vector中的下标
        if (r.isNone(1)) {  // parameter 'out' is None
            auto size = r.intlist(0);
            auto dtype = r.scalartype(2);
            auto device = r.device(4);
            const auto options = TensorOptions()
                .dtype(dtype)
                .device(device)
                .layout(r.layout(3).layout)
                .requires_grad(r.toBool(5));
            return wrap(dispatch_empty(size, options));
        } else {
            check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2),
                                   r.layout(3), r.isNone(3),
                                   r.device(4), r.isNone(4));
            return wrap(dispatch_empty(r.intlist(0), r.tensor(1)).set_requires_grad(r.toBool(5)));
        }
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}
```
从以上代码中可见，要创建一个 empty 的 Tensor，首先检查调用者是否提供了一个 Tensor，如未提供，则先创建一个 Tensor：
1. `out` 参数为None，则需要根据参数 dtype, device, layout 和 requires_grad 创建 Tensor
2. `out` 参数不为None, 则检查 `out` 这个 Tensor 与参数 dtype, layout, device 是否匹配，如果匹配，还需要将 `out` 的 requires_grad 属性重置为参数 requires_grad

然后调用函数 dispatch_empty，这个函数总共有两个重载版本，位于 torch/csrc/autograd/generated/python_torch_functions_dispatch.h，这个文件与同目录下的 python_torch_function.cpp 一样也是动态生成的，生成逻辑也是一样的，将 tools/autograd/templates/python_torch_functions_dispatch.h 中的占位符替换掉，不再具体展开，可参见 tools/autograd/gen_python_functions.py 中的函数 gen_py_torch_functions。dispatch_empty 的两个重载版本为，
```
// empty 函数调用者提供了 Tensor 'out'
inline Tensor dispatch_empty(IntList size, Tensor result) {
    AutoNoGIL no_gil;
    return at::empty_out(result, size);
}
// empty 函数调用者未提供 Tensor 'out'，需要根据参数 options 创建
inline Tensor dispatch_empty(IntList size, const TensorOptions & options) {
    maybe_initialize_cuda(options);
    AutoNoGIL no_gil;
    return torch::empty(size, options);
}
```
### 有输出 Tensor
我们看第一个重置版本的定义体，即，调用者提供了输出 Tensor，首先构造一个结构实例 AutoNoGIL，这个结构的构造函数为
```
AutoNoGIL() : save(PyEval_SaveThread()) {}
```
可以看出，先释放 GIL，因为下一句执行的 at::empty_out 可能会慢很多，为了防止程序使用多线程，但仍然被阻塞在这里，所以释放 GIL，待 at::empty_out 执行完毕，再重新获取 GIL，
```
~AutoNoGIL() {
    PyEval_RestoreThread(save);
}
```
然后 at::empty_out 函数位于 torch/lib/include/Aten/Functions.h，
```
static inline Tensor & empty_out(Tensor & result, IntList size) {
    return detail::infer_type(result).empty_out(result, size);
}
```
在分析 at::empty_out 函数之前，我们需要知道这里的 Functions.h 也是动态生成的，在项目源码中稍作查询便知，在 aten/src/ATen/gen.py 中的 generate_outputs 函数中使用如下语句生成（与前面的 Declarations.yaml 文件的生成在同一处地方），
```
file_manager.write('Functions.h', FUNCTIONS_H, top_env)
```
现在回到 at::empty_out 函数定义上来，首先 detail::infer_type(result) 根据调用用传入的 Tensor 实例 result 得到 TypeExtendedInference 类型实例，然后调用实例函数 empty_out。这里相关的结构、类为 TypeExtendedInferface，TypeDefault，位于文件 torch/lib/include/ATen/TypeExtendedInferface.h， torch/lib/include/ATen/TypeDefault.h，此外，TypeDefault类方法实现源文件为 build/aten/src/ATen/TypeDefault.cpp，接口方法 empty_out 的实现正是位于此文件中，
```
Tensor & TypeDefault::empty_out(Tensor & result, IntList size) const {
    return at::native::empty_out(/* native_actuals */ result, size);
}
```
首先这三个文件是动态生成的（与 Declarations.yaml 相同）。然后我们看方法定义体中，直接调用另一个同名函数 at::native::empty_out 下，函数声明位于文件 torch/lib/include/ATen/NativeFunctions.h，此文件动态生成（与 Declarations.yaml 相同），函数实现位于 aten/src/ATen/native/TensorFactories.cpp，这个文件不是动态生成的（终于来了一个非动态生成的了），在此文件中查看函数定义，
```
namespace at {
namespace native {
...
Tensor& empty_out(Tensor& result, IntList size) {
    if (result.is_sparse()) {
        result.sparse_resize_and_clear_(size, size.size(), 0);
    } else {
        result.resize_(size);
    }
    return result;
}
...
}
}
```
显然，根据输出 Tensor 是否是稀疏的进行不同的处理。
1. 输出 Tensor 是稀疏的
   
   对输出 Tensor 调用方法 sparse_resize_and_clear_，声明位于 torch/lib/include/ATen/core/Tensor.h，此文件动态生成，与 Declarations.yaml 相同，见于 aten/src/ATen/gen.py，但是实际上源码中存在 aten/src/ATen/core/Tensor.h，并且这俩文件完全一样，还有 TensorMethods.h 和 Type.h 均存在这个现象，这里暂时不清楚为啥会这样。sparse_resize_and_clear_ 的函数实现位于 torch/lib/include/ATen/core/TensorMethods.h，
   ```
   inline Tensor & Tensor::sparse_resize_and_clear_(IntList size, int64_t sparse_dim, int64_t dense_dim) {
       return type().sparse_resize_and_clear_(*this, size, sparse_dim, dense_dim);
   }
   ```
   先根据当前 Tensor 获取对应的 Type，然后调用 Type 类型的 sparse_resize_and_clear_ 方法，Type 这个结构是一个接口，其接口函数的具体实现见各个具体 Type 的 .cpp 文件，Type 是由数值类型（如 int,float,double 等）和 Backend（CPU,CUDA,SparseCPU, SparseCUDA 等）组合而成，比如 SparseCPUByteType.h 和 SparseCPUByteType.cpp，此函数的的定义为
   ```
   Tensor & SparseCPUByteType::sparse_resize_and_clear_(Tensor & self, IntList size, int64_t sparse_dim, int64_t dense_dim) const {
       const OptionalDeviceGuard device_guard(device_of(self));
       return at::native::sparse_resize_and_clear_(/* actuals */ self, size, sparse_dim, dense_dim);
   }
   ```
   其中 at::native::sparse_resize_and_clear_ 函数声明位于 torch/lib/include/ATen/NativeFunctions.h，函数实现位于 aten/src/ATen/native/sparse/SparseTensor.cpp，
   ```
   SparseTensor& sparse_resize_and_clear_(SparseTensor& self, ArrayRef<int64_t> size, int64_t sparse_dim, int64_t dense_dim) {
       get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
       return self;
   }
   ```
   根据 Tensor 获取其底层实现 SparseTensorImpl 类对象，然后调用 SparseTensorImpl 的方法 resize_and_clear_。
2. 输出 Tensor 是密集的
   
   Tensor 的 resize_ 方法定义见 TensorMethods.h，为
   ```
   inline Tensor & Tensor::resize_(IntList size) {
       return type().resize_(*this, size);
   }
   ```
   调用这个 Tensor 的类型方法 resize_，以 CPUByteType.cpp 为例，定义如下
   ```
   Tensor & CPUByteType::resize_(Tensor & self, IntList size) const {
       return at::native::resize_cpu_(/* actuals */ self, size);
   }
   ```
   可见，对 Tensor 按给定 size 进行 resize 操作，这个 resize_cpu_ 方法定义为，
   ```
   Tensor& resize_cpu_(Tensor& self, IntList size) {
       auto* self = self.unsafeGetTensorImpl();         // 获取 Tensor 的底层实现类对象
       // 按给定 size 大小对 Tensor 进行 resize，当 size 大小比 Tensor size 大时，才分配一个更大的内存块
       resize_impl_cpu_(self_, size, c10::nullopt);     
       self_->maybe_zero_dim(size.size()==0);
       return self;
   }
   ```
### 无输出 Tensor
直接按给定的 size 参数新建一个 Tensor，具体过程略。

# PS
好吧，主要是因为内容太多了，樯橹灰飞烟灭，先到此为止吧，就当是梳理了一下方法调用过程，等以后熟悉了整个代码框架，再回头重新整理一番。