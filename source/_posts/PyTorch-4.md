---
title: PyTorch-4
date: 2019-08-22 14:34:33
tags: PyTorch
categories: DL Framework
---
## Tensor
torch 模块中包含了各种 Tensor：FloatTensor，DoubleTensor，HalfTensor，ByteTensor 等。这些 Tensor 是怎么来的呢？首先，入口是 `torch/__init__.py` 中的 `_C._initExtension(manager_path())`，其中 manager_path 用于获取 torch_shm_manager 的文件路径，shm 实现Domain Socket通信获得共享内存的句柄，解决多进程的内存分配问题，这里跳过。_initExtension 在 torch/csrc/Module.cpp 中初始化 _C 模块时注册到 _C 模块中，其底层 c++ 实现函数为 THPModule_initExtension，这个函数定义中初始化了很多东西，我们依次来看看。
### initializeLayouts
初始化内存布局。当前有三种布局（位于文件 c10/core/Layout.h 中）：
1. Strided，使用密集多维数组的内存布局
2. Sparse，使用稀疏多维数组的内存布局
3. Mkldnn，使用 Intel 的 Mkldnn 库加速 CPU 时，由于 Mkldnn 使用了内部特殊内存布局，所以增加对应的内存布局枚举

以最常用的 Strided 布局为例，使用 THPLayout_New 生成 THPLayoutType/THPLayout 的类型对象，指定 layout 为 Strided，name 为 "torch.strided"，然后 __将这个类型添加到 torch 模块中__，其他两种内存布局方式也类似处理。最后注册这些布局类型：
- CPU, CUDA, MSNPU, XLA, QuantizedCPU -> strided_layout
- SparseCPU, SparseCUDA -> sparse_coo_layout
- MkldnnCPU -> mkldnn_layout
  
即，将 Backend 与 Layout 关联起来，以便将来根据 Backend 获取对应的 Layout。

### initializeMemoryFormats
初始化内存格式。内存格式表明 Tensor 中的数据是如何组织的。当前有三种：Preserve, Contiguous 和 ChannelsLast。例如 ChannelsLast 表示内存中数据的格式为 NHWC，假设正常顺序 NCHW 的各维度值为 sizes，那么 ChannelsLast 下的各维度步幅 strides 应为：
```c++
strides[1]=1;           // ChannelsLast 中 C 为最低一级维度，故步幅为 1
strides[3]=sizes[1];    // ChannelsLast 中 W 为次低一级维度，故步幅为 C 维度即 sizes[1]
strides[2]=strides[3]*sizes[3]; // ChannelsLast 中 H 为再次低一级维度，步幅为 W*C
strides[0]=strides[2]*sizes[2]; // ChannelsLast 中 N 为最高一级维度，步幅为 H*W*C
```
注意，上面 strides 和 sizes 的顺序均为 NCHW。
创建三种内存格式类型对象 preserve_format, contiguous_format, channels_last 并 __添加到 torch 模块中__。

### initializeQScheme
初始化量化机制，量化是将连续型的输入限制为离散型，比如将浮点计算转为整型计算，显然使用小型整型比浮点型计算更高效，且内存占用更小，这在模型 inference 阶段尤其重要。关于量化的具体概念以及相关操作可参考 [Introducing-Quantized-Tensor](https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor)。当前量化机制有 5 种，类似以上布局和内存格式，分别创建对应的类型对象，然后 __添加到 torch 模块中__。

### initializeDtypes
初始化数据类型。直接看此函数的部分定义
```c++
#define DEFINE_SCALAR_TYPE(_1, n) at::ScalarType::n,
at:ScalarType all_scalar_type[] = {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};
```
其中 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS 这个宏罗列了所有的标量类型，包括 complex 类型和 quantization 类型，展开后为
```
at::ScalarType::Byte,
at::ScalarType::Char,
at::ScalarType::Short,
at::ScalarType::Int,
at::ScalarType::Long,
at::ScalarType::Half,
at::ScalarType::Float,
at::ScalarType::Double,
at::ScalarType::ComplexHalf,
at::ScalarType::ComplexFloat,
at::ScalarType::ComplexDouble,
at::ScalarType::Bool,
at::ScalarType::QInt8,
at::ScalarType::QUInt8,
at::ScalarType::QInt32
at::ScalarType::BFloat16
```
然后根据以上标量类型获取其主名称和传统旧名称，如没有传统旧名称，则默认为空字符串 `""`，然后创建各个对应的类型对象，并注册这些类型对象（即，将类型对象与 at::ScalarType 值关联起来，存储到字典，以便将来能根据 at::ScalarType 获取对应的类型对象），相关代码如下
```c++
std::tie(primary_name, legacy_name) = getDtypeName(scalarType);
PyObject *dtype = THPDtype_New(scalarType, primary_name);
torch::registerDtypeObject((THPDtype*)dtype, scalarType);
```
然后将类型对象（THPDtypeType/THPDtype 实例）__添加到 torch 模块中__，如果存在传统旧名称，则也同样添加到 torch 模块中。

### initialize_python_bindings
初始化 python 绑定
#### initialize_aten_types
根据 all_declared_types 函数获取所有声明过的类型，Backend 有 `CPU, CUDA, SparseCPU, SparseCUDA` 四种，ScalarType 除去 Complex 和 Quantization 类型，则一共有
```
Byte, Char, Double, Float, Int, Long, Short, Half, Bool, BFloat16
```
然后所有 Backend 与 ScalarType 不同的组合构成这里所需要的声明类型，不过 (SparseCUDA|SparseCPU,Bool) 除外，这样的组合一共有 4*10-2=38 种，根据这 38 种组合构建对应的 PyTensorType 类型，看看这个 PyTensorType 类型定义，
```c++
struct PyTensorType {
    PyTypeObject py_type;   // python 类型对应的扩展类型，这个字段后文会再次讲到
    THPDtype* dtype;        // 对应上文 initializeDtypes 中注册的某个数据标量类型
    THPLayout* layout;      // 对应上文 initializeLayouts 中注册的某个内存布局类型
    bool is_cuda;           // 指示是 cuda 还是 cpu
    char name[64];          // tensor 类型名称
    int backend;            // CPU, CUDA, SparseCPU, SparseCUDA 四种之一
    int scalar_type;        // Byte 等十种之一
};
```
上述 38 种组合，每种组合构建一个 PyTensorType 对象。Python 接口中某种 Tensor，比如 FloatTensor 其底层就对应这里的某个 PyTensorType 对象。
- layout，根据 backend 字段获取，initializeLayouts 中注册了所有 Backend 与 Layout 的映射关系
- is_cuda，当 backend = CUDA|SparseCUD 时为 true
- name，名称构成为 `[模块名].[ScalarType名]Tensor`。  
  模块名：
  ```
  CPU -> torch
  CUDA -> torch.cuda
  SparseCPU -> torch.sparse
  SparseCUDA -> torch.cuda.sparse
  ```
  ScalarType名 就是 ScalarType 字面量的字符串形式，如 Byte -> "Byte", Float -> "Float" 等。例如组合 (CPU, Float) 对应的 PyTensorType 对象名为 "torch.FloatTensor"，(SparseCUDA, Double) 对应的 PyTensorType 对象名为 "torch.cuda.sparse.DoubleTensor"。

注意到，
```c++
if (backend==Backend::CPU && scalar_type==at::kFloat) {
    set_default_tensor_type(&tensor_type);
}
```
可见默认 Tensor 类型为 torch.FloatTensor（其 Backend 为 CPU，注意对应 CUDA 的为 torch.cuda.FloatTensor）。

总结：initialize_aten_types 根据 38 种组合构建 PyTensorType 类型对象，并保存到 tensor_types 这个 vector 中。

#### py_initialize_metaclass(metaclass)
初始化元类，这是一个 python 的扩展类型 PyTypeObject，对应的 python 类型名为 "torch.tensortype"，顾名思义表示 tensor 类型类，即，tensor 各种类型如 torch.FloatTensor 等的元类型，这个元类具有的属性为
- dtype  
    对应 initializeDtypes 中某个 THPDtype 对象
- layout  
    对应 initializeLayouts 中某个内存布局类型 THPLayout 对象
- is_cuda  
    是否使用 cuda
- is_sparse  
    是否是稀疏存储

具有方法
- `__instancecheck__` 检测某个 Tensor 是否与当前 tensor 类型类匹配，当 type_id 和 scalar_type 这两个字段均分别相同时，则匹配，否则不匹配。

PyTensorType 是表示 python 的 Tensor 类型，我们指定 python 的类型本身也是一种对象，这种类型对象的类型为元类型，也就是这里的 metaclass。

总结：初始化 PyTypeObject 类型对象 metaclass，它表示 tensor 类型的元类，且具有上述属性和方法。

#### get_tensor_dict
获取 torch.Tensor 以及其基类 _C._TensorBase 的初始属性（名称与值构成的字典）

#### py_initialize_tensor_type
对于前面构造的 38 个 PyTensorType 对象，设置每个对象的 py_type 字段。py_type 类型为 PyTypeObject，表示一个类型对象，也就是 python 中的某个类型，为这个类型对象设置元类 metaclass，名称，以及将上一小节中的属性字典并入这个类型，从而使得类型具有 torch.Tensor 的全部初始属性。
```
dir(torch.Tensor)
dir(torch.FloatTensor)
```
以上两个指令输出内容一样。

#### py_bind_tensor_types
至此，以上 38 种 PyTensorType 对象均已准备好，将他们添加进相应的模块中，前文可能说 "添加进 torch 模块中"，因为当时没有讨论到 PyTensorType 对象名，所以笼统的那么说了一下，实际上应为 "添加进相应的模块中"，比如 "torch.FloatTensor"，则将相应的 PyTensorType 对象以 FloatTensor 作为 python 端的名称添加进 torch 模块中，"torch.cuda.sparse.DoubleTensor" 则将相应的 PyTensorType 对象以 DoubleTensor 作为 python 端的名称添加进 torch.cuda.sparse 模块中，即最后一个 `.` 后面的部分表示类型，而之前的部分表示模块。

但是，现在还存在一个问题，那就是这些 torch.FloatTensor, torch.IntTensor 等类型与 torch.Tensor 是什么关系？
```python
a=torch.empty(1,2,dtype=torch.int)
isinstance(a, torch.IntTensor)  # True
isinstance(a, torch.Tensor)     # True
issubclass(torch.IntTensor, torch.Tensor)   # False
issubclass(torch.Tensor, torch.IntTensor)   # False
```
根据 [Pytorch-3](2019/06/18/Pytorch-3) 最后的分析，我们知道 torch.empty 函数最后使用 THPVariable_Wrap 将 c++ Variable 类型包装成 python 的 torch.Tensor 类型，甚至直接调用 torch.IntTensor 构造的对象最后也是经过 THPVariable_Wrap 包装成 torch.Tensor 类型，
```
>>> type(torch.IntTensor([1,2]))
<class 'torch.Tensor'>
```
既然返回的都是 torch.Tensor 类型，那怎么跟 torch.IntTensor 联系起来的呢？其实，torch.IntTensor 等 38 个 Tensor 类型与 torch.Tensor 没有直接关系
```
>>> torch.IntTensor.__bases__
(<class 'object'>)
>>> torch.Tensor.__bases__
(<class 'torch._C._TensorBases'>)
```
上面所说的各种构造 Tensor 的方法返回的类型也确实是 torch.Tensor，但是 `isinstance(a, torch.IntTensor)` 结果为 True 也没错，因为 `isinstance` 实际上内部调用 `__instancecheck__` 进行判断，前面讨论 metaclass 时讲到这个方法的 c++ 底层实现函数为 Tensor_instancecheck，其定义如下
```c++
static PyObject *Tensor_instancecheck(PyTensorType *self, PyObject * arg) {
    try{
        if(THPVariable_Check(arg)) {    // 检测参数是否是 THPVariable 类型
            auto& var = ((THPVariable*)arg)->cdata; // 获取内部的 Variable 类型对象
            if (var.type_id() == self->get_type_id() &&
                var.scalar_type() == static_cast<ScalarType>(self->scalar_type)) {
                Py_RETURN_TRUE;     // 如果 type_id 和 ScalarType 均分别相同，则返回 True
            }
        }
        Py_RETURN_FALSE;
    } catch(python_error & e){
        return nullptr;
    }
}
```
所以，不难理解 `isinstance(a, torch.IntTensor)=True`。

__总结：在 [PyTorch-2]() 中，我们讨论了 PyTorch 中的函数返回或直接构造的 Tensor 均为 torch.Tensor，其继承自 `torch._C.Tensor`，所以要理解 Tensor 类的各种方法，需要从 `torch._C.Tensor` 的类型构造开始着手。__

接下来是一系列的 THPxxxStorage_postInit 函数执行，这在 [PyTorch-2](2019/06/13/PyTorch-2) 中已经进行了介绍（THPxxxStorage_init），这里仅仅给出结论：
1. 函数声明
    ```c++
    #define THPStorage_(NAME) TH_CONCAT_4(THP,Real,Storage_,NAME) //torch/csrc/Storage.h
    bool THPStorage_(postInit)(PyObject *module);   // torch/csrc/generic/Storage.h
    ```
    THPStorage_(NAME) 这个宏展开后就得到 THPxxxStorage_init， 其中 Real 在宏展开时被替换为具体的 ScalarType，NAME 被替换为 init。于是，最终得到的函数声明为 THPxxxStorage_init(PyObject *module);

2. torch/csrc/generic/Storage.cpp 中
   ```c++
   PyObject *THPStorageClass = nullptr;
   bool THPStorage_(postInit)(PyObject *module){
       // 从 torch 模块中获取名为 RealStorage 的属性，其中 Real 可为 Float, Bool, Double 等 ScalarType
       THPStorageClass = PyObject_GetAttrString(module, (char*)TH_CONCAT_STRING_2(Real, Storage));
       at::Backend backend = at::Backend::CPU;
       #ifdef THC_GENERIC_FILE
       backend = at::Backend::CUDA;
       #endif
       #ifdef THQUANTIZED
       backend = at::Backend::QuantizedCPU;
       #endif
       torch::registerStoragePyTypeObject((PyTypeObject*)THPStorageClass, backend, TH_CONCAT_2(at::k, Real));
   }
   ```
   注意到在 `torch/__init__.py` 中定义了 FloatStorage 等类型，
   ```python
   class FloatStorage(_C.FloatStorageBase, _StorageBase):
       pass
   ```
   torch._C.FloatStorageBase 等类型是在 THPxxxStorage_init 函数中被添加到模块 torch._C 中。THPxxxStorage_postInit 函数先是从 torch 模块中获取 RealStorage 类型对象，然后进行注册即，将 RealStorage 类型对象与对应的组合 (Backend, ScalarType) 进行映射，这样以后就可以根据 (Backend, ScalarType) 获取对应的 RealStorage 类型对象，反过来亦可。

