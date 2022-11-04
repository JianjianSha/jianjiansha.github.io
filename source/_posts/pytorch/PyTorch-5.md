---
title: PyTorch 自动求导
p: pytorch/PyTorch-5
date: 2019-08-27 14:07:27
tags: PyTorch
categories: DL Framework
---

# 1. 自动求导

本篇主要分析 PyTorch 中自动求导是如何进行的。要使得能够自动求导，需要设置 Tensor 的 `requires_grad=True`，例如
<!-- more -->
```python
x=torch.ones(1, requires_grad=True)
```
根据前面 torch.empty 的底层 C++ 实现的分析，易知 torch.ones 的 C++ 底层实现由位于 torch/csrc/autograd/generated/python_torch_functions.cpp 中的 THPVariable_ones 函数实现，此函数定义的部分代码为
```c++
auto size = r.intlist(0);
auto dtype = r.scalartype(2);
auto device = r.device(4);
const auto options = TensorOptions()
    .dtype(dtype)
    .device(device)
    .layout(r.layout(3).layout)
    .requires_grad(r.toBool(5));
return wrap(dispatch_ones(size, options));
```
`wrap` 则是将 C++ 的 Tensor 包装为 python 的 Tensor 类型 `torch.Tensor`。dispatch_ones 函数其内部调用 torch::ones 函数，此函数定义的关键部分为
```c++
at::Tensor tensor = at::ones(size, at::TensorOptions(options).is_variable(false));
auto result = autograd::make_variable(tensor, options.requires_grad());
```
上面代码第一句是构造全 1 的 Tensor，第二句代码则是将 Tensor 转为 Variable。Variable 继承 Tensor，Tensor 内部有 c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> 类型字段 `impl_`，Variable::Impl 类型继承 TensorImpl，并且 Variable 的字段 `impl_` 实际上相当于指向 Variable::Impl 的指针类型，而 Variable::Impl 内部包含了字段 `requires_grad_` 记录了 Variable 是否支持自动求导。

```python
y=torch.ones(1) # y.requires_grad False
z=x+y           # z.requires_grad True
```

torch.Tensor 的基类 torch._C._TensorBase 的方法位于 torch/csrc/autograd/generated/python_variable_methods.cpp 中的 variable_methods，其中我们发现重载运算符 `__add__` 的实现函数为 THPVariable_add，调用栈为
```
THPVariable_add -> dispatch_add -> Tensor::add -> TypeDefault::add -> native::add -> native::add_out
```

# 2. backward()

前面内容是很久之前写的，没写完。今天重拾自动求导，但故事不接上一节内容，而是从 backward 函数出发，抽丝剥茧理清楚反向传播中自动求导过程。

`_tensor.py` 文件中 backward 函数定义，

```python
def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    ...
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
```

1. `gradient`，如果 tensor 不是一个标量，那么求导结果为 Jacobian 矩阵，需要提供一个梯度向量，与 Jacobian 矩阵相乘（链式法则），使得最终梯度依然为向量
2. `retain_graph`，是否保持计算图，默认情况下计算完梯度后，计算图被释放以节省内存。此计算图用于计算梯度。
3. `create_graph`，是否创建梯度图，用于计算高阶梯度。

来自模块 torch.autograd 的函数 backward 内部对参数进行判断以及整理，然后调用 `Variable._execution_engine.run_backward` 继续求导。

这个执行引擎来自 
```python
from torch._C import _ImperativeEngine as ImperativeEngine
```

涉及到 torch._C 的东西都首先去 Module.cpp 的 initModule() 函数中查找，正好发现了疑似相关的一句代码，
```c++
ASSERT_TRUE(THPEngine_initModule(module));
```

幸运地，从 `THPEngine_initModule` 函数内部发现了向 torch._C 模块中添加类型 _ImperativeEngine 的代码，这个类型对应到 C++ 中的类型为 `THPEngineType`，其相关方法为

```c++
static struct PyMethodDef THPEngine_methods[] = {
    {(char*)"run_backward",
     castPyCFunctionWithKeywords(THPEngine_run_backwards),
     METH_VARARGS | METH_KEYWORDS, nullptr},
    ...
    {nullptr}
}
```

继续查看 THPEngine_run_backward 方法定义，

```python
# ... 从参数中提取 对象和变量值，这是 Python 扩展方法的常规操作

# run_backward 的参数中，tensors 和 grad_tensors 是 tuple 类型
# tensors 数量必须要与 grads 数量相等。如果一开始未提供 grad，那么
# grads 是全 None 的 tuple。
Py_ssize_t num_tensors = PyTuple_GET_SIZE(tensors);
Py_ssize_t num_gradients = PyTuple_GET_SIZE(grad_tensors);

edge_list roots;
roots.reserve(num_tensors);
variable_list grads;
grads.reserve(num_tensors);

for (const auto i : c10:irange(num_tensors)) {
    PyObject *_tensor = PyTuple_GET_ITEM(tensors, i);
    # ... 对 _tensor 做一些检查
    # 类型转换
    const auto& variable = THPVariable_Unpack(_tensor);
    # 根据tensor创建计算图的边
    # 计算图中，通常一个操作 op 视作一个节点，一个数据(tensor)视作 edge
    # Forward 时，data tensor 为 edge，backward 时，grad tensor 为 edge
    auto gradient_edge = torch::autograd::impl::gradient_edge(variable);
    roots.push_back(std::move(gradient_edge));

    PyObject *grad = PyTuple_GET_ITEM(grad_tensors, i);     # 获取一个提供的梯度向量
    if (THPVariable_Check(grad)) {
        # 是 Tensor 类型，那么转换为 Tensor(即 Variable)
        const Variable& grad_var = THPVariable_Unpack(grad);
        grads.push_back(grad_var);
    } else {    # 当 tensor 是标量时，不需要额外提供梯度向量，此时 grad 默认为与 tensor
                # 同形的全 1 张量，当 tensor.requires_grad 为 False 时，grad 为 None
        THPUtils_assert(grad == Py_None, "...");
        THPUtils_assert(!variable.requires_grad(), "...");
    }
}

std::vector<Edge> output_edges;
# inputs 是反向传播时，op 节点的输入数据
if (inputs != nullptr) {
    int num_inputs = PyTuple_GET_SIZE(inputs);
    output_edges.reserve(num_inputs);
    for (const auto i : c10::irange(num_inputs)) {
        PyObject *input = PyTuple_GET_ITEM(inputs, i);
        const auto& tensor = THPVariable_Unpack(input);
        const auto output_nr = tensor.output_nr();      # 这个 tensor 是 op 的第几个输出
        auto grad_fn = tensor.grad_fn();
        if (!grad_fn)   {
            grad_fn = torch::autograd::impl::try_get_grad_accumulator(tensor);
        }
        if (accumulate_grad) {  # 指示累加梯度，默认为 true
            tensor.retain_grad();
        }
        if (!grad_fn) {
            output_edges.emplace_back(std::make_shared<Identity>(), 0);
        }
        else {
            output_edges.emplace_back(grad_fn, output_nr);
        }
    }
}
variable_list outputs;
{   # 实际的反向传播梯度计算为 engine.execute() 方法调用
    pybind11::gil_scoped_release no_gil;
    auto& engine = python::PythonEngine::get_python_engine();
    outputs = engine.execute(roots, grads, keep_graph, create_graph, accumulate_grad, output_edges);
}
```

当使用 标量 tensor 进行 backward 时，grad 可不显式提供，此时默认 grad 为与 tensor 同形的全 1 张量。

与 tensors 和 grad_tensors 一样，inputs 也需要转换为 tuple 形式，如 inputs 未提供，那么为 None，转换后为 `tuple()` ，即 empty tuple，如 inputs 为张量，那么转换为 `(inputs,)`，否则转换为 `tuple(inputs)` 。

接着看 `Engine::execute` 函数定义部分，

```c++

```
