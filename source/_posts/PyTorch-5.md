---
title: PyTorch-5
date: 2019-08-27 14:07:27
tags: PyTorch
categories: DL Framework
---
本篇主要分析 PyTorch 中自动求导是如何进行的。要使得能够自动求导，需要设置 Tensor 的 `requires_grad=True`，例如
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


