---
title: Tensor 加法运算中的梯度计算
date: 2023-03-14 14:07:05
tags: pytorch source code
---


# 1. Autograd
在 [Tensor add 方法源码分析](2023/03/16/pytorch/source/add) 一文中，我们说明了 Tensor 加法的计算过程，但是没有涉及到梯度计算，所以这篇文章仍以 Tensor 加法计算为例，看看梯度是如何计算的。

## 1.1 再看分发过程

我们从 Dispatcher 对 operator 根据参数进行分发开始，代码如下（如果不理解的，可以再看看 [Tensor add 方法源码分析](2023/03/16/pytorch/source/add) 这篇文章），

```c++
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

其中 `create_add_Tensor_typed_handle` 是根据 operator 的 name+overload_name 得到 operator 的 handle，然后调用 call 方法，

```c++
template<class Return, class... Args>
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);           // 获取 DispatchKeySet

  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  auto step_callbacks = at::getStepCallbacksUnlessEmpty(at::RecordScope::FUNCTION);
  if (C10_UNLIKELY(step_callbacks.has_value() && op.operatorDef_->op.isObserved())) {
    return callWithDispatchKeySlowPath<Return, Args...>(op, *step_callbacks, dispatchKeySet, kernel, std::forward<Args>(args)...);
  }
#endif  // PYTORCH_DISABLE_PER_OP_PROFILING
  return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
}
```

**# 获取 DispatchKeySet**

DispatchKeyExtractor::getDispatchKeySetUnboxed(args...) 方法中借助 `MultiDispatchKeySet` 这个类获取 DispatchKeySet，

```c++
// 位于 aten/src/ATen/core/dispatch/DispatchKeyExtractor.h
DispatchKeySet ts;
void operator()(const at::Tensor& x) {
    ts = ts | x.key_set();
}
```

参数 `args...` 为 `Tensor self, Tensor other, Scalar alpha=1`，依次对参数应用 MultiDispatchKeySet::operator() ，便能得到最终的 DispatchKeySet。`Tensor::key_set` 方法返回的是 `TensorImpl::key_set_` 字段，我们创建 Tensor 时（ 例如使用 python 创建一个 Tensor：`a = torch.Tensor(1,2)` ），准确的说创建 TensorImpl 完成后，对应的 DispatchKeySet 并不仅仅包含 DispatchKey::CPU，参见下方的 TensorImpl 构造函数，还包含 DispatchKey::AutogradCPU，DispatchKey::AutocastCPU 和 DispatchKey::ADInplaceOrView 。

<details>
<summary>TensorImpl 构造函数</summary>

```c++
TensorImpl::TensorImpl(
    Storage&& storage,
    DispatchKeySet key_set,     // ===============> 对于 a = torch.Tensor(1,2)，此参数仅包含 CPU
    const caffe2::TypeMeta data_type,
    c10::optional<c10::Device> device_opt)
    : storage_(std::move(storage)),
      pyobj_interpreter_(nullptr),
      pyobj_(nullptr),
      storage_offset_(0),
      numel_(0),
      data_type_(data_type),
      device_opt_(device_opt) {
  init_bitfields();
  ...
  bool inference_mode = c10::InferenceMode::is_enabled();   // 默认 false

  // TODO: be more explicit about the full key set at call sites so we
  // don't have to keep recomputing it here
  auto k = key_set.highestBackendKey(); // backend 为 CPU

  key_set = key_set | getAutocastRelatedKeySetFromBackend(k);   // enable AutocastCPU 这个 bit

  // See [Note: Python key removal]
  key_set = key_set - c10::python_ks;

  // Inference tensor doesn't have autograd related keys.
  if (inference_mode) {
    // See Note [Expected TLS state in InferenceMode] for why we exclude
    // Autograd & ADInplaceOrView keys. Normally key_set only contains backend
    // keys but we do the substraction here to make sure.
    key_set_ = key_set - c10::autograd_dispatch_keyset_with_ADInplaceOrView;
  } else {
    // TODO: Ideally we only add AutogradBackend key when the tensor requires
    // grad.
    //       See Note [Dream: skip VariableType kernel when requires_grad=false]
    // 对于 CPU 这个 backend，enable ADInplaceOrView 和 AutogradCPU
    key_set_ = key_set | getAutogradRelatedKeySetFromBackend(k);        
  }
  ...
}
```

</details>

所以，并不是直接分发到 DispatchKey::CPU 对应的 kernel 也就是 wrapper_add_Tensor 这个方法。

<details>
<summary>wrapper_add_Tensor -> Dispatch::CPU</summary>
```c++
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    ...
    m.impl("add.Tensor", TORCH_FN(wrapper_add_Tensor));
}
```

</details>


### 1.1.1 不分发到 AutocastCPU

Tensor 的三个 functionality key 为 `AutocastCPU, AutogradFunctionality, ADInplaceOrView`，看着是最先分发到 AutocastCPU 上，实际上不会。这是因为注册了 AutocastCPU 的 fallthrough 方法，

<details><summary>AutocastCPU 的 fallthrough 注册语句</summary>

```c++
// 位于 aten/src/ATen/autocast_mode.cpp

TORCH_LIBRARY_IMPL(_, AutocastCPU, m) {  
  m.fallback(torch::CppFunction::makeFallthrough());
}


TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {
  // lower_precision_fp cast policy
  KERNEL_CPU(conv1d, lower_precision_fp)
  ...
}
```
第一个 TORCH_LIBRARY_IMPL 在 AutocastCPU 这一列中将所有 operator kernel 均注册为 fallthrough 方法。第二个 TORCH_LIBRARY_IMPL 则在 AutocastCPU 这一列中将部分 operator kernel 注册为有效方法，由于我们知道后面注册的 kernel 始终排在先注册的 kernel 之前，即最新注册的方法总是位于 `OperatorEntry::kernels_[AutocastCPU].front()`，最新注册的 kernel 总是更新到 `OperatorEntry::dispatchTable_` 中，故 conv1d 等一系列 operator 的 AutocastCPU 的 kernel 是 nonFallthrough 的。

不过，我们的 (AutocastCPU, add) 这个坐标处的 kernel 是 fallthrough 的。

</details>

注册 fallthrough 调用 `OperatorEntry::updateFallback` 方法，这里不详细分析代码，只提一个关键的方法调用，如下所示，

```c++
// 位于 aten/src/ATen/core/dispatch/OperatorEntry.cpp
dispatchKeyExtractor_.setOperatorHasFallthroughForKey(dispatch_key, dispatchTable_[dispatch_ix].isFallthrough());
```

这句调用设置了 DispatchKeyExtractor::nonFallthroughKeys_ 中 AutocastCPU 这个 bit 位为 0，

```c++
// 位于 aten/src/ATen/core/dispatch/DispatchKeyExtractor.h
template<class... Args>
DispatchKeySet getDispatchKeySetUnboxed(const Args&... args) const {
    auto ks = detail::multi_dispatch_key_set(args...);  // 根据 Tensor 参数计算 DispatchKeySet，包含 AutocastCPU，AutogradCPU，ADInplaceorview，CPU
    // Keys that are fallthrough should be skipped
    if (requiresBitsetPerBackend_) {
        auto backend_idx = ks.getBackendIndex();
        return impl::computeDispatchKeySet(ks, nonFallthroughKeysPerBackend_[backend_idx]);
    } else {
        return impl::computeDispatchKeySet(ks, nonFallthroughKeys_);
    }
}
```

根据参数计算处相关的 DispatchKeySet，根据函数 `computeDispatchKeySet` 再合并上 TLS 的 dispatchkey，然后使用 nonFallthroughKeys_ 对得到的 DispatchKeySet 进行 mask 操作，即，**去掉 DispatchKeySet 中具有 fallthrough 的 dispatchkey`，于是 AutocastCPU 就被移除了。

同样地，还有 `ADInplaceOrView` 对于 add 这个 operator 也添加到 nonFallthroughKeys_ 中，故也不分发到 ADInplaceOrView。

<details><summary>为 ADInplaceOrView 注册 Fallthrought 方法</summary>

```c++
// 位于 aten/src/ATen/core/VariableFallbackKernel.cpp
TORCH_LIBRARY_IMPL(_, ADInplaceOrView, m) {
      m.fallback(torch::CppFunction::makeFallthrough());
}

// 位于 torch/csrc/autograd/VariableTypeManual.cpp
TORCH_LIBRARY_IMPL(aten, ADInplaceOrView, m) {
  m.impl(
      "copy_",
      torch::dispatch(
          DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::copy_)));
  m.impl(
      "detach",
      torch::dispatch(
          DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::detach)));
  m.impl(
      "_fw_primal",
      torch::dispatch(
          DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::_fw_primal)));
  m.impl(
      "_make_dual",
      torch::dispatch(
          DispatchKey::ADInplaceOrView, TORCH_FN(ADInplaceOrView::_make_dual)));
}
```
</details>

### 1.1.2 分发到 DispatchKey::AutogradCPU

此时最终计算得到的 DispatchKeySet enabled 的 bit 位包括 `AutogradFunctionality, ADInplaceOrView, CPU`，根据 DispatchKeySet::getDispatchTableIndexForDispatchKeySet() 方法，得到 AutogradFunctionality 与 CPU 组合的 key 即 DispatchKey::AutogradCPU 对应的 kernel 在 OperatorEntry::dispatchTable_ 中的 idnex。


为 operator 注册 DispatchKey::Autograd 的 Impl。

```c++
// 位于 torch/csrc/autograd/generated/VariableType_2.cpp
TORCH_LIBRARY_IMPL(aten, Autograd, m) {
    ...
    m.impl("add.Tensor",
       TORCH_FN(VariableType::add_Tensor)
    );
}
```

这个注册过程最后是调用如下方法（此时已经向 `OperatorEntry::kernels_[Autograd]` 的 front 位置插入了 kernel `VariableType::add_Tensor`），

```c++
void OperatorEntry::updateDispatchTable_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key/*实参 Autograd*/) {
  ... // 省略无关代码
  for (auto k : c10::getRuntimeDispatchKeySet(dispatch_key)) {
    updateDispatchTableEntry_(dispatcher, k);
  }
}
```

getRuntimeDispatchKeySet 函数将 DispatchKey::Autograd 映射为 runtime DispatchKeySet，其中被 enabled 的 bit 位为

```shell
AutogradNestedTensor
AutogradFunctionality
AutogradOther
# 以下是 14 个 backend
PrivateUse3
PrivateUse2
PrivateUse1
Meta
Lazy
...
HIP
CUDA
CPU
```

此 DispatchKeySet 经 for 循环时，得到一系列的 runtime DispatchKey，

```shell
AutogradNestedTensor,
# === 14 个 key 开始 ===
AutogradPrivateUse3,
AutogradPrivateUse2,
...
AutogradCUDA,
AutogradCPU,
# === 14 个 key 结束 ===
AutogradOther
```

循环体中，对每一个 runtime DispatchKey（例如AutogradCPU），获取到 kernel，然后更新到 `OperatorEntry::dispatchTable_` 对应位置，获取 kernel 的相关语句调用为，

```c++
// 位于 aten/src/ATen/core/dispatch/OperatorEntry.cpp
if (isIncludedInAlias(dispatch_key, DispatchKey::Autograd)) {
  if (auto autograd_registration = getKernelForDispatchKey(DispatchKey::Autograd)) {
    return {*autograd_registration, "autograd kernel"};
  }
}
```

## 1.2 VariableType::add_Tensor

为了拥有梯度，python 端代码改为如下，

```python
a = torch.Tensor([1.,2.])
a.requires_grad = True
b = a + 1
```

Dispatcher::call 中最后调用 `kernel.call` 方法，就是调用 VariableType::add_Tensor，这个方法的定义代码较多，

```c++
// 位于 torch/csrc/autograd/generated/VariableType_2.cpp
at::Tensor add_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );   // 例子中，self requires_grad=True
  
  (void)_any_requires_grad;
  // self 设置了 requires_grad，故 autograd_meta_ 有值，但是 fw_grad 仍尚未定义；other autograd_meta_ 无值
  auto _any_has_forward_grad_result = (isFwGradDefined(self) || isFwGradDefined(other)); // false
  (void)_any_has_forward_grad_result;
  std::shared_ptr<AddBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddBackward0>(new AddBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_scalar_type = other.scalar_type();
    grad_fn->alpha = alpha;
    grad_fn->self_scalar_type = self.scalar_type();
  }

  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::add(ks & c10::after_autograd_keyset, self_, other_, alpha);
  })();
  auto result = std::move(_tmp);

  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  c10::optional<at::Tensor> result_new_fw_grad_opt = c10::nullopt;
  if (_any_has_forward_grad_result && (result.defined())) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_tensor = toNonOptTensor(self);
      auto self_t = (self_t_raw.defined() || !self_tensor.defined())
        ? self_t_raw : at::_efficientzerotensor(self_tensor.sizes(), self_tensor.options());
      auto other_t_raw = toNonOptFwGrad(other);
      auto other_tensor = toNonOptTensor(other);
      auto other_t = (other_t_raw.defined() || !other_tensor.defined())
        ? other_t_raw : at::_efficientzerotensor(other_tensor.sizes(), other_tensor.options());
      result_new_fw_grad_opt = self_t + maybe_multiply(other_t, alpha);
  }
  if (result_new_fw_grad_opt.has_value() && result_new_fw_grad_opt.value().defined() && result.defined()) {
    // The hardcoded 0 here will need to be updated once we support multiple levels.
    result._set_fw_grad(result_new_fw_grad_opt.value(), /* level */ 0, /* is_inplace_op */ false);
  }
  return result;
}
```

### 1.2.1 创建 grad_fn

对于 add 操作，创建相应的计算梯度的类 `AddBackward0` 的实例，在梯度图 graph 中，`grad_fn` 是一个 node，根据 add 的两个操作数 `self, other` 分别创建两条 edge，

<details>

<summary>创建 edge 的代码</summary>

```c++
// 位于 torch/csrc/autograd/variable.cpp
Edge gradient_edge(const Variable& self) {
  if (const auto& gradient = self.grad_fn()) {
    return Edge(gradient, self.output_nr());
  } else {
    return Edge(grad_accumulator(self), 0);
  }
}

std::shared_ptr<Node> grad_accumulator(const Variable& self) {
  auto autograd_meta = get_autograd_meta(self);
  if (!autograd_meta) {         // other 参数没有 autograd_meta
    return nullptr;
  }
  if (autograd_meta->grad_fn_) {
    throw std::logic_error(
        "grad_accumulator() should be only called on leaf Variables");
  }
  if (!autograd_meta->requires_grad_) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(autograd_meta->mutex_);

  auto result = autograd_meta->grad_accumulator_.lock();
  if (result)
    return result;

  c10::raw::intrusive_ptr::incref(self.unsafeGetTensorImpl());
  auto intrusive_from_this =
      c10::intrusive_ptr<at::TensorImpl>::reclaim(self.unsafeGetTensorImpl());
  result = std::make_shared<AccumulateGrad>(
      Variable(std::move(intrusive_from_this)));
  autograd_meta->grad_accumulator_ = result;
  return result;
}
```
</details>

由于 `self, other` 的 grad_fn 均为 null，所以这里使用 grad_accumulator 为 `self, other` 创建两个 edge。

edge 连接两个 node，以 `out=self+other` 为例，一个 edge 连接 out 和 self 的 grad_fn，另一个 edge 连接 out 和 other 的 grad_fn，这里 self 和 other 的 grad_fn 均为 AccumulateGrad 类型。 `output_nr` tensor 是 function 的第几个输出，例如第二个输出，那么 `output_nr()=1`，显然对于 self 和 other 对于各自的 AccumulateGrad（一个 dummy function），都是第一且唯一的输出，所以 `Edge(grad_accumulator(self), 0)` 第二个参数为 0。

上述代码中新建的 `grad_fn`（AddBackward0 类实例）则作为 out 的 grad_fn，也就是说 grad_fn 是 add 这个操作对应的用于计算梯度的函数，而 out 是 add 这个参数的输出，由于是 add 操作的第一个且唯一的输出，所以 `out.output_nr()=0` 。


<details>
<summary>Node, Edge, Variable 三者关系图</summary>

```shell
+------------------+                                  +------------------+
| Variable out     |                                  | Variable self    |
| 1. grad_fn -->-+ |                                  | 1. grad_fn -->-+ |
| 2. output_nr   | |                                  | 2. output_nr   | |
+----------------+-+                                  +----------------+-+
                 |                                                     |
                 v              +---------------+                      v
+---------------------+         | Edge          |     +--------------------+
| Node                |  +----> | 1. function --+---> | Node               |
| 1. next_edges_ ---+-|--+      | 2. input_nr   |     | (AccumulateGrad)   |
| 2. topological_nr=1 |  |      +---------------+     | 1. topological_nr=0|
+---------------------+  |                            +--------------------+
                         |      +-------------------+     +----------------------+
                         +----> | Edge              |     | Variable other       | 
                                | 1. function =NULL |     |1. autograd_meta_=NULL|
                                | 2. input_nr       |     |2. output_nr          |
                                +-------------------+     +----------------------+ 
```
</details>

variable 的 output_nr 表示 variable 是相关操作的第一个输出，对于 leaf variable，虚拟出一个操作，leaf variable 是虚拟操作的第 1 个输出，output_nr=0，虚拟操作的梯度计算函数为 accumulategrad。

edge 的 input_nr 表示反向梯度计算时，edge 是其梯度计算函数 function 的第几个输入。

`other` 这个 Variable 是对 scalar `1` 的 wapper，所以 `autograd_meta_=NULL`，不需要计算梯度，故不创建 AccumulateGrad。


**Node**

每个节点 Node 有一个序列号 `sequence_nr`，创建节点时对其自增赋值，注意是线程内自增；每个节点还有一个拓扑号 `topological_nr`，表示当前 node 与所有 leaf node 之间的最长距离，对于leaf node 即 AccumulateGrad，其 topological_nr=0。

topological_nr 更新规则：

1. leaf node 的 topological_nr=0
2. parent node 的 topological_nr 是其最大 child node 的 topological_nr 再加上 1
3. 如果某个 node 在确定其有 parent node 之后，这个 node 的 topological_nr 不能再改变



### 1.2.2 分发到 DispatchKey::CPU

创建并初始化 grad_fn （即 AddBackward0类实例）后，执行如下的 lambda 函数，
```c++
auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::add(ks & c10::after_autograd_keyset, self_, other_, alpha);
})();
```

上面代码中 `AutoDispatchBelowADInplaceOrView guard`，在 AutoDispatchBelowADInplaceOrView 构造函数中，将 TLS（线程本地状态）的 DispatchKeySet 设置 excludeed_ 包含以下几个 key，

```shell
AutogradFunctionality, AutogradOther, AutogradNestedTensor, ADInplaceOrView
```

**分发时计算 DispatchKeySet 的过程综合了实参 Tensor 、tls 和 global 的 DispatchKeySet**

`at::redispatch::add` 方法中，`ks & c10::after_autograd_keyset` 操作对 DispatchKeySet ks 进行 mask，使得仅有 autograd 之后（不包括任何 autograd）的 bit 值被保留，之前的（包括所有 autograd）的 bit 全部被 disable。而上面已经说明 `ks` 中的 ADInplaceOrView bit 已经被 disable，所以当前将分发到 DispatchKey::CPU 上，注意 DispatchKey::CPU 是 DispatchKey::Dense 与 backendcomponent CPUBit 合成的 key。


`at::redispatch::add` 方法定义为，

```c++
// 位于 torch/include/ATen/RedispatchFunctions.h
inline at::Tensor add(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1) {
    return at::_ops::add_Tensor::redispatch(dispatchKeySet, self, other, alpha);
}

// 位于 build/aten/src/ATen/Operators_2.cpp
at::Tensor add_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_add_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}
```

现在，`at::redispatch::add` 方法参数 dispatchKeySet 的最高位 functionality_key 为 Dense，backendcomponent 为 CPU，所以根据 dispatchKeySet 得到的 dispatchTable_ 中 kernel 下标是对应 DispatchKey::CPU 的。 `op.redispatch` 方法调用 `Dispatcher::redispatch`，为了方便，这里再贴出其代码，由于简单，不再多说。

```c++
template<class Return, class... Args>
inline Return Dispatcher::redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKeySet currentDispatchKeySet, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  const KernelFunction& kernel = op.operatorDef_->op.lookup(currentDispatchKeySet); // 找到 DispatchKey::CPU 对应的 kernel
  return kernel.template call<Return, Args...>(op, currentDispatchKeySet, std::forward<Args>(args)...);
}
```


### 1.2.3 设置 output Tensor 的 grad

计算得到 output Tensor 之后，我们再回到 AutogradCPU 的 kernel 方法实现并接着往下看，

```c++
if (grad_fn) { // grad_fn 是 output Tensor 的 grad_fn
    set_history(flatten_tensor_args( result ), grad_fn);
}
```

flatten_tensor_args(result) 是将 Variable 全部展平放入一个 `vector<Variable>` 中，因为有时候一个 Node 有多个 output Tensor。然后使用 `set_history` 设置每个 Variable 的 grad_fn，相关函数定义如下，

```c++
// 位于 torch/csrc/autograd/functions/utils.h
inline void set_history(
    at::Tensor& variable,
    const std::shared_ptr<Node>& grad_fn) {
  AT_ASSERT(grad_fn);
  if (variable.defined()) {
    TORCH_INTERNAL_ASSERT(isDifferentiableType(variable.scalar_type())); // 确保是可导数据类型，例如 float。int 类型不可导
    auto output_nr = grad_fn->add_input_metadata(variable);
    impl::set_gradient_edge(variable, {grad_fn, output_nr});
  } else {
    grad_fn->add_input_metadata(Node::undefined_input());
  }
}
```

上面代码中，`add_input_metadata` 为当前 Node 设置 input_metadata_。  Node 通过 input_metadata_ 关联 Tensor，Tensor 通过 autograd_meta_ 关联 Node。 output_nr 为当前 Variable 是此 Node 的第一个输出，例如第一个输出那么 output_nr=0。


# 2. AddBackward0

分析本例中的用于计算梯度的函数 `AddBackward0` ，首先给出其定义，

```c++
struct TraceableFunction : public Node {
  using Node::Node;
  bool is_traceable() final {
    return true;
  }
};

struct TORCH_API AddBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward0"; }
  void release_variables() override {
  }

  at::ScalarType other_scalar_type;
  at::Scalar alpha;
  at::ScalarType self_scalar_type;
};
```

其中 `apply` 方法定义为

```c++
// grads 传递到这个 node 的 梯度集合
variable_list AddBackward0::apply(variable_list&& grads) {
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  // grads 本质是 Tensor list，是否至少有一个 Tensor 是 defined
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {   // other Tensor 是否需要计算 grad
    auto grad_result = any_grad_defined ? (handle_r_to_c(other_scalar_type, maybe_multiply(grad, alpha.conj()))) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
```

传递到一个 Node 的梯度通常是一个列表，因为 Node 所关联的 Tensor（Tensor.grad_fn == Node）可能作为多个 operator 的输入，那么反向传播时，就有多个 梯度传递到这个 Tensor 的 grad_fn
即 Node 上 。

```shell
              (0)   +-Edge--+      +------Node-------+
+-Node-+    +-----> | self -+----> | AccumulatedGrad |
|      |    |       +-------+      +-----------------+
| out -+----+
|      |    | (1)   +-Edge--+
+------+    +-----> | other |
                    +-------+
```

本文例子中，other 是包装 `1` 而来的 Tensor，这个路径分支不需要计算 grad 。对于 self 分支， add 操作的梯度计算是 1，即下式中的 $\partial o/\partial x$。

$$\frac {\partial L}{\partial x} = \frac {\partial L}{\partial o} \cdot \frac {\partial o}{\partial x}= \frac {\partial L}{\partial o} \cdot 1 = \frac {\partial L}{\partial o}$$

上式中，

1. L -> loss
2. o -> output
3. x -> self
4. $\partial L/ \partial o$ -> 上述代码中的 `grad`

`copy_range` 则是将计算出来的梯度结果 copy 到 `grad_inputs` 中，然后传递到下一个 Node 上，这里下一个 Node 是 `AccumulatedGrad` 。

