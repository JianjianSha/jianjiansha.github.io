---
title: DispatchKeySet（PyTorch）
date: 2023-02-17 12:00:50
tags: pytorch source code
p: pytorch/source/dispatchkeyset
summary: DispatchKeySet 知识点
---

## 1.1 几个构造函数

1. 
    ```c++
    constexpr DispatchKeySet(Full)
        : repr_((1ULL << (num_backends + num_functionality_keys - 1)) - 1) {}
    ```

    我认为 `Full` 没有包含 `DispatchKey::EndOfFunctionalityKeys` 。

    记住，`repr_` 的最低 `num_backends` 个 bits 用于表示 backends 的 mask。

2. 
    ```c++
    constexpr explicit DispatchKeySet(DispatchKey k) {...}
    ```

    - k=DispatchKey.Undefined, `repr_=0`
    - k<DispatchKey.EndOfFunctionalityKeys, 设置 `repr_` 中对应 k 的 bit 为 1
    - k<DispatchKey::EndOfRuntimeBackendKeys, 此时 k 是由 functionality key 和 backend 的组合，将 k 分解为一个 functionality key 和一个 backend，然后设置 `repr_` 中相应的两个 bit 为 1
    - otherwise，`repr_=0`

3. 
    ```c++
    // 设置 t 之后的 functionality keys，即 `repr_` 中比 t 低的 bits
    // functionality keys 顺序为：Autograd > Sparse > Quantized > Dense
    constexpr DispatchKeySet(FullAfter, DispatchKey t)
      : repr_(  // 从 t 中抽出 functionality key，其在`repr_` 对应的 bit 位置记为 d，enable `repr_` 比 d 低位的 bit
            (1ULL
             << (num_backends + static_cast<uint8_t>(toFunctionalityKey(t)) -   
                 1)) -
            1) {
    *this = add(DispatchKey::PythonDispatcher); // 增加一个 functionality_key，用于跳过C++ dispatcher
    }    
    ```

4. 
    ```c++
    constexpr DispatchKeySet(uint64_t repr) : repr_(repr) {}
    ```


`repr_` 各个 bit 含义：

```shell
# 从左往右，表示 bit 从高到低。 最低 bit index 从 0 开始计数，
# 那么 enable functionality_key 所在 bit： 1 << (num_backend + functionality_key - 1)
# 55                      54                    15    14        13          12              1     0
EndOfFunctionalityKeys, PythonDispatcher, ..., FPGA, Dense, PrivateUse3, PrivateUse2, ..., CUDA, CPU
```

## 1.2 类方法

1. 
    ```c++
    // 获取 repr_ 最高bit=1 的位置 index
    // e.g. index=0, all bit 为 0；index=1  最低bit=1；index=64，最高bit=1
    uint8_t indexOfHighestBit() const {
        return 64 - llvm::countLeadingZeros(repr_);
    }
    ```

2. `initializeFunctionalityOffsetsAndMasks`。初始化每个 functionality key 的 offset 和 mask。

    - 总共 `num_functionality_keys=42` 个 DispatchKey。第一个 DispatchKey 为 `Undefined`，其 offset 和 mask 均为 0。
    - 以后每个 DispatchKey 的 offset 则在前一个 DispatchKey 的 offset 上 `+1`，如果前一个 Dispatchkey 需要与 backends 结合，那么 `+num_backends`，相当于本来一个 DispatchKey 与 backends 结合后变成 `num_backends` 个 DispatchKey。
    - 以后每个 DispatchKey，如果需要与 backend 结合，那么其 mask 为 `full_backend_mask=(1ULL<<num_backends)-1`，否则 `mask=0`。

3. `getDispatchTableIndexForDispatchKeySet` (这里讨论非 mobile 情况)

    根据 `repr_` 表示的 functionality key 和 backend，得到对应的在 DispatchTable 中的 index。根据每个 functionality key 的 offset 和 backend 的 index 可以计算出来。
    例如：
    - `repr_=0`，表示 Undefined，那么返回值为 `0`。
    - `repr_=(1ULL << num_backends)+(1ULL << 1)`，那么 functionality key 为 Dense，其 offset=1，backend 为 cuda，index=1（index 从 0 开始计算，即CPU 的 index=0），于是最终返回值为 `offset+index=2`
    - `repr_=(1ULL << num_backends+Batched-1)`，那么 functionality key 为 Batched，其在 DispatchKey 枚举中值为 `33`，Batched 前面有 5 个需要与 backend 结合，所以 Batched 的 offset=`33+5*(num_backends-1)=98`，Batched 没有 backend，那么 `index=0`，所以返回值为 `offset+index=98`。
    

    DispatchKeySet(Dense) 和 DispatchKeySet(CPU) 的 `repr_` 值不同，后者大1，但是两者的 `getDispatchTableIndexForDispatchKeySet` 返回值相同，这表示两者都将被 dispatch 到 table 同一 index 处（backend 均为 CPU）。


    |DispatchKeySet最高functionality_key 和最高 backend 的组合|最高 functionality key offset/value|最高 backend 位置|dispatchTableIndex|
    |--|--|--|--|
    Undefined |0/0|0|0+0|
    Dense|1/1|0|1+0|
    Dense+CPU | 1/1 | 0（index从0开始计数）|1+0
    Dense+CUDA | 1/1|1|1+1| 
    Dense+PrivateUser3|1/1|13|1+13|
    FPGA|1+14（前面有Dense）/2|0（backend的mask=0，故设置 FPGA 最低的 14 个 backend bit=1也没用）|1+14
    ORT|2+14/3|0| 2+14
    Vulkan|3+14/4|0|3+14
    Metal|4+14/5|0|4+14
    Quantized|5+14/6|0|5+14|
    Quantized+CPU|5+14/6|0|5+14|
    Quantized+PrivateUser3|5+14/6|13|5+14+13|
    CustomRNGKeyId|5+2*14（前面有Dense, Quantized）/7| 0|5+2 *14|
    ...|
    |AutogradFunctionality+CPU|24+4*13（前面有 4 个特殊Functionality_Key）/24|0|24+4 * 13|
    ...|
    <center>表 1.</center>

```c++
using schema = at::Tensor (c10::SymIntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>);
```


```c++
// Return 为 Tensor
// Args... 为 SymIntArrayRef, optional<ScalarType>, optional<Layout>, optional<Device>
template<class Return, class... Args>
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);   // 根据参数获取对应的 dispatchKeySet
#ifndef NDEBUG
  DispatchTraceNestingGuard debug_guard;
  if (show_dispatch_trace()) {
      auto nesting_value = dispatch_trace_nesting_value();
      for (int64_t i = 0; i < nesting_value; ++i) std::cerr << " ";
      std::cerr << "[call] op=[" << op.operator_name() << "], key=[" << toString(dispatchKeySet.highestPriorityTypeId()) << "]" << std::endl;
  }
#endif
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

根据参数获取 dispatchKeySet 的具体实现位于 `MultiDispatchKeySet` 内，有关 `Tensor` 和 `Generator` 的参数会影响 dispatchKeySet，这里我们的 schema 函数参数均与 `Tensor, Generator` 无关，所以 dispatchKeySet 就是默认值 `repr_=0` 。

```c++
