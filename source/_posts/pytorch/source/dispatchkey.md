---
title: DispatchKey
date: 2023-02-17 11:59:33
tags: pytorch source code
---


`num_functionality_keys = DispatchKey::EndOfFunctionalityKeys=42`

`num_backends = BackendComponent::EndOfBackendKeys=14`

`num_runtime_entries = num_functionality_keys + 5 * (num_backends - 1) = 107`

这里 5 是指需要与 backends 结合的 DispatchKey 数量，分别是 `Dense, Quantized, Sparse, NestedTensor, AutogradFunctionality`。（注意，`Dense` 与 backend 结合时，省去了 `Dense`，例如，使用 `CPU` 代替 `DenseCPU`，`CUDA` 代替 `DesneCUDA`）

`num_backends - 1` 中 `-1` 是因为原本 `num_functionality_keys` 已经算上这 5 个特殊的 DispatchKey 了，但实际上这 5 个 functionality key 不单独存在，而是要与某个 backend 结合，这与其他 `42-5` 个 functionality key 不同。也就是说，运行时条目数量为：`42-5` 个单独存在的 functionality key，以及 `5*num_backends` 个与 backend 结合的 functionality key。例如，`Dense` 不作为运行时条目，而 `CPU` `CUDA` 等则是运行时条目。


1. `toBackendComponent` 

    - 如果参数是与 backend 结合的 DispatchKey，获取相应的 backend。例如 `CUDA` 返回 `CUDABit`
    - 如果参数未与 backend 结合，那么返回 `InvalidBit`。例如 `Batched` 返回 `InvalidBit`。

2. `toFunctionalityKey` 

    - 如果参数是与 backend 结合的 DispathKey，获取相应的基本的 DispatchKey。例如 `CUDA` 返回 `Dense`，`SparseCPU` 返回 `Sparse`
    - 如果参数是 `num_functionality_keys` 之内的 DispatchKey，那么返回参数自身。例如 `Batched` 返回 `Batched` 
    - otherwise，返回 `Undefined` 

3. `toRuntimePerBackendFunctionalityKey(DispatchKey, BackendComponent)`

    - 如果是 5 个需要与 Backend 结合的 DispatchKey，那么返回结合后的 DispatchKey，例如 `Dense` 与 `CPUBit` 就得到 `CPU`，`Sparse` 与 `CUDABit` 得到 `SparseCUDA` 
    - otherwise，返回 `Undefined`

    