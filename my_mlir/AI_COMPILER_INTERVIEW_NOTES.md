# NorthStar MLIR 项目面试速记

## 1. 一句话介绍

我做了一个基于 MLIR 的 AI 编译器原型，设计了自定义 `NorthStar` 方言来表达带设备语义和数据并行策略的张量计算，并通过多级 pass 把高层 IR lowering 到 Linalg、SCF、GPU/NVVM、runtime stub，最后生成 PTX 和 runtime JSON，由 CUDA runner 完成端到端执行验证。

## 2. 30 秒版本

这个项目不是只写一个 MLIR pass，而是一条比较完整的 AI compiler pipeline。

我先定义了 `NorthStar` 自定义方言，让 tensor type 和 op 显式带上 `device_id`、数据并行属性和 `device_region` 这类设备执行语义。然后实现多级 lowering pass，把高层张量 IR 降到 Linalg、SCF、GPU/NVVM，并把设备区域 outline 成独立 kernel。最后针对 CUDA 后端生成 PTX 和 runtime JSON descriptor，由 runner 加载 descriptor 和 PTX 去 launch kernel。

所以这个项目覆盖了方言设计、IR 变换、kernel outline、GPU lowering、runtime bridge 和端到端执行验证。

## 3. 2 分钟版本

这个项目主要想解决的是：AI 编译器里，高层张量计算如何携带设备放置和并行策略，并最终落到后端执行。

我设计了一个 `NorthStar` 方言，里面有自定义的 `NSTensorType`，类型里包含 shape、element type 和 `device_id`；还有 `DataParallelism` 属性，用来表达数据并行的设备组；以及 `device_region` op，用来标记某段计算应该运行在指定 device 上。这样设备信息不是靠外部表或者注释维护，而是在 IR 里一等可见，可以被 pass 分析、验证和改写。

在编译流程上，我实现了一条多阶段 pipeline。前面先做数据并行标记和 tensor 切分，比如把一个 host tensor 按 DP 策略拆到多个 device；中间把 `device_region` outline 成独立 kernel；后面根据目标后端继续 lowering 到 Linalg、SCF、GPU dialect 和 NVVM/PTX。

对 CUDA 路径，我还做了 runtime bridge。编译器会生成 runtime stub 和 JSON descriptor，descriptor 里记录 kernel name、descriptor id、device id、输入输出 shape/type 和 C API 名字。运行时 runner 只需要读取 JSON 和 PTX，就可以调用统一 CUDA launch API 执行对应 kernel。

这个项目最有价值的地方是三点：第一，设备语义在 IR 层显式表达；第二，pass pipeline 是分层的，不是孤立 rewrite；第三，编译器和 runtime 之间有清晰接口，能完成从高层 IR 到 CUDA 执行的闭环。

## 4. 项目结构怎么讲

```text
my_mlir/
├── include/Dialect/NorthStar/IR/         # NorthStar 方言的 type、attr、op 定义
├── include/Dialect/NorthStar/Transforms/ # 方言级 transform pass 声明
├── include/Conversion/                   # lowering pass 声明
├── include/Interfaces/                   # 并行和 fusion 相关接口
├── include/Utils/                        # CUDA/H350 runtime API 头文件
├── src/Dialect/NorthStar/IR/             # 方言实现和 verifier
├── src/Dialect/NorthStar/Transforms/     # 数据并行、device region fusion
├── src/Conversion/NorthStarToLinalg/     # 主要 lowering pipeline
├── src/Utils/                            # CUDA/H350 runtime 支持
├── src/NS-opt/                           # 自定义 mlir-opt 工具
└── test/                                 # FileCheck 和 runtime smoke test
```

面试里不需要逐个文件背，重点讲这四块：

1. `NorthStar` Dialect：表达高层张量、设备和并行语义。
2. Transform Pass：做数据并行标记、切分、融合。
3. Conversion Pass：做 Linalg/GPU/runtime lowering。
4. Runtime：生成 JSON descriptor，配合 PTX 运行。

## 5. 核心 IR 设计

### `NSTensorType`

位置：

```text
include/Dialect/NorthStar/IR/NorthStarTypes.td
```

它类似一个带设备信息的 tensor：

```text
shape + elementType + device_id
```

可以这样讲：

> 普通 tensor 只能表达形状和 dtype，但 AI 编译器还需要知道 tensor 在哪个 device 上。`NSTensorType` 把 `device_id` 编进类型系统，让后续 pass 可以直接依据类型做设备相关分析和 rewrite。

### `DataParallelism` 属性

位置：

```text
include/Dialect/NorthStar/IR/NorthStarAttrs.td
```

示例：

```mlir
#north_star.DP<DP = 3 : 0, 1, 2>
```

含义是：使用 3 路数据并行，设备列表是 0、1、2。

可以这样讲：

> 我用 attribute 表达并行策略，而不是把策略写死在 pass 里。这样 IR 本身携带策略，pass 只是消费策略并做变换。

### `device_region`

位置：

```text
include/Dialect/NorthStar/IR/NorthStarOps.td
```

作用是表示某一段计算属于某个 device：

```mlir
"north_star.device_region"(...) <{device_id = 1 : i64, sym_name = "..."}> ({
  ...
})
```

可以这样讲：

> `device_region` 明确划分 host/device 边界。后续 pass 可以把它 outline 成独立 kernel，再由 host 侧调度执行。这和真实 AI 编译器里的 kernel extraction、device placement 很接近。

### `buffer_cast`

作用：

- 一个 tensor 切成多个 device 上的 tensor
- 多个 device tensor 合并回一个 tensor

可以这样讲：

> `buffer_cast` 是数据并行切分和重组的 IR 表达。它让分布式张量在 IR 里有明确边界，而不是隐式藏在 runtime 里。

## 6. Pipeline 怎么讲

推荐画这个流程：

```text
NorthStar 高层 IR
  -> 数据并行标记和切分
  -> device_region fusion
  -> NorthStar op lowering 到 Linalg/Tensor
  -> device_region outline 成 func.func kernel
  -> lowering 到 SCF loop 或 GPU dialect
  -> GPU -> NVVM -> PTX
  -> runtime stub -> CUDA bridge
  -> runtime JSON descriptor
  -> CUDA runner 执行
```

更短的口径：

```text
NorthStar -> Linalg/SCF -> GPU/NVVM/PTX -> Runtime JSON -> CUDA Runner
```

## 7. 主要 Pass 及作用

### 方言级变换

```text
src/Dialect/NorthStar/Transforms/MarkDistributeParallelParameters.cpp
src/Dialect/NorthStar/Transforms/ApplyDistributeTransform.cpp
src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp
```

作用：

- 标记哪些参数参与数据并行
- 根据 DP 属性切分 tensor
- 把同一 device 上的连续计算融合进 `device_region`

面试说法：

> 这部分解决的是高层并行策略如何落到 IR 结构上。先用属性表达策略，再通过 pass 把策略转成显式的 tensor split、device region 和 merge。

### NorthStar 到 Linalg

```text
src/Conversion/NorthStarToLinalg/NorthStarToLinalgPass.cpp
```

作用：

- 把 `softmax`、elementwise op 等高层算子降到更标准的 Linalg/Tensor/Math 表达

面试说法：

> 这一步把自定义高层语义转到 MLIR 生态里更通用的 dialect，后续就可以复用 MLIR 的 bufferization、loop lowering 和 GPU lowering 能力。

### Kernel outline

```text
src/Conversion/NorthStarToLinalg/OutlineDeviceKernelsPass.cpp
```

作用：

- 把 `north_star.device_region` 抽成独立 `func.func`
- host 侧留下调用点

面试说法：

> 这是 host/device 解耦的关键。outline 之后 kernel 可以独立 lowering，host 侧只负责调度。

### GPU lowering

```text
src/Conversion/NorthStarToLinalg/LowerDeviceKernelsToGPUPass.cpp
```

作用：

- 生成 `gpu.module`
- 生成 `gpu.func`
- 生成 `gpu.launch_func`
- 给 kernel 附加 target、entry、block/grid 等 metadata

面试说法：

> 这一步把抽象的 device kernel 映射到 MLIR GPU dialect，为后续转 NVVM/PTX 做准备。

### Runtime bridge

```text
src/Conversion/NorthStarToLinalg/LowerRuntimeStubsToCUDACallsPass.cpp
src/Conversion/NorthStarToLinalg/GenerateRuntimeJsonPass.cpp
```

作用：

- 把 runtime stub 改写成统一 CUDA C API 调用
- 生成 runtime JSON descriptor

面试说法：

> 编译器不直接执行 kernel，而是生成稳定的元数据协议。runtime 根据 descriptor 找到 kernel entry、输入输出签名和 device 信息，再去 launch。

## 8. CUDA 端到端链路

脚本：

```text
run_cuda_pipeline.sh
run_cuda_on_5060.sh
```

`run_cuda_pipeline.sh` 做：

```text
NorthStar IR -> GPU IR -> NVVM IR -> PTX
```

`run_cuda_on_5060.sh` 做：

```text
生成 PTX
生成 runtime JSON
提取 JSON payload
调用 northstar-cuda-runner 执行
```

面试说法：

> 我不是只把 IR 打印出来，而是把 CUDA 路径串到了 PTX 和 runner 执行。这样可以验证 pass 生成的 kernel metadata、PTX entry 和 runtime descriptor 是否真的能对齐。

## 9. Runtime JSON 设计

位置：

```text
src/Conversion/NorthStarToLinalg/GenerateRuntimeJsonPass.cpp
```

JSON 里包含：

```text
target
c_api
descriptor_id
bridge
kernel_name
device_id
num_inputs
num_outputs
inputs shape/type/element_type
outputs shape/type/element_type
```

可以这样讲：

> PTX 只包含代码，但 runtime launch 还需要知道调用哪个 kernel、参数 shape/type 是什么、在哪个 device 上执行。所以我额外生成 runtime descriptor，把这些信息结构化传给 runtime。

## 10. 和 AI 编译器的关系

这个项目可以对应 AI 编译器里的几个核心问题：

| AI 编译器问题 | 项目里的对应实现 |
| --- | --- |
| 高层算子表达 | `NorthStar` op，例如 `softmax`、elementwise op |
| Tensor shape/type 管理 | `NSTensorType` |
| Device placement | `device_id`、`device_region` |
| 数据并行 | `DataParallelism` attr、`buffer_cast` |
| Kernel extraction | `OutlineDeviceKernelsPass` |
| Backend lowering | Linalg、SCF、GPU、NVVM、PTX |
| Runtime dispatch | runtime stub、CUDA bridge、JSON descriptor |
| 端到端验证 | CUDA runner、smoke test、FileCheck |

总结句：

> AI 编译器的关键不是只 lowering 单个算子，而是要让高层图/张量语义、并行策略、设备放置、kernel 生成和 runtime 调度形成闭环。这个项目就是围绕这个闭环做的一个小型原型。

## 11. 常见追问答案

### Q1：为什么要自定义 Dialect，不直接用 Linalg 或 StableHLO？

因为这个项目的重点不是只表达通用算子，而是表达设备语义和并行策略。Linalg 更适合结构化计算，StableHLO 更适合高层算子图，但我需要在 IR 里直接表达 `device_id`、`device_region`、数据并行切分和 host/device 调度边界。自定义 Dialect 可以把这些信息前置，后续 pass 更容易分析和改写。

### Q2：你的 pipeline 怎么分层？

我会分四层：

1. 高层语义层：`NorthStar` 表达 tensor、设备、并行和高层 op。
2. IR 变换层：做数据并行切分、device region fusion、kernel outline。
3. 后端 lowering 层：降到 Linalg、SCF、GPU、NVVM/PTX。
4. Runtime 执行层：生成 runtime stub 和 JSON descriptor，由 CUDA runner 调用。

### Q3：为什么要有 `device_region`？

`device_region` 是 host/device 边界的显式表示。它让编译器知道哪段计算属于哪个 device，后续可以把它 outline 成 kernel，并单独做 GPU lowering 或 runtime 调度。

### Q4：为什么要生成 runtime JSON？

因为 runtime 执行不只需要 PTX 代码，还需要结构化元数据，比如 kernel name、descriptor id、device id、输入输出 shape/type。JSON descriptor 相当于编译器和 runtime 之间的协议，runtime 不需要理解完整 MLIR，只需要消费 descriptor。

### Q5：项目最难的点是什么？

最难的是让 IR 语义、pass lowering 和 runtime metadata 三者保持一致。比如 `device_region` outline 后，kernel name、device id、descriptor id、PTX entry 和 runtime JSON 里的信息必须对齐，否则编译结果无法被 runtime 正确执行。

### Q6：如果继续优化，你会做什么？

我会从几个方向继续做：

1. 扩展更多算子和 dtype，比如 matmul、attention、更多 elementwise。
2. 完善动态 shape 支持。
3. 把 runtime descriptor schema 固化，增加版本和校验。
4. 做更完整的性能 benchmark 和 profiling。
5. 抽象更多后端，不只支持 CUDA 原型，也可以接 H350 或其他 NPU runtime。
6. 增加更真实的调度策略，比如 tiling、fusion、memory planning。

## 12. 最推荐你背的最终模板

> 我这个项目做的是一个基于 MLIR 的小型 AI 编译器 pipeline。核心思路是先用自定义 `NorthStar` 方言表达高层张量计算，同时把设备 ID、数据并行策略和 `device_region` 这种 host/device 边界显式放进 IR。然后我通过一系列 pass 做数据并行切分、device region fusion、kernel outline 和多级 lowering，把高层 IR 降到 Linalg、SCF、GPU/NVVM，最后生成 PTX。
>
> 对 CUDA 路径，我还设计了 runtime bridge 和 JSON descriptor。descriptor 里记录 kernel name、device id、输入输出 shape/type 等元数据，runner 可以读取 descriptor 和 PTX，调用统一 CUDA launch API 执行。这个项目让我完整实践了 MLIR 方言设计、pass pipeline、GPU lowering 和 runtime 接口设计，而不是只停留在单个 rewrite pass。

## 13. 面试时不要这样讲

不要只说：

> 我写了一个 MLIR 方言。

这样太弱。

应该说：

> 我做了一条从自定义高层 IR 到 CUDA runtime 执行的 mini AI compiler pipeline。

不要过度声称：

> 这是一个完整生产级 AI 编译器。

更稳妥的说法：

> 这是一个面向 AI 编译器核心问题的原型，重点验证了设备语义建模、kernel outline、GPU lowering 和 runtime dispatch 这条链路。

## 14. 可以主动展示的命令

生成 GPU/NVVM/PTX：

```bash
my_mlir/run_cuda_pipeline.sh
```

端到端跑 CUDA runner：

```bash
my_mlir/run_cuda_on_5060.sh
```

跑测试：

```bash
cmake --build build --target check-ch
```

注意：项目里的 pass 名当前是：

```bash
--convert-north-satr-to-linalg
```

这里 `satr` 是当前代码里的实际命令名，演示时按项目现状使用即可。

## 15. 最后压轴总结

如果面试官问“这个项目最能体现你什么能力”，可以答：

> 这个项目体现的是我对 MLIR 编译器分层的理解：高层 Dialect 负责表达语义，pass pipeline 负责逐步 lowering，backend dialect 负责贴近硬件，runtime descriptor 负责连接编译期和执行期。尤其是在 AI 编译器场景下，我把 tensor、device、parallelism、kernel 和 runtime launch 串成了一条完整链路。

## 16. 针对这个岗位 JD 的讲法

这个岗位关键词是：

```text
大模型图编译器
算子编译器
量化压缩
Triton / Tilelang / Cutlass / Cute / torch.compile
LLVM GPGPU 指令编译
指令选择 / 指令调度
C++ / Python
```

你的项目最匹配的点是：

| JD 方向 | 你的项目怎么对应 |
| --- | --- |
| 图编译器 | `NorthStar` 方言表达高层 tensor IR，并通过 pass pipeline 做 lowering |
| 算子编译器 | 支持 `softmax`、elementwise op，并降到 Linalg/GPU |
| GPGPU 编译 | 从 GPU dialect 继续降到 NVVM/PTX |
| LLVM 生态 | 基于 MLIR/LLVM pass、dialect、TableGen、FileCheck |
| runtime 验证 | 生成 PTX 和 runtime JSON，用 CUDA runner 执行 |
| C++ 能力 | Dialect、pass、runtime bridge 都是 C++ 实现 |
| Python 能力 | 用 Python 脚本提取 runtime JSON / PTX，辅助端到端验证 |

这个岗位还提到量化压缩、Triton、Cutlass/Cute、指令选择/调度。你的项目没有直接做这些，所以不要硬说做过。更好的说法是：

> 我现在这个项目主要覆盖 MLIR 图/算子 lowering 和 GPU runtime 执行链路，还没有深入到量化压缩和底层指令调度。但我理解它们在整体 AI 编译器中的位置：量化属于 graph/operator level 的数值表示优化，Triton/Cutlass/Cute 属于高性能 kernel 生成和模板化实现，LLVM 后端的指令选择/调度则更靠近机器码生成。我这个项目为继续往这些方向扩展打了基础。

## 17. 面这个岗位时最该讲的 5 个重点

### 重点 1：你理解 AI 编译器不是单一层

可以这样说：

> 我理解 AI 编译器通常分几层：前端图表示，比如 PyTorch FX、torch.compile、StableHLO；中间层做 shape、fusion、layout、parallel strategy；后端层做 kernel generation，比如 Triton、GPU dialect、LLVM/NVVM；最后 runtime 负责 launch、memory 和 profiling。我这个项目主要实践的是中间 IR 到 GPU/runtime 的这段链路。

### 重点 2：你做过 MLIR Dialect 和 pass pipeline

可以这样说：

> 我用 TableGen 定义了 `NorthStar` 的 op/type/attr，比如带 `device_id` 的 tensor type、数据并行属性和 `device_region`。然后写 C++ pass 做 IR rewrite 和 lowering，包括数据并行切分、device region outline、Linalg lowering、GPU lowering 和 runtime stub 生成。

### 重点 3：你碰过 GPU/NVVM/PTX 这条路

可以这样说：

> 项目里我把 outlined kernel 映射到 MLIR GPU dialect，生成 `gpu.module`、`gpu.func`、`gpu.launch_func`，再通过 MLIR 的 GPU-to-NVVM pipeline 生成 PTX。这让我对 GPU 编译路径中 IR 分层、kernel entry、block/grid metadata、PTX 产物都有实践。

### 重点 4：你不只会 IR，也考虑 runtime

可以这样说：

> 编译器产物不只是代码，还需要 runtime metadata。所以我设计了 JSON descriptor，记录 kernel name、descriptor id、device id、输入输出 shape/type 和 C API。runner 加载 JSON 和 PTX 后才能正确 launch。这部分体现的是编译期和运行期接口设计。

### 重点 5：你知道自己项目和 Triton/Cutlass 的关系

可以这样说：

> Triton 和 Cutlass 更偏高性能 kernel 生成。我的项目更偏编译器框架和 lowering pipeline。如果继续扩展，我会考虑两条路：一条是把 `device_region` lowering 到 Triton 风格 kernel；另一条是针对 matmul/attention 这类算子，生成 Cutlass/Cute 调用或用它们做 kernel backend。

## 18. 这个岗位可能问你的问题

### Q1：你这个项目和 torch.compile / Triton 有什么关系？

回答：

> torch.compile 更偏 PyTorch 前端图捕获和图优化，Triton 更偏 GPU kernel DSL 和代码生成。我的项目是一个 MLIR 编译器原型，位置更像中间层和后端桥接：先用自定义 IR 表达 tensor、device 和 parallelism，再 lowering 到 GPU/NVVM/PTX。它没有替代 Triton，而是可以理解为未来能接 Triton/CUDA backend 的一层 compiler IR pipeline。

### Q2：为什么用 MLIR？

回答：

> MLIR 很适合 AI 编译器，因为它允许多层 IR 共存。高层可以表达 tensor 和 graph 语义，中层可以表达 linalg、scf、bufferization，后端可以表达 gpu、llvm、nvvm。我的项目正好利用了这个分层能力：高层用 `NorthStar` 表达设备和并行语义，后面逐步降到 Linalg、GPU 和 NVVM。

### Q3：你的 softmax lowering 是高性能实现吗？

稳妥回答：

> 当前项目重点不是极致优化 softmax kernel，而是打通从高层 op 到 GPU/runtime 的编译链路。softmax 作为代表性 AI 算子，用来验证 shape、device region、kernel outline、GPU lowering 和 runtime descriptor 是否能贯通。如果要做性能优化，下一步会考虑 tiling、shared memory、warp-level reduction、vectorization，或者接入 Triton/Cutlass 风格 kernel。

### Q4：你做过指令选择和指令调度吗？

稳妥回答：

> 我没有手写过完整 LLVM 后端里的 instruction selection 和 scheduling pass。我的项目目前到 GPU/NVVM/PTX 层，主要是 MLIR 到 LLVM/NVVM 这条路径。但我理解指令选择是把 LLVM IR 或机器无关 IR 映射到目标机器指令，指令调度是根据依赖、latency、资源约束重排指令以提高吞吐。这个岗位如果需要，我可以从 LLVM 后端或者 MLIR LLVM dialect 继续往下深入。

### Q5：量化压缩你做过吗？

稳妥回答：

> 这个项目没有直接实现量化压缩。但我理解量化在编译器里一般涉及 dtype 转换、scale/zero point 传播、校准参数、算子替换和后端 kernel 适配。我的 `NSTensorType` 和 lowering pipeline 可以继续扩展 dtype 和 quantized attr，比如引入 int8/fp8 tensor type、量化参数属性，再在 lowering 时选择对应 runtime/kernel。

### Q6：你项目里最像真实 AI 编译器的地方是什么？

回答：

> 最像真实 AI 编译器的是分层和边界设计：高层 IR 表达 tensor/device/parallelism，pass pipeline 逐步做 lowering，device region 被 outline 成 kernel，后端生成 GPU/PTX，runtime descriptor 连接编译期和执行期。这几个环节都是大模型编译器里真实存在的问题。

### Q7：如果让你把这个项目扩展成大模型编译器，你会怎么做？

回答：

> 我会先补三块。第一是前端接入，比如从 PyTorch FX、ONNX 或 StableHLO 导入，而不是手写 NorthStar IR。第二是算子和优化，补 matmul、attention、layernorm，并做 fusion、tiling、layout transform。第三是高性能后端，接 Triton/Cutlass/Cute 或继续走 MLIR GPU/NVVM，同时加 benchmark 和 profiling。最后再加入量化 dtype 和 runtime memory planning。

## 19. 针对 JD 的自我介绍版本

> 我之前做过一个基于 MLIR 的 AI 编译器原型，主要围绕自定义 `NorthStar` 方言和 GPU lowering pipeline。项目里我用 C++ 和 TableGen 定义了 tensor type、设备属性、数据并行属性和 `device_region`，然后实现 pass pipeline，把高层张量计算逐步 lower 到 Linalg、SCF、GPU/NVVM，最后生成 PTX。为了验证端到端执行，我还设计了 runtime JSON descriptor 和 CUDA runner，让编译结果可以被 runtime 加载执行。
>
> 这个项目和岗位里的图编译器、算子编译器、LLVM GPGPU 编译比较相关。虽然我还没有直接做 Triton/Cutlass 或量化压缩，但我理解它们在 AI 编译器栈里的位置，也希望在实习里继续深入高性能 kernel 生成、量化和更底层的 GPU 编译优化。

## 20. 面试时的取舍

优先讲：

```text
MLIR / Dialect / Pass / Lowering / GPU / NVVM / PTX / Runtime
```

少讲：

```text
H350
tutorial 学习过程
普通 CMake 结构
过细的文件名
```

不要硬吹：

```text
我精通 Triton
我做过完整量化压缩
我做过 LLVM 后端指令调度
这是生产级大模型编译器
```

更稳的表达是：

> 我已经实践过 AI 编译器的 MLIR lowering 和 GPU runtime 链路，接下来希望进一步深入 Triton/Cutlass、量化压缩和 LLVM GPGPU 后端优化。

## 21. 测试工具是怎么构建的

这个项目的测试体系主要是基于：

```text
CMake + LLVM lit + FileCheck + 自定义 NS-opt
```

可以这样概括：

```text
CMake 配置项目
  -> 构建自定义 NS-opt 工具
  -> 构建 runtime smoke 可执行文件
  -> 生成 lit.site.cfg.py
  -> 注册 check-ch 测试 target
  -> lit 扫描 test/ 下的 .mlir / .py 文件
  -> 执行 RUN 命令
  -> 用 FileCheck 校验输出
```

### 关键文件

```text
my_mlir/test/CMakeLists.txt
my_mlir/test/lit.cfg.py
my_mlir/test/lit.site.cfg.py.in
my_mlir/src/NS-opt/CMakeLists.txt
```

### `NS-opt` 怎么构建

`NS-opt` 是这个项目自定义版的 `mlir-opt`，在：

```text
my_mlir/src/NS-opt/CMakeLists.txt
```

里通过 `add_mlir_tool` 构建：

```cmake
add_mlir_tool(NS-opt
  NS-opt.cpp

  DEPENDS
  ${LIBS}
)
```

它链接了项目自己的 dialect、transform、conversion 和 runtime utility：

```text
MLIRNorthStarDialect
MLIRNorthStarTransforms
MLIRNorthStarToLinalg
MLIRTutorialUtils
```

所以测试里可以直接用 `NS-opt` 跑自定义 pass pipeline。

面试说法：

> 我没有直接改官方 `mlir-opt`，而是做了一个项目自己的 `NS-opt`。它注册了 `NorthStar` dialect 和相关 pass，测试时通过这个工具来驱动 lowering pipeline。

### `check-ch` 怎么构建

测试入口在：

```text
my_mlir/test/CMakeLists.txt
```

里面定义了：

```cmake
set(TEST_DEPENDS
  NS-opt
)

add_lit_testsuite(check-ch "Running the lit regression tests..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
)
```

这表示：

- `check-ch` 是整个项目的 lit 测试 target
- 它依赖 `NS-opt`
- 跑测试前会先确保 `NS-opt` 已经构建完成

执行命令是：

```bash
cmake --build build --target check-ch
```

### lit 配置做了什么

`lit.site.cfg.py.in` 是 CMake 模板，配置构建路径：

```text
LLVM tools 路径
MLIR binary 路径
NS-opt 路径
动态库路径
Python 路径
```

CMake configure 后会生成真正的：

```text
build/my_mlir/test/lit.site.cfg.py
```

然后它加载：

```text
my_mlir/test/lit.cfg.py
```

`lit.cfg.py` 里主要做几件事：

1. 指定测试文件后缀：

```python
config.suffixes = ['.mlir', ".py"]
```

2. 把工具路径加入 `PATH`。

3. 注册测试里可以使用的工具：

```text
ns-opt
h350-runtime-smoke
cuda-runtime-smoke
cuda-runtime-artifacts-smoke
FileCheck
count
not
```

所以测试文件里可以直接写：

```mlir
// RUN: ns-opt %s --convert-north-satr-to-linalg | FileCheck %s
// CHECK: ...
```

或者 runtime smoke test：

```python
# RUN: cuda-runtime-smoke | FileCheck %s
# CHECK: descriptor=11 output=6.0,8.0
```

### 测试覆盖哪些内容

测试分三类：

```text
test/NorthStar/     # 方言级 transform 测试
test/Conversion/    # lowering pass 测试
test/Runtime/       # runtime smoke 测试
```

#### 方言级测试

例如：

```text
apply-distribute-transform.mlir
device-region-fusion.mlir
mark-distribute-parallel-parameters.mlir
```

验证数据并行标记、切分和 device region fusion 是否符合预期。

#### Conversion 测试

例如：

```text
north_star_to_linalg.mlir
device_kernels_to_gpu.mlir
generate_runtime_json_cuda.mlir
runtime_stubs_to_cuda_calls.mlir
```

验证 lowering pipeline 是否生成期望的 Linalg、GPU、runtime stub 和 JSON metadata。

#### Runtime 测试

例如：

```text
cuda_runtime_smoke.py
h350_runtime_smoke.py
cuda_runtime_artifacts_smoke.py
```

这些 `.py` 文件本身不是复杂 Python 脚本，而是 lit 测试文件，通过 `RUN:` 调用已经构建好的 smoke binary，再用 `FileCheck` 检查输出。

### 面试推荐回答

如果面试官问“你们测试怎么做的”，可以答：

> 我用 LLVM 生态常见的 lit + FileCheck 搭了回归测试。CMake 里定义了 `check-ch` target，它依赖自定义 `NS-opt`。`NS-opt` 类似项目自己的 `mlir-opt`，注册了 `NorthStar` dialect 和 pass。lit 会扫描 `test/` 下的 `.mlir` 和 `.py` 文件，执行里面的 `RUN:` 命令，然后用 `FileCheck` 检查输出 IR 或 runtime 输出。测试覆盖三块：方言级 transform、conversion lowering 和 runtime smoke test。

如果继续追问“为什么用 lit/FileCheck”，可以答：

> 因为 MLIR pass 的输出本质上是文本 IR，`FileCheck` 很适合检查关键结构是否出现，比如 `gpu.module`、`gpu.func`、`runtime_json`、`descriptor_id` 等。它比写完整 C++ 单元测试更轻量，也符合 LLVM/MLIR 社区常见做法。
