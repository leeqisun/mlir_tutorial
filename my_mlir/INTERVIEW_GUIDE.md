# Interview Guide

这份文档是给你面试时直接复述用的，不追求完整，只追求好讲、能打。

## 一句话介绍

我做了一个基于 MLIR 的自定义编译器原型，围绕 `NorthStar` 方言，把带有设备语义的高层张量计算，逐步 lowering 成 Linalg/GPU/runtime 调用，并打通了 CUDA 端到端执行验证。

## 30 秒版本

这个项目的重点不是单个算子，而是完整链路。我先定义了一个 `NorthStar` 方言，让张量类型和算子带上设备 ID、并行切分和 `device_region` 这类信息；然后通过多级 pass，把高层 IR 逐步 lowering 成 Linalg、loop、GPU 和 runtime stub；最后再生成 runtime JSON 和 PTX，由 runner 加载执行。也就是说，我同时做了方言、pass pipeline 和 runtime bridge，而不是只写了几个 rewrite pattern。

## 2 分钟版本

这个项目想解决的是“多设备语义怎么在 MLIR 里被表达并落到执行层”。我设计了 `NorthStar` 方言，里面有自定义的 `NSTensorType`、并行属性和 `device_region` 等 op，设备信息不是靠注释或者外部表保存，而是直接在 IR 里可见、可验证。

在编译流程上，我做了一个自定义 `NS-opt` 驱动，把自己的 pass 注册进去。主要链路是先把 `NorthStar` 高层 op 降到更标准的 tensor/linalg 形式，再把设备区域 outline 成独立 kernel，然后根据后端选择继续 lowering：一条路走 GPU dialect 和 NVVM/PTX，另一条路走 runtime stub 和 bridge。对 CUDA 这条路，我还生成了 runtime descriptor JSON，让运行时知道 kernel name、descriptor id、输入输出 shape/type 和 device id，最后由 runner 把 JSON 和 PTX 一起加载执行。

我觉得这个项目最有价值的地方有三个：第一，我把设备语义设计进了 IR，而不是后处理；第二，我做的是一条完整 pass pipeline，不是孤立 pass；第三，我考虑了编译器和 runtime 的边界，定义了比较清晰的 descriptor/bridge 接口。项目里也配了 `FileCheck` 测试、runtime smoke test 和端到端脚本，说明不是纸面设计。

## 你负责了什么

如果面试官问“你具体做了什么”，你可以按下面答：

- 设计并实现 `NorthStar` 自定义方言，包括 type、attr、op 和 verify 逻辑
- 实现分布式/多设备相关的 IR 变换，例如并行参数标记、设备区域融合、张量切分重组
- 实现从 `NorthStar` 到 Linalg/GPU/runtime 的 lowering pass
- 设计 runtime bridge，生成 CUDA/H350 风格的统一调用接口
- 生成 runtime JSON descriptor，并实现加载 PTX/descriptor 的运行时辅助代码
- 写 conversion test、runtime smoke test 和执行脚本

## 这项目的技术亮点

### 1. IR 里显式建模设备语义

不是只有 `tensor<f32>` 这种纯数据类型，而是有带设备信息的 `NSTensorType` 和 `device_region`。这让 pass 能直接依据设备语义做 rewrite、fusion 和 outline。

### 2. Kernel outline 做了编译阶段解耦

我把设备区域抽成独立 kernel function，再由 host 侧去调度。这样 host/device 的责任边界更清晰，也更接近真实编译器后端架构。

### 3. Runtime descriptor 设计

我没有把运行依赖写死在 pass 里，而是把 kernel 元数据导出成 JSON，包括：

- descriptor id
- kernel name
- target
- device id
- 输入输出 tensor 的 shape/type

这样 runtime 可以独立消费这些元数据，后续也更容易替换执行后端。

### 4. 编译器不是停在 IR dump

我把 CUDA 这条链路真正串起来了：`NorthStar IR -> GPU/NVVM -> PTX -> runtime JSON -> runner 执行`。这比只展示 lowering 结果更有说服力。

## 常见面试题怎么答

### Q1：你为什么要自定义方言，不直接用 linalg/tosa/stablehlo？

因为这个项目想表达的是设备语义和并行切分，不只是通用算子语义。标准 dialect 更适合表达通用计算，但不一定适合直接承载 `device_id`、`device_region`、分布策略这些概念。自定义方言可以把这些信息前置，让后续 pass 更容易分析和改写。

### Q2：你的 pipeline 怎么分层？

我会分成四层：

1. 高层语义层：`NorthStar` 方言表达设备和并行信息
2. 中间 lowering 层：转成 Linalg/Tensor/SCF 这类更标准的表示
3. 后端适配层：outline kernel，改写成 GPU func 或 runtime stub
4. 执行层：生成 descriptor，交给 CUDA/H350 runtime bridge 调用

### Q3：为什么要生成 runtime JSON？

因为执行阶段除了代码本身，还需要稳定的执行元数据。JSON descriptor 把 kernel identity、输入输出签名和设备信息显式化，runtime 不必重新理解完整 IR，只需要消费统一协议即可。

### Q4：这个项目最大的难点是什么？

最大的难点不是写 op，而是保证“IR 语义、pass 切分和 runtime 接口”三者对齐。尤其是 kernel outline 之后，host 侧调用、descriptor id、kernel name、device id 和 PTX entry 要保持一致，否则编译阶段和运行阶段会断开。

### Q5：如果继续做，你会往哪扩展？

- 扩更多 dtype 和多输入多输出 kernel
- 把 runtime descriptor schema 固化下来
- 补性能 benchmark 和 profiling
- 做更通用的后端抽象，不只支持 CUDA 原型
- 清理 tutorial 代码和生产代码边界

## 建议避免的讲法

- 不要把重点放在 `main.cpp` 的学习代码上
- 不要只说“我写了一个 MLIR 方言”
- 不要只讲 API 名字，要讲为什么这样分层
- 不要把 H350 讲成已经 fully productized，当前更像 runtime bridge 原型

## 最后的表达模板

你可以直接背这一段：

“这个项目里，我主要做的是一个基于 MLIR 的多设备编译器原型。我的思路是先通过 `NorthStar` 方言把设备信息、并行策略和高层算子语义编码进 IR，然后用自定义 pass 做分布式改写、kernel outline 和多级 lowering。对 CUDA 后端，我不仅把 IR 降到了 GPU/NVVM/PTX，还额外设计了 runtime bridge 和 descriptor JSON，让编译结果能被运行时稳定消费，最后通过 runner 完成端到端执行验证。这个项目让我比较完整地做到了方言设计、pass pipeline 和 runtime 接口协同。” 
