# my_mlir

`my_mlir` 是一个基于 MLIR 的小型编译器项目，核心目标是为自定义 `NorthStar` 方言提供一条从高层张量表达到底层设备执行的完整链路。项目覆盖了方言定义、IR 变换、kernel outline、GPU/NPU 风格 runtime bridge，以及 CUDA 端到端验证脚本。

如果你是拿它准备面试，可以把它理解成一句话：

“我做了一个自定义 MLIR 编译器原型，把 `NorthStar` 高层算子和多设备语义，逐步 lowering 成 Linalg/GPU/runtime 调用，并打通了 CUDA 运行验证链路。”

## 项目结构

```text
my_mlir/
├── include/
│   ├── Dialect/NorthStar/IR/         # 方言、类型、属性、Op 定义
│   ├── Dialect/NorthStar/Transforms/ # 方言级 pass 声明
│   ├── Conversion/                   # lowering pass 声明
│   ├── Interfaces/                   # 自定义接口
│   └── Utils/                        # runtime/API 辅助头文件
├── src/
│   ├── Dialect/NorthStar/IR/         # 方言实现
│   ├── Dialect/NorthStar/Transforms/ # 分布式变换、fusion
│   ├── Conversion/NorthStarToLinalg/ # lowering 主链路
│   ├── Utils/                        # CUDA/H350 runtime 支持
│   └── NS-opt/                       # 自定义 mlir-opt 驱动
├── test/
│   ├── NorthStar/                    # 方言/变换测试
│   ├── Conversion/                   # lowering 测试
│   └── Runtime/                      # runtime smoke test
├── run_cuda_pipeline.sh              # 生成 GPU/NVVM/PTX
├── run_cuda_on_5060.sh               # 端到端 CUDA 运行脚本
└── northstar_cuda_runner.cpp         # runtime JSON + PTX 执行器
```

## 这个项目解决什么问题

普通 MLIR tutorial 往往停留在：

- 定义一个方言
- 写几个简单 rewrite
- 做到标准 dialect lowering

这个项目比 tutorial 更进一步，重点在于把“设备语义”真正落到执行层：

- `NorthStar` 方言显式表达张量所在设备、并行切分和 `device_region`
- 通过 pass 把设备侧计算 outline 成独立 kernel
- 针对不同后端生成 runtime stub/bridge
- 为 CUDA 生成 runtime descriptor JSON 和 PTX
- 通过 runner 把 descriptor + PTX 真正加载并运行

## 核心组件

### 1. 自定义方言 `NorthStar`

关键文件：

- `include/Dialect/NorthStar/IR/NorthStarOps.td`
- `src/Dialect/NorthStar/IR/NorthStarDialect.cpp`
- `src/Dialect/NorthStar/IR/NorthStarOps.cpp`

核心能力：

- 自定义 `NSTensorType`，附带设备信息
- 自定义属性表达布局和并行语义
- 自定义算子如 `softmax`、`buffer_cast`、`device_region`、`npu_call`
- 对 op/type 做 verify，保证 IR 语义一致

面试里可以强调：

“我不是只做了一个语法层的 toy dialect，而是把设备 ID、张量切分和设备区域这些后端相关语义前置到了 IR 设计里。”

### 2. 方言级变换

关键文件：

- `src/Dialect/NorthStar/Transforms/MarkDistributeParallelParameters.cpp`
- `src/Dialect/NorthStar/Transforms/ApplyDistributeTransform.cpp`
- `src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp`

作用：

- 标记数据并行参数
- 按设备对张量做切分和重组
- 将相同 device 的计算融合到 `device_region`

可以这样讲：

“我把并行策略先编码成属性，再通过 pass 做 IR 级重写，这样策略和执行解耦，后续更容易换后端。”

### 3. Lowering 主链路

关键文件：

- `src/Conversion/NorthStarToLinalg/NorthStarToLinalgPass.cpp`
- `src/Conversion/NorthStarToLinalg/OutlineDeviceKernelsPass.cpp`
- `src/Conversion/NorthStarToLinalg/LowerDeviceKernelsToLoopsPass.cpp`
- `src/Conversion/NorthStarToLinalg/LowerDeviceKernelsToGPUPass.cpp`
- `src/Conversion/NorthStarToLinalg/LowerHostToNPUCallsPass.cpp`
- `src/Conversion/NorthStarToLinalg/LowerNPUCallsToRuntimePass.cpp`
- `src/Conversion/NorthStarToLinalg/LowerRuntimeStubsToCUDACallsPass.cpp`
- `src/Conversion/NorthStarToLinalg/GenerateRuntimeJsonPass.cpp`

整体链路可以概括成：

```text
NorthStar 高层 IR
-> Linalg/Tensor/SCF
-> 设备 kernel outline
-> GPU func / runtime stub
-> CUDA bridge / H350 bridge
-> runtime JSON / PTX / 执行
```

其中最值得讲的几个点：

- `OutlineDeviceKernelsPass`：把 `north_star.device_region` 抽成独立 `func.func kernel`
- `LowerDeviceKernelsToGPUPass`：生成 `gpu.module` / `gpu.func` / `gpu.launch_func`
- `LowerRuntimeStubsToCUDACallsPass`：把 runtime stub 改写成统一 CUDA C API 调用
- `GenerateRuntimeJsonPass`：导出 descriptor 元数据，给 runtime 层消费

### 4. Runtime 与后端桥接

关键文件：

- `src/Utils/CUDARuntime.cpp`
- `src/Utils/CUDARuntimeAPI.cpp`
- `src/Utils/CUDARuntimeArtifacts.cpp`
- `src/Utils/CUDADriverBackend.cpp`
- `src/Utils/H350Runtime.cpp`
- `src/Utils/H350RuntimeAPI.cpp`
- `northstar_cuda_runner.cpp`

这里的核心思路是：

- 编译阶段负责生成“可执行元数据”
- runtime 负责加载 descriptor、PTX 和 kernel entry
- 统一 launch API 负责真正调度设备执行

这部分很适合面试，因为它说明你不是只停在编译 IR，而是考虑了编译器和运行时的接口设计。

## `NS-opt` 是项目真正入口

文件：

- `src/NS-opt/NS-opt.cpp`

它本质上是一个自定义版 `mlir-opt`，负责：

- 注册 `NorthStar` 方言
- 注册自定义 transform/conversion pass
- 用命令行方式拼接 pass pipeline

这意味着你可以像标准 MLIR 工具链一样调试自己的 pass，例如：

```bash
build/my_mlir/src/NS-opt/NS-opt \
  my_mlir/test/Conversion/device_kernels_to_gpu_loops.mlir \
  --convert-north-satr-to-linalg \
  --outline-north-star-device-kernels \
  --lower-north-star-device-kernels-to-loops \
  --lower-north-star-device-kernels-to-gpu
```

## 可以怎么验证

### 1. 跑 lit 测试

```bash
cmake --build build --target check-ch
```

测试目录：

- `test/NorthStar`
- `test/Conversion`
- `test/Runtime`

### 2. 跑 CUDA lowering 管线

```bash
my_mlir/run_cuda_pipeline.sh
```

这个脚本会做三件事：

- 用 `NS-opt` 把 `NorthStar` lowering 到 GPU IR
- 用 `mlir-opt` 把 GPU IR 进一步降到 NVVM
- 尝试导出完整 PTX

### 3. 跑 CUDA 端到端执行

```bash
my_mlir/run_cuda_on_5060.sh
```

这个脚本会：

- 生成 PTX
- 生成 runtime JSON
- 提取 JSON 属性
- 调用 `northstar-cuda-runner` 执行指定 descriptor

## 面试时推荐强调的亮点

### 亮点 1：IR 设计不是只描述算子，还描述设备语义

你可以强调 `NSTensorType`、`device_region`、`device_id`、并行属性这些设计，让设备相关决策在 IR 层可见、可变换、可验证。

### 亮点 2：不是单点 pass，而是一条可串联的 lowering pipeline

这个项目已经有比较完整的阶段划分：

- 方言变换
- kernel outline
- loop/gpu lowering
- runtime stub 生成
- bridge 重写
- descriptor 导出

这比“只写一个 canonicalization pass”更有说服力。

### 亮点 3：考虑了编译器和 runtime 的边界

`GenerateRuntimeJsonPass` + `CUDARuntimeArtifacts` 这套设计，可以很好地说明你理解：

- 编译产物不只是代码
- 还包括执行元数据
- runtime 需要稳定、清晰的 ABI 和 descriptor 协议

### 亮点 4：有测试和 smoke 验证

很多候选人讲编译器项目只能讲“理论上能跑”，你这个项目至少已经做到：

- `FileCheck` 验证 IR 形态
- smoke test 验证 runtime 接口
- shell 脚本串起端到端流程

## 当前项目的不足

面试里主动承认这些点，反而显得成熟：

- 命名里有少量 tutorial/实验痕迹，例如 `convert-north-satr-to-linalg` 拼写问题
- `main.cpp` 更像学习代码，不是核心产品入口
- 某些 runtime 路径当前偏向 `f32`、单输入单输出等简化场景
- H350 后端目前更偏 bridge/接口原型，而不是完整可运行后端
- README 和开发文档原先缺失，不利于团队接手

如果面试官问“你接下来会怎么做”，可以答：

- 统一命名和 pass 入口
- 清理 tutorial 样例与主工程边界
- 扩展多输入/多输出和更多 dtype 支持
- 补 benchmark 与错误处理
- 补齐 runtime descriptor schema 文档

## 建议你怎么讲这段项目

推荐按这个顺序：

1. 先讲目标：做一个支持多设备语义的 MLIR 编译器原型
2. 再讲 IR：我设计了 `NorthStar` 方言，把设备和并行信息编码进类型/属性/Op
3. 再讲 pass：我把高层 IR 逐层 lowering 到 Linalg/GPU/runtime
4. 再讲 runtime：我定义了 bridge 和 descriptor，让编译产物能真正被 CUDA runtime 消费
5. 最后讲验证：我写了 conversion test、smoke test 和端到端脚本

如果你需要更直接的面试话术，见 [INTERVIEW_GUIDE.md](/home/lqy/mlir-tutorial/my_mlir/INTERVIEW_GUIDE.md)。
