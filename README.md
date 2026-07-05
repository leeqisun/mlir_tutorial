# Hands-On MLIR Tutorial & Custom Compiler Project

[![MLIR](https://img.shields.io/badge/Dialect-MLIR-orange.svg?style=flat-square)](https://mlir.llvm.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg?style=flat-square)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg?style=flat-square)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-Apache%202.0-red.svg?style=flat-square)](LICENSE)

这是一个一站式 **MLIR 编译技术学习与实践平台**。本仓库包含两大部分：

1. 📚 **MLIR 系统化实战教程**：从 MLIR 基础概念到 Dialect 转换、TableGen 语法、Pattern Rewrite 与类型转换的详细原理解析与实战教程。
2. 🚀 **`my_mlir` 编译器项目**：一个完整的、支持自定义方言 `NorthStar` 并端到端 Lowering 到 GPU 并在 CUDA 设备上实际运行验证的编译器原型。

---

## 📂 仓库目录结构

```text
.
├── my_mlir/                 # 🚀 自定义 NorthStar 编译器主工程
│   ├── include/             #   - 方言定义、变换与 lowering 头文件
│   ├── src/                 #   - 方言实现、编译 pass 与 runtime 桥接层
│   ├── test/                #   - lit 测试用例与 runtime smoke test
│   ├── USAGE.md             #   - 编译器端到端详细运行与调试手册
│   └── INTERVIEW_GUIDE.md   #   - 编译器项目面试话术与亮点提炼
│
├── TUTORIAL.md              # 📚 1800+ 行 MLIR 实践教程（原 README.md 移至此处）
│
├── CMakeLists.txt           # 整个仓库的 CMake 构建文件
├── LICENSE                  # 项目开源许可证
└── README.md                # 本文件（仓库全局入口门户）
```

---

## 📚 一、MLIR 系统化实战教程 (TUTORIAL.md)

本教程是专为快速上手 MLIR 并在实际工程中应用而编写的系统化教程，已移动至 **[TUTORIAL.md](file:///home/lqy/mlir-tutorial/TUTORIAL.md)**。

### 📖 教程核心大纲：
* **1. MLIR 简介**：编译管线、常见 Dialect 与 MLIR 的优缺点。
* **2. MLIR 基本用法**：IR 基本结构、工程模板配置与读入输出。
* **3. MLIR Op 的结构**：Attribute、Operand、Value 与 Type 的关系与设计。
* **4. MLIR 的类型转换**：Op 与 Type/Attribute 的转换管线实现。
* **5. MLIR 的图结构**：数据流图、控制流图的遍历与修改。
* **6. 基本的 Dialect 工程**：TableGen 工程模板与 IR 的定义实现。
* **7. TableGen Op 定义详解**：Attribute/Type 约束、Verifier 校验、AssemblyFormat 打印排版、Builder 自定义创建。
* **8. Dialect 转换管线详解**：Pattern Rewrite 匹配重写、Type Conversion 类型转换。

👉 **[立即阅读完整 MLIR 教程 ->](file:///home/lqy/mlir-tutorial/TUTORIAL.md)**

---

## 🚀 二、`my_mlir` 自定义多设备编译器项目 (my_mlir)

`my_mlir` 是本仓库的实战项目，实现了一个名为 `NorthStar` 的自定义高阶方言，它不仅具有普通的算子语义，还能在 IR 层描述**设备 ID**、**数据并行切分**与**设备执行区域**。

### 🌟 项目核心亮点
* **显式设备语义表达**：设计了自定义类型 `NSTensorType` 与方言算子，将硬件执行的物理信息在编译高阶直接编码。
* **方言级并行变换**：通过自定义 Pass（如 `DeviceRegionFusion` 和 `ApplyDistributeTransform`）将算子按目标设备进行自动切分和区域融合。
* **设备 Kernel Outline**：在 Lowering 管线中，将 `device_region` 计算区域自动抽取并降解为独立的 GPU module / GPU kernel。
* **双后端运行时桥接**：为 CUDA 和自研 NPU（H350 原型）构建了统一的运行时桥接 API，支持元数据描述文件（JSON descriptor）与可执行 PTX 文件的动态加载和调度运行。

### 🔄 Lowering 编译管线流向

```text
      NorthStar 高层算子与设备语义 IR
                 │
                 ▼  (convert-north-star-to-linalg)
      Linalg / Tensor / SCF 标准三方方言
                 │
                 ▼  (outline-north-star-device-kernels)
      独立封装的 GPU Kernel Module (gpu.module)
                 │
                 ▼  (lower-device-kernels-to-gpu & nvvm)
      LLVM / NVVM / GPU 汇编 IR 表达
                 │
                 ▼  (GenerateRuntimeJson & ptxas)
  编译产物：Runtime JSON Descriptor + GPU PTX 汇编代码
                 │
                 ▼
     [CUDA 执行器] -> 动态加载 PTX 并利用 CUDA Driver API 调度执行
```

👉 **[查看 `my_mlir` 编译器详细介绍 ->](file:///home/lqy/mlir-tutorial/my_mlir/README.md)**

---

## 🛠️ 三、快速上手与构建验证

### 1. 克隆与环境配置
项目依赖 LLVM 与 MLIR。推荐在包含 LLVM 编译环境的容器或本地环境下构建：
```bash
# 根目录下配置与编译
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_EXTERNAL_LIT=$(pwd)/build/bin/llvm-lit
cmake --build build -j$(nproc)
```

### 2. 运行自动化测试
项目中配备了完善的 `lit` 自动化 IR 验证测试：
```bash
cmake --build build --target check-ch
```

### 3. 端到端 CUDA lowering 与执行测试
如果您在带有 NVIDIA 显卡和 CUDA 驱动的环境下，可以直接运行以下脚本验证端到端编译与执行：
```bash
# 执行完整 lowering 管线并生成 PTX 与 Descriptor JSON
./my_mlir/run_cuda_pipeline.sh

# 运行端到端 CUDA 运行验证
./my_mlir/run_cuda_on_5060.sh
```

有关更高级的开发调试、Pass 管道设计及运行细节，请参阅 **[`my_mlir` 使用指南](file:///home/lqy/mlir-tutorial/my_mlir/USAGE.md)** 与 **[面试指引手册](file:///home/lqy/mlir-tutorial/my_mlir/INTERVIEW_GUIDE.md)**。
