# my_mlir 使用说明

这份文档说明如何构建、测试和运行 `my_mlir` 项目。项目核心是一个基于 MLIR 的 `NorthStar` 自定义方言和 lowering pipeline，可以把高层张量 IR 降到 Linalg、GPU/NVVM、PTX，并通过 runtime JSON + CUDA runner 做端到端验证。

## 1. 目录位置

假设当前仓库路径是：

```bash
/home/lqy/mlir-tutorial
```

项目主体在：

```bash
/home/lqy/mlir-tutorial/my_mlir
```

后续命令默认都在仓库根目录执行：

```bash
cd /home/lqy/mlir-tutorial
```

## 2. 依赖

需要准备：

```text
CMake
Ninja 或 Make
Clang/LLVM/MLIR
Python3
lit / FileCheck
CUDA Toolkit，可选，用于 CUDA/PTX 端到端路径
```

项目顶层 `CMakeLists.txt` 里使用：

```cmake
find_package(MLIR REQUIRED CONFIG)
```

所以配置 CMake 时需要能找到 MLIR 的 CMake package。常见方式是指定：

```bash
-DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
```

如果当前环境已经配置好 `MLIR_DIR`，可以省略。

## 3. 构建项目

推荐使用 Ninja：

```bash
cmake -S . -B build \
  -G Ninja \
  -DMLIR_DIR=/home/lqy/mlir-tutorial/llvm-project/build/lib/cmake/mlir

cmake --build build
```

如果你的 MLIR 安装路径不同，把 `MLIR_DIR` 改成自己的路径。

构建完成后，重要产物包括：

```text
build/my_mlir/src/NS-opt/NS-opt      # 自定义 mlir-opt 工具
build/northstar-cuda-runner          # CUDA runner
build/cuda-runtime-smoke             # CUDA runtime smoke test
build/h350-runtime-smoke             # H350 runtime smoke test
```

## 4. 单独构建 NS-opt

`NS-opt` 是项目自己的 `mlir-opt`，注册了 `NorthStar` dialect 和相关 pass。

只构建 `NS-opt`：

```bash
cmake --build build --target NS-opt
```

构建产物：

```bash
build/my_mlir/src/NS-opt/NS-opt
```

可以用它直接跑 pass：

```bash
build/my_mlir/src/NS-opt/NS-opt \
  my_mlir/test/Conversion/north_star_to_linalg.mlir \
  --convert-north-satr-to-linalg
```

注意：当前代码里的 pass 命令名是：

```bash
--convert-north-satr-to-linalg
```

这里 `satr` 是项目现有命令名，使用时按这个拼写。

## 5. 运行测试

项目测试基于：

```text
LLVM lit + FileCheck
```

运行全部测试：

```bash
cmake --build build --target check-ch
```

测试入口定义在：

```text
my_mlir/test/CMakeLists.txt
```

测试配置在：

```text
my_mlir/test/lit.cfg.py
my_mlir/test/lit.site.cfg.py.in
```

测试目录：

```text
my_mlir/test/NorthStar/     # 方言级 transform 测试
my_mlir/test/Conversion/    # lowering pass 测试
my_mlir/test/Runtime/       # runtime smoke 测试
```

测试文件里通常通过 `RUN:` 调用 `ns-opt`，再用 `FileCheck` 校验输出：

```mlir
// RUN: ns-opt %s --convert-north-satr-to-linalg | FileCheck %s
// CHECK: ...
```

## 6. 常用 lowering 命令

### 6.1 NorthStar 到 Linalg

```bash
build/my_mlir/src/NS-opt/NS-opt \
  my_mlir/test/Conversion/north_star_to_linalg.mlir \
  --convert-north-satr-to-linalg
```

作用：

```text
NorthStar 高层 op -> Linalg / Tensor / Math / Arith
```

### 6.2 device_region outline

```bash
build/my_mlir/src/NS-opt/NS-opt \
  my_mlir/test/Conversion/device_kernel_to_loops.mlir \
  --convert-north-satr-to-linalg \
  --outline-north-star-device-kernels
```

作用：

```text
north_star.device_region -> 独立 func.func kernel
```

### 6.3 lowering 到 loop

```bash
build/my_mlir/src/NS-opt/NS-opt \
  my_mlir/test/Conversion/device_kernels_to_gpu_loops.mlir \
  --convert-north-satr-to-linalg \
  --outline-north-star-device-kernels \
  --lower-north-star-device-kernels-to-loops
```

作用：

```text
Linalg / Tensor -> SCF / MemRef 风格 loop
```

### 6.4 lowering 到 GPU dialect

```bash
build/my_mlir/src/NS-opt/NS-opt \
  my_mlir/test/Conversion/device_kernels_to_gpu.mlir \
  --convert-north-satr-to-linalg \
  --outline-north-star-device-kernels \
  --lower-north-star-device-kernels-to-gpu
```

作用：

```text
outlined kernel -> gpu.module / gpu.func / gpu.launch_func
```

## 7. 运行 CUDA lowering pipeline

脚本：

```bash
my_mlir/run_cuda_pipeline.sh
```

默认输入：

```text
my_mlir/test/Conversion/device_kernels_to_gpu_loops.mlir
```

运行：

```bash
my_mlir/run_cuda_pipeline.sh
```

它会执行三步：

```text
1. 使用 NS-opt 把 NorthStar device kernel 降到 GPU dialect
2. 使用 mlir-opt 把 GPU dialect 降到 NVVM/LLVM
3. 尝试通过 gpu-lower-to-nvvm-pipeline 生成 PTX
```

默认输出在 `/tmp`：

```text
/tmp/northstar_gpu.mlir
/tmp/northstar_nvvm.mlir
/tmp/northstar_full_pipeline.mlir
/tmp/northstar_kernel.ptx
/tmp/northstar_gpu_pipeline.log
```

注意：生成 PTX 需要 CUDA Toolkit 里的 `libdevice.10.bc`。脚本会检查：

```text
CUDA_HOME
CUDA_PATH
/usr/local/cuda
/usr/lib/cuda
```

如果找不到 `libdevice.10.bc`，脚本会保留 GPU/NVVM IR，但跳过完整 PTX 生成。

## 8. 运行 CUDA 端到端执行

脚本：

```bash
my_mlir/run_cuda_on_5060.sh
```

运行：

```bash
my_mlir/run_cuda_on_5060.sh
```

这个脚本会：

```text
1. 调用 run_cuda_pipeline.sh 生成 PTX
2. 使用 NS-opt 生成 runtime bridge IR
3. 从 IR 中提取 runtime JSON
4. 调用 northstar-cuda-runner 执行指定 descriptor
```

默认 descriptor id 是：

```text
0
```

也可以指定输入：

```bash
my_mlir/run_cuda_on_5060.sh \
  my_mlir/test/Conversion/device_kernels_to_gpu_loops.mlir \
  0 \
  2 2 2
```

参数含义：

```text
第 1 个参数：输入 MLIR 文件
第 2 个参数：descriptor_id
第 3-5 个参数：输入 tensor 的 dim0 dim1 dim2
后续参数：输入数据 values
```

示例：

```bash
my_mlir/run_cuda_on_5060.sh \
  my_mlir/test/Conversion/device_kernels_to_gpu_loops.mlir \
  0 \
  2 2 2 \
  1 2 0 1 3 0.5 1 -1
```

## 9. 生成 runtime JSON

可以手动运行：

```bash
build/my_mlir/src/NS-opt/NS-opt \
  my_mlir/test/Conversion/generate_runtime_json_cuda.mlir \
  --convert-north-satr-to-linalg \
  --outline-north-star-device-kernels \
  --lower-north-star-device-kernels-to-loops \
  --lower-north-star-host-to-npu-calls \
  --lower-north-star-npu-calls-to-runtime \
  --lower-north-star-runtime-stubs-to-cuda-calls \
  --generate-north-star-runtime-json
```

输出 IR 的 module attribute 中会包含：

```text
north_star.runtime_json
```

JSON 里记录：

```text
target
c_api
descriptor_id
kernel_name
device_id
inputs shape/type
outputs shape/type
```

## 10. 运行 runtime smoke test

构建后可以直接运行 smoke binary：

```bash
build/cuda-runtime-smoke
build/h350-runtime-smoke
build/cuda-runtime-artifacts-smoke
```

也可以通过 lit 统一跑：

```bash
cmake --build build --target check-ch
```

## 11. 常见问题

### 11.1 找不到 MLIR_DIR

报错类似：

```text
Could not find a package configuration file provided by "MLIR"
```

解决：

```bash
cmake -S . -B build \
  -G Ninja \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
```

### 11.2 找不到 NS-opt

报错类似：

```text
missing NS-opt: build/my_mlir/src/NS-opt/NS-opt
```

先构建：

```bash
cmake --build build --target NS-opt
```

### 11.3 找不到 mlir-opt

`run_cuda_pipeline.sh` 默认查找：

```text
install/bin/mlir-opt
```

如果你的 `mlir-opt` 在别的位置，可以修改脚本里的：

```bash
MLIR_OPT="${ROOT_DIR}/install/bin/mlir-opt"
```

或者确保该路径存在。

### 11.4 找不到 libdevice.10.bc

说明 CUDA Toolkit 没配置好，或者脚本找不到 CUDA 路径。

可以设置：

```bash
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
```

然后重新运行：

```bash
my_mlir/run_cuda_pipeline.sh
```

### 11.5 FileCheck 找不到

`check-ch` 需要 LLVM 工具链里的 `FileCheck`。确认：

```bash
which FileCheck
```

如果找不到，需要把 LLVM build 的 `bin` 目录加入 `PATH`，或检查 `lit.site.cfg.py.in` 中的工具路径配置。

## 12. 面试时怎么讲使用流程

可以这样说：

> 这个项目通过 CMake 构建，核心工具是自定义的 `NS-opt`，它类似 `mlir-opt`，注册了 `NorthStar` 方言和所有 pass。测试通过 `check-ch` target 触发 lit + FileCheck，覆盖方言 transform、conversion lowering 和 runtime smoke。CUDA 路径通过 `run_cuda_pipeline.sh` 生成 GPU/NVVM/PTX，再通过 `run_cuda_on_5060.sh` 生成 runtime JSON 并调用 runner 执行，完成从高层 IR 到设备执行的端到端验证。

