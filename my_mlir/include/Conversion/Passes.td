#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD
include "mlir/Pass/PassBase.td"

def ConvertNorthStarToLinalgPass : Pass<"convert-north-satr-to-linalg","::mlir::ModuleOp"> {
  let summary = "标记全局并行参数";
  let description = [{
    "标记全局并行参数。";
  }];
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect"
  ];
}

def OutlineNorthStarDeviceKernelsPass : Pass<"outline-north-star-device-kernels","::mlir::ModuleOp"> {
  let summary = "outline north_star.device_region into func.func kernels";
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::func::FuncDialect"
  ];
}

def LowerNorthStarDeviceKernelsToLoopsPass : Pass<"lower-north-star-device-kernels-to-loops","::mlir::ModuleOp"> {
  let summary = "bufferize outlined device kernels and lower them to scf loops";
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::math::MathDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def LowerNorthStarDeviceKernelsToGPUPass : Pass<"lower-north-star-device-kernels-to-gpu","::mlir::ModuleOp"> {
  let summary = "mirror outlined device kernels into gpu.module/gpu.func containers";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::func::FuncDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::north_star::NorthStarDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def LowerNorthStarHostToNPUCallsPass : Pass<"lower-north-star-host-to-npu-calls","::mlir::ModuleOp"> {
  let summary = "rewrite outlined kernel calls into explicit H350 NPU launch ops";
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::func::FuncDialect"
  ];
}

def SerializeNorthStarNPUKernelsPass : Pass<"serialize-north-star-npu-kernels","::mlir::ModuleOp"> {
  let summary = "serialize H350 NPU kernel metadata into module attributes";
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::func::FuncDialect"
  ];
}

def LowerNorthStarNPUCallsToRuntimePass : Pass<"lower-north-star-npu-calls-to-runtime","::mlir::ModuleOp"> {
  let summary = "rewrite north_star.npu_call into runtime stub func.call ops";
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::func::FuncDialect"
  ];
}

def MaterializeNorthStarRuntimeStubsPass : Pass<"materialize-north-star-runtime-stubs","::mlir::ModuleOp"> {
  let summary = "materialize mock runtime stub bodies that dispatch to kernel funcs";
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::func::FuncDialect"
  ];
}

def LowerNorthStarRuntimeStubsToH350CallsPass : Pass<"lower-north-star-runtime-stubs-to-h350-calls","::mlir::ModuleOp"> {
  let summary = "rewrite runtime stubs into external H350 runtime bridge calls";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::north_star::NorthStarDialect",
    "::mlir::func::FuncDialect"
  ];
}

def LowerNorthStarRuntimeStubsToCUDACallsPass : Pass<"lower-north-star-runtime-stubs-to-cuda-calls","::mlir::ModuleOp"> {
  let summary = "rewrite runtime stubs into external CUDA runtime bridge calls";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::north_star::NorthStarDialect",
    "::mlir::func::FuncDialect"
  ];
}

def GenerateNorthStarRuntimeJsonPass : Pass<"generate-north-star-runtime-json","::mlir::ModuleOp"> {
  let summary = "materialize runtime descriptor metadata as a JSON string attr";
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::func::FuncDialect"
  ];
}


#endif  // CONVERSION_PASSES_TD
