// RUN: ns-opt %s --lower-north-star-npu-calls-to-runtime --materialize-north-star-runtime-stubs | sed '/^initializing north_star$/d;/^register north_star /d;/^destroying north_star$/d' | mlir-opt --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -e main -entry-point-result=f32 | FileCheck %s
module {
  // CHECK: 4.000000e+00
  func.func @main() -> f32 {
    %cst = arith.constant 3.000000e+00 : f32
    %0 = north_star.npu_call @h350_scalar_add1 target = "amd_h350_npu" device = 0(%cst) : (f32) -> f32
    return %0 : f32
  }

  func.func private @h350_scalar_add1(%arg0: f32) -> f32 attributes {device_id = 0 : i64, north_star.kernel, north_star.target = "amd_h350_npu"} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = arith.addf %arg0, %cst : f32
    return %0 : f32
  }
}
