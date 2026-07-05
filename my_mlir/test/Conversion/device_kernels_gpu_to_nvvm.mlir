// RUN: ns-opt %s --convert-north-satr-to-linalg --outline-north-star-device-kernels --lower-north-star-device-kernels-to-loops --lower-north-star-device-kernels-to-gpu | sed '/^initializing north_star$/d;/^register north_star /d;/^destroying north_star$/d' | mlir-opt --allow-unregistered-dialect '--pass-pipeline=builtin.module(nvvm-attach-target{chip=sm_120 O=3 verify-target-arch=false},gpu.module(strip-debuginfo,convert-gpu-to-nvvm,convert-index-to-llvm{index-bitwidth=32},canonicalize,cse))' | FileCheck %s

module @NorthStar {
  // CHECK: gpu.module @northstar_cuda_kernels [#nvvm.target
  // CHECK-DAG: llvm.func @__nv_expf(f32) -> f32
  // CHECK-DAG: llvm.func @softmax_2_d_d_softmax_2_d_d__device_1__kernel
  // CHECK-DAG: nvvm.kernel
  // CHECK-DAG: llvm.func @softmax_2_d_d_softmax_2_d_d__device_2__kernel
  // CHECK-DAG: nvvm.kernel
  // CHECK: llvm.call @__nv_expf
  func.func @main(%arg0: !north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0> attributes {dp_attr = #north_star.DP<DP = 3 : 0, 1, 2>, host_func} {
    %0:3 = "north_star.buffer_cast"(%arg0) <{distribute_attr = #north_star.DP<DP = 3 : 0, 1, 2>}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<2x?x?xf32,1>, !north_star.ns_tensor<2x?x?xf32,2>)
    %1 = "north_star.softmax"(%0#0) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,0>) -> !north_star.ns_tensor<1x?x?xf32,0>
    %2 = "north_star.softmax"(%1) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,0>) -> !north_star.ns_tensor<1x?x?xf32,0>
    %3 = "north_star.device_region"(%0#1) <{device_id = 1 : i64, sym_name = "softmax_2_d_d_softmax_2_d_d_"}> ({
    ^bb0(%arg1: !north_star.ns_tensor<2x?x?xf32,1>):
      %52 = "north_star.softmax"(%arg1) <{axis = 1 : i64}> : (!north_star.ns_tensor<2x?x?xf32,1>) -> !north_star.ns_tensor<2x?x?xf32,1>
      %62 = "north_star.softmax"(%52) <{axis = 1 : i64}> : (!north_star.ns_tensor<2x?x?xf32,1>) -> !north_star.ns_tensor<2x?x?xf32,1>
      north_star.return %62 : !north_star.ns_tensor<2x?x?xf32,1>
    }) : (!north_star.ns_tensor<2x?x?xf32,1>) -> !north_star.ns_tensor<2x?x?xf32,1>
    %4 = "north_star.device_region"(%0#2) <{device_id = 2 : i64, sym_name = "softmax_2_d_d_softmax_2_d_d_"}> ({
    ^bb0(%arg1: !north_star.ns_tensor<2x?x?xf32,2>):
      %53 = "north_star.softmax"(%arg1) <{axis = 1 : i64}> : (!north_star.ns_tensor<2x?x?xf32,2>) -> !north_star.ns_tensor<2x?x?xf32,2>
      %63 = "north_star.softmax"(%53) <{axis = 1 : i64}> : (!north_star.ns_tensor<2x?x?xf32,2>) -> !north_star.ns_tensor<2x?x?xf32,2>
      north_star.return %63 : !north_star.ns_tensor<2x?x?xf32,2>
    }) : (!north_star.ns_tensor<2x?x?xf32,2>) -> !north_star.ns_tensor<2x?x?xf32,2>
    %5 = "north_star.buffer_cast"(%2, %3, %4) <{distribute_attr = #north_star.DP<DP = 3 : 0, 1, 2>}> : (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<2x?x?xf32,1>, !north_star.ns_tensor<2x?x?xf32,2>) -> !north_star.ns_tensor<5x?x?xf32,0>
    return %5 : !north_star.ns_tensor<5x?x?xf32,0>
  }
}
