// RUN: ns-opt %s --convert-north-satr-to-linalg --outline-north-star-device-kernels --lower-north-star-device-kernels-to-gpu --split-input-file | sed '/^initializing north_star$/d;/^register north_star /d;/^destroying north_star$/d' | FileCheck %s
module @NorthStar {
  // CHECK: module @NorthStar attributes {gpu.container_module}
  // CHECK: gpu.module @northstar_cuda_kernels attributes {north_star.target = "nvidia_rtx_5060_cuda"}
  // CHECK-DAG: gpu.func @softmax_2_d_d_softmax_2_d_d__device_1
  // CHECK-DAG: north_star.kernel_name = "softmax_2_d_d_softmax_2_d_d__device_1"
  // CHECK-DAG: north_star.target = "nvidia_rtx_5060_cuda"
  // CHECK-DAG: north_star.gpu_kernel_abi = "tensor_mirror"
  // CHECK-DAG: device_id = 1 : i64
  // CHECK-DAG: known_block_size = array<i32: 256, 1, 1>
  // CHECK-DAG: known_grid_size = array<i32: 1, 1, 1>
  // CHECK-DAG: gpu.return
  // CHECK-DAG: gpu.func @softmax_2_d_d_softmax_2_d_d__device_1__kernel(
  // CHECK-DAG: memref<2x?x?xf32>
  // CHECK-DAG: kernel attributes
  // CHECK-DAG: north_star.gpu_kernel_entry
  // CHECK-DAG: north_star.gpu_kernel_abi = "void_out_buffer"
  // CHECK-DAG: %[[TIN1:.*]] = bufferization.to_tensor
  // CHECK-DAG: linalg.fill
  // CHECK-DAG: bufferization.materialize_in_destination
  // CHECK-DAG: func.func private @softmax_2_d_d_softmax_2_d_d__device_1__gpu_launch(%[[ARG1:.*]]: tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  // CHECK-DAG: bufferization.to_buffer %[[ARG1]]
  // CHECK-DAG: %[[ALLOC1:.*]] = memref.alloc
  // CHECK-DAG: gpu.launch_func @northstar_cuda_kernels::@softmax_2_d_d_softmax_2_d_d__device_1__kernel blocks in
  // CHECK-DAG: args(%{{.*}} : memref<2x?x?xf32>, %[[ALLOC1]] : memref<2x?x?xf32>)
  // CHECK-DAG: bufferization.to_tensor %[[ALLOC1]]
  // CHECK-DAG: gpu.func @softmax_2_d_d_softmax_2_d_d__device_2
  // CHECK-DAG: north_star.kernel_name = "softmax_2_d_d_softmax_2_d_d__device_2"
  // CHECK-DAG: device_id = 2 : i64
  // CHECK-DAG: north_star.gpu_kernel_abi = "tensor_mirror"
  // CHECK-DAG: gpu.func @softmax_2_d_d_softmax_2_d_d__device_2__kernel(
  // CHECK-DAG: memref<2x?x?xf32>
  // CHECK-DAG: kernel attributes
  // CHECK-DAG: north_star.gpu_kernel_entry
  // CHECK-DAG: %[[TIN2:.*]] = bufferization.to_tensor
  // CHECK-DAG: linalg.fill
  // CHECK-DAG: bufferization.materialize_in_destination
  // CHECK-DAG: func.func private @softmax_2_d_d_softmax_2_d_d__device_2__gpu_launch(%[[ARG2:.*]]: tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  // CHECK-DAG: gpu.launch_func @northstar_cuda_kernels::@softmax_2_d_d_softmax_2_d_d__device_2__kernel blocks in
  // CHECK-DAG: call @softmax_2_d_d_softmax_2_d_d__device_1__gpu_launch
  // CHECK-DAG: call @softmax_2_d_d_softmax_2_d_d__device_2__gpu_launch
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
