// RUN: ns-opt %s --convert-north-satr-to-linalg --outline-north-star-device-kernels --lower-north-star-device-kernels-to-loops --lower-north-star-host-to-npu-calls --lower-north-star-npu-calls-to-runtime --lower-north-star-runtime-stubs-to-h350-calls --split-input-file | FileCheck %s
module @NorthStar {
  // CHECK-DAG: func.func private @north_star_h350_launch_f32(i64, tensor<*xf32>) -> tensor<*xf32> attributes {north_star.runtime_c_api, north_star.target = "amd_h350_npu"}
  // CHECK-DAG: func.func private @north_star_h350_launch_softmax_2_d_d_softmax_2_d_d__device_1(
  // CHECK-DAG: attributes {device_id = 1 : i64, north_star.descriptor_id = 0 : i64, north_star.kernel_name = "softmax_2_d_d_softmax_2_d_d__device_1", north_star.runtime_bridge, north_star.target = "amd_h350_npu"}
  // CHECK-DAG: %[[DESC0:.+]] = arith.constant 0 : i64
  // CHECK-DAG: %[[ERASE0:.+]] = tensor.cast %arg0 : tensor<2x?x?xf32> to tensor<*xf32>
  // CHECK-DAG: %[[RAW0:.+]] = call @north_star_h350_launch_f32(%[[DESC0]], %[[ERASE0]]) : (i64, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-DAG: %[[RANK0:.+]] = tensor.cast %[[RAW0]] : tensor<*xf32> to tensor<2x?x?xf32>
  // CHECK-DAG: return %[[RANK0]] : tensor<2x?x?xf32>
  // CHECK-DAG: func.func private @north_star_h350_launch_softmax_2_d_d_softmax_2_d_d__device_2(
  // CHECK-DAG: attributes {device_id = 2 : i64, north_star.descriptor_id = 1 : i64, north_star.kernel_name = "softmax_2_d_d_softmax_2_d_d__device_2", north_star.runtime_bridge, north_star.target = "amd_h350_npu"}
  // CHECK-DAG: %[[DESC1:.+]] = arith.constant 1 : i64
  // CHECK-DAG: %[[ERASE1:.+]] = tensor.cast %arg0 : tensor<2x?x?xf32> to tensor<*xf32>
  // CHECK-DAG: %[[RAW1:.+]] = call @north_star_h350_launch_f32(%[[DESC1]], %[[ERASE1]]) : (i64, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-DAG: %[[RANK1:.+]] = tensor.cast %[[RAW1]] : tensor<*xf32> to tensor<2x?x?xf32>
  // CHECK-DAG: return %[[RANK1]] : tensor<2x?x?xf32>
  // CHECK-DAG: func.func private @north_star_runtime_launch_softmax_2_d_d_softmax_2_d_d__device_1
  // CHECK-DAG: attributes {device_id = 1 : i64, north_star.descriptor_id = 0 : i64
  // CHECK-DAG: call @north_star_h350_launch_softmax_2_d_d_softmax_2_d_d__device_1
  // CHECK-DAG: func.func private @north_star_runtime_launch_softmax_2_d_d_softmax_2_d_d__device_2
  // CHECK-DAG: attributes {device_id = 2 : i64, north_star.descriptor_id = 1 : i64
  // CHECK-DAG: call @north_star_h350_launch_softmax_2_d_d_softmax_2_d_d__device_2
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
