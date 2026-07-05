// RUN: ns-opt %s --convert-north-satr-to-linalg --outline-north-star-device-kernels --lower-north-star-device-kernels-to-loops --lower-north-star-host-to-npu-calls --lower-north-star-npu-calls-to-runtime --split-input-file | FileCheck %s
module @NorthStar {
  // CHECK-DAG: func.func private @north_star_runtime_launch_softmax_2_d_d_softmax_2_d_d__device_1
  // CHECK-DAG: attributes {device_id = 1 : i64, north_star.kernel_name = "softmax_2_d_d_softmax_2_d_d__device_1", north_star.runtime_stub, north_star.target = "amd_h350_npu"}
  // CHECK-DAG: func.func private @north_star_runtime_launch_softmax_2_d_d_softmax_2_d_d__device_2
  // CHECK-DAG: attributes {device_id = 2 : i64, north_star.kernel_name = "softmax_2_d_d_softmax_2_d_d__device_2", north_star.runtime_stub, north_star.target = "amd_h350_npu"}
  // CHECK: func.func @main
  // CHECK: %[[ARG0:.+]] = builtin.unrealized_conversion_cast
  // CHECK: %[[CALL0:.+]] = call @north_star_runtime_launch_softmax_2_d_d_softmax_2_d_d__device_1(%[[ARG0]])
  // CHECK: builtin.unrealized_conversion_cast
  // CHECK: call @north_star_runtime_launch_softmax_2_d_d_softmax_2_d_d__device_2(
  // CHECK-NOT: north_star.npu_call
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
