/home/lqy/mlir-tutorial/build/my_mlir/src/NS-opt/NS-opt \
  /home/lqy/mlir-tutorial/my_mlir/test/Conversion/device_kernel_to_loops.mlir \
  --convert-north-satr-to-linalg \
  --outline-north-star-device-kernels \
  --lower-north-star-device-kernels-to-loops
initializing north_star
register north_star  Type
register north_star  Attr
register north_star  Op
module @NorthStar {
  func.func @main(%arg0: !north_star.ns_tensor<5x?x?xf32,0>) -> !north_star.ns_tensor<5x?x?xf32,0> attributes {dp_attr = #north_star.DP<DP = 3 : 0, 1, 2>, host_func} {
    %0:3 = "north_star.buffer_cast"(%arg0) <{distribute_attr = #north_star.DP<DP = 3 : 0, 1, 2>}> : (!north_star.ns_tensor<5x?x?xf32,0>) -> (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<2x?x?xf32,1>, !north_star.ns_tensor<2x?x?xf32,2>)
    %1 = "north_star.softmax"(%0#0) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,0>) -> !north_star.ns_tensor<1x?x?xf32,0>
    %2 = "north_star.softmax"(%1) <{axis = 1 : i64}> : (!north_star.ns_tensor<1x?x?xf32,0>) -> !north_star.ns_tensor<1x?x?xf32,0>
    %3 = builtin.unrealized_conversion_cast %0#1 : !north_star.ns_tensor<2x?x?xf32,1> to tensor<2x?x?xf32>
    %4 = call @softmax_2_d_d_softmax_2_d_d__device_1(%3) : (tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
    %5 = builtin.unrealized_conversion_cast %4 : tensor<2x?x?xf32> to !north_star.ns_tensor<2x?x?xf32,1>
    %6 = builtin.unrealized_conversion_cast %0#2 : !north_star.ns_tensor<2x?x?xf32,2> to tensor<2x?x?xf32>
    %7 = call @softmax_2_d_d_softmax_2_d_d__device_2(%6) : (tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
    %8 = builtin.unrealized_conversion_cast %7 : tensor<2x?x?xf32> to !north_star.ns_tensor<2x?x?xf32,2>
    %9 = "north_star.buffer_cast"(%2, %5, %8) <{distribute_attr = #north_star.DP<DP = 3 : 0, 1, 2>}> : (!north_star.ns_tensor<1x?x?xf32,0>, !north_star.ns_tensor<2x?x?xf32,1>, !north_star.ns_tensor<2x?x?xf32,2>) -> !north_star.ns_tensor<5x?x?xf32,0>
    return %9 : !north_star.ns_tensor<5x?x?xf32,0>
  }
  func.func private @softmax_2_d_d_softmax_2_d_d__device_1(%arg0: tensor<2x?x?xf32>) -> tensor<2x?x?xf32> attributes {device_id = 1 : i64, north_star.kernel} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFFC00000 : f32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_buffer %arg0 : tensor<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %1 = bufferization.to_buffer %arg0 : tensor<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %2 = bufferization.to_buffer %arg0 : tensor<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %3 = bufferization.to_buffer %arg0 : tensor<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %4 = bufferization.to_buffer %arg0 : tensor<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %dim = memref.dim %4, %c1 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %dim_1 = memref.dim %3, %c2 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %alloc = memref.alloc(%dim, %dim_1) {alignment = 64 : i64} : memref<2x?x?xf32>
    %dim_2 = memref.dim %2, %c2 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %alloc_3 = memref.alloc(%dim_2) {alignment = 64 : i64} : memref<2x?xf32>
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_2 step %c1 {
        memref.store %cst_0, %alloc_3[%arg1, %arg2] : memref<2x?xf32>
      }
    }
    %dim_4 = memref.dim %1, %c1 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %dim_5 = memref.dim %1, %c2 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_4 step %c1 {
        scf.for %arg3 = %c0 to %dim_5 step %c1 {
          %6 = memref.load %1[%arg1, %arg2, %arg3] : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
          %7 = memref.load %alloc_3[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.maxnumf %6, %7 : f32
          memref.store %8, %alloc_3[%arg1, %arg3] : memref<2x?xf32>
        }
      }
    }
    %dim_6 = memref.dim %0, %c1 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %dim_7 = memref.dim %0, %c2 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_6 step %c1 {
        scf.for %arg3 = %c0 to %dim_7 step %c1 {
          %6 = memref.load %0[%arg1, %arg2, %arg3] : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
          %7 = memref.load %alloc_3[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.subf %6, %7 : f32
          %9 = math.exp %8 : f32
          memref.store %9, %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
        }
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_2 step %c1 {
        memref.store %cst, %alloc_3[%arg1, %arg2] : memref<2x?xf32>
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_3[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.addf %6, %7 : f32
          memref.store %8, %alloc_3[%arg1, %arg3] : memref<2x?xf32>
        }
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_3[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.divf %6, %7 : f32
          memref.store %8, %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
        }
      }
    }
    %alloc_8 = memref.alloc(%dim, %dim_1) {alignment = 64 : i64} : memref<2x?x?xf32>
    %alloc_9 = memref.alloc(%dim_1) {alignment = 64 : i64} : memref<2x?xf32>
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_1 step %c1 {
        memref.store %cst_0, %alloc_9[%arg1, %arg2] : memref<2x?xf32>
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_9[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.maxnumf %6, %7 : f32
          memref.store %8, %alloc_9[%arg1, %arg3] : memref<2x?xf32>
        }
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_9[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.subf %6, %7 : f32
          %9 = math.exp %8 : f32
          memref.store %9, %alloc_8[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
        }
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_1 step %c1 {
        memref.store %cst, %alloc_9[%arg1, %arg2] : memref<2x?xf32>
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc_8[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_9[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.addf %6, %7 : f32
          memref.store %8, %alloc_9[%arg1, %arg3] : memref<2x?xf32>
        }
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc_8[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_9[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.divf %6, %7 : f32
          memref.store %8, %alloc_8[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
        }
      }
    }
    %5 = bufferization.to_tensor %alloc_8 : memref<2x?x?xf32> to tensor<2x?x?xf32>
    return %5 : tensor<2x?x?xf32>
  }
  func.func private @softmax_2_d_d_softmax_2_d_d__device_2(%arg0: tensor<2x?x?xf32>) -> tensor<2x?x?xf32> attributes {device_id = 2 : i64, north_star.kernel} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFFC00000 : f32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_buffer %arg0 : tensor<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %1 = bufferization.to_buffer %arg0 : tensor<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %2 = bufferization.to_buffer %arg0 : tensor<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %3 = bufferization.to_buffer %arg0 : tensor<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %4 = bufferization.to_buffer %arg0 : tensor<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %dim = memref.dim %4, %c1 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %dim_1 = memref.dim %3, %c2 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %alloc = memref.alloc(%dim, %dim_1) {alignment = 64 : i64} : memref<2x?x?xf32>
    %dim_2 = memref.dim %2, %c2 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %alloc_3 = memref.alloc(%dim_2) {alignment = 64 : i64} : memref<2x?xf32>
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_2 step %c1 {
        memref.store %cst_0, %alloc_3[%arg1, %arg2] : memref<2x?xf32>
      }
    }
    %dim_4 = memref.dim %1, %c1 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %dim_5 = memref.dim %1, %c2 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_4 step %c1 {
        scf.for %arg3 = %c0 to %dim_5 step %c1 {
          %6 = memref.load %1[%arg1, %arg2, %arg3] : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
          %7 = memref.load %alloc_3[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.maxnumf %6, %7 : f32
          memref.store %8, %alloc_3[%arg1, %arg3] : memref<2x?xf32>
        }
      }
    }
    %dim_6 = memref.dim %0, %c1 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    %dim_7 = memref.dim %0, %c2 : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_6 step %c1 {
        scf.for %arg3 = %c0 to %dim_7 step %c1 {
          %6 = memref.load %0[%arg1, %arg2, %arg3] : memref<2x?x?xf32, strided<[?, ?, ?], offset: ?>>
          %7 = memref.load %alloc_3[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.subf %6, %7 : f32
          %9 = math.exp %8 : f32
          memref.store %9, %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
        }
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_2 step %c1 {
        memref.store %cst, %alloc_3[%arg1, %arg2] : memref<2x?xf32>
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_3[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.addf %6, %7 : f32
          memref.store %8, %alloc_3[%arg1, %arg3] : memref<2x?xf32>
        }
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_3[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.divf %6, %7 : f32
          memref.store %8, %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
        }
      }
    }
    %alloc_8 = memref.alloc(%dim, %dim_1) {alignment = 64 : i64} : memref<2x?x?xf32>
    %alloc_9 = memref.alloc(%dim_1) {alignment = 64 : i64} : memref<2x?xf32>
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_1 step %c1 {
        memref.store %cst_0, %alloc_9[%arg1, %arg2] : memref<2x?xf32>
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_9[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.maxnumf %6, %7 : f32
          memref.store %8, %alloc_9[%arg1, %arg3] : memref<2x?xf32>
        }
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_9[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.subf %6, %7 : f32
          %9 = math.exp %8 : f32
          memref.store %9, %alloc_8[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
        }
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim_1 step %c1 {
        memref.store %cst, %alloc_9[%arg1, %arg2] : memref<2x?xf32>
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc_8[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_9[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.addf %6, %7 : f32
          memref.store %8, %alloc_9[%arg1, %arg3] : memref<2x?xf32>
        }
      }
    }
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %dim step %c1 {
        scf.for %arg3 = %c0 to %dim_1 step %c1 {
          %6 = memref.load %alloc_8[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
          %7 = memref.load %alloc_9[%arg1, %arg3] : memref<2x?xf32>
          %8 = arith.divf %6, %7 : f32
          memref.store %8, %alloc_8[%arg1, %arg2, %arg3] : memref<2x?x?xf32>
        }
      }
    }
    %5 = bufferization.to_tensor %alloc_8 : memref<2x?x?xf32> to tensor<2x?x?xf32>
    return %5 : tensor<2x?x?xf32>
  }
}
