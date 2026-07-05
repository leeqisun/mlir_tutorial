#ifndef INTERFACES_DISTRIBUTED_PARALLELISM_INTERFACES_H
#define INTERFACES_DISTRIBUTED_PARALLELISM_INTERFACES_H
#include "Dialect/NorthStar/IR/NorthStarEunms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#define FIX
#include "Interfaces/DistributeParallelismAttrInterfaces.h.inc"
#include "Interfaces/DistributeParallelismOpInterfaces.h.inc"
#undef FIX
#endif  // INTERFACES_DISTRIBUTED_PARALLELISM_INTERFACES_H