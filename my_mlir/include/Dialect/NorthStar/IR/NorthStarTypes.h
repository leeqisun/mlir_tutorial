#ifndef DIALECT_NORTH_STAR_TYPES_H
#define DIALECT_NORTH_STAR_TYPES_H
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#define FIX
#define GET_TYPEDEF_CLASSES
#include "Dialect/NorthStar/IR/NorthStarTypes.h.inc"
#undef FIX

#endif  // DIALECT_NORTH_STAR_TYPES_H