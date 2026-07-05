#include "Dialect/NorthStar/IR/NorthStarDialect.h"

#include "llvm/Support/raw_ostream.h"
#define FIX
#include "Dialect/NorthStar/IR/NorthStarDialect.cpp.inc"
#undef FIX

namespace mlir::north_star {
// 实现方言的初始化方法
void NorthStarDialect::initialize() {
  llvm::outs() << "initializing " << getDialectNamespace() << "\n";
  registerTypes();
  registerAttrs();
  registerOps();
}

// 实现方言的析构函数
NorthStarDialect::~NorthStarDialect() {
  llvm::outs() << "destroying " << getDialectNamespace() << "\n";
}

// 实现在extraClassDeclaration 声明当中生命的方法。
void NorthStarDialect::sayHello() {
  llvm::outs() << "Hello in " << getDialectNamespace() << "\n";
}

}  // namespace mlir::north_star