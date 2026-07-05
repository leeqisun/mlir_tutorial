#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace llvm;

int main(int argc, char ** argv) {
  MLIRContext ctx;

  ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();

  // 创建 OpBuilder
  auto loc = UnknownLoc::get(&ctx);
  OpBuilder opBuilder(&ctx);
  auto mod = ModuleOp::create(loc);

  // 使用隐式位置构建器
  ImplicitLocOpBuilder builder(loc, opBuilder);
  builder.setInsertionPointToEnd(mod.getBody());

  // 创建 func
  auto i32 = builder.getI32Type();
  auto funcType = builder.getFunctionType({i32, i32}, {i32});
  auto func = func::FuncOp::create(loc, "test", funcType);
  mod.push_back(func);  // 将函数添加到模块

  // 添加基本块
  auto entry = func.addEntryBlock();
  auto args = entry->getArguments();

  // 设置插入点
  builder.setInsertionPointToEnd(entry);

  // 创建 arith.addi (位置自动管理)
  auto addi = arith::AddIOp::create(builder, args[0], args[1]);

  // 创建 func.return (位置自动管理)
  func::ReturnOp::create(builder, ValueRange({addi}));
  mod->print(llvm::outs());
  return 0;
}