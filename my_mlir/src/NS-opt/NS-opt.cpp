#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Config/mlir-config.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir-c/Debug.h"
#include "Conversion/Passes.h"


// 1. 实现opt 工具
// 2. 利用opt 运行pass
// '/home/lqy/mlir-tutorial/build/my_mlir/src/NS-opt/NS-opt' '/home/lqy/mlir-tutorial/my_mlir/test/softmax.mlir' 
// --apply-distribute-transform --mark-distribute-parallel-parameters="DP=5 TP=1"

// 3. 将IR dump 下来 [ir after and tree] && pm option=
// '/home/lqy/mlir-tutorial/build/my_mlir/src/NS-opt/NS-opt' '/home/lqy/mlir-tutorial/my_mlir/test/softmax.mlir'
// --mlir-print-ir-after-all --apply-distribute-transform


//--convert-north-satr-to-linalg  --reconcile-unrealized-casts --split-input-file > '/home/lfr/MLIR_Tutorial/log.txt'
// 4. debug 选项 debug\debug-only
// '/home/lqy/mlir-tutorial/build/my_mlir/src/NS-opt/NS-opt' '/home/lqy/mlir-tutorial/my_mlir/test/softmax.mlir' 
// --mark-distribute-parallel-parameters="DP=5 TP=1" --apply-distribute-transform --device-region-fusion --debug
int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::north_star::NorthStarDialect>();
  registerAllExtensions(registry);
  mlir::north_star::registerNorthStarOptPasses();
  mlir::north_star::registerNorthStarConversionPasses();
  // mlirEnableGlobalDebug(true);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NS modular optimizer driver\n", registry));
}
