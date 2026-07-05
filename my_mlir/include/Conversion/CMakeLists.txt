set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NorthStarConversion)
add_public_tablegen_target(MLIRNorthStarConversionPassesIncGen)
add_dependencies(mlir-headers MLIRNorthStarConversionPassesIncGen)
