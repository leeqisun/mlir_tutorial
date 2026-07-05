set(LLVM_TARGET_DEFINITIONS NorthStarDialect.td)
# 生成NorthStar Dialect 的声明
mlir_tablegen(NorthStarDialect.h.inc --gen-dialect-decls -dialect=north_star)
# 生成NorthStar Dialect 的实现
mlir_tablegen(NorthStarDialect.cpp.inc --gen-dialect-defs -dialect=north_star)

set(LLVM_TARGET_DEFINITIONS NorthStarTypes.td)
# 生成NorthStar Type 的声明
mlir_tablegen(NorthStarTypes.h.inc -gen-typedef-decls -typedefs-dialect=north_star)
# 生成NorthStar Type 的实现
mlir_tablegen(NorthStarTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=north_star)

set(LLVM_TARGET_DEFINITIONS NorthStarAttrs.td)
# 生成NorthStar Attrs 的声明
mlir_tablegen(NorthStarAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=north_star)
# 生成NorthStar Attrs 的实现
mlir_tablegen(NorthStarAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=north_star)

set(LLVM_TARGET_DEFINITIONS NorthStarEunms.td)
# 生成NorthStar Eunms 的声明
mlir_tablegen(NorthStarEunms.h.inc -gen-enum-decls)
# 生成NorthStar Eunms 的实现
mlir_tablegen(NorthStarEunms.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS NorthStarOps.td)
# 生成NorthStar Op 的声明
mlir_tablegen(NorthStarOps.h.inc -gen-op-decls -dialect=north_star)
# 生成NorthStar Op 的实现
mlir_tablegen(NorthStarOps.cpp.inc -gen-op-defs -dialect=north_star)

# 将生成的命令们定义为为target
add_public_tablegen_target(NorthStarDialectIncGen)
