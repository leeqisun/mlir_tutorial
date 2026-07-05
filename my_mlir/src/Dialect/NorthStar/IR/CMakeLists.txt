add_mlir_dialect_library(MLIRNorthStarDialect
    NorthStarDialect.cpp
    NorthStarTypes.cpp
    NorthStarAttrs.cpp
    NorthStarOps.cpp

    DEPENDS
    NorthStarDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
    MLIRDistributeParallelismInterfaces
    
    
)