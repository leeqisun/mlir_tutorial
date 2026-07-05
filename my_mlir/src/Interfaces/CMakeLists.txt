set(LLVM_OPTIONAL_SOURCES
    DistributeParallelismInterfaces.cpp
    FusionRegionInterfaces.cpp
)

add_mlir_library(MLIRDistributeParallelismInterfaces
    DistributeParallelismInterfaces.cpp

    DEPENDS
    MLIRDistributeParallelismInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
add_mlir_library(MLIRFusionRegionInterfaces
    FusionRegionInterfaces.cpp

    DEPENDS
    MLIRFusionRegionInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)