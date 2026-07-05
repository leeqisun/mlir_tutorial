add_mlir_dialect_library(MLIRNorthStarTransforms
  ApplyDistributeTransform.cpp
  MarkDistributeParallelParameters.cpp
  DeviceRegionFusion.cpp
  
  DEPENDS
  MLIRNorthStarPassesIncGen

  LINK_LIBS PUBLIC 
  MLIRNorthStarDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  )
