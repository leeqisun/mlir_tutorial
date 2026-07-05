add_mlir_conversion_library(MLIRNorthStarToLinalg
  NorthStarToLinalg.cpp
  NorthStarToLinalgPass.cpp
  OutlineDeviceKernelsPass.cpp
  LowerDeviceKernelsToLoopsPass.cpp
  LowerDeviceKernelsToGPUPass.cpp
  LowerHostToNPUCallsPass.cpp
  SerializeNPUKernelsPass.cpp
  LowerNPUCallsToRuntimePass.cpp
  MaterializeRuntimeStubsPass.cpp
  LowerRuntimeStubsToH350CallsPass.cpp
  LowerRuntimeStubsToCUDACallsPass.cpp
  GenerateRuntimeJsonPass.cpp

  DEPENDS
  MLIRNorthStarConversionPassesIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRNorthStarDialect
  MLIRSCFDialect
  MLIRTransformUtils
  )
