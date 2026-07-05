set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRRegisterAllPasses
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions



  # TODO: Remove when registerAllGPUToLLVMIRTranslations is no longer
  # registered directly in mlir-opt.cpp.
  MLIRToLLVMIRTranslationRegistration
  MLIRNorthStarDialect
  MLIRNorthStarTransforms
  MLIRTutorialUtils
  MLIRCAPIDebug
  MLIRNorthStarToLinalg
)

add_mlir_tool(NS-opt
  NS-opt.cpp

  DEPENDS
  ${LIBS}
)

# 覆盖全局输出目录：放到当前子目录对应的 build 路径
set_target_properties(NS-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)


target_link_libraries("NS-opt" PRIVATE ${LIBS})
llvm_update_compile_flags(NS-opt)

mlir_check_all_link_libraries(NS-opt)
export_executable_symbols_for_plugins(NS-opt)