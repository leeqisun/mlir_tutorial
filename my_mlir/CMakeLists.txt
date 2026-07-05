include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
# 告诉 CMake 优先使用 lld 作为链接器
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")

set(MY_LLVM_TOOLS_DIR "${CMAKE_SOURCE_DIR}/llvm-project/build/bin")
set(TUTORIA_TOOL_DIR "${CMAKE_BINARY_DIR}/my_mlir/src/NS-opt")
set(TUTORIA_NS_OPT "${CMAKE_BINARY_DIR}/my_mlir/src/NS-opt/NS-opt")


add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)
add_executable(CH-8 "main.cpp")
target_link_libraries(CH-8 
    MLIRNorthStarDialect
    MLIRNorthStarTransforms
    MLIRTutorialUtils
    MLIRGPUDialect 
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLinalgDialect
    MLIRParser
    )

add_executable(h350-runtime-smoke "runtime_smoke.cpp")
target_link_libraries(h350-runtime-smoke
    MLIRTutorialUtils
)

add_executable(cuda-runtime-smoke "cuda_runtime_smoke.cpp")
target_link_libraries(cuda-runtime-smoke
    MLIRTutorialUtils
)

add_executable(cuda-driver-runtime-smoke "cuda_driver_runtime_smoke.cpp")
target_link_libraries(cuda-driver-runtime-smoke
    MLIRTutorialUtils
)

add_executable(cuda-generated-ptx-smoke "cuda_generated_ptx_smoke.cpp")
target_link_libraries(cuda-generated-ptx-smoke
    MLIRTutorialUtils
)

add_executable(cuda-runtime-bundle-smoke "cuda_runtime_bundle_smoke.cpp")
target_link_libraries(cuda-runtime-bundle-smoke
    MLIRTutorialUtils
)

add_executable(cuda-runtime-artifacts-smoke "cuda_runtime_artifacts_smoke.cpp")
target_link_libraries(cuda-runtime-artifacts-smoke
    MLIRTutorialUtils
)

add_executable(northstar-cuda-runner "northstar_cuda_runner.cpp")
target_link_libraries(northstar-cuda-runner
    MLIRTutorialUtils
)
