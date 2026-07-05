cmake_minimum_required(VERSION 3.13.4)

project(mlir-toy VERSION 0.0.0)

# 防止 cmake 生成一堆 warning
cmake_policy(SET CMP0116 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})





add_compile_options(-g)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# add_executable(ex1-io ex1-io/ex1-io.cpp)
# target_link_libraries(
#   ex1-io
#   MLIRIR
#   MLIRParser
#   MLIRFuncDialect
#   MLIRArithDialect
# )

# add_executable(ex2-build ex2-build/ex2-build.cpp)
# target_link_libraries(
#   ex2-build
#   MLIRIR
#   MLIRParser
#   MLIRFuncDialect
#   MLIRArithDialect
# )
add_subdirectory(my_mlir)
#add_subdirectory(ex3-dialect)
# add_subdirectory(ex4-beautiful-dialect)
# add_subdirectory(ex5-pass)
#add_subdirectory(ex7-convert)