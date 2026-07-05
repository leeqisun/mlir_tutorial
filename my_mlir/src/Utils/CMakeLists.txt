find_library(CUDA_DRIVER_LIBRARY NAMES cuda
  PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib/wsl/lib
)

set(MLIR_TUTORIAL_UTILS_SOURCES
    File.cpp
    CUDARuntime.cpp
    CUDARuntimeAPI.cpp
    CUDARuntimeArtifacts.cpp
    H350Runtime.cpp
    H350RuntimeAPI.cpp
)

set(MLIR_TUTORIAL_UTILS_LINK_LIBS
    MLIRIR
)

if(CUDA_DRIVER_LIBRARY)
  list(APPEND MLIR_TUTORIAL_UTILS_SOURCES
      CUDADriverBackend.cpp
  )
endif()

add_mlir_library(MLIRTutorialUtils
    ${MLIR_TUTORIAL_UTILS_SOURCES}

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/Utils

    DEPENDS

    LINK_LIBS PUBLIC
    ${MLIR_TUTORIAL_UTILS_LINK_LIBS}
)

if(CUDA_DRIVER_LIBRARY)
  target_link_libraries(MLIRTutorialUtils PUBLIC ${CUDA_DRIVER_LIBRARY})
endif()
