cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(LIB_NAME NeAT)
project(${LIB_NAME} VERSION 1.0.0 LANGUAGES C CXX CUDA)


set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/External/saiga/cmake/")

include(helper_macros)
include(ExternalProject)
DefaultBuildType(RelWithDebInfo)


message("Build Options")

OptionsHelper(HEADLESS "Skips the viewer and other apps that require a window." OFF)

############# Required LIBRARIES ###############


### OR Tools for linear integer optimization
# Dep: https://www.scipopt.org/index.php#download
# -> Download .deb and install
set(BUILD_absl ON)
set(BUILD_Protobuf OFF)
set(BUILD_gflags OFF)
set(USE_COINOR ON)
set(USE_SCIP OFF)
set(BUILD_SAMPLES OFF)
set(BUILD_EXAMPLES OFF)
add_subdirectory(External/or-tools)
PackageHelperTarget(ortools::ortools OR_TOOLS_FOUND)

add_subdirectory(External/tensorboard_logger)
#target_include_directories(tensorboard_logger PUBLIC External/tensorboard_logger/include ${CMAKE_CURRENT_BINARY_DIR}/External/tensorboard_logger)
set_property(TARGET tensorboard_logger PROPERTY POSITION_INDEPENDENT_CODE ON)
PackageHelperTarget(tensorboard_logger TBL_FOUND)

set(SAIGA_BUILD_SAMPLES OFF)
set(SAIGA_BUILD_TESTS OFF)
set(SAIGA_MODULE_VULKAN OFF)
set(SAIGA_MODULE_VISION OFF)
if (HEADLESS)
    set(SAIGA_MODULE_OPENGL OFF)
    set(SAIGA_BUILD_GLFW OFF)
endif ()
set(SAIGA_NO_INSTALL ON)
set(SAIGA_USE_SUBMODULES ON)
set(SAIGA_WITH_FFMPEG OFF)
set(SAIGA_WITH_FREETYPE OFF)
set(SAIGA_WITH_YAMLCPP OFF)
set(SAIGA_WITH_OPENAL OFF)
set(SAIGA_WITH_OPENMESH OFF)
set(SAIGA_WITH_OPENVR OFF)
set(SAIGA_WITH_G2O OFF)
set(SAIGA_WITH_CHOLMOD OFF)
set(SAIGA_WITH_OPENNI OFF)
set(SAIGA_WITH_K4A OFF)
set(SAIGA_WITH_TINY_EIGEN OFF)
add_subdirectory(External/saiga)

PackageHelperTarget(saiga_core SAIGA_FOUND)
if (NOT HEADLESS)
    PackageHelperTarget(saiga_opengl SAIGA_FOUND)
endif ()
 PackageHelperTarget(saiga_cuda SAIGA_FOUND)


find_package(Torch REQUIRED)
PackageHelperTarget(torch TORCH_FOUND)
#find_package(TorchVision CONFIG REQUIRED)
#PackageHelperTarget(TorchVision::TorchVision TORCHVISION_FOUND)

if (NOT SAIGA_FOUND)
    message(FATAL_ERROR "Saiga not found D:")
endif ()


#openmp
find_package(OpenMP REQUIRED)
PackageHelperTarget(OpenMP::OpenMP_CXX OPENMP_FOUND)


include_directories(.)
include_directories(${PACKAGE_INCLUDES})

############# COMPILER FLAGS ###############
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread")

if (MSVC)
    #multiprocessor compilation for visual studio
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=return-type")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=0")
endif ()


message(STATUS CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
message(STATUS CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})

set(LIBS ${LIBS} ${LIB_TARGETS})

############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

############# SOURCE ###############

add_subdirectory(src)
add_subdirectory(samples)







