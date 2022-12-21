IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(rnnt_release)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

include_directories(include)

FIND_PACKAGE(CUDA)
MESSAGE(STATUS "cuda found ${CUDA_FOUND}")

option(USE_NAIVE_KERNEL "use naive alpha-beta kernel" OFF)
option(DEBUG_TIME "output kernel time" OFF)
option(DEBUG_KERNEL "output alpha beta" OFF)
if (USE_NAIVE_KERNEL)
    add_definitions(-DUSE_NAIVE_KERNEL)
endif()
if (DEBUG_TIME)
    add_definitions(-DDEBUG_TIME)
endif()
if (DEBUG_KERNEL)
    add_definitions(-DDEBUG_KERNEL)
endif()

option(WITH_GPU "compile warp-rnnt with cuda." ${CUDA_FOUND})
option(WITH_OMP "compile warp-rnnt with openmp." ON)
option(BUILD_TESTS  "build warp-rnnt unit tests."      ON)
option(BUILD_SHARED "build warp-rnnt shared library."  ON)
option(WITH_ROCM   "Compile PaddlePaddle with ROCM platform" OFF)

if(WITH_ROCM)
    add_definitions(-DWARPRNNT_WITH_HIP)
    include(hip)
endif(WITH_ROCM)

if(BUILD_SHARED)
    set(WARPRNNT_SHARED "SHARED")
else()
    set(WARPRNNT_SHARED "STATIC")
endif()


if(WIN32)
    set(CMAKE_STATIC_LIBRARY_PREFIX lib)
else()
    # Set c++ flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2")
endif()

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    add_definitions(-DAPPLE)
endif()

if(NOT WITH_OMP)
    add_definitions(-DRNNT_DISABLE_OMP)
endif()
if (WITH_OMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
endif()


# need to be at least 30 or __shfl_down in reduce wont compile
IF (CUDA_VERSION VERSION_LESS "11.0")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
ENDIF()

# sm35 is deprecated after cuda 12.0
IF (CUDA_VERSION VERSION_LESS "12.0")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
ENDIF()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

IF (CUDA_VERSION VERSION_GREATER "7.6")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "9.0") OR (CUDA_VERSION VERSION_EQUAL "9.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "10.0") OR (CUDA_VERSION VERSION_EQUAL "10.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "11.0") OR (CUDA_VERSION VERSION_EQUAL "11.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "11.2") OR (CUDA_VERSION VERSION_EQUAL "11.2"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "11.8") OR (CUDA_VERSION VERSION_EQUAL "11.8"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_90,code=sm_90")
ENDIF()

if (NOT APPLE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
ENDIF()

IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()


IF (WITH_GPU OR WITH_ROCM)
    MESSAGE(STATUS "Building shared library with GPU support")

    IF (WITH_GPU)
        MESSAGE(STATUS "NVCC_ARCH_FLAGS" ${CUDA_NVCC_FLAGS})
    ENDIF()

    if (WIN32)
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler \"/wd 4068 /wd 4244 /wd 4267 /wd 4305 /wd 4819\"")
    endif()

    IF (WITH_GPU)
        CUDA_ADD_LIBRARY(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cu)
    ELSE()
        HIP_ADD_LIBRARY(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cu)
        TARGET_LINK_LIBRARIES(warprnnt PUBLIC ${ROCM_HIPRTC_LIB})
    ENDIF()

    IF (!Torch_FOUND)
        MESSAGE(STATUS "Link rand library")

        IF (WITH_GPU)
            MESSAGE(STATUS "Link cuda rand library: ${CUDA_curand_LIBRARY}")
            TARGET_LINK_LIBRARIES(warprnnt ${CUDA_curand_LIBRARY})
        ELSE()
            MESSAGE(STATUS "Link hip rand library: ${hiprand_LIBRARY_DIRS}")
            TARGET_LINK_LIBRARIES(warprnnt ${hiprand_LIBRARY_DIRS}/libhiprand.so)
        ENDIF()
    ENDIF()

    if(BUILD_TESTS)
        MESSAGE(STATUS "Build tests")
        IF (WITH_GPU)
            cuda_add_executable(test_time_gpu tests/test_time.cu)
            TARGET_LINK_LIBRARIES(test_time_gpu warprnnt ${CUDA_curand_LIBRARY})
            SET_TARGET_PROPERTIES(test_time_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

            cuda_add_executable(test_gpu tests/test_gpu.cu)
            TARGET_LINK_LIBRARIES(test_gpu warprnnt ${CUDA_curand_LIBRARY})
            SET_TARGET_PROPERTIES(test_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
        ELSE()
            hip_add_executable(test_time_gpu tests/test_time.cu)
            TARGET_LINK_LIBRARIES(test_time_gpu warprnnt ${hiprand_LIBRARY_DIRS}/libhiprand.so)
            SET_TARGET_PROPERTIES(test_time_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

            hip_add_executable(test_gpu tests/test_gpu.cu)
            TARGET_LINK_LIBRARIES(test_gpu warprnnt ${hiprand_LIBRARY_DIRS}/libhiprand.so)
            SET_TARGET_PROPERTIES(test_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
        ENDIF()
    endif()

ELSE()
    MESSAGE(STATUS "Building shared library with no GPU support")

    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    ENDIF()

    ADD_LIBRARY(warprnnt ${WARPRNNT_SHARED} src/rnnt_entrypoint.cpp)

ENDIF()

if(BUILD_TESTS)
    add_executable(test_cpu tests/test_cpu.cpp)
    TARGET_LINK_LIBRARIES(test_cpu warprnnt)
    SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    add_executable(test_time tests/test_time.cpp)
    TARGET_LINK_LIBRARIES(test_time warprnnt)
    SET_TARGET_PROPERTIES(test_time PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
endif()

INSTALL(TARGETS warprnnt
        RUNTIME DESTINATION "bin"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib")

INSTALL(FILES include/rnnt.h DESTINATION "include")
