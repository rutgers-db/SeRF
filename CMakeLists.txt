cmake_minimum_required(VERSION 3.12)
project(RangeFiltering-ANNS)
include(CheckCXXCompilerFlag)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

MESSAGE(STATUS ${CMAKE_CXX_FLAGS})

set(CMAKE_CXX_STANDARD 11)

MESSAGE(${CMAKE_SYSTEM})


set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3 ")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
    if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
    else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
            SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
            message("set -mcpu=apple-m1 flag")
        endif()
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
endif()


option(NOSIMD "build hnsw on SIMD")
if(NOSIMD)
    add_definitions(-DNO_MANUAL_VECTORIZATION)
endif(NOSIMD)

option(NOPARALLEL "no parallel build")
if(NOPARALLEL)
    add_definitions(-DNO_PARALLEL_BUILD)
endif(NOPARALLEL)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/common)
include_directories(${PROJECT_SOURCE_DIR}/src)



add_subdirectory(src/common)
add_subdirectory(benchmark)

