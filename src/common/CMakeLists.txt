cmake_minimum_required(VERSION 3.9)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB_RECURSE native_srcs *.cc *.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include/common)

add_library(UTIL STATIC ${native_srcs} )

target_compile_options(UTIL PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(UTIL ${OpenMP_CXX_FLAGS})
target_link_libraries(UTIL OpenMP::OpenMP_CXX)