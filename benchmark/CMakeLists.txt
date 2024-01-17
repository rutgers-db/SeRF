set(CMAKE_CXX_STANDARD 17)

add_executable(generate_groundtruth generate_groundtruth.cc)
target_link_libraries(generate_groundtruth UTIL)
target_compile_options(generate_groundtruth PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(generate_groundtruth ${OpenMP_CXX_FLAGS})
target_link_libraries(generate_groundtruth OpenMP::OpenMP_CXX)


add_executable(deep_halfbound benchmark_deep_halfbounded.cc)
target_link_libraries(deep_halfbound UTIL)
target_compile_options(deep_halfbound PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(deep_halfbound ${OpenMP_CXX_FLAGS})
target_link_libraries(deep_halfbound OpenMP::OpenMP_CXX)

add_executable(deep_arbitrary benchmark_deep_arbitrary.cc)
target_link_libraries(deep_arbitrary UTIL)
target_compile_options(deep_arbitrary PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(deep_arbitrary ${OpenMP_CXX_FLAGS})
target_link_libraries(deep_arbitrary OpenMP::OpenMP_CXX)