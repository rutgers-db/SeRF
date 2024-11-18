set(CMAKE_CXX_STANDARD 17)

add_executable(generate_groundtruth generate_groundtruth.cc)
target_link_libraries(generate_groundtruth UTIL)
target_compile_options(generate_groundtruth PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(generate_groundtruth ${OpenMP_CXX_FLAGS})
target_link_libraries(generate_groundtruth OpenMP::OpenMP_CXX)


add_executable(serf_halfbound serf_halfbound.cc)
target_link_libraries(serf_halfbound UTIL)
target_compile_options(serf_halfbound PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(serf_halfbound ${OpenMP_CXX_FLAGS})
target_link_libraries(serf_halfbound OpenMP::OpenMP_CXX)

add_executable(serf_arbitrary serf_arbitrary.cc)
target_link_libraries(serf_arbitrary UTIL)
target_compile_options(serf_arbitrary PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(serf_arbitrary ${OpenMP_CXX_FLAGS})
target_link_libraries(serf_arbitrary OpenMP::OpenMP_CXX)