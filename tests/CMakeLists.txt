set(CMAKE_CXX_STANDARD 11)

add_executable(exp_default exp_search_outbound.cc)
target_link_libraries(exp_default UTIL)

add_executable(exp_prune_reverse exp_search_prune_reverse.cc)
target_link_libraries(exp_prune_reverse UTIL)



add_executable(exp_test exp_test.cc)
target_link_libraries(exp_test UTIL)

add_executable(exp_duplicate exp_duplicate_pos.cc)
target_link_libraries(exp_duplicate UTIL)


add_executable(run_baseline run_baselines.cc)
target_link_libraries(run_baseline UTIL)

add_executable(exp_halfbound exp_halfbound.cc)
target_link_libraries(exp_halfbound UTIL)


add_executable(halfbound_scalability exp_halfbound_scalability.cc)
target_link_libraries(halfbound_scalability UTIL)

add_executable(twoside_scalability exp_twoside_scalability.cc)
target_link_libraries(twoside_scalability UTIL)


add_executable(twoside_sensitivity exp_twoside_sensitivity.cc)
target_link_libraries(twoside_sensitivity UTIL)

add_executable(twoside_params exp_twoside_sensitivity_params.cc)
target_link_libraries(twoside_params UTIL)

add_executable(halfbound_params exp_halfbound_params.cc)
target_link_libraries(halfbound_params UTIL)

add_executable(twoside_search exp_twoside_search.cc)
target_link_libraries(twoside_search UTIL)


# find_package(BLAS REQUIRED)
# find_package(LAPACK REQUIRED)

# add_library(libfaiss STATIC IMPORTED)


# if (${CMAKE_SYSTEM} MATCHES "Linux-5.15.0-73-generic")
# set_target_properties(libfaiss PROPERTIES IMPORTED_LOCATION /common/home/cz296/workspace/multi-vector-search/third_party/faiss/build/faiss/libfaiss.a)
# else()
# set_target_properties(libfaiss PROPERTIES IMPORTED_LOCATION /home/cz296/workspace/range-knn-search/third_party/faiss/build/faiss/libfaiss.a)
# endif()


# add_executable(exp_ivfpq_halfbound EXCLUDE_FROM_ALL faiss_ivfpq_halfbound.cc)
# target_link_libraries(exp_ivfpq_halfbound PRIVATE libfaiss ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} UTIL)


# find_package(BLAS REQUIRED)
# find_package(LAPACK REQUIRED)

# add_library(libfaiss STATIC IMPORTED)
# set_target_properties(libfaiss PROPERTIES IMPORTED_LOCATION /common/home/cz296/workspace/multi-vector-search/third_party/faiss/build/faiss/libfaiss.a)

# add_executable(test_imi EXCLUDE_FROM_ALL test_faiss_imi.cc)
# target_link_libraries(test_imi PRIVATE libfaiss ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} UTIL)
