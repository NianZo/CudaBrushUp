set_source_files_properties(Hello.cpp VectorAdd.cpp MatrixMu.cpp PROPERTIES LANGUAGE CUDA)

add_executable(Hello Hello.cpp)
set_target_properties(Hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(VectorAdd VectorAdd.cpp)
set_target_properties(VectorAdd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(MatrixMul MatrixMul.cpp)
set_target_properties(MatrixMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)