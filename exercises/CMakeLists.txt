set_source_files_properties(Hello.cpp VectorAdd.cpp MatrixMul.cpp Stencil1D.cpp MatrixMulShared.cpp VectorAddGridStride.cpp PROPERTIES LANGUAGE CUDA)

add_executable(Hello Hello.cpp)
set_target_properties(Hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(VectorAdd VectorAdd.cpp)
set_target_properties(VectorAdd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(MatrixMul MatrixMul.cpp)
set_target_properties(MatrixMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(Stencil1D Stencil1D.cpp)
set_target_properties(Stencil1D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(MatrixMulShared MatrixMulShared.cpp)
set_target_properties(MatrixMulShared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(VectorAddGridStride VectorAddGridStride.cpp)
set_target_properties(VectorAddGridStride PROPERTIES CUDA_SEPARABLE_COMPILATION ON)