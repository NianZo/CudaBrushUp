set(BINARY AddVectors)
set(SOURCES AddVectors.cpp)
set_source_files_properties(AddVectors.cpp PROPERTIES LANGUAGE CUDA)

add_executable(${BINARY}_run ${SOURCES})
# add_library(${BINARY}_lib STATIC ${SOURCES})

# Link warning flags
# target_link_libraries(${BINARY}_run PRIVATE project_warnings)

set_target_properties(${BINARY}_run PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
