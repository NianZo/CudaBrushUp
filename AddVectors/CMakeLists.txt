set(BINARY AddVectors)

set(SOURCES AddVectors.cpp)

add_executable(${BINARY}_run ${SOURCES})

#add_library(${BINARY} _lib STATIC ${SOURCES})
add_library(${BINARY}_lib STATIC ${SOURCES})

# Link warning flags
target_link_libraries(${BINARY}_run PRIVATE project_warnings)
