cmake_minimum_required(VERSION 3.15)

project(OpenACCIntroduction)

# set_property(TARGET tgt PROPERTY CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

# Standard compiler warnings
add_library(project_warnings INTERFACE)
include(CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# Add 2 vectors (arrays) together using Cuda to accelerate the operation
add_subdirectory(AddVectors)
