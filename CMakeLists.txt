cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)
include("${CMAKE_CURRENT_SOURCE_DIR}/external/madrona/cmake/madrona_init.cmake")

# set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project(MadronaGridWorld LANGUAGES C CXX)

include(setup)
set(MADRONA_REQUIRE_PYTHON ON)
include(dependencies)

add_subdirectory(external)
add_subdirectory(src)
