cmake_minimum_required(VERSION 3.16)
project(snc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)

add_library(snc_core
  src/brain_grid.cpp
  src/energy_field.cpp
  src/simulator.cpp
)
target_include_directories(snc_core PUBLIC include)
target_compile_options(snc_core PRIVATE -Wall -Wextra -O3 -march=native)
if(OpenMP_CXX_FOUND)
  target_link_libraries(snc_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(snc_demo src/main.cpp)
target_link_libraries(snc_demo PRIVATE snc_core)

add_executable(snc_learn src/learn_demo.cpp)
target_link_libraries(snc_learn PRIVATE snc_core)

add_executable(snc_test_stdp src/test_stdp.cpp)
target_link_libraries(snc_test_stdp PRIVATE snc_core)

add_executable(snc_mom_dad src/mom_dad_demo.cpp)
target_link_libraries(snc_mom_dad PRIVATE snc_core)
