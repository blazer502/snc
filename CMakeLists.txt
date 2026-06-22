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

add_executable(snc_aversive src/aversive_demo.cpp)
target_link_libraries(snc_aversive PRIVATE snc_core)

add_executable(snc_vocab src/vocab_demo.cpp)
target_link_libraries(snc_vocab PRIVATE snc_core)

add_executable(snc_chat src/chat_demo.cpp)
target_link_libraries(snc_chat PRIVATE snc_core)

# ---------------------------------------------------------------------------
# Structure-aware SNN substrate (new-plan.md): graph + encoders + runtimes +
# benchmark. Kept as a separate library so the existing demos are untouched.
# ---------------------------------------------------------------------------
option(SNC_ENABLE_CUDA "Build the CUDA runtime backend (requires nvcc)" OFF)

set(SNC_SUBSTRATE_SRC
  src/graph/snn_graph.cpp
  src/structure/connectome.cpp
  src/training/encoders.cpp
  src/training/datasets.cpp
  src/training/trainer.cpp
  src/runtime/cpu_backend.cpp
)
if(SNC_ENABLE_CUDA)
  enable_language(CUDA)
  list(APPEND SNC_SUBSTRATE_SRC src/runtime/cuda_backend.cu)
else()
  list(APPEND SNC_SUBSTRATE_SRC src/runtime/cuda_stub.cpp)
endif()

add_library(snc_substrate ${SNC_SUBSTRATE_SRC})
target_include_directories(snc_substrate PUBLIC include)
# snc_core provides Simulator/Neuron for compile_from_simulator.
target_link_libraries(snc_substrate PUBLIC snc_core)
# Apply host warning/opt flags to C++ TUs only; nvcc has its own flag dialect.
target_compile_options(snc_substrate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-march=native>
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(snc_substrate PUBLIC OpenMP::OpenMP_CXX)
endif()
if(SNC_ENABLE_CUDA)
  target_compile_definitions(snc_substrate PUBLIC SNC_HAVE_CUDA)
  set_target_properties(snc_substrate PROPERTIES CUDA_ARCHITECTURES native)
endif()

add_executable(snc_bench src/bench/snc_bench.cpp)
target_link_libraries(snc_bench PRIVATE snc_substrate snc_core)

add_executable(snc_train src/training/train_main.cpp)
target_link_libraries(snc_train PRIVATE snc_substrate snc_core)

add_executable(snc_cotrain src/training/cotrain_main.cpp)
target_link_libraries(snc_cotrain PRIVATE snc_substrate snc_core)
