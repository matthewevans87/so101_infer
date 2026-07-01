cmake_minimum_required(VERSION 3.20)
project(so101_infer CXX)

set(CMAKE_CXX_STANDARD 17)          # LibTorch needs >=14; 17 is safe
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)        # found via -DCMAKE_PREFIX_PATH=$LIBTORCH

add_executable(so101_infer
    src/main.cpp src/model_runner.cpp src/preprocess.cpp src/tensor_io.cpp)
target_include_directories(so101_infer PRIVATE include)
target_link_libraries(so101_infer PRIVATE "${TORCH_LIBRARIES}")

# --- tests ---
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(parity_test tests/parity_test.cpp
    src/model_runner.cpp src/preprocess.cpp src/tensor_io.cpp)
target_include_directories(parity_test PRIVATE include)
target_link_libraries(parity_test PRIVATE "${TORCH_LIBRARIES}" GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(parity_test)