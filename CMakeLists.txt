cmake_minimum_required(VERSION 2.8)
project(Project_CV)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add libraries
add_library(utility headers/utility.h implementations/utility.cpp))
add_library(player_detection headers/player_detection.h implementations/player_detection.cpp))
add_library(segmentation headers/segmentation.h implementations/segmentation.cpp)
add_library(write_results headers/write_results.h implementations/write_results.cpp))
add_library(court_detection headers/court_detection.h implementations/court_detection.cpp))
add_library(evaluation headers/Eval.h implementations/Eval.cpp)
add_library(base headers/header.h)

# Add the executable
add_executable(${PROJECT_NAME} main.cpp)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Link other libraries
target_link_libraries(${PROJECT_NAME} utility)
target_link_libraries(${PROJECT_NAME} evaluation)
target_link_libraries(${PROJECT_NAME} player_detection)
target_link_libraries(${PROJECT_NAME} segmentation)
target_link_libraries(${PROJECT_NAME} write_results)
target_link_libraries(${PROJECT_NAME} court_detection)
target_link_libraries(${PROJECT_NAME} base)
