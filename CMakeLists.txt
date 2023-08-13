cmake_minimum_required(VERSION 2.8)
project(Project_CV)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add libraries
add_library(utility utility.h utility.cpp))
#add_library(recognition recognition.h recognition.cpp))

# Add the executable
add_executable(${PROJECT_NAME} Project_CV.cpp)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Link other libraries
target_link_libraries(${PROJECT_NAME} utility)
#target_link_libraries(${PROJECT_NAME} recognition)