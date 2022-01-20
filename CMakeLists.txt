cmake_minimum_required(VERSION 3.5)

project(stere_orb_slam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Find Eigen3
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen3 found.")
else()
  message(FATAL_ERROR "Eigen3 > 3.1.0 not found.")
endif()

# Find Sophus
find_package(Sophus REQUIRED)
if (Sophus_FOUND)
    include_directories(${Sophus_INCLUDE_DIR})
    message(STATUS "Sophus found.")
else()
  message(FATAL_ERROR "Sophus not found.")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV found.")
else()
    message(FATAL_ERROR "OpenCV not found.")
endif()

# Find gflags
find_package(gflags REQUIRED)
if (gflags_FOUND)
    include_directories(${gflags_INCLUDE_DIRS})
    message(STATUS "gflags found.")
else()
    message(FATAL_ERROR "gflags not found.")
endif()

include_directories(./include)
include_directories(./src)
include_directories(../ransac_solver/include)

add_executable(visual_odometry
    ./src/visual_odometry.cpp
    ./src/camera_model.cpp
    ./src/visual_odometer.cpp
    ./src/utils.cpp
    ./src/dataset.cpp
    ./src/camera_utils.cpp
)

target_link_libraries(visual_odometry
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Sophus::Sophus
    gflags
)
