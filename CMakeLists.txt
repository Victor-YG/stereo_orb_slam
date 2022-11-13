cmake_minimum_required(VERSION 3.5)

project(stere_orb_slam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# find Eigen3
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen3 found.")
else()
  message(FATAL_ERROR "Eigen3 > 3.1.0 not found.")
endif()

# find Sophus
find_package(Sophus REQUIRED)
if (Sophus_FOUND)
    include_directories(${Sophus_INCLUDE_DIR})
    message(STATUS "Sophus found.")
else()
  message(FATAL_ERROR "Sophus not found.")
endif()

# find OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV found.")
else()
    message(FATAL_ERROR "OpenCV not found.")
endif()

# find gflags
find_package(gflags REQUIRED)
if (gflags_FOUND)
    include_directories(${gflags_INCLUDE_DIRS})
    message(STATUS "gflags found.")
else()
    message(FATAL_ERROR "gflags not found.")
endif()

# find ceres
find_package(Ceres REQUIRED COMPONENTS SuiteSparse EigenSparse)
if (Ceres_FOUND)
    include_directories(${Ceres_INCLUDE_DIRS})
    message(STATUS "Ceres found.")
else()
    message(FATAL_ERROR "Ceres not found.")
endif()

# find DBoW2
find_package(DBoW2 REQUIRED)
if (DBoW2_FOUND)
    include_directories(${DBoW2_INCLUDE_DIRS})
    message(STATUS "DBoW2 found.")
else()
    message(FATAL_ERROR "DBoW2 not found.")
endif()

# #find g2o
# find_package(g2o REQUIRED)
# if (g2o)
#     include_directories(${g2o_INCLUDE_DIRS})
#     message(STATUS "g2o found.")
# else()
#     message(FATAL_ERROR "g2o not found.")
# endif()

include_directories(./include)
include_directories(./src)
include_directories(./ransac_solver/include)
include_directories(../../libs/g2o)

# visual odometry
add_executable(visual_odometry
    ./src/visual_odometry.cpp
    ./src/camera_model.cpp
    ./src/visual_odometer.cpp
    ./src/dataset.cpp
    ./src/camera_utils.cpp
)

target_link_libraries(visual_odometry
    ${OpenCV_LIBS}
    Eigen3::Eigen
    gflags
)

# slam
add_executable(slam
    ./src/slam.cpp
    ./src/camera_model.cpp
    ./src/visual_odometer.cpp
    ./src/bundle_adjuster.cpp
    ./src/dataset.cpp
    ./src/camera_utils.cpp
    ./src/loop_detector.cpp
    ./src/pose_graph_optimizer.cpp
)

target_link_libraries(slam
    ${OpenCV_LIBS}
    Ceres::ceres
    Eigen3::Eigen
    gflags
    DBoW2
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_core.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_stuff.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_types_slam3d.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_solver_eigen.so
)
