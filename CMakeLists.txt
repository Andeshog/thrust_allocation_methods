cmake_minimum_required(VERSION 3.8)
project(thrust_allocation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(casadi REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)

include_directories(include)

add_executable(thrust_allocation_node
  src/thrust_allocation_node.cpp
  src/nlp.cpp
  src/maneuvering_allocator.cpp
  src/qp.cpp
  src/pseudo_inverse.cpp
)

target_include_directories(thrust_allocation_node PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${OsqpEigen_INCLUDE_DIRS}
)

ament_target_dependencies(thrust_allocation_node
rclcpp
geometry_msgs
std_msgs
casadi
fmt
spdlog
Eigen3
OsqpEigen
)

target_link_libraries(thrust_allocation_node fmt::fmt spdlog::spdlog casadi::casadi OsqpEigen::OsqpEigen)

install(TARGETS 
  thrust_allocation_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/thrust_allocation_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
