cmake_minimum_required(VERSION 3.5)
project(omo_local_planner)

# C++ 14 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(message_filters REQUIRED)

# Python 스크립트 실행 가능하게 설정
#install(
#  PROGRAMS
#  DESTINATION lib/${PROJECT_NAME}
#)
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)
install(
  PROGRAMS scripts/robot_test.py
  DESTINATION lib/${PROJECT_NAME}
)
# DWA modules
install(
  PROGRAMS scripts/dwa_convert_global_path.py
  DESTINATION lib/${PROJECT_NAME}
)
install(
  PROGRAMS scripts/dwa_cost_evaluator.py
  DESTINATION lib/${PROJECT_NAME}
)
install(
  PROGRAMS scripts/dwa_data_provider.py
  DESTINATION lib/${PROJECT_NAME}
)
install(
  PROGRAMS scripts/dwa_executor.py
  DESTINATION lib/${PROJECT_NAME}
)
install(
  PROGRAMS scripts/dwa_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(
  PROGRAMS scripts/dwa_obstacle_cost.py
  DESTINATION lib/${PROJECT_NAME}
)
install(
  PROGRAMS scripts/dwa_trj_generator.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)


install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.yaml"
)

ament_package()
