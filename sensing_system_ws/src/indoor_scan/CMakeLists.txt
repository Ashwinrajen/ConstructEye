cmake_minimum_required(VERSION 3.0.2)
project(indoor_scan)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

add_service_files(
  FILES
  ProcessPointCloud.srv
)

add_message_files(
  FILES
  PointCloudDetections.msg
)


generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/analyse_model.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
