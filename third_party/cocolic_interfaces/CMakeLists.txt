cmake_minimum_required(VERSION 3.8)
project(cocolic_interfaces)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FeatureCloud.msg"
  "msg/ImuArray.msg"
  "msg/LioInput.msg"
  "msg/LivoxCustomPoint.msg"
  "msg/LivoxCustomMsg.msg"
  "msg/PoseArray.msg"
  DEPENDENCIES geometry_msgs sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
