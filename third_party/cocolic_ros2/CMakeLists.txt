cmake_minimum_required(VERSION 3.16)
project(cocolic_ros2)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O2 -msse4.2)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(cocolic_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem iostreams program_options system serialization thread)
file(GLOB CAMERA src/camera/loam/*.cpp src/camera/optical_flow/*.cpp src/camera/rgb_map/*.cpp src/camera/*.cpp)
list(FILTER CAMERA EXCLUDE REGEX "LiDAR_front_end.cpp$")
add_executable(cocolic_original_odometry_node
 src/odometry_node.cpp src/spline/trajectory.cpp
 src/lidar/lidar_handler.cpp src/lidar/velodyne_feature_extraction.cpp src/lidar/livox_feature_extraction.cpp
 src/imu/imu_state_estimator.cpp src/imu/imu_initializer.cpp
 src/odom/msg_manager.cpp src/odom/trajectory_manager.cpp src/odom/trajectory_estimator.cpp src/odom/odometry_manager.cpp
 src/odom/factor/analytic_diff/marginalization_factor.cpp src/utils/parameter_struct.cpp ${CAMERA})
target_include_directories(cocolic_original_odometry_node PRIVATE src src/camera src/camera/loam/include src/camera/tools src/camera/rgb_map ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(cocolic_original_odometry_node rclcpp rosbag2_cpp cocolic_interfaces sensor_msgs geometry_msgs nav_msgs visualization_msgs cv_bridge pcl_conversions tf2_ros)
target_link_libraries(cocolic_original_odometry_node yaml-cpp Ceres::ceres ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)
install(TARGETS cocolic_original_odometry_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
# Only expose launch files for the original LIC/R3LIVE executable built above.
install(FILES
 launch/cocolic_original.launch.py
 launch/cocolic_real_odometry.launch.py
 DESTINATION share/${PROJECT_NAME}/launch)
ament_package()
