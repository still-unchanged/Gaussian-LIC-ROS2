#include <ros2_io.hpp>
#include <odom/odometry_manager.h>
int main(int argc,char **argv) {
 google::InitGoogleLogging(argv[0]);
 rclcpp::init(argc,argv);
 ros2_io::node()=std::make_shared<rclcpp::Node>("cocolic");
 ros2_io::NodeHandle nh;
 std::string config;
 nh.param<std::string>("config_path",config,"");
 try {
  if(config.empty()) throw std::runtime_error("config_path is required");
  auto cfg=YAML::LoadFile(config);
  // Same bag interval parameters, optionally overridden for reproducible tests.
  cfg["bag_start"]=ros2_io::node()->declare_parameter<double>("bag_start",cfg["bag_start"].as<double>());
  cfg["bag_durr"]=ros2_io::node()->declare_parameter<double>("bag_duration",cfg["bag_durr"].as<double>());
  cocolic::OdometryManager manager(cfg,nh);
  manager.RunBag();
  manager.SaveOdometry();
  RCLCPP_INFO(ros2_io::node()->get_logger(),"Original LIC/R3LIVE offline processing completed");
 } catch(const std::exception &e) {
  RCLCPP_ERROR(ros2_io::node()->get_logger(),"%s",e.what()); rclcpp::shutdown(); return 1;
 }
 rclcpp::shutdown();return 0;
}
