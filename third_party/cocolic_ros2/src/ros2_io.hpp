#pragma once
// Native ROS2 transport boundary for the original offline estimator.
// No legacy runtime, bridge, estimator, or synthetic messages are used here.
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cassert>
#include <cmath>
#include <Eigen/Geometry>
namespace ros2_io {
inline rclcpp::Node::SharedPtr &node() { static rclcpp::Node::SharedPtr value; return value; }
inline bool ok() { return rclcpp::ok(); }
inline void shutdown() { rclcpp::shutdown(); }
inline void spinOnce() { rclcpp::spin_some(node()); }
inline double seconds(const builtin_interfaces::msg::Time &t) { return t.sec + t.nanosec * 1e-9; }
class Time {
  int64_t ns_{0};
public:
  Time() = default;
  Time(const builtin_interfaces::msg::Time &t):ns_(int64_t(t.sec)*1000000000LL+t.nanosec){}
  static Time now() { Time t; t.ns_=node()->now().nanoseconds(); return t; }
  Time &fromNSec(int64_t ns) { ns_=ns; return *this; }
  Time &fromSec(double s) { ns_=int64_t(std::llround(s*1e9)); return *this; }
  int64_t toNSec() const { return ns_; }
  double toSec() const { return ns_*1e-9; }
  operator builtin_interfaces::msg::Time() const { return rclcpp::Time(ns_).operator builtin_interfaces::msg::Time(); }
};
class Publisher {
  rclcpp::PublisherBase::SharedPtr pub_;
public:
  Publisher() = default;
  explicit Publisher(rclcpp::PublisherBase::SharedPtr p):pub_(std::move(p)){}
  size_t getNumSubscribers() const { return pub_ ? pub_->get_subscription_count():0; }
  template<class T> void publish(const T &msg) const {
    auto p=std::dynamic_pointer_cast<rclcpp::Publisher<T>>(pub_);
    if(!p) throw std::runtime_error("Uninitialized or mismatched ROS2 publisher");
    p->publish(msg);
  }
  template<class T> void publish(const std::shared_ptr<T> &msg) const { publish(*msg); }
};
using Subscriber=rclcpp::SubscriptionBase::SharedPtr;
class NodeHandle {
public:
  explicit NodeHandle(const std::string & = "") {}
  template<class T> void param(const std::string &key,T &value,const T &fallback) {
    value=node()->has_parameter(key)?node()->get_parameter(key).get_value<T>():node()->declare_parameter<T>(key,fallback);
  }
  template<class T> bool getParam(const std::string &key,T &value) { return node()->get_parameter(key,value); }
  template<class T> Publisher advertise(const std::string &topic,size_t depth) { return Publisher(node()->create_publisher<T>(topic,rclcpp::QoS(depth))); }
};
template<class V,class M> void vectorEigenToMsg(const V &v,M &m) { m.x=v.x();m.y=v.y();m.z=v.z(); }
template<class Q,class M> void quaternionEigenToMsg(const Q &q,M &m) { m.x=q.x();m.y=q.y();m.z=q.z();m.w=q.w(); }
}
