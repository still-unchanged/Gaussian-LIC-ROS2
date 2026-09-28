#pragma once
#include <utils/mypcl_cloud_type.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cstring>
#include <stdexcept>
namespace ros2_io {
// ROS2 PointField has no INT64. Relative nanoseconds are exact as FLOAT64
// for |time| <= 2^53; reject values outside this domain instead of truncating.
inline void toTimedROSMsg(const RTPointCloud &cloud, sensor_msgs::msg::PointCloud2 &msg) {
 sensor_msgs::PointCloud2Modifier modifier(msg);
 modifier.setPointCloud2Fields(6,"x",1,7,"y",1,7,"z",1,7,"intensity",1,7,"ring",1,4,"time",1,8);
 modifier.resize(cloud.size());
 sensor_msgs::PointCloud2Iterator<float> x(msg,"x"),y(msg,"y"),z(msg,"z"),intensity(msg,"intensity");
 sensor_msgs::PointCloud2Iterator<uint16_t> ring(msg,"ring");
 // Access double via memcpy: time follows uint16 ring and may be unaligned.
 const auto time_offset=msg.fields.back().offset;
 size_t i=0;
 for(const auto &p:cloud) {
  if(p.time < -9007199254740992LL || p.time > 9007199254740992LL) throw std::runtime_error("Relative point timestamp exceeds exact ROS2 wire range");
  *x=p.x;*y=p.y;*z=p.z;*intensity=p.intensity;*ring=p.ring;
  const double time=p.time;std::memcpy(msg.data.data()+i*msg.point_step+time_offset,&time,sizeof(time));
  ++x;++y;++z;++intensity;++ring;++i;
 }
 msg.is_dense=cloud.is_dense;
}
inline void fromTimedROSMsg(const sensor_msgs::msg::PointCloud2 &msg, RTPointCloud &cloud) {
 const sensor_msgs::msg::PointField *time_field=nullptr;
 for(const auto &f:msg.fields)if(f.name=="time")time_field=&f;
 if(!time_field || time_field->datatype!=8 || time_field->offset+8>msg.point_step || msg.is_bigendian)
  throw std::runtime_error("Expected little-endian FLOAT64 relative nanoseconds");
 if(msg.row_step!=msg.width*msg.point_step || msg.data.size()!=size_t(msg.row_step)*msg.height)throw std::runtime_error("Invalid timed cloud layout");
 cloud.resize(size_t(msg.width)*msg.height);
 sensor_msgs::PointCloud2ConstIterator<float>x(msg,"x"),y(msg,"y"),z(msg,"z"),intensity(msg,"intensity");
 sensor_msgs::PointCloud2ConstIterator<uint16_t>ring(msg,"ring");
 size_t i=0;
 for(auto &p:cloud){double t;std::memcpy(&t,msg.data.data()+i*msg.point_step+time_field->offset,8);
  if(!std::isfinite(t)||std::abs(t)>9007199254740992.0||std::trunc(t)!=t)throw std::runtime_error("Invalid relative nanoseconds");
  p.x=*x;p.y=*y;p.z=*z;p.intensity=*intensity;p.ring=*ring;p.time=int64_t(t);
  ++x;++y;++z;++intensity;++ring;++i;
 }
 cloud.is_dense=msg.is_dense;
}
}
