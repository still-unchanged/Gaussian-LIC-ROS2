#include <utils/ros2_timed_cloud.h>
#include <cassert>
#include <iostream>
int main(){
 RTPointCloud a,b;a.resize(4);
 const int64_t times[]={-1,0,135470252209LL,9007199254740992LL};
 for(size_t i=0;i<a.size();++i){a[i].x=1+i;a[i].y=-2;a[i].z=3;a[i].intensity=57;a[i].ring=5;a[i].time=times[i];}
 sensor_msgs::msg::PointCloud2 msg;ros2_io::toTimedROSMsg(a,msg);ros2_io::fromTimedROSMsg(msg,b);
 assert(a.size()==b.size());
 for(size_t i=0;i<a.size();++i)assert(a[i].time==b[i].time && a[i].x==b[i].x && a[i].y==b[i].y && a[i].z==b[i].z && a[i].intensity==b[i].intensity && a[i].ring==b[i].ring);
 a[0].time=9007199254740993LL;bool rejected=false;
 try{ros2_io::toTimedROSMsg(a,msg);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 std::cout<<"PASS: relative nanoseconds round trip exactly; out-of-range values rejected\n";
}
