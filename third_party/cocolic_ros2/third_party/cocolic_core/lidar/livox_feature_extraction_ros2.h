// ROS 2 transport wrapper for the original Coco-LIC Livox feature core.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "lidar_feature.h"
#include <utils/cloud_tool.h>
#include <utils/mypcl_cloud_type.h>
#include <utils/parameter_struct.h>

namespace cocolic {

// Point data required by the ROS2 Livox interface.  A PointCloud2 adapter
// fills this structure without changing the original feature calculations.
struct LivoxPointRos2 {
  float x{0.F}, y{0.F}, z{0.F};
  float reflectivity{0.F};
  std::uint8_t tag{0};
  std::uint8_t line{0};
  std::uint32_t offset_time{0};
};

struct LivoxScanRos2 {
  std::vector<LivoxPointRos2> points;
};

enum Feature { Nor, Poss_Plane, Real_Plane, Edge_Jump, Edge_Plane, Wire, ZeroPoint };
enum Surround { Prev, Next };
enum E_jump { Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind };

struct orgtype {
  double range{0};
  double dista{0};
  double angle[2]{};
  double intersect{2};
  E_jump edj[2]{Nr_nor, Nr_nor};
  Feature ftype{Nor};
};

class LivoxFeatureExtractionRos2 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<LivoxFeatureExtractionRos2>;

  explicit LivoxFeatureExtractionRos2(const YAML::Node &node);
  bool ParsePointCloudR3LIVE(const LivoxScanRos2 &lidar_msg, RTPointCloud::Ptr out_cloud);
  RTPointCloud::Ptr GetCornerFeature() const { return p_corner_cloud; }
  RTPointCloud::Ptr GetSurfaceFeature() const { return p_surface_cloud; }
  int GetScanNumber() const { return n_scan; }

private:
  void AllocateMemory();
  void clearState();
  void giveFeatureR3LIVE(RTPointCloud &pl, std::vector<orgtype> &types,
                         RTPointCloud &pl_corn, RTPointCloud &pl_surf);
  int checkPlane(const RTPointCloud &pl, std::vector<orgtype> &types,
                 uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool checkCorner(const RTPointCloud &pl, std::vector<orgtype> &types,
                   uint i, Surround nor_dir);

  int n_scan{};
  double blind{}, inf_bound{};
  int group_size{};
  double disA{}, disB{}, limit_maxmid{}, limit_midmin{}, limit_maxmin{}, p2l_ratio{};
  double jump_up_limit{}, jump_down_limit{}, edgea{}, edgeb{}, smallp_intersect{}, smallp_ratio{};
  int point_filter_num{};
  double vx{}, vy{}, vz{};
  RTPointCloud::Ptr p_corner_cloud, p_surface_cloud, p_full_cloud;
};
}  // namespace cocolic
