#include "livox_feature_extraction_ros2.h"

#include <cmath>

namespace cocolic {
namespace {
// Kept identical to the upstream validity test (the upstream name is inverted).
#define IS_VALID(a) ((std::abs(a) > 1e8) ? true : false)
}

LivoxFeatureExtractionRos2::LivoxFeatureExtractionRos2(const YAML::Node &node)
    : vx(0), vy(0), vz(0) {
  const auto &livox_node = node["Livox"];
  n_scan = livox_node["n_scan"].as<int>();
  blind = livox_node["blind"].as<double>();
  inf_bound = livox_node["inf_bound"].as<double>();
  group_size = livox_node["group_size"].as<int>();
  disA = livox_node["disA"].as<double>();
  disB = livox_node["disB"].as<double>();
  limit_maxmid = livox_node["limit_maxmid"].as<double>();
  limit_midmin = livox_node["limit_midmin"].as<double>();
  limit_maxmin = livox_node["limit_maxmin"].as<double>();
  p2l_ratio = livox_node["p2l_ratio"].as<double>();
  jump_up_limit = std::cos(livox_node["jump_up_limit"].as<double>() / 180 * M_PI);
  jump_down_limit = std::cos(livox_node["jump_down_limit"].as<double>() / 180 * M_PI);
  edgea = livox_node["edgea"].as<double>();
  edgeb = livox_node["edgeb"].as<double>();
  smallp_intersect = std::cos(livox_node["smallp_intersect"].as<double>() / 180 * M_PI);
  smallp_ratio = livox_node["smallp_ratio"].as<double>();
  point_filter_num = livox_node["point_filter_num"].as<int>();
  AllocateMemory();
}

bool LivoxFeatureExtractionRos2::ParsePointCloudR3LIVE(
    const LivoxScanRos2 &lidar_msg, RTPointCloud::Ptr out_cloud) {
  clearState();
  std::vector<RTPointCloud::Ptr> in_cloud_vec(n_scan);
  std::vector<std::vector<orgtype>> typess(n_scan);
  const std::size_t plsize = lidar_msg.points.size();
  p_corner_cloud->reserve(plsize);
  p_surface_cloud->reserve(plsize);
  p_full_cloud->resize(plsize);
  for (int i = 0; i < n_scan; ++i) {
    in_cloud_vec[i] = RTPointCloud::Ptr(new RTPointCloud());
    in_cloud_vec[i]->reserve(plsize);
  }
  // The loop and filter below are the upstream ParsePointCloudR3LIVE body;
  // only the ROS message access has become a transport-neutral LivoxScanRos2.
  for (std::size_t i = 1; i < plsize; ++i) {
    const auto &src = lidar_msg.points[i];
    if (src.line < n_scan && !IS_VALID(src.x) && !IS_VALID(src.y) &&
        !IS_VALID(src.z) && src.x > 0.7) {
      if (src.x > 2.0 && ((src.tag & 0x03) != 0x00 || (src.tag & 0x0C) != 0x00)) continue;
      auto &dst = (*p_full_cloud)[i];
      dst.x = src.x; dst.y = src.y; dst.z = src.z;
      dst.intensity = src.reflectivity; dst.time = static_cast<int64_t>(src.offset_time);
      const auto &previous = (*p_full_cloud)[i - 1];
      if (std::abs(dst.x - previous.x) > 1e-7 || std::abs(dst.y - previous.y) > 1e-7 ||
          std::abs(dst.z - previous.z) > 1e-7) in_cloud_vec[src.line]->push_back(dst);
    }
  }
  if (in_cloud_vec.size() != static_cast<std::size_t>(n_scan) || in_cloud_vec[0]->size() <= 7) return false;
  for (int j = 0; j < n_scan; ++j) {
    RTPointCloud &pl = *in_cloud_vec[j];
    auto &types = typess[j];
    if (pl.size() < 7) continue;
    types.resize(pl.size());
    for (std::size_t i = 0; i + 1 < pl.size(); ++i) {
      types[i].range = pl[i].x * pl[i].x + pl[i].y * pl[i].y;
      vx = pl[i].x - pl[i + 1].x; vy = pl[i].y - pl[i + 1].y; vz = pl[i].z - pl[i + 1].z;
    }
    types.back().range = pl.back().x * pl.back().x + pl.back().y * pl.back().y;
    giveFeatureR3LIVE(pl, types, *p_corner_cloud, *p_surface_cloud);
  }
  p_corner_cloud->push_back((*p_full_cloud)[0]);
  for (const auto &cloud : in_cloud_vec) *out_cloud += *cloud;
  return true;
}

void LivoxFeatureExtractionRos2::clearState() {
  vx = vy = vz = 0; p_full_cloud->clear();
  p_corner_cloud.reset(new RTPointCloud); p_surface_cloud.reset(new RTPointCloud);
}
void LivoxFeatureExtractionRos2::AllocateMemory() {
  p_full_cloud.reset(new RTPointCloud); p_corner_cloud.reset(new RTPointCloud); p_surface_cloud.reset(new RTPointCloud);
}

void LivoxFeatureExtractionRos2::giveFeatureR3LIVE(
    RTPointCloud &pl, std::vector<orgtype> &types, RTPointCloud &pl_corn, RTPointCloud &pl_surf) {
  const std::size_t plsize = pl.size();
  if (plsize == 0) return;
  std::size_t head = 0;
  while (head < types.size() && types[head].range < blind) ++head;
  // Upstream R3LIVE branch intentionally exports the valid scan samples as
  // surface features (instead of the general giveFeature classifier).
  const std::size_t plsize2 = plsize > static_cast<std::size_t>(group_size) ? plsize - group_size : 0;
  RTPoint ap;
  for (std::size_t i = head; i < plsize2; ++i) {
    if (types[i].range > blind) {
      ap.x = pl[i].x; ap.y = pl[i].y; ap.z = pl[i].z;
      ap.time = pl[i].time; ap.intensity = pl[i].intensity;
      pl_surf.push_back(ap);
    }
  }
}
}  // namespace cocolic
