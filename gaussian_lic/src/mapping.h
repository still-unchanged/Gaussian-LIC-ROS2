/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "yaml_utils.h"

#include <chrono>
#include <deque>
#include <queue>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <cv_bridge/cv_bridge.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

class Params
{
public:
    Params(const YAML::Node &node)
    {
        height = node["height"].as<int>();
        width = node["width"].as<int>();
        fx = node["fx"].as<double>();
        fy = node["fy"].as<double>();
        cx = node["cx"].as<double>();
        cy = node["cy"].as<double>();

        select_every_k_frame = node["select_every_k_frame"].as<int>();
        // Missing keys retain the published fixed-interval selector so older
        // configurations remain comparable. Adaptive selection is expressed
        // in image-space quantities, rather than a dataset-specific world
        // distance threshold.
        keyframe_mode = node["keyframe_mode"] ? node["keyframe_mode"].as<std::string>() : "fixed";
        keyframe_min_frame_gap = node["keyframe_min_frame_gap"] ? node["keyframe_min_frame_gap"].as<int>() : 1;
        keyframe_min_interval_s = node["keyframe_min_interval_s"] ? node["keyframe_min_interval_s"].as<double>() : 0.0;
        keyframe_max_interval_s = node["keyframe_max_interval_s"] ? node["keyframe_max_interval_s"].as<double>() : 1.0;
        keyframe_min_normalized_parallax = node["keyframe_min_normalized_parallax"] ?
            node["keyframe_min_normalized_parallax"].as<double>() : 0.02;
        // keyframe_min_normalized_parallax is retained as a backwards-
        // compatible alias for configurations written before the explicit
        // information-density policy was introduced.
        keyframe_target_normalized_parallax = node["keyframe_target_normalized_parallax"] ?
            node["keyframe_target_normalized_parallax"].as<double>() : keyframe_min_normalized_parallax;
        keyframe_emergency_normalized_parallax = node["keyframe_emergency_normalized_parallax"] ?
            node["keyframe_emergency_normalized_parallax"].as<double>() : 2.0 * keyframe_target_normalized_parallax;
        keyframe_min_overlap = node["keyframe_min_overlap"] ? node["keyframe_min_overlap"].as<double>() : 0.80;
        keyframe_min_exposure_score = node["keyframe_min_exposure_score"] ?
            node["keyframe_min_exposure_score"].as<double>() : 0.60;
        keyframe_min_sharpness_ratio = node["keyframe_min_sharpness_ratio"] ?
            node["keyframe_min_sharpness_ratio"].as<double>() : 0.50;
        keyframe_min_depth_coverage = node["keyframe_min_depth_coverage"] ?
            node["keyframe_min_depth_coverage"].as<double>() : 0.002;
        keyframe_depth_consistency_rel = node["keyframe_depth_consistency_rel"] ?
            node["keyframe_depth_consistency_rel"].as<double>() : 0.05;
        // Test views are kept only for final evaluation.  Bounding this cache
        // is essential for long recordings on machines with limited RAM.
        max_test_cameras = node["max_test_cameras"] ? node["max_test_cameras"].as<int>() : 0;
        if (keyframe_mode != "fixed" && keyframe_mode != "adaptive")
            throw std::runtime_error("keyframe_mode must be 'fixed' or 'adaptive'");
        if (select_every_k_frame < 1 || keyframe_min_frame_gap < 0 || keyframe_min_interval_s < 0.0 ||
            keyframe_max_interval_s <= 0.0 ||
            keyframe_max_interval_s < keyframe_min_interval_s ||
            keyframe_min_normalized_parallax < 0.0 ||
            keyframe_target_normalized_parallax < 0.0 ||
            keyframe_emergency_normalized_parallax < keyframe_target_normalized_parallax ||
            keyframe_min_overlap < 0.0 || keyframe_min_overlap > 1.0 ||
            keyframe_min_exposure_score < 0.0 || keyframe_min_exposure_score > 1.0 ||
            keyframe_min_sharpness_ratio < 0.0 ||
            keyframe_min_depth_coverage < 0.0 || keyframe_min_depth_coverage > 1.0 ||
            keyframe_depth_consistency_rel < 0.0 || max_test_cameras < 0)
            throw std::runtime_error("invalid keyframe-selection parameters");
        depth_completion = node["depth_completion"].as<bool>();
        patch_size = node["patch_size"].as<int>();
        max_depth = node["max_depth"].as<double>();
        // Export the exact RGB/depth/pose triplets used by keyframes so an
        // offline TSDF fusion stage can produce a mesh.  Disabled by default
        // to preserve the original mapping behaviour and disk usage.
        tsdf_export = node["tsdf_export"] ? node["tsdf_export"].as<bool>() : false;
        std::string pkg_path = ament_index_cpp::get_package_share_directory("gaussian_lic");
        if (height == 512 && width == 640) engine_path = pkg_path + "/ckpt/spnet_512_640.engine";
        if (height == 480 && width == 640) engine_path = pkg_path + "/ckpt/spnet_480_640.engine";

        sh_degree = node["sh_degree"].as<int>();
        white_background = node["white_background"].as<bool>();
        random_background = node["random_background"].as<bool>();
        convert_SHs_python = node["convert_SHs_python"].as<bool>();
        compute_cov3D_python = node["compute_cov3D_python"].as<bool>();
        lambda_erank = node["lambda_erank"].as<double>();
        scaling_scale = node["scaling_scale"].as<double>();
        // A positive value bounds persistent model and optimizer memory for
        // long sequences; zero retains the original unlimited behavior.
        max_gaussians = node["max_gaussians"] ? node["max_gaussians"].as<int64_t>() : 0;
        // Limit one insertion as well, so a long trajectory receives map
        // coverage throughout the bag instead of spending the full budget at
        // its first few keyframes.
        max_gaussians_per_keyframe = node["max_gaussians_per_keyframe"] ?
            node["max_gaussians_per_keyframe"].as<int64_t>() : 0;
        // Leave physical VRAM headroom for renderer workspaces and final
        // evaluation.  Zero disables this runtime guard.
        gpu_min_free_mb = node["gpu_min_free_mb"] ? node["gpu_min_free_mb"].as<int>() : 0;
        if (max_gaussians < 0 || max_gaussians_per_keyframe < 0 || gpu_min_free_mb < 0)
            throw std::runtime_error("Gaussian memory limits must be non-negative");

        position_lr = node["position_lr"].as<double>();
        feature_lr = node["feature_lr"].as<double>();
        opacity_lr = node["opacity_lr"].as<double>();
        scaling_lr = node["scaling_lr"].as<double>();
        rotation_lr = node["rotation_lr"].as<double>();
        lambda_dssim = node["lambda_dssim"].as<double>();
        optimize_depth = node["optimize_depth"].as<bool>();
        lambda_depth = node["lambda_depth"].as<double>();
        iteration_decay = node["iteration_decay"].as<bool>();

        apply_exposure = node["apply_exposure"].as<bool>();
        exposure_lr = node["exposure_lr"].as<double>();
        skybox_points_num = node["skybox_points_num"].as<int>();
        skybox_radius = node["skybox_radius"].as<int>();
    }

    /// dataset
    int height;
    int width;
    double fx;
    double fy;
    double cx;
    double cy;

    int select_every_k_frame;
    std::string keyframe_mode;
    int keyframe_min_frame_gap;
    double keyframe_min_interval_s;
    double keyframe_max_interval_s;
    double keyframe_min_normalized_parallax;
    double keyframe_target_normalized_parallax;
    double keyframe_emergency_normalized_parallax;
    double keyframe_min_overlap;
    double keyframe_min_exposure_score;
    double keyframe_min_sharpness_ratio;
    double keyframe_min_depth_coverage;
    double keyframe_depth_consistency_rel;
    int max_test_cameras;
    bool depth_completion;
    int patch_size;
    double max_depth;
    bool tsdf_export;
    std::string engine_path;

    /// gaussian
    int sh_degree;
    bool white_background;
    bool random_background;
    bool convert_SHs_python;
    bool compute_cov3D_python;
    float lambda_erank;
    double scaling_scale;
    int64_t max_gaussians;
    int64_t max_gaussians_per_keyframe;
    int gpu_min_free_mb;

    double position_lr;
    double feature_lr;
    double opacity_lr;
    double scaling_lr;
    double rotation_lr;
    double lambda_dssim;
    bool optimize_depth;
    double lambda_depth;
    bool iteration_decay;

    bool apply_exposure;
    double exposure_lr;
    int skybox_points_num;
    int skybox_radius;
};

struct Frame 
{
    sensor_msgs::msg::PointCloud2::ConstSharedPtr point_msg;
    geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_msg;
    sensor_msgs::msg::Image::ConstSharedPtr image_msg;
    sensor_msgs::msg::Image::ConstSharedPtr depth_msg;
};
