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

#include "mapping.h"
#include "gaussian.h"

#include <atomic>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

std::mutex m_buf;
std::condition_variable con;

std::queue<sensor_msgs::msg::PointCloud2::ConstSharedPtr> point_buf;
std::queue<geometry_msgs::msg::PoseStamped::ConstSharedPtr> pose_buf;
std::queue<sensor_msgs::msg::Image::ConstSharedPtr> image_buf;
std::queue<sensor_msgs::msg::Image::ConstSharedPtr> depth_buf;

std::atomic<bool> exit_flag(false);
std::atomic<double> last_point_time(0.0);
std::atomic<bool> gaussians_initialized(false);

double steadyTimeSeconds()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void pointCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr point_msg)
{
    std::lock_guard<std::mutex> lock(m_buf);
    point_buf.push(point_msg);
    last_point_time = steadyTimeSeconds();
}

void poseCallback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_msg)
{
    std::lock_guard<std::mutex> lock(m_buf);
    pose_buf.push(pose_msg);
}

void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr image_msg)
{
    std::lock_guard<std::mutex> lock(m_buf);
    image_buf.push(image_msg);
}

void depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr depth_msg)
{
    std::lock_guard<std::mutex> lock(m_buf);
    depth_buf.push(depth_msg);
}

bool getAlignedData(Frame& cur_frame)
{
    if (point_buf.empty() || pose_buf.empty() || image_buf.empty() || depth_buf.empty()) 
    {
        return false;
    }

    double frame_time = rclcpp::Time(point_buf.front()->header.stamp).seconds();

    while (1) 
    {
        if (rclcpp::Time(pose_buf.front()->header.stamp).seconds() < frame_time - 0.01)
        {
            pose_buf.pop();
            if (pose_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (rclcpp::Time(pose_buf.front()->header.stamp).seconds() > frame_time + 0.01)
    {
        point_buf.pop();
        return false;
    }

    while (1) 
    {
        if (rclcpp::Time(image_buf.front()->header.stamp).seconds() < frame_time - 0.01)
        {
            image_buf.pop();
            if (image_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (rclcpp::Time(image_buf.front()->header.stamp).seconds() > frame_time + 0.01)
    {
        point_buf.pop();
        return false;
    }

    while (1) 
    {
        if (rclcpp::Time(depth_buf.front()->header.stamp).seconds() < frame_time - 0.01)
        {
            depth_buf.pop();
            if (depth_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (rclcpp::Time(depth_buf.front()->header.stamp).seconds() > frame_time + 0.01)
    {
        point_buf.pop();
        return false;
    }

    auto cur_point = point_buf.front();
    auto cur_pose = pose_buf.front();
    auto cur_image = image_buf.front();
    auto cur_depth = depth_buf.front();

    cur_frame.point_msg = cur_point;
    cur_frame.pose_msg = cur_pose;
    cur_frame.image_msg = cur_image;
    cur_frame.depth_msg = cur_depth;

    point_buf.pop();
    pose_buf.pop();
    image_buf.pop();
    depth_buf.pop();

    return true;
}

void mapping(const YAML::Node& node, const std::string& result_path, const std::string& lpips_path)
{
    torch::jit::setGraphExecutorOptimize(false);

    Params prm(node);
    std::shared_ptr<GaussianModel> gaussians = std::make_shared<GaussianModel>(prm);
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(prm);
    dataset->configureTsdfExport(result_path);

    std::chrono::steady_clock::time_point t_start, t_end;
    double total_mapping_time = 0;
    double total_adding_time = 0;
    double total_extending_time = 0;

    Frame cur_frame;
    while (!exit_flag)
    {
        /// [1] data alignment
        m_buf.lock();
        bool align_flag = getAlignedData(cur_frame);
        m_buf.unlock();
        if (!align_flag)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        /// [2] add every frame
        t_start = std::chrono::steady_clock::now();
        dataset->addFrame(cur_frame);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        if (dataset->is_keyframe_current_)
        {
            total_adding_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
            std::cout << "\033[1;33m     Cur Frame " << dataset->all_frame_num_ - 1 << ",\033[0m";
        }
        else continue;

        if (!gaussians->is_init_)
        {
            /// [3] initialize map
            gaussians->is_init_ = true;
            gaussians_initialized = true;
            gaussians->initialize(dataset);
            gaussians->trainingSetup();
        }
        else 
        {
            /// [4] extend map
            t_start = std::chrono::steady_clock::now();
            extend(dataset, gaussians);
            torch::cuda::synchronize();
            t_end = std::chrono::steady_clock::now();
            total_extending_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        }

        /// [5] optimize map
        t_start = std::chrono::steady_clock::now();
        double updated_num = optimize(dataset, gaussians);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        total_mapping_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        std::cout << std::fixed << std::setprecision(2) 
                  << "\033[1;36m Update " << updated_num / 10000 
                  << "w GS per Iter \033[0m" << std::endl;
    }

    /// [6] evaluation
    std::cout << "\n     🎉 Runtime Statistics 🎉\n";
    std::cout << std::fixed << std::setprecision(2) << "\n        [Total Mapping Time] " << total_mapping_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         1) Forward " << gaussians->t_forward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         2) Backward " << gaussians->t_backward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         3) Step " << gaussians->t_step_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         4) CPU2GPU " << gaussians->t_tocuda_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Adding Time] " << total_adding_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Extending Time] " << total_extending_time << "s" << std::endl;
    torch::NoGradGuard no_grad;
    evaluateVisualQuality(dataset, gaussians, result_path, lpips_path);
    gaussians->saveMap(result_path);
    dataset->writeKeyframeSelectionReport(result_path);

    std::cout << "\n\n😋 Gaussian-LIC Done!\n\n\n";
}

int main(int argc, char** argv)
{
    std::cout << "\n\n😋 Gaussian-LIC Ready!\n\n\n";
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("gs_mapping");

    const auto queue_depth = rclcpp::QoS(rclcpp::KeepLast(10000));
    auto sub_point = node->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/points_for_gs", queue_depth, pointCallback);
    auto sub_pose = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/pose_for_gs", queue_depth, poseCallback);
    auto image_sub = node->create_subscription<sensor_msgs::msg::Image>(
        "/image_for_gs", queue_depth, imageCallback);
    auto depth_sub = node->create_subscription<sensor_msgs::msg::Image>(
        "/depth_for_gs", queue_depth, depthCallback);

    const std::string config_path = node->declare_parameter<std::string>("config_path", "");
    if (config_path.empty())
    {
        RCLCPP_FATAL(node->get_logger(), "The required 'config_path' parameter is empty.");
        rclcpp::shutdown();
        return 1;
    }
    YAML::Node config_node = YAML::LoadFile(config_path);
    const std::string result_path = node->declare_parameter<std::string>("result_path", "");
    const std::string lpips_path = node->declare_parameter<std::string>("lpips_path", "");

    std::thread mapping_process(mapping, config_node, result_path, lpips_path);
    std::thread monitor_thread([](){
        while (!exit_flag) 
        {
            double now = steadyTimeSeconds();
            if (gaussians_initialized && (now - last_point_time > 15.0)) 
            {
                std::lock_guard<std::mutex> lock(m_buf);
                if (point_buf.empty()) exit_flag = true;
                if (exit_flag) rclcpp::shutdown();
            } 
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });
    
    rclcpp::spin(node);

    mapping_process.join();
    monitor_thread.join();
    rclcpp::shutdown();

    return 0;
}
