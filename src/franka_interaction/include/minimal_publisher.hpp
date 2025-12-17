#pragma once

#include <memory>
#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <string>
#include <eigen3/Eigen/Dense>

class MinimalPublisher : public rclcpp::Node
{
public:
  explicit MinimalPublisher(
    Eigen::Affine3d & pose,
    std::mutex & pose_mutex,
    std::string ns,
    Eigen::Affine3d & partner_pose,
    std::mutex & partner_pose_mutex,
    std::string partner_ns);
  void init();

private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr partner_pose_sub_;

  Eigen::Affine3d & pose_;
  std::mutex & pose_mutex_;
  std::string ns_;
  
  Eigen::Affine3d & partner_pose_;
  std::mutex & partner_pose_mutex_;
  std::string partner_ns_;
};