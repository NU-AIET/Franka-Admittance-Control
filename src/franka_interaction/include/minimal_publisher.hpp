#pragma once

#include <memory>
#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <string>
#include <eigen3/Eigen/Dense>
#include <atomic>

struct affine_buffer{
  std::array<Eigen::Affine3d, 2> affines;
  std::atomic<int> active{0};
};

class MinimalPublisher : public rclcpp::Node
{
public:
  explicit MinimalPublisher(
    affine_buffer & EE_buffer,
    std::string ns,
    affine_buffer & Mirror_buffer,
    std::string partner_ns);
  void init();

private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr partner_pose_sub_;

  affine_buffer & EE_buffer_;
  std::string ns_;
  
  affine_buffer & Mirror_buffer_;
  std::string partner_ns_;
};