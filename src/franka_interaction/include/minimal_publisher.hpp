#pragma once

#include <memory>
#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <string>
#include <eigen3/Eigen/Dense>
#include <atomic>

typedef Eigen::Matrix<double, 6, 1> Vector6d;

struct affine_buffer{
  std::array<Eigen::Affine3d, 2> affines;
  std::atomic<int> active{0};
};

struct Vector6d_buffer
{
  std::array<Vector6d, 2> vectors{};
  std::atomic<int> active{0};
};


class MinimalPublisher : public rclcpp::Node
{
public:
  explicit MinimalPublisher(
    affine_buffer & EE_buffer,
    Vector6d_buffer & EE_twist,
    std::string ns,
    affine_buffer & Mirror_buffer,
    Vector6d_buffer & Mirror_twist,
    std::string partner_ns);
  void init();

private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr EE_config_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr Mirror_config_sub_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr EE_twist_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr Mirror_twist_sub_;

  affine_buffer & EE_buffer_;
  Vector6d_buffer & EE_twist_;
  std::string ns_;
  
  affine_buffer & Mirror_buffer_;
  Vector6d_buffer & Mirror_twist_;
  std::string partner_ns_;
};