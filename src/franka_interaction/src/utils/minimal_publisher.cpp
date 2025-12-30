#include "minimal_publisher.hpp"

MinimalPublisher::MinimalPublisher(
    affine_buffer & EE_buffer,
    Vector6d_buffer & EE_twist,
    std::string ns,
    affine_buffer & Mirror_buffer,
    Vector6d_buffer & Mirror_twist,
    std::string partner_ns)
: Node("minimal_publisher"),
  EE_buffer_(EE_buffer),
  EE_twist_(EE_twist),
  ns_(ns),
  Mirror_buffer_(Mirror_buffer),
  Mirror_twist_(Mirror_twist),
  partner_ns_(partner_ns)
{
  EE_config_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/"+ns_+"/config", 1000);
  EE_twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/"+ns_+"/twist", 1000);

  Mirror_config_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
  "/"+partner_ns_+"/config", 1000,
  [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    int next = 1 - Mirror_buffer_.active.load(std::memory_order_relaxed);
    Mirror_buffer_.affines[next].linear() = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z).toRotationMatrix();
    Mirror_buffer_.affines[next].translation() = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    Mirror_buffer_.active.store(next, std::memory_order_release);
  }
  );

  Mirror_twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
  "/"+partner_ns_+"/twist", 1000,
  [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
    int next = 1 - Mirror_twist_.active.load(std::memory_order_relaxed);
    Mirror_twist_.vectors[next].head(3) = Eigen::Vector3d(msg->linear.x, msg->linear.y, msg->linear.z);
    Mirror_twist_.vectors[next].tail(3) = Eigen::Vector3d(msg->angular.x, msg->angular.y, msg->angular.z);
    Mirror_twist_.active.store(next, std::memory_order_release);
  }
  );

  auto timer_callback = [this]() -> void {
    int idx = EE_buffer_.active.load(std::memory_order_acquire);
    Eigen::Affine3d pose = EE_buffer_.affines[idx];
    auto msg = geometry_msgs::msg::PoseStamped();
    msg.header.stamp = get_clock()->now();
    msg.pose.position.x = pose.translation()(0);
    msg.pose.position.y = pose.translation()(1);
    msg.pose.position.z = pose.translation()(2);

    Eigen::Matrix3d R = pose.linear();

    Eigen::Quaterniond q(R);

    q.normalize();

    msg.pose.orientation.x = q.x();
    msg.pose.orientation.y = q.y();
    msg.pose.orientation.z = q.z();
    msg.pose.orientation.w = q.w();

    EE_config_pub_->publish(msg);

    int idy = EE_twist_.active.load(std::memory_order_acquire);
    Vector6d V = EE_twist_.vectors[idy];

    auto twist = geometry_msgs::msg::Twist();

    twist.linear.x = V(0);
    twist.linear.y = V(1);
    twist.linear.z = V(2);
    twist.angular.x = V(3);
    twist.angular.y = V(4);
    twist.angular.z = V(5);

    EE_twist_pub_->publish(twist);
  };
  
  timer_ = this->create_wall_timer(std::chrono::milliseconds(1), timer_callback);
}
