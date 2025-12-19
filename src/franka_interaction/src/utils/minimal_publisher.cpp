#include "minimal_publisher.hpp"

MinimalPublisher::MinimalPublisher(
    affine_buffer & EE_buffer,
    std::string ns,
    affine_buffer & Mirror_buffer,
    std::string partner_ns)

: Node("minimal_publisher"),
  EE_buffer_(EE_buffer),
  ns_(ns),
  Mirror_buffer_(Mirror_buffer),
  partner_ns_(partner_ns)
{
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/"+ns_+"/pose", 1000);

  partner_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
  "/"+partner_ns_+"/pose", 1000,
  [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    int next = 1 - Mirror_buffer_.active.load(std::memory_order_relaxed);
    Mirror_buffer_.affines[next].linear() = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z).toRotationMatrix();
    Mirror_buffer_.affines[next].translation() = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    Mirror_buffer_.active.store(next, std::memory_order_release);
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

    pose_pub_->publish(msg);
  };
  
  timer_ = this->create_wall_timer(std::chrono::milliseconds(1), timer_callback);
}
