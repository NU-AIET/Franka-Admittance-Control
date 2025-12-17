#include "minimal_publisher.hpp"

MinimalPublisher::MinimalPublisher(
    Eigen::Affine3d & pose,
    std::mutex & pose_mutex,
    std::string ns,
    Eigen::Affine3d & partner_pose,
    std::mutex & partner_pose_mutex,
    std::string partner_ns)

: Node("minimal_publisher"),
  pose_(pose),
  pose_mutex_(pose_mutex),
  ns_(ns),
  partner_pose_(partner_pose),
  partner_pose_mutex_(partner_pose_mutex),
  partner_ns_(partner_ns)
{
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/"+ns_+"/pose", 1000);

  partner_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
  "/"+partner_ns_+"/pose", 1000,
  [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(partner_pose_mutex_);
    partner_pose_.linear() = Eigen::Quaterniond(msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w).toRotationMatrix();
    partner_pose_.translation() = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  }
  );


  auto timer_callback = [this]() -> void {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    auto msg = geometry_msgs::msg::PoseStamped();
    msg.header.stamp = get_clock()->now();
    msg.pose.position.x = pose_.translation()(0);
    msg.pose.position.y = pose_.translation()(1);
    msg.pose.position.z = pose_.translation()(2);

    Eigen::Matrix3d R = pose_.linear();

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
