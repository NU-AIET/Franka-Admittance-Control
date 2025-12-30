#include <array>
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <chrono>
#include <csignal>
#include <fstream>


#include <eigen3/Eigen/Dense>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "examples_common.h"
#include "net_ft/hardware_interface.hpp"
#include <rclcpp/rclcpp.hpp>


volatile bool robot_stop = false; // Global flag

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cerr << "\nCtrl+C detected. Initiating graceful shutdown..." << std::endl;
        robot_stop = true;
    }
}


Eigen::Matrix<double, 6, 1>bilateral_controller(const Eigen::Affine3d & T_BA, const double k_T, const double & k_R)
{
    // Generate translational stiffness tensor (N/m)
    Eigen::Matrix3d K_T = k_T * Eigen::Matrix3d::Identity();

    // Generate rotational stiffness tensor (Nm/rad)
    Eigen::Matrix3d K_R = k_R * Eigen::Matrix3d::Identity();

    const Eigen::Matrix3d R = T_BA.linear();

    Eigen::Matrix<double, 6, 1> Wrench{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    Eigen::Matrix3d W = K_R * R - R.transpose() * K_R.transpose();

    Wrench.head(3) = Eigen::Vector3d{-W(1,2), W(0,2), -W(0,1)};
    Wrench.tail(3) = R.transpose() * K_T * T_BA.translation();

    return Wrench;
}

class BilateralAdmittanceController : public rclcpp::Node
{
    public:
        BilateralAdmittanceController()
        : Node("BilateralAdmittanceController")
        {





        }

    private:
    rclcpp::Publisher<

};