// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <array>
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <chrono>
#include <csignal>
#include <fstream>
#include <atomic>
#include <eigen3/Eigen/Dense>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "examples_common.h"
#include "net_ft/hardware_interface.hpp"
#include "SafeQueue.hpp"
#include "json.hpp"
#include "traj_simulate.hpp"
#include "minimal_publisher.hpp"
#include "butterworth.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>


#define EIGEN_RUNTIME_NO_MALLOC


using json = nlohmann::json;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 7, 7> Matrix7d;

volatile bool robot_stop = false; // Global flag

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cerr << "\nCtrl+C detected. Initiating graceful shutdown..." << std::endl;
        robot_stop = true;
    }
}

struct franka_model_calculations
{
   std::array<double, 7> coriolis_array{};
   std::array<double, 7> gravity_array{};
   std::array<double, 42> jacobian_array{};
   std::array<double, 49> mass_array{};
};

void Vec2Skew(Eigen::Matrix3d & skew, const Eigen::Vector3d & vec)
{
  skew << 0.0, -vec(2), vec(1),
          vec(2), 0.0, -vec(0),
          -vec(1), vec(0), 0.0;
}

void Skew2Vec(Eigen::Vector3d & vec, const Eigen::Matrix3d & skew){
  vec(0) = -skew(1,2);
  vec(1) = skew(0,2);
  vec(2) = -skew(0,1);
}

void Adjoint(Matrix6d & Ad, const Eigen::Affine3d & T)
{
  Ad.setZero();
  Ad.topLeftCorner(3,3) = T.linear();
  Ad.bottomRightCorner(3,3) = T.linear();
  static Eigen::Matrix3d P = Eigen::Matrix3d::Zero();
  Vec2Skew(P, T.translation());
  Ad.bottomLeftCorner(3,3).noalias() = P * T.linear();
  return;
}


struct vec_buffer {
    std::array<std::array<double, 6>,2> ft_readings{};
    std::atomic<int> active{0};
};

vec_buffer ft_buffer{};
std::atomic<bool> ft_running(true);

net_ft_driver::NetFtHardwareInterface sensor;

void ft_read(){
  while(ft_running){
    sensor.read();
    int next = 1 - ft_buffer.active.load(std::memory_order_relaxed);
    ft_buffer.ft_readings[next] = sensor.ft_sensor_measurements_;
    ft_buffer.active.store(next, std::memory_order_release);
  }
}

// std::array<double, 7> joint_limit_upper{2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
// std::array<double, 7> joint_limit_lower{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
Vector7d joint_limit_middle = Vector7d{0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0};
// std::array<double, 7> joint_limit_range{5.7946, 3.5256, 5.7946, 3.002 ,5.7946, 3.77, 5.7946};

/**
 * An admittance controller designed to interface with a Axia M8 F/T sensor at the wrist. Bounding boxes,
 * velocity limits, and control parameters can be configured in the configuration named when the contoller
 * is called.
 *
 * @warning collision thresholds are set to high values. Make sure you have the user stop at hand!
 */
int main(int argc, char** argv) {

  std::signal(SIGINT, signal_handler);

  // Check whether the required arguments were passed
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <config-name>" << std::endl;
    return -1;
  }

  std::string config_name{argv[1]};

  std::string package_share_dir = ament_index_cpp::get_package_share_directory("franka_interaction");
  std::string config_path = package_share_dir + "/config/config.json";
  std::ifstream f(config_path);
  
  json config = json::parse(f);

  //how fast the torque is allowed to change
  const double torque_smoothing = config[config_name]["torque_smoothing"];
  const double max_torque_change = std::min(torque_smoothing, 1000.0) / 1000.0;

  double force_limit = config[config_name]["force_limit"];
  double torque_limit = config[config_name]["torque_limit"];

  bool swap_torque = config[config_name]["swap_torque"];

  //Virtual Stiffness
  std::array<double, 6> stiffness_values = config[config_name]["stiffness"];
  Vector6d stiffness_vec = Eigen::Map<Vector6d>(stiffness_values.data());
  const Matrix6d virtual_stiffness = stiffness_vec.asDiagonal();

  //Virtual Damping
  std::array<double, 6> damping_values = config[config_name]["damping"];
  Vector6d damping_vec = Eigen::Map<Vector6d>(damping_values.data());
  const Matrix6d virtual_damping = damping_vec.asDiagonal();

  //mass matrix
  std::array<double, 6> mass_values = config[config_name]["mass"];
  Vector6d mass_vec = Eigen::Map<Vector6d>(mass_values.data());
  Matrix6d virtual_mass = mass_vec.asDiagonal();
  const Matrix6d virtual_mass_inv = virtual_mass.inverse();

  //joint weights
  std::array<double, 7> weight_values = config[config_name]["joint_weight"];
  Vector7d joint_weights = Eigen::Map<Vector7d>(weight_values.data());
  const Matrix7d W_inv = joint_weights.asDiagonal().inverse();

  //friction comp
  bool use_friction_comp = config[config_name]["use_friction_comp"];
  double coulomb_epsilon = config[config_name]["friction_comp"]["friction_sign_epsilon"];
  std::array<double, 7> coulomb_values = config[config_name]["friction_comp"]["friction_coulomb"];
  const Vector7d coulomb_frictions = Eigen::Map<Vector7d>(coulomb_values.data());

  std::array<double, 7> viscous_values = config[config_name]["friction_comp"]["friction_viscous"];
  const Vector7d viscous_frictions = Eigen::Map<Vector7d>(viscous_values.data());

  //boundry conditions
  const bool use_boundry = config[config_name]["use_boundry"];

  std::array<double, 6> boundry_min_values = config[config_name]["boundry"]["min"];
  const Vector6d boundry_min = Eigen::Map<Vector6d>(boundry_min_values.data());

  std::array<double, 6> boundry_max_values = config[config_name]["boundry"]["max"];
  const Vector6d boundry_max = Eigen::Map<Vector6d>(boundry_max_values.data());

  const double boundry_trans_stiffness = config[config_name]["boundry"]["trans_stiffness"];
  const double boundry_rot_stiffness = config[config_name]["boundry"]["rot_stiffness"];
  const double boundry_trans_damping = config[config_name]["boundry"]["trans_damping"];
  const double boundry_rot_damping = config[config_name]["boundry"]["rot_damping"];

  //velocity limits
  bool use_velocity_max = config[config_name]["use_velocity_max"];
  std::array<double, 6> velocity_max_values = config[config_name]["velocity_max"]["max_velocity"];
  const Vector6d velocity_max = Eigen::Map<Vector6d>(velocity_max_values.data());

  std::array<double, 6> velocity_max_damping_values = config[config_name]["velocity_max"]["damping"];
  const Vector6d velocity_max_damping = Eigen::Map<Vector6d>(velocity_max_damping_values.data());

  const std::string ft_ip = config[config_name]["ft_ip"];
  const std::string ns = config[config_name]["ns"];
  const std::string partner_ns = config[config_name]["partner_ns"];

  const bool bilateral = config[config_name]["bilateral_enable"];
  const double bilateral_trans_stiffness = config[config_name]["bilateral_trans_stiff"];
  const double bilateral_rot_stiffness = config[config_name]["bilateral_rot_stiff"];

  std::array<double, 6> bilateral_damping_vec = config[config_name]["bilateral_damping"];
  Vector6d bilateral_damping = Eigen::Map<Vector6d>(bilateral_damping_vec.data());
  const Matrix6d bilateral_C = bilateral_damping.asDiagonal();

  const Eigen::Matrix3d K_T = Eigen::Matrix3d::Identity() * bilateral_trans_stiffness;
  const Eigen::Matrix3d K_R = Eigen::Matrix3d::Identity() * bilateral_rot_stiffness;

  const bool null_space_avoid = config[config_name]["null_enable"];
  const double joint_avoid_k = config[config_name]["joint_limit_stiffness"];
  const double joint_avoid_c = config[config_name]["joint_limit_damping"];

  Vector7d joint_limit_stiffness = joint_avoid_k * Vector7d{0.0297819462745, 0.0804514590908, 0.0297819462745, 0.11096311098, 0.0297819462745, 0.0703586178753, 0.0297819462745};
  const Matrix7d Joint_Avoidance_Stiffness = joint_limit_stiffness.asDiagonal();
  Vector7d joint_limit_damping = joint_avoid_c * Vector7d{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  const Matrix7d Joint_Avoidance_Damping = joint_limit_damping.asDiagonal();


  //connect to sensor, see data sheet for filter selection based on sampling rate.
  net_ft_driver::ft_info input;
  input.ip_address = ft_ip;
  input.sensor_type = "ati_axia";
  input.rdt_sampling_rate = 2000;
  input.use_biasing = "true";
  input.internal_filter_rate = 5;
  sensor = net_ft_driver::NetFtHardwareInterface(input);

  // setup sensor transform
  Eigen::Matrix<double, 3, 3> sensor_rotation;
  //rotated to align with sensor frame, 90 degrees counter clockwise
  sensor_rotation <<  std::cos(-M_PI_2), -std::sin(-M_PI_2), 0,
                      std::sin(-M_PI_2), std::cos(-M_PI_2), 0,
                      0,                0,                1;
  
  // shifted down in sensor frame (up to the user)
  // Luke this is right, stop changing it 
  const Eigen::Vector3d sensor_translation {0.0, 0.0, -0.0424};
  Eigen::Affine3d EE_to_Sensor;

  EE_to_Sensor.linear() = sensor_rotation;
  EE_to_Sensor.translation() = sensor_translation;

  Matrix6d Adjoint_EE_to_Sensor = Matrix6d::Zero();
  Adjoint(Adjoint_EE_to_Sensor, EE_to_Sensor);

  const double load_mass = config[config_name]["load_mass"];
  const double load_weight = load_mass * 9.81;

  const Eigen::Vector3d gravity_vec{0.0, 0.0, -9.81};

  const std::array<double, 3> payload_com{0.0, 0.0, 0.06};
  const std::array<double, 9> payload_inertia{0.0005, 0.0, 0.0, 0.0, 0.0005, 0.0, 0.0, 0.0, 0.0005};

  Eigen::Affine3d Load_to_EE;

  Load_to_EE.translation() = Eigen::Vector3d::Map(payload_com.data());
  Load_to_EE.linear() = Eigen::Matrix3d::Identity();

  Eigen::Affine3d EE_to_Load = Load_to_EE.inverse();

  Eigen::Affine3d Handle_to_EE;

  Handle_to_EE.translation() = Eigen::Vector3d{0.0, 0.0, 0.1};
  Handle_to_EE.linear() = Eigen::Matrix3d::Identity();

  Eigen::Affine3d EE_to_Handle = Handle_to_EE.inverse();

  Matrix6d Adjoint_EE_to_Handle = Matrix6d::Zero();

  Adjoint(Adjoint_EE_to_Handle, EE_to_Handle);

  const Matrix6d Virtual_Mass_EE = Adjoint_EE_to_Handle.transpose() * virtual_mass * Adjoint_EE_to_Handle;
  const Matrix6d Virtual_Mass_Inv_EE = Virtual_Mass_EE.inverse();
  const Matrix6d Virtual_Damping_EE = Adjoint_EE_to_Handle.transpose() * virtual_damping * Adjoint_EE_to_Handle;
  const Matrix6d Virtual_Stiffness_EE = Adjoint_EE_to_Handle.transpose() * virtual_stiffness * Adjoint_EE_to_Handle;

  Eigen::Affine3d Handle_to_World;

  // thread-safe queue to transfer robot data to ROS
  std::thread ros_thread;
  std::thread ft_thread;
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;

  Eigen::Affine3d EE_to_World;
  affine_buffer EE_buffer;

  Eigen::Affine3d Mirror_to_World;
  affine_buffer Mirror_buffer;

  Vector6d_buffer Mirror_twist_buffer;
  Vector6d_buffer EE_twist_buffer;


  constexpr int max_index = 100000;
  int index = 0;
  std::array<int, max_index> elapsed_time_;

  try {
    // connect to robot
    franka::Robot robot(config[config_name]["robot_ip"]);

    // Let it clear errors before we start 
    robot.automaticErrorRecovery();
    setDefaultBehavior(robot, 0.80);

    // First move the robot to a suitable joint configuration
    const std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    MotionGenerator motion_generator(0.5, q_goal);

    robot.control(motion_generator);
    std::cout << "Finished moving to initial joint configuration." << std::endl;

    // load the kinematics and dynamics model
    franka::Model model = robot.loadModel();

    robot.setLoad(load_mass, payload_com, payload_inertia);

    franka::RobotState initial_state = robot.readOnce();

    // equilibrium point is the initial position
    const Eigen::Affine3d EE_to_World_Original(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
    Eigen::Vector3d position_d(EE_to_World_Original.translation());

    //Pre-fill buffers for smooth start

    Mirror_buffer.affines[0] = EE_to_World_Original * Handle_to_EE;
    Mirror_buffer.affines[1] = EE_to_World_Original * Handle_to_EE;
    Mirror_to_World = EE_to_World_Original * Handle_to_EE;

    EE_buffer.affines[0] = EE_to_World_Original * Handle_to_EE;
    EE_buffer.affines[1] = EE_to_World_Original * Handle_to_EE;
    EE_to_World = EE_to_World_Original;

    Handle_to_World = EE_to_World * Handle_to_EE;

    Eigen::Quaterniond orientation_d(EE_to_World_Original.rotation());

    // set collision behavior
    robot.setCollisionBehavior({{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

    // ===============================================
    // BEGIN VARIABLE DECLARATION FOR CONTROL FUNCTION
    // ===============================================
    franka_model_calculations model_calculations;

    std::array<double, 6> ft_reading{};

    Vector6d spatial_accel_d = Vector6d::Zero();

    Vector6d spatial_position = Vector6d::Zero();
    Vector6d old_spatial_position = Vector6d::Zero();

    Vector6d spatial_error = Vector6d::Zero();

    Vector6d spatial_velocity_raw = Vector6d::Zero();
    Vector6d spatial_velocity = Vector6d::Zero();
    Vector6d Mirror_velocity = Vector6d::Zero();
    Vector6d old_spatial_velocity = Vector6d::Zero();

    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    // Vector6d position_d = Vector6d::Zero();

    Vector6d gravity_wrench_Load = Vector6d::Zero();

    Vector7d ddq_d = Vector7d::Zero();
    Vector7d tau_d = Vector7d::Zero();
    Vector7d old_tau_d = Vector7d::Zero();

    Eigen::Matrix<double, 6, 7> old_spatial_jacobian;
    Eigen::Matrix<double, 6, 7> djacobian;

    Matrix6d Adjoint_Load_to_EE = Matrix6d::Zero();
    Adjoint(Adjoint_Load_to_EE, Load_to_EE);

    Matrix6d Adjoint_EE_to_Load = Matrix6d::Zero();
    Adjoint(Adjoint_EE_to_Load, EE_to_Load);

    Vector6d fexternal_wrench_EE = Vector6d::Zero();
    Vector6d gravity_wrench_EE = Vector6d::Zero();
    Vector6d fexternal_wrench_EW = Vector6d::Zero();

    Vector6d damping_wrench_EW = Vector6d::Zero();
    Vector6d spring_wrench_EW = Vector6d::Zero();

    Vector6d fnet_wrench_EW = Vector6d::Zero();

    Vector6d boundary_correction = Vector6d::Zero();
    Vector6d boundary_decel = Vector6d::Zero();

    Vector6d damping_decel = Vector6d::Zero();

    Eigen::Matrix<double, 7, 6> J_inv_weighted = Eigen::Matrix<double, 7, 6>::Zero();

    Vector7d friction_comp_tau = Vector7d::Zero();
    Vector7d dq_smooth_sign = Vector7d::Zero();

    Eigen::Affine3d bilateral_error = Eigen::Affine3d::Identity();

    Eigen::Vector3d bilateral_force_Handle = Eigen::Vector3d::Zero();
    Eigen::Matrix3d bilateral_skew_Handle = Eigen::Matrix3d::Zero();
    Eigen::Vector3d bilateral_torque_Handle = Eigen::Vector3d::Zero();

    Vector6d bilateral_wrench_EW = Vector6d::Zero();

    Eigen::Matrix3d Handle_to_EE_Skew = Eigen::Matrix3d::Zero();
    Vec2Skew(Handle_to_EE_Skew, Eigen::Vector3d{0.0, 0.0, 0.1});

    Vector7d bilateral_tau = Vector7d::Zero();

    std::array<double, 7> tau_d_array{};

    Matrix6d A = Matrix6d::Zero();
    Eigen::LLT<Matrix6d> llt;
    Vector6d rhs = Vector6d::Zero();
    Vector6d lhs = Vector6d::Zero();

    Matrix7d mass_inv = Matrix7d::Zero();

    Matrix6d lambda = Matrix6d::Zero();

    Vector7d null_torque = Vector7d::Zero();

    const Matrix7d I_7 = Matrix7d::Identity();

    Matrix7d N = Matrix7d::Zero();
    Eigen::Matrix<double, 7, 6> J_bar = Eigen::Matrix<double, 7, 6>::Zero();



    constexpr double alpha = 0.1;

    // HARD REAL TIME THREAD 
    //=============================================================//
    //                        CONTROL
    //                        FUNCTION
    //                        START
    //=============================================================//
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback = [&](const franka::RobotState& robot_state,
                                         franka::Duration duration) -> franka::Torques {

      // auto start = std::chrono::high_resolution_clock::now();
      // get state variables
      // model_calculations.coriolis_array = model.coriolis(robot_state);
      // model_calculations.gravity_array = model.gravity(robot_state);
      model_calculations.jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
      model_calculations.mass_array = model.mass(robot_state);
                      
      // update sensor data
      ft_reading = ft_buffer.ft_readings[ft_buffer.active.load(std::memory_order_acquire)];

      // convert to Eigen
      // Eigen::Map<const Vector7d> coriolis(model_calculations.coriolis_array.data());
      // Eigen::Map<const Vector7d> gravity(model_calculations.gravity_array.data());
      Eigen::Map<const Eigen::Matrix<double, 6, 7>> spatial_jacobian(model_calculations.jacobian_array.data());
      Eigen::Map<const Matrix7d> mass(model_calculations.mass_array.data());

      Eigen::Map<const Vector7d> q(robot_state.q.data());
      Eigen::Map<const Vector7d> dq(robot_state.dq.data());
      Eigen::Map<const Vector7d> tau_J(robot_state.tau_J.data());
      Eigen::Map<const Vector7d> tau_J_d(robot_state.tau_J_d.data());

      Eigen::Map<Vector6d> fexternal_wrench_Sensor(ft_reading.data());

      // Pose to end of J7 of Franka T_EE^O
      EE_to_World = Eigen::Matrix4d::Map(robot_state.O_T_EE.data());

      Handle_to_World = EE_to_World * Handle_to_EE;

      int next = 1 - EE_buffer.active.load(std::memory_order_relaxed);
      EE_buffer.affines[next] = Handle_to_World;
      EE_buffer.active.store(next, std::memory_order_release);
      Mirror_to_World = Mirror_buffer.affines[Mirror_buffer.active.load(std::memory_order_acquire)];

      position = EE_to_World.translation();

      Eigen::Quaterniond orientation(EE_to_World.rotation());
      
      spatial_position.head(3) = position;

      // static Vector6d old_spatial_position = spatial_position;


      // Better filtering should probably be included
      // arbitrary cutoff for no duration, expected duration is 0.001
      if (duration.toSec() < 0.00000001) {
        djacobian.setZero();
      } else {
        djacobian = (spatial_jacobian - old_spatial_jacobian) / duration.toSec();
        // spatial_velocity_raw = (spatial_position - old_spatial_position) / duration.toSec();
        // spatial_velocity = alpha * spatial_velocity_raw + (1.0 - alpha) * old_spatial_velocity;
      }

      old_spatial_jacobian = spatial_jacobian;

      // Potentially add Force-Torque filtering

      // translate wrench from FT sensor as wrench in EE frame. MR 3.98
      fexternal_wrench_EE = Adjoint_EE_to_Sensor.transpose() * fexternal_wrench_Sensor;

      // This is the gravity wrench in the LOAD FRAME, we want to convert this to the EE frame
      gravity_wrench_Load.head(3).noalias() = load_mass * Load_to_EE.rotation().transpose() * EE_to_World.rotation().transpose() * gravity_vec;
     
      gravity_wrench_EE = Adjoint_EE_to_Load.transpose() * gravity_wrench_Load;


      fexternal_wrench_EE -= gravity_wrench_EE;
      // Sensor biases weight out so we need to add it back in
      fexternal_wrench_EE(2) += load_weight; 

      // We want to rotate this into a WORLD aligned frame located at the EE

      // translate gravity compensated wrench at EE to base frame to express acceleration in cartesian space.

      fexternal_wrench_EW.head(3) = EE_to_World.rotation() * fexternal_wrench_EE.head(3);
      fexternal_wrench_EW.tail(3) = EE_to_World.rotation() * fexternal_wrench_EE.tail(3);

      // Precompute velocity from jacobian for reuse
      // This is a world aligned twist located at EE
      spatial_velocity.noalias()  = spatial_jacobian * dq;

      int next_vel = 1 - EE_twist_buffer.active.load(std::memory_order_relaxed);
      EE_twist_buffer.vectors[next_vel] = spatial_velocity;
      EE_twist_buffer.active.store(next_vel, std::memory_order_release);
      Mirror_velocity = Mirror_twist_buffer.vectors[Mirror_twist_buffer.active.load(std::memory_order_acquire)];

      // damping_wrench_EW.noalias() = Virtual_Damping_EE * spatial_velocity;
      // damping_wrench_EW.noalias() = virtual_damping * spatial_velocity;

      // Maybe replace with a better error method / Bilateral Control
      // spring_wrench_EW.noalias() = Virtual_Stiffness_EE * spatial_error;


      // Very unclear why this would be needed?
      if (swap_torque) {
        fexternal_wrench_EW(3) = -fexternal_wrench_EW(3);
        fexternal_wrench_EW(4) = -fexternal_wrench_EW(4);
        fexternal_wrench_EW(5) = -fexternal_wrench_EW(5);
      }

      fnet_wrench_EW = fexternal_wrench_EW - damping_wrench_EW - spring_wrench_EW;

      // Clamp fext for saftey 
      for(int i = 0; i <3; ++i) {
        fnet_wrench_EW(i) = std::clamp(fnet_wrench_EW(i), -force_limit, force_limit);
      }

      for(int i = 3; i <6; ++i) {
        fnet_wrench_EW(i) = std::clamp(fnet_wrench_EW(i), -torque_limit, torque_limit);
      }

      // compute control MR 11.66 Virtual Dynamics 
      // a = F/m
      // spatial_accel_d.noalias() = Virtual_Mass_Inv_EE * fnet_wrench_EW;
      spatial_accel_d.noalias() = virtual_mass_inv * fnet_wrench_EW;

      // compute boundry acceleration to keep EE in bounds
      if (use_boundry) {
        boundary_correction.noalias()  = (spatial_position - boundry_max).cwiseMax(0.0) + (spatial_position - boundry_min).cwiseMin(0.0);

        boundary_decel.setZero();
        // if out of bounds anywhere, apply corrective force and damp user movement
        if ((boundary_correction.head(3).array().abs() > 0.001).any()) {
            boundary_decel.head(3).noalias()  = -boundary_correction.head(3) * boundry_trans_stiffness - boundry_trans_damping * spatial_velocity.head(3);
        }
        if ((boundary_correction.tail(3).array().abs() > 0.001).any()) {
            boundary_decel.tail(3).noalias()  = -boundary_correction.tail(3) * boundry_rot_stiffness - boundry_rot_damping * spatial_velocity.tail(3);
        }

        spatial_accel_d += boundary_decel;
      }

      // apply damping above maximum velocity if we are too fast
      if (use_velocity_max) {
          for (int i = 0; i < 6; ++i) {
            damping_decel(i) = -velocity_max_damping(i) * (spatial_velocity(i) - std::clamp(spatial_velocity(i), -velocity_max(i), velocity_max(i)));
          }
        spatial_accel_d += damping_decel;
      }

      // MR 6.7 weighted pseudoinverse
      A.noalias()  = (spatial_jacobian * W_inv * spatial_jacobian.transpose());

      llt = Eigen::LLT<Matrix6d>(A);

      rhs.noalias()= (spatial_accel_d - (djacobian * dq));
      lhs = llt.solve(rhs);
      // translate EE accel to joint accel MR 11.66
      ddq_d.noalias() = W_inv * spatial_jacobian.transpose() * lhs;
      
      // MR 8.1 : inverse dynamics, add all control elements together
      // F = ma
      tau_d.noalias() = (mass * ddq_d);

      if (use_friction_comp) {
        dq_smooth_sign = dq.array() / (dq.array().square() + coulomb_epsilon * coulomb_epsilon).sqrt();

        // total friction comp
        friction_comp_tau.noalias()  =  coulomb_frictions.cwiseProduct(dq_smooth_sign) + viscous_frictions.cwiseProduct(dq);
        tau_d += friction_comp_tau;
      }


      // Bilateral coupling

      //Pose = T^{O}_{EE_A}
      //Partner_Pose = T^{O}_{EE_B}
      //Difference = Pose^-1 * Partner_Pose = T^{EE_A}_{EE_B}

      bilateral_error = Mirror_to_World.inverse() * Handle_to_World;

      bilateral_force_Handle.noalias()  = -bilateral_error.rotation().transpose() * K_T * bilateral_error.translation();
      // bilateral_force_Handle = K_T * bilateral_error.translation();

      for (int i = 0; i<3; ++i){
        bilateral_force_Handle(i) = std::clamp(bilateral_force_Handle(i), -force_limit, force_limit);
      }
      
      // Normally the second K_R should be transposed, but as a diagonal matrix, it does not matter
      bilateral_skew_Handle.noalias()  = -(K_R * bilateral_error.rotation() - bilateral_error.rotation().transpose() * K_R);

      Skew2Vec(bilateral_torque_Handle, bilateral_skew_Handle);

      for (int i = 0; i<3; ++i){
        bilateral_torque_Handle(i) = std::clamp(bilateral_torque_Handle(i), -torque_limit, torque_limit);
      }

      bilateral_wrench_EW.head(3) = EE_to_World.rotation() * bilateral_force_Handle;
      bilateral_wrench_EW.tail(3) = EE_to_World.rotation() * bilateral_torque_Handle + Handle_to_EE_Skew * bilateral_wrench_EW.head(3);

      bilateral_wrench_EW += bilateral_C * (Mirror_velocity - spatial_velocity);

      if(bilateral){
        bilateral_tau.noalias()  = spatial_jacobian.transpose() * bilateral_wrench_EW;
        tau_d += bilateral_tau;
      }

      if(null_space_avoid){
        mass_inv = mass.inverse();
        lambda = spatial_jacobian * mass_inv * spatial_jacobian.transpose();

        J_bar = mass_inv * spatial_jacobian.transpose() * lambda.inverse();
        N = I_7 - spatial_jacobian.transpose() * J_bar.transpose();

        null_torque = -Joint_Avoidance_Stiffness * (q - joint_limit_middle) - Joint_Avoidance_Damping * dq;

        tau_d += N.transpose() * null_torque;
      }

      // Spec sheet lists 1000/sec as maximum but in practice should be much lower for smooth human use.
      for (int i = 0; i < tau_d.size(); ++i) {
        tau_d(i) = std::clamp(tau_d(i), old_tau_d(i) - max_torque_change, old_tau_d(i) + max_torque_change);
      }
      old_tau_d = tau_d;

      // output format
      Eigen::Map<Eigen::Matrix<double, 7, 1>>(tau_d_array.data()) = tau_d;
      
      franka::Torques torques = tau_d_array;

      // auto end = std::chrono::high_resolution_clock::now();

      // if(index < max_index)
      // {
      //   elapsed_time_[index] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      // } else{
      //   robot_stop = true;
      // }

      // index++;

      // if ctrl-c is pressed, robot should stop
      if (robot_stop) { return franka::MotionFinished(torques); }
      return torques;
    };

    // start real-time control loop
    std::cout << "WARNING: Collision thresholds are set to high values. "
              << "Make sure you have the user stop at hand!" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    sensor.re_bias();

    auto node = std::make_shared<MinimalPublisher>(EE_buffer, EE_twist_buffer, ns, Mirror_buffer, Mirror_twist_buffer, partner_ns);
    executor.add_node(node);
    ros_thread = std::thread([&executor, node]() { executor.spin(); });

    ft_thread = std::thread(ft_read);

    robot.control(impedance_control_callback, false, 200.0);
  } catch (const franka::Exception& ex) {
    std::cout << "Franka Exception: " << ex.what() << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "Misc Exception: " << ex.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception caught." << std::endl;
  }

  ft_running = false;

  rclcpp::shutdown();
  ros_thread.join();
  ft_thread.join();
  sensor.on_deactivate();

  // std::ofstream outfile("src/Data/"+ns+ "_timings.csv");
  // if (outfile.is_open()) {
  //       // Write each value of elapsed_time_ to the file
  //       for (int i = 0; i < max_index; ++i) {
  //           outfile << elapsed_time_[i];

  //           // If it's not the last element, add a comma to separate the values
  //           if (i < max_index - 1) {
  //               outfile << ",";
  //           }
  //       }
  //       outfile << "\n";  // Newline at the end of the row

  //       std::cout << "Data has been written to data.csv\n";
  //   } else {
  //       std::cerr << "Error opening file for writing\n";
  //   }

  //   outfile.close();  // Close the file

  return 0;
}
