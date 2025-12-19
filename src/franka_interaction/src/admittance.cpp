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
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <config-name>" << " <publish?>"<< std::endl;
    return -1;
  }
  std::string config_name{argv[1]};
  std::string ros2_publish{argv[2]};

  std::string package_share_dir = ament_index_cpp::get_package_share_directory("franka_interaction");
  std::string config_path = package_share_dir + "/config/config.json";
  std::ifstream f(config_path);
  
  json config = json::parse(f);

  //how fast the torque is allowed to change
  const double torque_smoothing = config[config_name]["torque_smoothing"];
  const double max_torque_change = torque_smoothing / 1000.0;

  double force_limit = config[config_name]["force_limit"];
  double torque_limit = config[config_name]["torque_limit"];

  bool swap_torque = config[config_name]["swap_torque"];

  //stiffness
  std::vector<double> stiffness_values = config[config_name]["stiffness"];
  Vector6d stiffness_vec = Eigen::Map<Vector6d>(stiffness_values.data(), stiffness_values.size());
  Matrix6d stiffness = stiffness_vec.asDiagonal();

  //damping
  std::vector<double> damping_values = config[config_name]["damping"];
  Vector6d damping_vec = Eigen::Map<Vector6d>(damping_values.data(), damping_values.size());
  Matrix6d damping = damping_vec.asDiagonal();

  //mass matrix
  std::vector<double> mass_values = config[config_name]["mass"];
  Vector6d mass_vec = Eigen::Map<Vector6d>(mass_values.data(), mass_values.size());
  Matrix6d virtual_mass = mass_vec.asDiagonal();
  const Matrix6d M_v_inv = virtual_mass.inverse();

  //joint weights
  std::vector<double> weight_values = config[config_name]["joint_weight"];
  Vector7d joint_weights = Eigen::Map<Vector7d>(weight_values.data(), weight_values.size());
  const Matrix7d W_inv = joint_weights.asDiagonal().inverse();

  //friction comp
  bool use_friction_comp = config[config_name]["use_friction_comp"];
  double coulomb_epsilon = config[config_name]["friction_comp"]["friction_sign_epsilon"];
  std::vector<double> coulomb_values = config[config_name]["friction_comp"]["friction_coulomb"];
  Vector7d coulomb_frictions = Eigen::Map<Vector7d>(coulomb_values.data(), coulomb_values.size());
  std::vector<double> viscous_values = config[config_name]["friction_comp"]["friction_viscous"];
  Vector7d viscous_frictions = Eigen::Map<Vector7d>(viscous_values.data(), viscous_values.size());

  //boundry conditions
  bool use_boundry = config[config_name]["use_boundry"];

  std::vector<double> boundry_min_values = config[config_name]["boundry"]["min"];
  Vector6d boundry_min = Eigen::Map<Vector6d>(boundry_min_values.data(), boundry_min_values.size());

  std::vector<double> boundry_max_values = config[config_name]["boundry"]["max"];
  Vector6d boundry_max = Eigen::Map<Vector6d>(boundry_max_values.data(), boundry_max_values.size());

  double boundry_trans_stiffness = config[config_name]["boundry"]["trans_stiffness"];
  double boundry_rot_stiffness = config[config_name]["boundry"]["rot_stiffness"];
  double boundry_trans_damping = config[config_name]["boundry"]["trans_damping"];
  double boundry_rot_damping = config[config_name]["boundry"]["rot_damping"];

  //velocity limits
  bool use_velocity_max = config[config_name]["use_velocity_max"];
  std::vector<double> velocity_max_values = config[config_name]["velocity_max"]["max_velocity"];
  Vector6d velocity_max = Eigen::Map<Vector6d>(velocity_max_values.data(), velocity_max_values.size());

  std::vector<double> velocity_max_damping_values = config[config_name]["velocity_max"]["damping"];
  Vector6d velocity_max_damping = Eigen::Map<Vector6d>(velocity_max_damping_values.data(), velocity_max_damping_values.size());

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
  // Double check?
  // Luke this is right, stop changing it 
  Eigen::Vector3d sensor_translation {0.0, 0.0, -0.0424};
  Eigen::Affine3d EE_to_Sensor;

  EE_to_Sensor.linear() = sensor_rotation;
  EE_to_Sensor.translation() = sensor_translation;

  Matrix6d Adjoint_EE_to_Sensor = Matrix6d::Zero();
  Adjoint(Adjoint_EE_to_Sensor, EE_to_Sensor);

  double load_mass = config[config_name]["load_mass"];
  double load_weight = load_mass * 9.81;

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

  try {
    // connect to robot
    franka::Robot robot(config[config_name]["robot_ip"]);
    robot.automaticErrorRecovery();
    setDefaultBehavior(robot, 0.80);

    // First move the robot to a suitable joint configuration
    std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
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

    // BEGIN VARIABLE DECLARATION FOR CONTROL FUNCTION
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

    Eigen::Vector3d bilateral_force_EE = Eigen::Vector3d::Zero();
    Eigen::Matrix3d bilateral_skew_EE = Eigen::Matrix3d::Zero();
    Eigen::Vector3d bilateral_torque_EE = Eigen::Vector3d::Zero();

    Vector6d bilateral_wrench_EW = Vector6d::Zero();

    Vector7d bilateral_tau = Vector7d::Zero();

    std::array<double, 7> tau_d_array{};


    Matrix6d A = Matrix6d::Zero();
    Eigen::LLT<Matrix6d> llt;
    Vector6d rhs = Vector6d::Zero();
    Vector6d lhs = Vector6d::Zero();

    constexpr double alpha = 0.1;

    
    
    // define callback for the torque control loop

    //=============================================================//
    //                        CONTROL
    //                        FUNCTION
    //                        START
    //=============================================================//
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback = [&](const franka::RobotState& robot_state,
                                         franka::Duration duration) -> franka::Torques {
      // get state variables
      model_calculations.coriolis_array = model.coriolis(robot_state);
      model_calculations.gravity_array = model.gravity(robot_state);
      model_calculations.jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
      model_calculations.mass_array = model.mass(robot_state);
                      
      // update sensor data
      ft_reading = ft_buffer.ft_readings[ft_buffer.active.load(std::memory_order_acquire)];

      // convert to Eigen
      Eigen::Map<const Vector7d> coriolis(model_calculations.coriolis_array.data());
      Eigen::Map<const Vector7d> gravity(model_calculations.gravity_array.data());
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

      // spatial_error.head(3) << position - position_d;

      // // This isn't wrong yet, but why?
      // // orientation error
      // // "difference" quaternion
      // if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
      //   orientation.coeffs() << -orientation.coeffs();
      // }

      // // This is wrong 
      // // "difference" quaternion for use in control
      // Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);
      // spatial_error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
      // // Transform to base frame
      // spatial_error.tail(3) << -EE_to_World.rotation() * spatial_error.tail(3);

      // // axis angle representation for use in boundaries and logging (in base frame)
      // Eigen::AngleAxisd angle_axis(error_quaternion);
      // Eigen::Vector3d orientation_error_axis_angle = -EE_to_World.rotation() * (angle_axis.angle() * angle_axis.axis());
      
      spatial_position.head(3) = position;

      // static Vector6d old_spatial_position = spatial_position;
      
      // arbitrary cutoff for no duration, expected duration is 0.001
      if (duration.toSec() < 0.00000001) {
        djacobian.setZero();
      } else {
        djacobian = (spatial_jacobian - old_spatial_jacobian) / duration.toSec();
        // spatial_velocity_raw = (spatial_position - old_spatial_position) / duration.toSec();
        // spatial_velocity = alpha * spatial_velocity_raw + (1.0 - alpha) * old_spatial_velocity;
      }

      // non static update
      old_spatial_jacobian = spatial_jacobian;
      // old_spatial_velocity = spatial_velocity;
      // old_spatial_position = spatial_position;

      // Potentially add Force-Torque filtering

      // translate wrench from FT sensor as wrench in EE frame. MR 3.98
      fexternal_wrench_EE = Adjoint_EE_to_Sensor.transpose() * fexternal_wrench_Sensor;

      // This is the gravity wrench in the LOAD FRAME, we want to convert this to the EE frame
      gravity_wrench_Load.head(3).noalias() = load_mass * Load_to_EE.rotation().transpose() * EE_to_World.rotation().transpose() * gravity_vec;
     
      gravity_wrench_EE = Adjoint_EE_to_Load.transpose() * gravity_wrench_Load;


      fexternal_wrench_EE -= gravity_wrench_EE;
      // Sensor biases weight out so we need to add it back in
      fexternal_wrench_EE(2) += load_weight; 

      //We want to rotate this into a WORLD aligned frame located at the EE

      // translate gravity compensated wrench at EE to base frame to express acceleration in cartesian space.

      fexternal_wrench_EW.head(3) = EE_to_World.rotation() * fexternal_wrench_EE.head(3);
      fexternal_wrench_EW.tail(3) = EE_to_World.rotation() * fexternal_wrench_EE.tail(3);

      //precompute velocity from jacobian for reuse
      // This is a world aligned twist located at EE
      spatial_velocity.noalias()  = spatial_jacobian * dq;

      int next_vel = 1 - EE_twist_buffer.active.load(std::memory_order_relaxed);
      EE_twist_buffer.vectors[next_vel] = spatial_velocity;
      EE_twist_buffer.active.store(next_vel, std::memory_order_release);
      Mirror_velocity = Mirror_twist_buffer.vectors[Mirror_twist_buffer.active.load(std::memory_order_acquire)];

      damping_wrench_EW.noalias() = damping * spatial_velocity;

      // Maybe replace with a better error method / Bilateral Control
      // spring_wrench_EW.noalias() = stiffness * spatial_error;

      fnet_wrench_EW = fexternal_wrench_EW - damping_wrench_EW - spring_wrench_EW;

      // Clamp fext to help prevent off-phase run away
      for(int i = 0; i <3; ++i) {
        fnet_wrench_EW(i) = std::clamp(fnet_wrench_EW(i), -force_limit, force_limit);
      }
      for(int i = 3; i <6; ++i) {
        fnet_wrench_EW(i) = std::clamp(fnet_wrench_EW(i), -torque_limit, torque_limit);
      }

      if (swap_torque) {
        fnet_wrench_EW(3) = -fnet_wrench_EW(3);
        fnet_wrench_EW(4) = -fnet_wrench_EW(4);
        fnet_wrench_EW(5) = -fnet_wrench_EW(5);
      }

      // compute control MR 11.66 Virtual Dynamics 
      // a = F/m
      spatial_accel_d.noalias() = M_v_inv * fnet_wrench_EW;

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
      tau_d.noalias() = (mass * ddq_d) + coriolis;

      if (use_friction_comp) {
        dq_smooth_sign= dq.array() / (dq.array().square() + coulomb_epsilon * coulomb_epsilon).sqrt();

        // total friction comp
        friction_comp_tau.noalias()  =  coulomb_frictions.cwiseProduct(dq_smooth_sign) + viscous_frictions.cwiseProduct(dq);
        tau_d += friction_comp_tau;
      }

      // Bilateral coupling

      //Pose = T^{O}_{EE_A}
      //Partner_Pose = T^{O}_{EE_B}
      //Difference = Pose^-1 * Partner_Pose = T^{EE_A}_{EE_B}

      bilateral_error = Mirror_to_World.inverse() * Handle_to_World;

      bilateral_force_EE.noalias()  = -bilateral_error.rotation().transpose() * K_T * bilateral_error.translation();
      // bilateral_force_EE = K_T * bilateral_error.translation();

      for (int i = 0; i<3; ++i){
        bilateral_force_EE(i) = std::clamp(bilateral_force_EE(i), -force_limit, force_limit);
      }
      
      // Normally the second K_R should be transposed, but as a diagonal matrix, it does not matter
      bilateral_skew_EE.noalias()  = -(K_R * bilateral_error.rotation() - bilateral_error.rotation().transpose() * K_R);

      Skew2Vec(bilateral_torque_EE, bilateral_skew_EE);

      for (int i = 0; i<3; ++i){
        bilateral_torque_EE(i) = std::clamp(bilateral_torque_EE(i), -torque_limit, torque_limit);
      }

      bilateral_wrench_EW.head(3) = EE_to_World.rotation() * bilateral_force_EE;
      bilateral_wrench_EW.tail(3) = EE_to_World.rotation() * bilateral_torque_EE;

      bilateral_wrench_EW += bilateral_C * (Mirror_velocity - spatial_velocity);


      if(bilateral){
        bilateral_tau.noalias()  = spatial_jacobian.transpose() * bilateral_wrench_EW;
        tau_d += bilateral_tau;
      }

      //Spec sheet lists 1000/sec as maximum but in practice should be much lower for smooth human use.
      for (int i = 0; i < tau_d.size(); ++i) {
        tau_d(i) = std::clamp(tau_d(i), old_tau_d(i) - max_torque_change, old_tau_d(i) + max_torque_change);
      }
      old_tau_d = tau_d;

      // output format
      Eigen::Map<Eigen::Matrix<double, 7, 1>>(tau_d_array.data()) = tau_d;
      
      franka::Torques torques = tau_d_array;

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

  
    robot.control(impedance_control_callback);
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
  return 0;
}
