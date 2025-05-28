#include "thrust_allocation/thrust_allocation_node.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

using std::placeholders::_1;

ThrustAllocationNode::ThrustAllocationNode()
    : Node("thrust_allocation_node"),
      dt_(0.01) {
    tau_desired_ = {0.0, 0.0, 0.0};
    allocate_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(dt_ * 1000)),
        std::bind(&ThrustAllocationNode::allocate_callback, this));

    init_parameters();
    init_maneuvering_allocator();
    init_nlp_allocator();
    init_qp_allocator();
    init_pseudo_inverse_allocator();
    init_subscribers_and_publishers();

    publish_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1 / publish_rate_ * 1000)),
        std::bind(&ThrustAllocationNode::publish_callback, this));

    spdlog::info("Thrust Allocation Node initialized");
    }

void ThrustAllocationNode::init_parameters() {
    this->declare_parameter<double>("rpm_thrust_mapping.p_propeller_rpm_to_command");
    this->declare_parameter<std::vector<double>>("rpm_thrust_mapping.p_thrust_to_propeller_rpm");
    this->declare_parameter<double>("single_thruster_min_force");
    this->declare_parameter<double>("single_thruster_max_force");
    this->declare_parameter<double>("publish_rate");

    p_propeller_rpm_to_command_ = this->get_parameter("rpm_thrust_mapping.p_propeller_rpm_to_command").as_double();
    p_thrust_to_propeller_rpm_ = this->get_parameter("rpm_thrust_mapping.p_thrust_to_propeller_rpm").as_double_array();
    single_thruster_min_force_ = this->get_parameter("single_thruster_min_force").as_double();
    single_thruster_max_force_ = this->get_parameter("single_thruster_max_force").as_double();
    publish_rate_ = this->get_parameter("publish_rate").as_double();

    this->declare_parameter<std::string>("topics.control_action_reference");
    this->declare_parameter<std::string>("topics.actuator_1_angle");
    this->declare_parameter<std::string>("topics.actuator_2_angle");
    this->declare_parameter<std::string>("topics.actuator_3_angle");
    this->declare_parameter<std::string>("topics.actuator_4_angle");
    this->declare_parameter<std::string>("topics.actuator_1_setpoint");
    this->declare_parameter<std::string>("topics.actuator_2_setpoint");
    this->declare_parameter<std::string>("topics.actuator_3_setpoint");
    this->declare_parameter<std::string>("topics.actuator_4_setpoint");

    // Maneuvering allocator parameters
    this->declare_parameter<double>("maneuvering.gamma");
    this->declare_parameter<double>("maneuvering.mu");
    this->declare_parameter<double>("maneuvering.rho");
    this->declare_parameter<double>("maneuvering.zeta");
    this->declare_parameter<double>("maneuvering.rate_limit");
    this->declare_parameter<double>("maneuvering.theta_min");
    this->declare_parameter<double>("maneuvering.theta_max");
    this->declare_parameter<double>("maneuvering.lambda");
    this->declare_parameter<bool>("maneuvering.power_management");

    // NLP allocator parameters

    // QP allocator parameters
    this->declare_parameter<double>("qp.u_bound");
    this->declare_parameter<double>("qp.max_rate");
    this->declare_parameter<double>("qp.max_force_rate");
    this->declare_parameter<double>("qp.beta");
}

void ThrustAllocationNode::init_maneuvering_allocator() {
    ManeuveringParams params;
    params.dt = dt_;
    params.gamma = this->get_parameter("maneuvering.gamma").as_double();
    params.mu = this->get_parameter("maneuvering.mu").as_double();
    params.rho = this->get_parameter("maneuvering.rho").as_double();
    params.zeta = this->get_parameter("maneuvering.zeta").as_double();
    params.rate_limit = this->get_parameter("maneuvering.rate_limit").as_double();
    params.theta_min = this->get_parameter("maneuvering.theta_min").as_double();
    params.theta_max = this->get_parameter("maneuvering.theta_max").as_double();
    params.lambda = this->get_parameter("maneuvering.lambda").as_double();
    params.power_management = this->get_parameter("maneuvering.power_management").as_bool();

    maneuvering_allocator_ = std::make_unique<ManeuveringAllocator>(params);
}

void ThrustAllocationNode::init_nlp_allocator() {
    nlp_allocator_ = std::make_unique<NLPAllocator>(dt_);
}

void ThrustAllocationNode::init_qp_allocator() {
    QPParameters params;
    params.dt = dt_;
    params.u_bound = this->get_parameter("qp.u_bound").as_double();
    params.max_rate = this->get_parameter("qp.max_rate").as_double();
    params.max_force_rate = this->get_parameter("qp.max_force_rate").as_double();
    params.beta = this->get_parameter("qp.beta").as_double();

    qp_allocator_ = std::make_unique<QPAllocator>(params);
}

void ThrustAllocationNode::init_pseudo_inverse_allocator() {
    pseudo_inverse_allocator_ = std::make_unique<PseudoInverseAllocator>();
}

void ThrustAllocationNode::init_subscribers_and_publishers() {
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos_sensor_data = rclcpp::QoS(
        rclcpp::QoSInitialization(qos_profile.history, 1), qos_profile);

    actuator_1_pub_ = this->create_publisher<custom_msgs::msg::ActuatorSetpoints>(
        this->get_parameter("topics.actuator_1_setpoint").as_string(), 1);
    actuator_2_pub_ = this->create_publisher<custom_msgs::msg::ActuatorSetpoints>(
        this->get_parameter("topics.actuator_2_setpoint").as_string(), 1);
    actuator_3_pub_ = this->create_publisher<custom_msgs::msg::ActuatorSetpoints>(
        this->get_parameter("topics.actuator_3_setpoint").as_string(), 1);
    actuator_4_pub_ = this->create_publisher<custom_msgs::msg::ActuatorSetpoints>(
        this->get_parameter("topics.actuator_4_setpoint").as_string(), 1);
    
    tau_actual_pub_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(
        "tau_actual", 1);
    tau_diff_pub_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(
        "tau_diff", qos_sensor_data);
    forces_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "forces", 1);

    tau_sub_ = this->create_subscription<geometry_msgs::msg::Wrench>(
        this->get_parameter("topics.control_action_reference").as_string(),
        1,
        std::bind(&ThrustAllocationNode::tau_callback, this, _1));
    mode_sub_ = this->create_subscription<std_msgs::msg::String>(
        "allocator_mode",
        1,
        std::bind(&ThrustAllocationNode::mode_callback, this, _1)
    );
}

void ThrustAllocationNode::tau_callback(const geometry_msgs::msg::Wrench::SharedPtr msg) {
    tau_desired_[0] = msg->force.x;
    tau_desired_[1] = msg->force.y;
    tau_desired_[2] = msg->torque.z;
}

void ThrustAllocationNode::mode_callback(const std_msgs::msg::String::SharedPtr msg) {
    static const std::vector<std::string> valid_modes = {
        "nlp", "qp", "maneuvering", "pseudo_inverse"
    };

    // Check if mode is valid
    if (std::find(valid_modes.begin(), valid_modes.end(), msg->data) != valid_modes.end())
    {
        allocator_mode_ = msg->data;

        if (allocator_mode_ == "maneuvering")
        {
            maneuvering_allocator_->theta_.setZero();
        }

        spdlog::info("Allocator mode set to: {}", allocator_mode_);
    }
    else
    {
        spdlog::error("Invalid allocator mode: {}", msg->data);
    }
}

void ThrustAllocationNode::allocate_callback() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    Eigen::Vector4d alpha_opt, u_opt;
    Eigen::Vector3d tau_opt;
    Eigen::VectorXd f_ext;
    if (allocator_mode_ == "nlp") {
        std::tie(alpha_opt, u_opt, tau_opt) = nlp_allocator_->allocate(tau_desired_);
    }
    else if (allocator_mode_ == "qp") {
        std::tie(f_ext, u_opt, alpha_opt, tau_opt) = qp_allocator_->allocate(tau_desired_);
    }
    else if (allocator_mode_ == "maneuvering") {
        std::tie(f_ext, u_opt, alpha_opt, tau_opt) = qp_allocator_->allocate(tau_desired_);
        std::tie(u_opt, alpha_opt, tau_opt) = maneuvering_allocator_->allocate(tau_desired_, f_ext);
    }
    else if (allocator_mode_ == "pseudo_inverse") {
        std::tie(alpha_opt, u_opt, tau_opt) = pseudo_inverse_allocator_->allocate(tau_desired_);
    }
    else {
        spdlog::error("Invalid allocator mode: {}", allocator_mode_);
        return;
    }

    std::vector<double> thrusts;
    for (int i = 0; i < 4; ++i) {
        thrusts.push_back(u_opt[i]);
    }
    forces_msg_.data = thrusts;

    tau_actual_msg_.header.stamp = this->now();
    tau_actual_msg_.wrench.force.x = tau_opt[0];
    tau_actual_msg_.wrench.force.y = tau_opt[1];
    tau_actual_msg_.wrench.torque.z = tau_opt[2];
   
    tau_diff_msg_.header.stamp = this->now();
    tau_diff_msg_.wrench.force.x = tau_desired_[0] - tau_opt[0];
    tau_diff_msg_.wrench.force.y = tau_desired_[1] - tau_opt[1];
    tau_diff_msg_.wrench.torque.z = tau_desired_[2] - tau_opt[2];

    auto rad_2_deg = [](double rad) {
        return rad * 180.0 / M_PI;
    };

    act1_.throttle_reference = thrust_to_command(u_opt[0]);
    act1_.angle_reference = rad_2_deg(alpha_opt[0]);
    act2_.throttle_reference = thrust_to_command(u_opt[1]);
    act2_.angle_reference = rad_2_deg(alpha_opt[1]);
    act3_.throttle_reference = thrust_to_command(u_opt[3]);
    act3_.angle_reference = rad_2_deg(alpha_opt[3]);
    act4_.throttle_reference = thrust_to_command(u_opt[2]);
    act4_.angle_reference = rad_2_deg(alpha_opt[2]);
}

void ThrustAllocationNode::publish_callback() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    actuator_1_pub_->publish(act1_);
    actuator_2_pub_->publish(act2_);
    actuator_3_pub_->publish(act3_);
    actuator_4_pub_->publish(act4_);

    tau_actual_pub_->publish(tau_actual_msg_);
    tau_diff_pub_->publish(tau_diff_msg_);

    forces_pub_->publish(forces_msg_);
}

double ThrustAllocationNode::thrust_to_rpm(double thrust) const {
    constexpr double coeff_pos[3] = {-0.005392628724401, 4.355160257309755, 0.0};
    constexpr double coeff_neg[3] = { 0.009497954053629, 5.891321999522722, 0.0};
    constexpr double tmax =  430.0;
    constexpr double tmin = -330.0;

    auto polyval = [](const double (&c)[3], double x) {
    return (c[2] + c[1]) * x + c[0] * x * x;
    };

    double rpm;
    if (thrust < 0.0) {
    rpm = polyval(coeff_neg, thrust);
    if (thrust < tmin) rpm = polyval(coeff_neg, tmin);
    } else {
    rpm = polyval(coeff_pos, thrust);
    if (thrust > tmax) rpm = polyval(coeff_pos, tmax);
    }
    return rpm;
}

double ThrustAllocationNode::rpm_to_command(double rpm) const {
    double command = p_propeller_rpm_to_command_ * rpm;
    return std::clamp(command, -1000.0, 1000.0);
}

double ThrustAllocationNode::thrust_to_command(double thrust) const {
    return rpm_to_command(thrust_to_rpm(thrust));
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ThrustAllocationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
}