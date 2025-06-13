#include "thrust_allocation/thrust_allocation_node.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <string>
#include <spdlog/spdlog.h>

using std::placeholders::_1;

ThrustAllocationNode::ThrustAllocationNode()
    : Node("thrust_allocation_node")
    {
    tau_desired_ = {0.0, 0.0, 0.0};
    init_parameters();

    allocate_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(dt_ * 1000)),
        std::bind(&ThrustAllocationNode::allocate_callback, this));

    init_maneuvering_allocator();
    init_nlp_allocator();
    init_qp_allocator();
    init_pseudo_inverse_allocator();
    init_subscribers_and_publishers();

    publish_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1 / publish_rate_ * 1000)),
        std::bind(&ThrustAllocationNode::publish_callback, this));

    spdlog::info("Thrust Allocation Node initialized with allocator mode: {}", allocator_mode_);
    }

void ThrustAllocationNode::init_parameters() {
    this->declare_parameter<double>("cpp.publish_rate");
    this->declare_parameter<double>("cpp.loop_rate");

    loop_rate_ = this->get_parameter("cpp.loop_rate").as_double();
    dt_ = 1.0 / loop_rate_;
    publish_rate_ = this->get_parameter("cpp.publish_rate").as_double();

    this->declare_parameter<std::string>("tau_topic");
    this->declare_parameter<std::string>("out_topic");

    this->declare_parameter<std::string>("allocator");
    allocator_mode_ = this->get_parameter("allocator").as_string();

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
    this->declare_parameter<double>("nlp.u_bound");
    this->declare_parameter<double>("nlp.w_angle");
    this->declare_parameter<double>("nlp.w_neg");
    this->declare_parameter<double>("nlp.max_rate");
    this->declare_parameter<double>("nlp.max_force_rate");
    this->declare_parameter<double>("nlp.w_alpha_change");

    // QP allocator parameters
    this->declare_parameter<double>("qp.u_bound");
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
    NLPParams params;
    params.dt = dt_;
    params.u_bound = this->get_parameter("nlp.u_bound").as_double();
    params.w_angle = this->get_parameter("nlp.w_angle").as_double();
    params.w_neg = this->get_parameter("nlp.w_neg").as_double();
    params.max_rate = this->get_parameter("nlp.max_rate").as_double();
    params.max_force_rate = this->get_parameter("nlp.max_force_rate").as_double();
    params.w_alpha_change = this->get_parameter("nlp.w_alpha_change").as_double();
    nlp_allocator_ = std::make_unique<NLPAllocator>(params);
}

void ThrustAllocationNode::init_qp_allocator() {
    QPParameters params;
    params.dt = dt_;
    params.u_bound = this->get_parameter("qp.u_bound").as_double();
    params.max_force_rate = this->get_parameter("qp.max_force_rate").as_double();
    params.beta = this->get_parameter("qp.beta").as_double();

    qp_allocator_ = std::make_unique<QPAllocator>(params);
}

void ThrustAllocationNode::init_pseudo_inverse_allocator() {
    pseudo_inverse_allocator_ = std::make_unique<PseudoInverseAllocator>();
}

void ThrustAllocationNode::init_subscribers_and_publishers() {
    tau_sub_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
        this->get_parameter("tau_topic").as_string(),
        1,
        std::bind(&ThrustAllocationNode::tau_callback, this, _1));
    out_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        this->get_parameter("out_topic").as_string(),
        1);
}

void ThrustAllocationNode::tau_callback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
    tau_desired_[0] = msg->wrench.force.x;
    tau_desired_[1] = msg->wrench.force.y;
    tau_desired_[2] = msg->wrench.torque.z;
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

    std::vector<double> thrusts_and_angles;
    for (int i = 0; i < 4; ++i) {
        thrusts_and_angles.push_back(u_opt[i]);
    }
    for (int i = 0; i < 4; ++i) {
        thrusts_and_angles.push_back(alpha_opt[i]);
    }
    out_msg_.data = thrusts_and_angles;

}

void ThrustAllocationNode::publish_callback() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    out_pub_->publish(out_msg_);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ThrustAllocationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
}