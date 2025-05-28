#ifndef THRUST_ALLOCATION_NODE_HPP
#define THRUST_ALLOCATION_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include "thrust_allocation/nlp.hpp"
#include "thrust_allocation/maneuvering_allocator.hpp"
#include "thrust_allocation/qp.hpp"
#include "thrust_allocation/pseudo_inverse.hpp"
#include <geometry_msgs/msg/wrench.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <custom_msgs/msg/actuator_setpoints.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <array>
#include <vector>
#include <string>

class ThrustAllocationNode : public rclcpp::Node {
public:
    ThrustAllocationNode();

private:
    void init_parameters();

    void init_maneuvering_allocator();
    void init_nlp_allocator();
    void init_pseudo_inverse_allocator();
    void init_qp_allocator();

    void init_subscribers_and_publishers();

    void tau_callback(const geometry_msgs::msg::Wrench::SharedPtr msg);

    void allocate_callback();

    void publish_callback();

    void mode_callback(const std_msgs::msg::String::SharedPtr msg);

    double thrust_to_rpm(double thrust) const;

    double rpm_to_command(double rpm) const;

    double thrust_to_command(double thrust) const;

    rclcpp::Publisher<custom_msgs::msg::ActuatorSetpoints>::SharedPtr actuator_1_pub_;
    rclcpp::Publisher<custom_msgs::msg::ActuatorSetpoints>::SharedPtr actuator_2_pub_;
    rclcpp::Publisher<custom_msgs::msg::ActuatorSetpoints>::SharedPtr actuator_3_pub_;
    rclcpp::Publisher<custom_msgs::msg::ActuatorSetpoints>::SharedPtr actuator_4_pub_;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr tau_actual_pub_;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr tau_diff_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr forces_pub_;

    rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr tau_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mode_sub_;

    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::TimerBase::SharedPtr allocate_timer_;

    custom_msgs::msg::ActuatorSetpoints act1_, act2_, act3_, act4_;
    std_msgs::msg::Float64MultiArray forces_msg_;
    geometry_msgs::msg::WrenchStamped tau_actual_msg_;
    geometry_msgs::msg::WrenchStamped tau_diff_msg_;

    std::mutex data_mutex_;
    double publish_rate_;

    double dt_;
    std::string allocator_mode_ = "pseudo_inverse";
    std::array<double, 3> tau_desired_;
    double p_propeller_rpm_to_command_;
    std::vector<double> p_thrust_to_propeller_rpm_;
    double single_thruster_min_force_;
    double single_thruster_max_force_;

    std::unique_ptr<NLPAllocator> nlp_allocator_;
    std::unique_ptr<ManeuveringAllocator> maneuvering_allocator_;
    std::unique_ptr<QPAllocator> qp_allocator_;
    std::unique_ptr<PseudoInverseAllocator> pseudo_inverse_allocator_;
};

#endif // THRUST_ALLOCATION_NODE_HPP