#ifndef THRUST_ALLOCATION_NODE_HPP
#define THRUST_ALLOCATION_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include "thrust_allocation/nlp.hpp"
#include "thrust_allocation/maneuvering_allocator.hpp"
#include "thrust_allocation/qp.hpp"
#include "thrust_allocation/pseudo_inverse.hpp"
#include <geometry_msgs/msg/wrench_stamped.hpp>
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

    void tau_callback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg);

    void allocate_callback();

    void publish_callback();

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr out_pub_;

    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr tau_sub_;

    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::TimerBase::SharedPtr allocate_timer_;

    std_msgs::msg::Float64MultiArray out_msg_;

    std::mutex data_mutex_;
    double publish_rate_;
    double loop_rate_;

    double dt_;
    std::string allocator_mode_;
    std::array<double, 3> tau_desired_;

    std::unique_ptr<NLPAllocator> nlp_allocator_;
    std::unique_ptr<ManeuveringAllocator> maneuvering_allocator_;
    std::unique_ptr<QPAllocator> qp_allocator_;
    std::unique_ptr<PseudoInverseAllocator> pseudo_inverse_allocator_;
};

#endif // THRUST_ALLOCATION_NODE_HPP