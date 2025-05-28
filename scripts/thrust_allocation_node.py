#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Float64
from geometry_msgs.msg import Wrench, WrenchStamped
from thrust_allocation.thrust_allocation import ThrustAllocation
from custom_msgs.msg import ActuatorSetpoints
from math import pi
import numpy as np
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

best_effort_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT
)

class ThrustAllocationNode(Node):
    def __init__(self):
        super().__init__('thrust_allocation_node')

        self.thrust_allocation_ = ThrustAllocation()

        self.init_values()

        self.init_params()

        self.init_subscribers()

        self.init_publishers()

        self.timer_ = self.create_timer(self.dt, self.timer_callback)

    def init_values(self):
        self.azi_angles_base_ : np.ndarray = np.array([3*pi/4, -3*pi/4, -pi/4, pi/4])
        self.last_angles_ : np.ndarray = self.azi_angles_base_
        self.last_thrust_ : np.ndarray = np.array([0, 0, 0, 0])
        self.tau_ : np.ndarray = np.array([2, 0, 0])

    def init_params(self):
        self.declare_parameter('dt', 0.1)
        self.dt = self.get_parameter('dt').value

        self.declare_parameter('min_rpm', -750.0)
        self.declare_parameter('max_rpm', 750.0)
        self.min_rpm = self.get_parameter('min_rpm').value
        self.max_rpm = self.get_parameter('max_rpm').value

        self.declare_parameter('configuration.Lx', 1.80)
        self.declare_parameter('configuration.Ly', 0.80)
        self.declare_parameter('scaling_factor', 3.5)
        self.declare_parameter('rpm_thrust_mapping.p_propeller_rpm_to_command', 0.842403)
        self.declare_parameter('rpm_thrust_mapping.p_thrust_to_propeller_rpm', None, ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('single_thruster_min_force', -294.0)
        self.declare_parameter('single_thruster_max_force', 500.0)
        self.declare_parameter('topics.control_action_reference', '_')
        self.declare_parameter('topics.actuator_1_angle', '_')
        self.declare_parameter('topics.actuator_2_angle', '_')
        self.declare_parameter('topics.actuator_3_angle', '_')
        self.declare_parameter('topics.actuator_4_angle', '_')
        self.declare_parameter('topics.actuator_1_setpoint', '_')
        self.declare_parameter('topics.actuator_2_setpoint', '_')
        self.declare_parameter('topics.actuator_3_setpoint', '_')
        self.declare_parameter('topics.actuator_4_setpoint', '_')
        
        self.thrust_allocation_.l_x = self.get_parameter('configuration.Lx').value
        self.thrust_allocation_.l_y = self.get_parameter('configuration.Ly').value
        self.thrust_allocation_.motor_rpm_factor = self.get_parameter('scaling_factor').value
        self.thrust_allocation_.p_propeller_rpm_to_command = self.get_parameter('rpm_thrust_mapping.p_propeller_rpm_to_command').value
        self.thrust_allocation_.p_thrust_to_propeller_rpm = np.array(self.get_parameter('rpm_thrust_mapping.p_thrust_to_propeller_rpm').value)
        self.thrust_allocation_.single_thruster_min_force = self.get_parameter('single_thruster_min_force').value
        self.thrust_allocation_.single_thruster_max_force = self.get_parameter('single_thruster_max_force').value

    def init_subscribers(self):
        azi_topic_1 = self.get_parameter('topics.actuator_1_angle').value
        azi_topic_2 = self.get_parameter('topics.actuator_2_angle').value
        azi_topic_3 = self.get_parameter('topics.actuator_3_angle').value
        azi_topic_4 = self.get_parameter('topics.actuator_4_angle').value
        self.azi_angle_sub_1_ = self.create_subscription(Float64, azi_topic_1, self.azi_angle_callback_1, 10)
        self.azi_angle_sub_2_ = self.create_subscription(Float64, azi_topic_2, self.azi_angle_callback_2, 10)
        self.azi_angle_sub_3_ = self.create_subscription(Float64, azi_topic_3, self.azi_angle_callback_3, 10)
        self.azi_angle_sub_4_ = self.create_subscription(Float64, azi_topic_4, self.azi_angle_callback_4, 10)

        tau_topic = self.get_parameter('topics.control_action_reference').value
        self.control_input_sub_ = self.create_subscription(Wrench, tau_topic, self.control_input_callback, 10)

    def init_publishers(self):
        act_topic_1 = self.get_parameter('topics.actuator_1_setpoint').value
        act_topic_2 = self.get_parameter('topics.actuator_2_setpoint').value
        act_topic_3 = self.get_parameter('topics.actuator_3_setpoint').value
        act_topic_4 = self.get_parameter('topics.actuator_4_setpoint').value
        self.actuator_1_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_1, 10)
        self.actuator_2_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_2, 10)
        self.actuator_3_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_3, 10)
        self.actuator_4_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_4, 10)

        self.tau_actual_publisher_ = self.create_publisher(WrenchStamped, 'tau_actual', 1)
        self.tau_diff_publisher_ = self.create_publisher(WrenchStamped, 'tau_diff', best_effort_qos)

    def timer_callback(self):
        u_cmd, tau_actual = self.thrust_allocation_.allocate_thrust(self.tau_, self.last_thrust_, self.last_angles_)

        tau_actual_msg = WrenchStamped()
        tau_actual_msg.header.stamp = self.get_clock().now().to_msg()
        tau_actual_msg.wrench.force.x = tau_actual[0]
        tau_actual_msg.wrench.force.y = tau_actual[1]
        tau_actual_msg.wrench.torque.z = tau_actual[2]
        self.tau_actual_publisher_.publish(tau_actual_msg)

        tau_diff_msg = WrenchStamped()
        tau_diff_msg.header.stamp = self.get_clock().now().to_msg()
        tau_diff_msg.wrench.force.x = self.tau_[0] - tau_actual[0]
        tau_diff_msg.wrench.force.y = self.tau_[1] - tau_actual[1]
        tau_diff_msg.wrench.torque.z = self.tau_[2] - tau_actual[2]
        self.tau_diff_publisher_.publish(tau_diff_msg)

        actuator_msg_1 = ActuatorSetpoints()
        actuator_msg_2 = ActuatorSetpoints()
        actuator_msg_3 = ActuatorSetpoints()
        actuator_msg_4 = ActuatorSetpoints()

        actuator_msg_1.throttle_reference = int(self.thrust_allocation_.thrust_to_command(u_cmd[0]))
        actuator_msg_1.angle_reference = int(np.rad2deg(u_cmd[4]))
        actuator_msg_2.throttle_reference = int(self.thrust_allocation_.thrust_to_command(u_cmd[1]))
        actuator_msg_2.angle_reference = int(np.rad2deg(u_cmd[5]))
        actuator_msg_3.throttle_reference = int(self.thrust_allocation_.thrust_to_command(u_cmd[2]))
        actuator_msg_3.angle_reference = int(np.rad2deg(u_cmd[6]))
        actuator_msg_4.throttle_reference = int(self.thrust_allocation_.thrust_to_command(u_cmd[3]))
        actuator_msg_4.angle_reference = int(np.rad2deg(u_cmd[7]))
        
        self.last_thrust_ = u_cmd[0:4]
        self.last_angles_ = u_cmd[4:8]

        self.actuator_1_ref_pub_.publish(actuator_msg_1)
        self.actuator_2_ref_pub_.publish(actuator_msg_2)
        self.actuator_3_ref_pub_.publish(actuator_msg_3)
        self.actuator_4_ref_pub_.publish(actuator_msg_4)

    def azi_angle_callback_1(self, msg: Float64) -> None:
        self.last_angles_[0] = msg.data

    def azi_angle_callback_2(self, msg: Float64) -> None:
        self.last_angles_[1] = msg.data

    def azi_angle_callback_3(self, msg: Float64) -> None:
        self.last_angles_[2] = msg.data

    def azi_angle_callback_4(self, msg: Float64) -> None:
        self.last_angles_[3] = msg.data

    def control_input_callback(self, msg: Wrench) -> None:
        self.tau_[0] = msg.force.x
        self.tau_[1] = msg.force.y
        self.tau_[2] = msg.torque.z

    

def main(args=None):
    rclpy.init(args=args)
    node = ThrustAllocationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()