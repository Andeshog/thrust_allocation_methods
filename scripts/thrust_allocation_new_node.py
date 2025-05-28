#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from thrust_allocation.thrust_allocation_new import PSQPAllocator
from geometry_msgs.msg import Wrench, WrenchStamped
from custom_msgs.msg import ActuatorSetpoints
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

best_effort_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT
)

class ThrustAllocationNewNode(Node):
    def __init__(self):
        super().__init__('thrust_allocation_new_node')

        self.thrust_allocator_ = PSQPAllocator()

        self.init_params()
        self.init_topics()
        self.init_subscribers()
        self.init_publishers()

    def init_topics(self):
        self.declare_parameter('topics.control_action_reference', '_')
        self.declare_parameter('topics.actuator_1_angle', '_')
        self.declare_parameter('topics.actuator_2_angle', '_')
        self.declare_parameter('topics.actuator_3_angle', '_')
        self.declare_parameter('topics.actuator_4_angle', '_')
        self.declare_parameter('topics.actuator_1_setpoint', '_')
        self.declare_parameter('topics.actuator_2_setpoint', '_')
        self.declare_parameter('topics.actuator_3_setpoint', '_')
        self.declare_parameter('topics.actuator_4_setpoint', '_')

    def init_params(self):
        self.declare_parameter('rpm_thrust_mapping.p_propeller_rpm_to_command', 0.842403)
        self.declare_parameter('rpm_thrust_mapping.p_thrust_to_propeller_rpm', None, ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('single_thruster_min_force', -294.0)
        self.declare_parameter('single_thruster_max_force', 500.0)
        self.p_propeller_rpm_to_command = self.get_parameter('rpm_thrust_mapping.p_propeller_rpm_to_command').value
        self.p_thrust_to_propeller_rpm = self.get_parameter('rpm_thrust_mapping.p_thrust_to_propeller_rpm').value
        self.single_thruster_min_force = self.get_parameter('single_thruster_min_force').value
        self.single_thruster_max_force = self.get_parameter('single_thruster_max_force').value

    def init_subscribers(self):
        tau_topic = self.get_parameter('topics.control_action_reference').value

        self.tau_sub_ = self.create_subscription(Wrench, tau_topic, self.tau_callback, 10)
    
    def init_publishers(self):
        act_topic_1 = self.get_parameter('topics.actuator_1_setpoint').value
        act_topic_2 = self.get_parameter('topics.actuator_2_setpoint').value
        act_topic_3 = self.get_parameter('topics.actuator_3_setpoint').value
        act_topic_4 = self.get_parameter('topics.actuator_4_setpoint').value
        self.actuator_1_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_1, 10)
        self.actuator_2_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_2, 10)
        self.actuator_3_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_3, 10)
        self.actuator_4_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_4, 10)

        self.thruster_1_pub_ = self.create_publisher(WrenchStamped, 'thruster_1', best_effort_qos)
        self.thruster_2_pub_ = self.create_publisher(WrenchStamped, 'thruster_2', best_effort_qos)
        self.thruster_3_pub_ = self.create_publisher(WrenchStamped, 'thruster_3', best_effort_qos)
        self.thruster_4_pub_ = self.create_publisher(WrenchStamped, 'thruster_4', best_effort_qos)

    def tau_callback(self, msg: Wrench):
        tau = np.array([msg.force.x, msg.force.y, msg.torque.z], dtype=np.float32)
        solution = self.thrust_allocator_.solve(tau)
        x_forces = np.array(solution['cmdix'])
        y_forces = np.array(solution['cmdiy'])

        scale = 1 / 200.0

        thruster_1_msg = WrenchStamped()
        thruster_1_msg.header.stamp = self.get_clock().now().to_msg()
        thruster_1_msg.header.frame_id = 'bow_port_thruster'
        thruster_1_msg.wrench.force.x = x_forces[2] * scale
        thruster_1_msg.wrench.force.y = y_forces[2] * scale
        self.thruster_1_pub_.publish(thruster_1_msg)

        thruster_2_msg = WrenchStamped()
        thruster_2_msg.header.stamp = self.get_clock().now().to_msg()
        thruster_2_msg.header.frame_id = 'bow_starboard_thruster'
        thruster_2_msg.wrench.force.x = x_forces[1] * scale
        thruster_2_msg.wrench.force.y = y_forces[1] * scale
        self.thruster_2_pub_.publish(thruster_2_msg)

        thruster_3_msg = WrenchStamped()
        thruster_3_msg.header.stamp = self.get_clock().now().to_msg()
        thruster_3_msg.header.frame_id = 'stern_starboard_thruster'
        thruster_3_msg.wrench.force.x = x_forces[0] * scale
        thruster_3_msg.wrench.force.y = y_forces[0] * scale
        self.thruster_3_pub_.publish(thruster_3_msg)

        thruster_4_msg = WrenchStamped()
        thruster_4_msg.header.stamp = self.get_clock().now().to_msg()
        thruster_4_msg.header.frame_id = 'stern_port_thruster'
        thruster_4_msg.wrench.force.x = x_forces[3] * scale
        thruster_4_msg.wrench.force.y = y_forces[3] * scale
        self.thruster_4_pub_.publish(thruster_4_msg)

        actuator_1_msg = ActuatorSetpoints()
        angle_1 = (np.rad2deg(np.arctan2(y_forces[2], x_forces[2])))
        w_1 = int(self.thrust_to_command(np.linalg.norm([x_forces[2], y_forces[2]])))
        if angle_1 < 0:
            w_1 = -w_1
            angle_1+=180
        actuator_1_msg.throttle_reference = w_1
        actuator_1_msg.angle_reference = int(angle_1)

        actuator_2_msg = ActuatorSetpoints()
        angle_2 = (np.rad2deg(np.arctan2(y_forces[1], x_forces[1])))
        w_2 = int(self.thrust_to_command(np.linalg.norm([x_forces[1], y_forces[1]])))
        if angle_2 > 0:
            w_2 = -w_2
            angle_2-=180
        actuator_2_msg.throttle_reference = w_2
        actuator_2_msg.angle_reference = int(angle_2)

        actuator_3_msg = ActuatorSetpoints()
        angle_3 = (np.rad2deg(np.arctan2(y_forces[0], x_forces[0])))
        w_3 = int(self.thrust_to_command(np.linalg.norm([x_forces[0], y_forces[0]])))
        if angle_3 > 0:
            w_3 = -w_3
            angle_3-=180
        actuator_3_msg.throttle_reference = w_3
        actuator_3_msg.angle_reference = int(angle_3)

        actuator_4_msg = ActuatorSetpoints()
        angle_4 = (np.rad2deg(np.arctan2(y_forces[3], x_forces[3])))
        w_4 = int(self.thrust_to_command(np.linalg.norm([x_forces[3], y_forces[3]])))
        if angle_4 < 0:
            w_4 = -w_4
            angle_4+=180
        actuator_4_msg.throttle_reference = w_4
        actuator_4_msg.angle_reference = int(angle_4)

        self.actuator_1_ref_pub_.publish(actuator_1_msg)
        self.actuator_2_ref_pub_.publish(actuator_2_msg)
        self.actuator_3_ref_pub_.publish(actuator_3_msg)
        self.actuator_4_ref_pub_.publish(actuator_4_msg)

    def thrust_to_rpm(self, F):
        coeff_fit_pos = np.array([-0.005392628724401, 4.355160257309755, 0])
        coeff_fit_neg = np.array([0.009497954053629, 5.891321999522722, 0])
        tmax = 430
        tmin = -330
		
        if F < 0:
            rpm = np.polyval(coeff_fit_neg, F)
        else:
            rpm = np.polyval(coeff_fit_pos, F)

        if F > tmax:
            rpm = np.polyval(coeff_fit_pos, tmax)

        if F < tmin:
            rpm = np.polyval(coeff_fit_neg, tmin)

        return rpm 
    
    def rpm_to_command(self, rpm: float) -> float:
        p = self.p_propeller_rpm_to_command

        command = np.clip(p * rpm, -1000, 1000)

        return command
    
    def thrust_to_command(self, thrust: float) -> float:
        rpm = self.thrust_to_rpm(thrust)
        command = self.rpm_to_command(rpm)
        return command
    
def main(args=None):
    rclpy.init(args=args)

    thrust_allocation_new_node = ThrustAllocationNewNode()

    rclpy.spin(thrust_allocation_new_node)

    thrust_allocation_new_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()