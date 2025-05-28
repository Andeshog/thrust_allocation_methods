#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from thrust_allocation.nlp import NLPAllocator
from thrust_allocation.qp import QPAllocator
from thrust_allocation.maneuvering_allocator import ManeuveringAllocator
from thrust_allocation.pseudo_inverse import PsuedoInverseAllocator
from geometry_msgs.msg import Wrench, WrenchStamped
from custom_msgs.msg import ActuatorSetpoints
from std_msgs.msg import String
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

best_effort_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT
)

class ThrustAllocationNode(Node):
    def __init__(self):
        super().__init__('thrust_allocation_new_node')

        self.nlp_allocator_ = NLPAllocator()
        self.qp_allocator_ = QPAllocator()
        self.maneuvering_allocator_ = ManeuveringAllocator()
        self.pseudo_inverse_allocator_ = PsuedoInverseAllocator()

        self.dt = 0.1
        self.tau = np.zeros(3)
        self.allocator_mode = 'pseudo_inverse'  # Options: 'nlp', 'qp', 'maneuvering', 'pseudo_inverse'

        self.init_params()
        self.init_topics()
        self.init_subscribers()
        self.init_publishers()

        self.timer_ = self.create_timer(self.dt, self.timer_callback)

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

        self.tau_sub_ = self.create_subscription(Wrench, tau_topic, self.tau_callback, 1)
        self.mode_sub_ = self.create_subscription(String, 'allocator_mode', self.mode_callback, 1)
    
    def init_publishers(self):
        act_topic_1 = self.get_parameter('topics.actuator_1_setpoint').value
        act_topic_2 = self.get_parameter('topics.actuator_2_setpoint').value
        act_topic_3 = self.get_parameter('topics.actuator_3_setpoint').value
        act_topic_4 = self.get_parameter('topics.actuator_4_setpoint').value
        self.actuator_1_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_1, 1)
        self.actuator_2_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_2, 1)
        self.actuator_3_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_3, 1)
        self.actuator_4_ref_pub_ = self.create_publisher(ActuatorSetpoints, act_topic_4, 1)

        self.thruster_1_pub_ = self.create_publisher(WrenchStamped, 'thruster_1', best_effort_qos)
        self.thruster_2_pub_ = self.create_publisher(WrenchStamped, 'thruster_2', best_effort_qos)
        self.thruster_3_pub_ = self.create_publisher(WrenchStamped, 'thruster_3', best_effort_qos)
        self.thruster_4_pub_ = self.create_publisher(WrenchStamped, 'thruster_4', best_effort_qos)

        self.tau_actual_publisher_ = self.create_publisher(WrenchStamped, 'tau_actual', 1)
        self.tau_diff_publisher_ = self.create_publisher(WrenchStamped, 'tau_diff', best_effort_qos)

    def mode_callback(self, msg: String):
        if msg.data in ['nlp', 'qp', 'maneuvering', 'pseudo_inverse']:
            self.allocator_mode = msg.data
            if self.allocator_mode == 'maneuvering':
                self.maneuvering_allocator_.theta = np.zeros(5)
            self.get_logger().info(f"Allocator mode set to: {self.allocator_mode}")
        else:
            self.get_logger().error(f"Invalid allocator mode: {msg.data}. Valid modes are: nlp, qp, maneuvering, pseudo_inverse")

    def tau_callback(self, msg: Wrench):
        self.tau = np.array([msg.force.x, msg.force.y, msg.torque.z], dtype=np.float32)

    def timer_callback(self):
        if self.allocator_mode == 'nlp':
            alpha, u, tau_actual = self.nlp_allocator_.allocate(self.tau, self.dt)
        
        elif self.allocator_mode == 'pseudo_inverse':
            u, tau_actual = self.pseudo_inverse_allocator_.allocate(self.tau)
            alpha = np.array([3*np.pi/4, -3*np.pi/4, np.pi/4, -np.pi/4])

        elif self.allocator_mode == 'qp':
            f_ext, u, alpha, tau_actual = self.qp_allocator_.allocate(self.tau, self.dt)

        elif self.allocator_mode == 'maneuvering':
            xi_p, _, _, _ = self.qp_allocator_.allocate(self.tau, self.dt)
            self.get_logger().info(f"theta: {self.maneuvering_allocator_.theta}")
            f_ext, u, alpha, tau_actual = self.maneuvering_allocator_.allocate(self.tau, xi_p, self.dt)
        
        else:
            self.get_logger().error(f"Invalid allocator mode: {self.allocator_mode}")
            return

        tau_actual_msg = WrenchStamped()
        tau_actual_msg.header.stamp = self.get_clock().now().to_msg()
        tau_actual_msg.wrench.force.x = tau_actual[0]
        tau_actual_msg.wrench.force.y = tau_actual[1]
        tau_actual_msg.wrench.torque.z = tau_actual[2]
        self.tau_actual_publisher_.publish(tau_actual_msg)

        tau_diff_msg = WrenchStamped()
        tau_diff_msg.header.stamp = self.get_clock().now().to_msg()
        tau_diff_msg.wrench.force.x = self.tau[0] - tau_actual[0]
        tau_diff_msg.wrench.force.y = self.tau[1] - tau_actual[1]
        tau_diff_msg.wrench.torque.z = self.tau[2] - tau_actual[2]
        self.tau_diff_publisher_.publish(tau_diff_msg)

        actuator_1_msg = ActuatorSetpoints()
        angle_1 = (np.rad2deg(alpha[0]))
        w_1 = int(self.thrust_to_command(u[0]))
        actuator_1_msg.throttle_reference = w_1
        actuator_1_msg.angle_reference = int(angle_1)

        actuator_2_msg = ActuatorSetpoints()
        angle_2 = (np.rad2deg(alpha[1]))
        w_2 = int(self.thrust_to_command(u[1]))
        actuator_2_msg.throttle_reference = w_2
        actuator_2_msg.angle_reference = int(angle_2)

        actuator_3_msg = ActuatorSetpoints()
        angle_3 = (np.rad2deg(alpha[3]))
        w_3 = int(self.thrust_to_command(u[3]))
        actuator_3_msg.throttle_reference = w_3
        actuator_3_msg.angle_reference = int(angle_3)

        actuator_4_msg = ActuatorSetpoints()
        angle_4 = (np.rad2deg(alpha[2]))
        w_4 = int(self.thrust_to_command(u[2]))
        actuator_4_msg.throttle_reference = w_4
        actuator_4_msg.angle_reference = int(angle_4)

        self.actuator_1_ref_pub_.publish(actuator_1_msg)
        self.actuator_2_ref_pub_.publish(actuator_2_msg)
        self.actuator_3_ref_pub_.publish(actuator_3_msg)
        self.actuator_4_ref_pub_.publish(actuator_4_msg)

        # if self.allocator_mode == 'pseudo_inverse' or self.allocator_mode == 'nlp':
        #     return

        # scale = 1 / 200.0

        # thruster_1_msg = WrenchStamped()
        # thruster_1_msg.header.stamp = self.get_clock().now().to_msg()
        # thruster_1_msg.header.frame_id = 'bow_port_thruster'
        # thruster_1_msg.wrench.force.x = f_ext[0] * scale
        # thruster_1_msg.wrench.force.y = f_ext[1] * scale
        # self.thruster_1_pub_.publish(thruster_1_msg)

        # thruster_2_msg = WrenchStamped()
        # thruster_2_msg.header.stamp = self.get_clock().now().to_msg()
        # thruster_2_msg.header.frame_id = 'bow_starboard_thruster'
        # thruster_2_msg.wrench.force.x = f_ext[2] * scale
        # thruster_2_msg.wrench.force.y = f_ext[3] * scale
        # self.thruster_2_pub_.publish(thruster_2_msg)

        # thruster_3_msg = WrenchStamped()
        # thruster_3_msg.header.stamp = self.get_clock().now().to_msg()
        # thruster_3_msg.header.frame_id = 'stern_starboard_thruster'
        # thruster_3_msg.wrench.force.x = f_ext[6] * scale
        # thruster_3_msg.wrench.force.y = f_ext[7] * scale
        # self.thruster_3_pub_.publish(thruster_3_msg)
        
        # thruster_4_msg = WrenchStamped()
        # thruster_4_msg.header.stamp = self.get_clock().now().to_msg()
        # thruster_4_msg.header.frame_id = 'stern_port_thruster'
        # thruster_4_msg.wrench.force.x = f_ext[4] * scale
        # thruster_4_msg.wrench.force.y = f_ext[5] * scale
        # self.thruster_4_pub_.publish(thruster_4_msg)

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

    thrust_allocation_new_node = ThrustAllocationNode()

    rclpy.spin(thrust_allocation_new_node)

    thrust_allocation_new_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()