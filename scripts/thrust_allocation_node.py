#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from thrust_allocation.nlp import NLPAllocator
from thrust_allocation.qp import QPAllocator
from thrust_allocation.maneuvering_allocator import ManeuveringAllocator
from thrust_allocation.pseudo_inverse import PsuedoInverseAllocator
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray
import numpy as np

class ThrustAllocationNode(Node):
    def __init__(self):
        super().__init__('thrust_allocation_node_py')

        self.tau = np.zeros(3)
        
        self.pseudo_inverse_allocator_ = PsuedoInverseAllocator()
        self.nlp_allocator_ = NLPAllocator()
        self.qp_allocator_ = QPAllocator()
        self.maneuvering_allocator_ = ManeuveringAllocator()

        self.init_params()
        self.init_topics()
        self.init_subscribers_and_publishers()

        self.timer_ = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info(f"Thrust Allocation Node initialized with allocator mode: {self.allocator_mode}")

    def init_topics(self):
        self.declare_parameter('tau_topic', '_')
        self.declare_parameter('out_topic', '_')

    def init_params(self):
        self.declare_parameter('py.dt', 0.0)
        self.dt = self.get_parameter('py.dt').value

        self.declare_parameter('allocator', 'pseudo_inverse')
        self.allocator_mode = self.get_parameter('allocator').value

    def init_subscribers_and_publishers(self):
        tau_topic = self.get_parameter('tau_topic').value
        out_topic = self.get_parameter('out_topic').value

        self.tau_sub_ = self.create_subscription(WrenchStamped, tau_topic, self.tau_callback, 1)
        self.out_pub_ = self.create_publisher(Float64MultiArray, out_topic, 1)

    def tau_callback(self, msg: WrenchStamped):
        self.tau = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.torque.z])

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
        
        out_msg = Float64MultiArray()
        out_msg.data = np.concatenate((u, alpha)).tolist()
        self.out_pub_.publish(out_msg)

    
def main(args=None):
    rclpy.init(args=args)

    thrust_allocation_node = ThrustAllocationNode()

    rclpy.spin(thrust_allocation_node)

    thrust_allocation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()