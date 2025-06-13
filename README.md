## Thrust Allocation Methods

This repository contains a collection of thrust allocation methods and a ROS 2 node. Each method is implemented in both C++ and Python. The methods are implemented on milliAmpere1, which has four azimuth thrustersFor information about the theory, check out my thesis (link comes soon).

### Prerequisites

The following dependencies are needed:
- [Eigen](https://github.com/PX4/eigen)
- [spdlog](https://github.com/gabime/spdlog)
- [CasADi](https://github.com/casadi/casadi)
- [OSQP](https://github.com/osqp/osqp)
- [OSQP-Eigen](https://github.com/robotology/osqp-eigen)

### Methods

There are five methods:
- __Pseudoinverse__: Simple unconstrained, fixed-angle thrust allocation.
- __QP__: The baseline method. Optimization-based.
- __Maneuvering-based__: A thrust allocation method based on the maneuvering-problem. Solves the thrust allocation problem as a tracking problem, and is able to handle secondary objectives.
- __Power-aware maneuvering-based__: An implementation of the maneuvering-based method where the secondary objective is to minimize the power consumption by utilizing the most power-efficient thrust direction of each thruster. 

### Usage