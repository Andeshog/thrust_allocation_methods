#ifndef PSEUDO_INVERSE_ALLOCATOR_HPP
#define PSEUDO_INVERSE_ALLOCATOR_HPP

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <array>

class PseudoInverseAllocator {
public:
    PseudoInverseAllocator();
    std::tuple<Eigen::Vector4d, Eigen::Vector4d, Eigen::Vector3d> allocate(
        const std::array<double, 3>& tau_cmd);
private:
    Eigen::Matrix<double, 3, 4> thrust_matrix_;
    Eigen::Matrix<double, 4, 3> thrust_matrix_inv_;
};

#endif // PSEUDO_INVERSE_ALLOCATOR_HPP