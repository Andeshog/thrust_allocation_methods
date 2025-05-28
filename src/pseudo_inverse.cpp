#include "thrust_allocation/pseudo_inverse.hpp"

PseudoInverseAllocator::PseudoInverseAllocator() {
    constexpr double sqrt2over2 = std::sqrt(2.0) / 2.0;
    thrust_matrix_ << -sqrt2over2, -sqrt2over2, sqrt2over2, sqrt2over2,
                       sqrt2over2, -sqrt2over2, sqrt2over2, -sqrt2over2,
                       sqrt2over2, -sqrt2over2, -sqrt2over2, sqrt2over2;
    thrust_matrix_inv_ = thrust_matrix_.transpose() * (thrust_matrix_ * thrust_matrix_.transpose()).inverse();
}

std::tuple<Eigen::Vector4d, Eigen::Vector4d, Eigen::Vector3d> PseudoInverseAllocator::allocate(
    const std::array<double, 3>& tau_cmd) {
    Eigen::Vector3d tau;
    tau << tau_cmd[0], tau_cmd[1], tau_cmd[2];

    Eigen::Vector4d u = thrust_matrix_inv_ * tau;
    Eigen::Vector4d alpha;
    alpha << 3*M_PI/4, -3*M_PI/4, M_PI/4, -M_PI/4;
    Eigen::Vector3d tau_out = thrust_matrix_ * u;

    return std::make_tuple(alpha, u, tau_out);
}
    