#ifndef THRUST_ALLOCATION_NLP_HPP
#define THRUST_ALLOCATION_NLP_HPP

#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Dense>
#include <array>
#include <vector>
#include <cmath>
#include <tuple>

class NLPAllocator {
public:
    NLPAllocator(double dt);

    std::tuple<Eigen::Vector4d, Eigen::Vector4d, Eigen::Vector3d> allocate(
        const std::array<double, 3>& tau_desired);
private:
    void build_nlp();
    
    double dt_;
    const std::size_t n_thrusters_;
    double half_length_;
    double half_width_;
    double u_bound_;
    double w_angle_;
    double w_neg_;
    double max_rate_;
    double max_force_rate_;
    double w_alpha_change_;

    std::array<std::array<double, 2>, 4> thruster_positions_;

    std::vector<double> alpha_des_;
    std::vector<double> alpha_prev_;
    std::vector<double> u_prev_;

    std::vector<double> u_lb_, u_ub_;
    std::vector<double> static_alpha_lb_, static_alpha_ub_;

    casadi::Function solver_;
    std::vector<double> x0_;
};

#endif // THRUST_ALLOCATION_NLP_HPP