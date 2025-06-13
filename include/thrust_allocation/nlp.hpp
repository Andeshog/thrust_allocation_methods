#ifndef THRUST_ALLOCATION_NLP_HPP
#define THRUST_ALLOCATION_NLP_HPP

#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Dense>
#include <array>
#include <vector>
#include <cmath>
#include <tuple>

struct NLPParams {
    double dt = 0.01;
    double u_bound = 400.0;
    double w_angle = 1000.0;
    double w_neg = 10.0;
    double max_rate = 0.5;
    double max_force_rate = 200.0;
    double w_alpha_change = 50000.0;
};

class NLPAllocator {
public:
    NLPAllocator(NLPParams params);

    std::tuple<Eigen::Vector4d, Eigen::Vector4d, Eigen::Vector3d> allocate(
        const std::array<double, 3>& tau_desired);
private:
    void build_nlp();
    
    double dt_;
    const std::size_t n_thrusters_ = 4;
    double half_length_ = 1.8;
    double half_width_ = 0.8;
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