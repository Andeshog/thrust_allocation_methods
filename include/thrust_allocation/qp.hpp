#ifndef QP_ALLOCATOR_HPP
#define QP_ALLOCATOR_HPP

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <array>

struct QPParameters {
    double dt = 0.01;
    double u_bound = 500.0;
    double max_rate = 1.0;
    double max_force_rate = 50.0;
    double beta = 1.1;
};

class QPAllocator {
public:
    QPAllocator(const QPParameters& params);

    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::Vector3d> allocate(
        std::array<double, 3> tau_cmd);

private:
    const int num_thrusters_ = 4;
    static constexpr size_t z_dim_ = 12;
    double half_length_ = 1.8;
    double half_width_ = 0.8;
    Eigen::Matrix<double, 3, 8> thrust_matrix_;

    Eigen::Vector4d static_alpha_lb_;
    Eigen::Vector4d static_alpha_ub_;
    Eigen::Vector4d u_lb_, u_ub_;

    Eigen::MatrixXd W_;
    Eigen::MatrixXd Q_;

    double dt_;
    double u_bound_;
    double max_rate_;
    double max_force_rate_;
    double beta_;

    Eigen::Matrix<double, z_dim_, z_dim_> H_;
    Eigen::Matrix<double, z_dim_, 1> c_;

    Eigen::Vector4d alpha_des_;
    Eigen::Vector4d alpha_prev_;
    Eigen::Vector4d u_prev_;
};

#endif // QP_ALLOCATOR_HPP