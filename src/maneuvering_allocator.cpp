#include "thrust_allocation/maneuvering_allocator.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

ManeuveringAllocator::ManeuveringAllocator(ManeuveringParams& params)
    : dt_(params.dt),
      gamma_(params.gamma),
      mu_(params.mu),
      rho_(params.rho),
      zeta_(params.zeta),
      rate_limit_(params.rate_limit),
      theta_min_(params.theta_min),
      theta_max_(params.theta_max),
      lambda_(params.lambda),
      power_management_(params.power_management)
    {
    c_ << 1.0, 1.0, 1.0, 1.0;
    w_matrix_.setIdentity();
    min_forces_ << -430.0,  0.1, -430.0, -430.0,
                    0.1,    0.1,   0.1, -430.0;

    max_forces_ <<  -0.1, 430.0,   -0.1,   -0.1,
                    430.0, 430.0,  430.0,   -0.1;
    reference_angles_ <<  3*M_PI/4.0,  -3*M_PI/4.0,
                        M_PI/4.0,   -M_PI/4.0;
    constexpr double half_length = 1.8;
    constexpr double half_width  = 0.8;

    std::array<Eigen::Vector2d,4> pos = {
        Eigen::Vector2d( half_length, -half_width ),   // front-left
        Eigen::Vector2d( half_length,  half_width ),   // front-right
        Eigen::Vector2d(-half_length, -half_width ),   // rear-left
        Eigen::Vector2d(-half_length,  half_width )    // rear-right
    };

    T_.setZero();
    for (int i=0;i<kNumThrusters;++i)
    {
        const double x = pos[i].x();
        const double y = pos[i].y();

        T_(0,2*i  ) = 1.0;
        T_(1,2*i+1) = 1.0;
        T_(2,2*i  ) = -y;
        T_(2,2*i+1) =  x;
    }

    T_pinv_    = pseudo_inverse(T_);
    q_matrix_  = null_space(T_);

    last_xi_p_.setZero();
    xi_.setZero();
    theta_.setZero();
}

std::tuple<Eigen::Vector4d,Eigen::Vector4d,VecTau>
    ManeuveringAllocator::allocate(const std::array<double, 3>& tau_cmd, Eigen::VectorXd xi_p) {
    VecTau tau = VecTau::Zero();
    tau << tau_cmd[0], tau_cmd[1], tau_cmd[2];
    // VecXi xi_p_eigen = T_pinv_ * tau;
    VecXi xi_p_eigen = xi_p;
    VecXi xi_p_dot = (xi_p_eigen - last_xi_p_) / dt_;
    last_xi_p_ = xi_p_eigen;
    VecXi xi_0 = q_matrix_ * theta_;
    VecXi xi_d = xi_p_eigen + xi_0;
    VecXi xi_tilde = xi_ - xi_d;
    VecTheta v = -gamma_ * calculate_j_theta(xi_d).transpose();
    VecTheta theta_dot = v - mu_ * calculate_v_theta(xi_tilde);
    theta_ += theta_dot * dt_;
    theta_ = theta_.cwiseMin(theta_max_).cwiseMax(theta_min_);
    
    VecXi kappa = calculate_kappa(xi_tilde, xi_p_dot, v);
    kappa = cbf(kappa, rho_);
    xi_ += kappa * dt_;
    
    VecTau tau_out = T_ * xi_;
    Eigen::Vector4d alpha;
    Eigen::Vector4d u;

    for (int i=0;i<kNumThrusters;++i)
    {   
        const double fx = xi_(2*i);
        const double fy = xi_(2*i+1);

        alpha(i) = std::atan2(fy, fx);
        u(i) = std::hypot(fx, fy);

        if ((i == 0 || i == 1) && fx > 0.0) { // Front thrusters negative thrust
            if (alpha(i) < 0.0) {
                alpha(i) += M_PI;
            }
            else {
                alpha(i) -= M_PI;
            }
            u(i) = -u(i);
        }
        else if ((i == 2 || i == 3) && fx < 0.0) { // Rear thrusters negative thrust
            if (alpha(i) < 0.0) {
                alpha(i) += M_PI;
            }
            else {
                alpha(i) -= M_PI;
            }
            u(i) = -u(i);
        }
    }

    return std::make_tuple(u, alpha, tau_out);
}

VecTheta ManeuveringAllocator::calculate_j_theta(const VecXi& xi_d) const {
    VecTheta j_theta = VecTheta::Zero();

    const double a1_pos = 0.85;
    const double a2_pos = 0.0083;

    const double a1_neg = -1.45;
    const double a2_neg = 0.0115;

    double a1;
    double a2;
    double sign;

    for (int i=0;i<kNumThrusters;++i) {
        Eigen::Vector2d xi_di = xi_d.segment<2>(2*i);
        double xi_di_norm = xi_di.norm();
        if (xi_di_norm == 0.0) {
            continue;
        }
        if (!power_management_) {
            double a_ref = reference_angles_(i);
            Eigen::Vector2d ref_vec(std::cos(a_ref), std::sin(a_ref));
            Eigen::Matrix<double, 2, kThetaSize> Q_i = q_matrix_.block<2, kThetaSize>(2*i, 0);
            j_theta += c_(i) * Q_i.transpose() * (xi_di/xi_di_norm - lambda_ * ref_vec);
        }
        else {
            if (i == 2 || i == 3) { // Rear thrusters
                if (xi_di(0) > 0) { // Positive thrust
                    a1 = a1_pos;
                    a2 = a2_pos;
                    sign = 1.0;
                }
                else { // Negative thrust
                    a1 = a1_neg;
                    a2 = a2_neg;
                    sign = -1.0;
                }
            }
            else { // Front thrusters
                if (xi_di(0) < 0) { // negative thrust
                    a1 = a1_neg;
                    a2 = a2_neg;
                    sign = -1.0;
                }
                else { // positive thrust
                    a1 = a1_pos;
                    a2 = a2_pos;
                    sign = 1.0;
                }
            }
            
            Eigen::Matrix<double, 2, kThetaSize> Q_i = q_matrix_.block<2, kThetaSize>(2*i, 0);
            double denom = xi_di_norm + 0.001;
            j_theta += (a1 * sign + 2.0 * a2 * xi_di_norm) * Q_i.transpose() * xi_di / denom;
        }
    }
    return j_theta;
}

VecTheta ManeuveringAllocator::calculate_v_theta(const VecXi& xi_tilde) const {
    VecTheta v_theta = VecTheta::Zero();

    for (int i=0;i<kNumThrusters;++i) {
        Eigen::Vector2d xi_tilde_i = xi_tilde.segment<2>(2*i);
        Eigen::Matrix<double, 2, kThetaSize> Q_i = q_matrix_.block<2, kThetaSize>(2*i, 0);
        v_theta -= c_(i) * Q_i.transpose() * xi_tilde_i;
    }
    return v_theta;
}

VecXi ManeuveringAllocator::calculate_kappa(const VecXi& xi_tilde, const VecXi& xi_p_dot, const VecTheta& v) const {
    VecXi kappa = VecXi::Zero();
    for (int i=0;i<kNumThrusters;++i) {
        Eigen::Vector2d xi_tilde_i = xi_tilde.segment<2>(2*i);
        double denom = xi_tilde_i.norm() + zeta_;

        Eigen::Vector2d term = -rate_limit_ * xi_tilde_i / denom + xi_p_dot.segment<2>(2*i) + q_matrix_.block<2, kThetaSize>(2*i, 0) * v;
        kappa.segment<2>(2*i) = term;
    }
    return kappa;
}

VecXi ManeuveringAllocator::cbf(const VecXi& u,
                                double time_constant) const
{
    constexpr double T_MAX  = 430.0;
    const double     T_MAX2 = T_MAX * T_MAX;

    VecXi kappa = VecXi::Zero();

    for (int i = 0; i < kNumThrusters; ++i)
    {
        const Eigen::Vector2d xi_i = xi_.segment<2>(2 * i);
        Eigen::Vector2d       du   = u.segment<2>(2 * i);

        const double xi_norm2 = xi_i.squaredNorm();
        const double h        = T_MAX2 - xi_norm2;

        if (h < 0.0) {
            du = -time_constant * xi_i;
        } else {
            const double dhdot = -2.0 * xi_i.dot(du);
            const double rhs   = -2.0 * time_constant * h;
            if (dhdot < rhs) {
                const double alpha = (rhs - dhdot) / (2.0 * xi_norm2 + 1e-12);
                du += alpha * xi_i;
            }
        }

        Eigen::Vector2d xi_next = xi_i + du;
        const Eigen::Vector2d xi_min = min_forces_.segment<2>(2 * i);
        const Eigen::Vector2d xi_max = max_forces_.segment<2>(2 * i);

        xi_next = xi_next.cwiseMin(xi_max).cwiseMax(xi_min);

        if (xi_next.squaredNorm() > T_MAX2) {
            xi_next = xi_next.normalized() * T_MAX;
        }

        kappa.segment<2>(2 * i) = xi_next - xi_i;
    }

    return kappa;
}