#include "thrust_allocation/qp.hpp"
#include <cmath>
#include <stdexcept>
#include <OsqpEigen/OsqpEigen.h>

constexpr double eps_azimuth = 1e-4;
constexpr double f_min = 1.0;
constexpr double reg_fb = 1.0e-6;

QPAllocator::QPAllocator(const QPParameters& params)
    : dt_(params.dt),
      u_bound_(params.u_bound),
      max_force_rate_(params.max_force_rate),
      beta_(params.beta) {
    const double x[]{ half_length_,  half_length_, -half_length_, -half_length_ };
    const double y[]{-half_width_ ,  half_width_ , -half_width_ ,  half_width_  };
    thrust_matrix_.setZero();
    for (int i=0; i < num_thrusters_; ++i) {
        thrust_matrix_(0, 2*i    ) =  1.0;
        thrust_matrix_(1, 2*i + 1) =  1.0;
        thrust_matrix_(2, 2*i    ) = -y[i];
        thrust_matrix_(2, 2*i + 1) =  x[i];
    }

    alpha_des_ <<  3*M_PI/4, -3*M_PI/4,  M_PI/4, -M_PI/4;
    alpha_prev_ = alpha_des_;
    u_prev_     = Eigen::Vector4d::Zero();

    u_lb_.setZero();
    u_ub_.setConstant(u_bound_);
    double epsilon = 0.25;
    static_alpha_lb_ <<  M_PI/2 + epsilon, -M_PI + epsilon,     0.0 + epsilon , -M_PI/2 + epsilon;
    static_alpha_ub_ <<      M_PI - epsilon, -M_PI/2 - epsilon, M_PI/2 - epsilon,      0.0 - epsilon;

    W_ = Eigen::MatrixXd::Identity(num_thrusters_ * 2, num_thrusters_ * 2);
    Q_ = 1000.0 * Eigen::MatrixXd::Identity(3, 3);

    H_.setZero();
    H_.block<8,8>(0,0) = 2.0 * W_;
    H_.block<3,3>(8,8) = 2.0 * Q_;
    H_(z_dim_ - 1,z_dim_ - 1)= reg_fb;

    c_.setZero();
    c_(z_dim_ - 1) = beta_;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::Vector3d> QPAllocator::allocate(std::array<double, 3> tau_cmd) {
    const double du_max = max_force_rate_ * dt_;
    const Eigen::Vector4d dyn_u_ub = (u_prev_.array() + du_max).min(u_ub_.array());

    const Eigen::Vector4d dyn_alpha_lb = static_alpha_lb_;
    const Eigen::Vector4d dyn_alpha_ub = static_alpha_ub_;

    Eigen::Vector3d tau_desired = Eigen::Map<const Eigen::Vector3d>(tau_cmd.data());

    const int m =
    /* equality rows   */ 3 +
    /* Fx/Fy bounds    */ 4 * 2 * num_thrusters_ +
    /* f-bar inequality*/ 1 +
    /* ‖f‖₁ slack rows */ 2 * 8 +
    /* azimuth rows    */ 2 * num_thrusters_;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, z_dim_);
    Eigen::VectorXd l = Eigen::VectorXd::Constant(m, -OSQP_INFTY);
    Eigen::VectorXd u = Eigen::VectorXd::Constant(m,  OSQP_INFTY);

    int row = 0;
    for (int r=0; r<3; ++r) {
        A.block(row, 0, 1, 8) = thrust_matrix_.row(r);
        A(row, 8 + r)         = -1.0;         // s components
        l(row) = u(row) = tau_desired(r);
        ++row;
    }

    for (int i=0;i<num_thrusters_;++i) {
        const int fx = 2*i, fy = 2*i+1;
        A(row,fx)= 1.0; u(row)=dyn_u_ub(i); ++row;
        A(row,fx)=-1.0; u(row)=dyn_u_ub(i); ++row;
        A(row,fy)= 1.0; u(row)=dyn_u_ub(i); ++row;
        A(row,fy)=-1.0; u(row)=dyn_u_ub(i); ++row;
    }

    A(row,z_dim_-1) = 1.0;
    l(row) = 1.0;
    ++row;

    for (int j=0;j<8;++j) {
        A(row,j)= 1.0; A(row,z_dim_-1)=-1.0; u(row)=0.0; ++row;
        A(row,j)=-1.0; A(row,z_dim_-1)=-1.0; u(row)=0.0; ++row;
    }

    for (int i=0;i<num_thrusters_;++i) {
        const int fx = 2*i, fy = 2*i+1;
        const double th_min = dyn_alpha_lb(i);
        const double th_max = dyn_alpha_ub(i);

        A(row,fy) =  std::cos(th_min);
        A(row,fx) = -std::sin(th_min);
        l(row)    = -eps_azimuth;
        ++row;

        A(row,fx) =  std::sin(th_max);
        A(row,fy) = -std::cos(th_max);
        l(row)    = -eps_azimuth;
        ++row;
    }

    Eigen::SparseMatrix<double> Hs = H_.sparseView();
    Eigen::SparseMatrix<double> As = A.sparseView();

    OsqpEigen::Solver solver;
    solver.settings()->setWarmStart(true);
    solver.settings()->setVerbosity(false);

    solver.data()->setNumberOfVariables(z_dim_);
    solver.data()->setNumberOfConstraints(m);
    solver.data()->setHessianMatrix(Hs);
    solver.data()->setGradient(c_);
    solver.data()->setLinearConstraintsMatrix(As);
    solver.data()->setLowerBound(l);
    solver.data()->setUpperBound(u);

    if (!solver.initSolver())
        throw std::runtime_error("OSQP initialisation failed");

    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
        throw std::runtime_error("OSQP: solver failure");

    const Eigen::VectorXd z = solver.getSolution();
    const Eigen::VectorXd f = z.head(8);

    Eigen::Matrix<double, 4, 2> f_mat;
    for (int i=0;i<num_thrusters_;++i) {
        f_mat(i,0) = f(2*i);
        f_mat(i,1) = f(2*i+1);
    }

    Eigen::Vector4d thrusts, angles;
    for (int i=0;i<num_thrusters_;++i) {
        thrusts(i) = std::hypot(f_mat(i,0), f_mat(i,1));
        angles (i) = std::atan2( f_mat(i,1), f_mat(i,0) );

        if (thrusts(i) >= 0.0 && thrusts(i) < f_min) {
            angles (i)  = alpha_des_(i);
            f_mat(i,0)  =  std::cos(alpha_des_(i)) * f_min;
            f_mat(i,1)  =  std::sin(alpha_des_(i)) * f_min;
            thrusts(i)  = f_min;
        }
    }

    alpha_prev_ = angles;
    u_prev_     = thrusts;

    const Eigen::Vector3d tau_actual = thrust_matrix_ * f;

    return std::make_tuple(f, thrusts, angles, tau_actual);
}