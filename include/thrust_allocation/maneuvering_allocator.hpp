#ifndef MANEUVERING_ALLOCATOR_HPP
#define MANEUVERING_ALLOCATOR_HPP

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <array>
#include <cmath>
#include <algorithm>

struct ManeuveringParams {
    double dt = 0.01;
    double gamma = 0.1;
    double mu = 10.0;
    double rho = 1.0;
    double zeta = 0.001;
    double rate_limit = 100.0;
    double theta_min = -50.0;
    double theta_max = 50.0;
    double lambda = 1.0;
    bool power_management = false;
};

static constexpr int kNumThrusters = 4;     ///< m
static constexpr int kXiSize       = 8;    ///< 2*m
static constexpr int kThetaSize    = 5;    ///< p-n  (8–3)

using VecXi  = Eigen::Matrix<double,kXiSize ,1>;
using VecTau = Eigen::Matrix<double,3,1>;
using VecTheta = Eigen::Matrix<double,kThetaSize,1>;
using MatT   = Eigen::Matrix<double,3,kXiSize>;
using MatQ   = Eigen::Matrix<double,kXiSize,kThetaSize>;
using MatW   = Eigen::Matrix<double,kXiSize,kXiSize>;

class ManeuveringAllocator {
public:
    ManeuveringAllocator(ManeuveringParams& params);
    std::tuple<Eigen::Vector4d,Eigen::Vector4d,VecTau> allocate(
        const std::array<double, 3>& tau_cmd, Eigen::VectorXd xi_p);
    VecTheta theta_;
private:
    Eigen::MatrixXd pseudo_inverse(const Eigen::MatrixXd& A) {
        return A.transpose() * (A * A.transpose()).inverse();
    }
    Eigen::MatrixXd null_space(const Eigen::MatrixXd& A) {
        using namespace Eigen;
        
        JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
        const auto& S = svd.singularValues();
        const int   m = A.rows();
        const int   n = A.cols();

        double tol = std::numeric_limits<double>::epsilon() *
                    std::max(m, n) * S(0);

        int rank = 0;
        while (rank < S.size() && S(rank) > tol) ++rank;

        MatrixXd N = svd.matrixV().rightCols(n - rank);

        HouseholderQR<MatrixXd> qr(N);
        N = qr.householderQ() * MatrixXd::Identity(n, n - rank);

        return N;
    }
    VecTheta calculate_j_theta(const VecXi& xi_d) const;
    VecTheta calculate_v_theta(const VecXi& xi_tilde) const;
    VecXi calculate_kappa(const VecXi& xi_tilde, const VecXi& xi_p_dot, const VecTheta& v) const;
    VecXi cbf(const VecXi& u, double time_constant) const;

    double dt_;
    double gamma_;
    double mu_;
    double rho_;
    double zeta_;
    double rate_limit_;
    double theta_min_;
    double theta_max_;
    double lambda_;
    bool power_management_{false};

    MatW w_matrix_;
    VecXi xi_;
    VecXi last_xi_p_;

    Eigen::Vector4d c_;

    VecXi min_forces_;
    VecXi max_forces_;
    Eigen::Vector4d reference_angles_;

    MatT T_;
    Eigen::Matrix<double, kXiSize, 3> T_pinv_;
    MatQ q_matrix_;
};

#endif // MANEUVERING_ALLOCATOR_HPP