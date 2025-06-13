#include "thrust_allocation/nlp.hpp"

NLPAllocator::NLPAllocator(NLPParams params)
    : dt_(params.dt),
      u_bound_(params.u_bound),
      w_angle_(params.w_angle),
      w_neg_(params.w_neg),
      max_rate_(params.max_rate),
      max_force_rate_(params.max_force_rate),
      w_alpha_change_(params.w_alpha_change)
    {
        thruster_positions_ = {{
            { half_length_,  half_width_},   // front-left
            { half_length_, -half_width_},   // front-right
            {-half_length_,  half_width_},   // rear-left
            {-half_length_, -half_width_}    // rear-right
        }};

        alpha_des_  = {  3*M_PI/4, -3*M_PI/4,  M_PI/4, -M_PI/4 };
        alpha_prev_ = alpha_des_;                   // warm start
        u_prev_.assign(n_thrusters_, 0.0);

        static_alpha_lb_ = {  M_PI/2, -M_PI,      0.0,     -M_PI/2};
        static_alpha_ub_ = {      M_PI, -M_PI/2,  M_PI/2,        0};

        u_lb_.assign(n_thrusters_, -u_bound_);
        u_ub_.assign(n_thrusters_,  u_bound_);

        build_nlp();

        x0_.reserve(2*n_thrusters_);
        x0_.insert(x0_.end(), alpha_des_.begin(), alpha_des_.end());
        x0_.insert(x0_.end(), n_thrusters_, 0.0);
    }

std::tuple<Eigen::Vector4d, Eigen::Vector4d, Eigen::Vector3d> NLPAllocator::allocate(
    const std::array<double, 3>& tau_desired) {

    std::vector<double> alpha_lb_dyn(n_thrusters_), alpha_ub_dyn(n_thrusters_);
    std::vector<double> u_lb_dyn(n_thrusters_),    u_ub_dyn(n_thrusters_);

    const double d_alpha_max = max_rate_       * dt_;
    const double d_u_max     = max_force_rate_ * dt_;

    for (std::size_t i = 0; i < n_thrusters_; ++i) {
        alpha_lb_dyn[i] = std::max(static_alpha_lb_[i],
                                    alpha_prev_[i] - d_alpha_max);
        alpha_ub_dyn[i] = std::min(static_alpha_ub_[i],
                                    alpha_prev_[i] + d_alpha_max);

        u_lb_dyn[i]     = std::max(u_lb_[i], u_prev_[i] - d_u_max);
        u_ub_dyn[i]     = std::min(u_ub_[i], u_prev_[i] + d_u_max);
    }

    std::vector<double> lbx, ubx;
    lbx.reserve(2*n_thrusters_);
    ubx.reserve(2*n_thrusters_);
    lbx.insert(lbx.end(), alpha_lb_dyn.begin(), alpha_lb_dyn.end());
    ubx.insert(ubx.end(), alpha_ub_dyn.begin(), alpha_ub_dyn.end());
    lbx.insert(lbx.end(),    u_lb_dyn.begin(),    u_lb_dyn.end());
    ubx.insert(ubx.end(),    u_ub_dyn.begin(),    u_ub_dyn.end());

    std::vector<double> lbg(3, 0.0), ubg(3, 0.0);

    std::vector<double> p;
    p.reserve(3 + n_thrusters_);
    p.insert(p.end(), tau_desired.begin(), tau_desired.end());
    p.insert(p.end(), alpha_prev_.begin(), alpha_prev_.end());

    casadi::DMDict arg {
        {"x0",  x0_},
        {"lbx", lbx},
        {"ubx", ubx},
        {"lbg", lbg},
        {"ubg", ubg},
        {"p",   p }
    };
    casadi::DMDict res = solver_(arg);
    std::vector<double> x_opt = std::vector<double>(res.at("x"));

    std::vector<double> alpha_opt(x_opt.begin(),                    x_opt.begin()+n_thrusters_);
    std::vector<double> u_opt    (x_opt.begin()+n_thrusters_,       x_opt.end());

    double Fx = 0.0, Fy = 0.0, Mz = 0.0;
    for (std::size_t i = 0; i < n_thrusters_; ++i) {
        const double c = std::cos(alpha_opt[i]);
        const double s = std::sin(alpha_opt[i]);
        const double ui = u_opt[i];
        Fx += c * ui;
        Fy += s * ui;
        Mz += ( thruster_positions_[i][0] * s - thruster_positions_[i][1] * c ) * ui;
    }
    std::array<double,3> tau_out{Fx, Fy, Mz};

    x0_          = x_opt;
    alpha_prev_  = alpha_opt;
    u_prev_      = u_opt;

    Eigen::Vector4d alpha, u;
    for (std::size_t i = 0; i < n_thrusters_; ++i) {
        alpha(i) = alpha_opt[i];
        u(i)     = u_opt[i];
    }
    Eigen::Vector3d tau;
    for (std::size_t i = 0; i < 3; ++i) {
        tau(i) = tau_out[i];
    }

    return std::make_tuple(alpha, u, tau);
}

void NLPAllocator::build_nlp() {

    casadi::SX alpha = casadi::SX::sym("alpha", n_thrusters_);
    casadi::SX u     = casadi::SX::sym("u",     n_thrusters_);
    casadi::SX x     = casadi::SX::vertcat({alpha, u});

    casadi::SX tau_sym        = casadi::SX::sym("tau",        3);
    casadi::SX alpha_prev_sym = casadi::SX::sym("alpha_prev", n_thrusters_);
    casadi::SX p              = casadi::SX::vertcat({tau_sym, alpha_prev_sym});

    casadi::SX Fx = 0, Fy = 0, Mz = 0;
    for (std::size_t i = 0; i < n_thrusters_; ++i) {
        casadi::SX ai = alpha(i);
        Fx += casadi::SX::cos(ai) * u(i);
        Fy += casadi::SX::sin(ai) * u(i);
        Mz += (thruster_positions_[i][0] * casadi::SX::sin(ai)
         - thruster_positions_[i][1] * casadi::SX::cos(ai)) * u(i);
    }
    casadi::SX F_net = casadi::SX::vertcat({Fx, Fy, Mz});

    casadi::SX cost = 0.5 * casadi::SX::sumsqr(u);
    casadi::SX alpha_des_sx = casadi::SX(casadi::DM(alpha_des_));
    cost += w_angle_ * casadi::SX::sumsqr(alpha - alpha_des_sx);
    casadi::SX u_neg = casadi::SX::fmax(casadi::SX(0), -u);
    cost += w_neg_ * casadi::SX::sumsqr(u_neg);
    cost += w_alpha_change_ * casadi::SX::sumsqr(alpha - alpha_prev_sym);

    casadi::SX g = F_net - tau_sym;
    
    casadi::SXDict nlp {{"x", x}, {"f", cost}, {"g", g}, {"p", p}};
    casadi::Dict opts;
    opts["print_time"]          = false;
    opts["ipopt.print_level"]   = 0;
    solver_ = casadi::nlpsol("solver", "ipopt", nlp, opts);
}