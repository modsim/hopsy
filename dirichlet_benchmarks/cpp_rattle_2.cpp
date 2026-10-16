#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>

namespace py = pybind11;

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

struct RattleIntegrator {
    py::object model;
    Mat metric_inv;
    Mat S;
    Vec c;
    Vec lower;
    Vec upper;
    Mat proj_matr;
    Vec particular_solution;
    double stepsize;
    int integration_length;
    bool include_gram_term;
    int n_inner_step;
    bool adaptive_inner_steps;
    bool include_squash_logdet;

    Vec as_vector(const py::object& obj) const {
        py::array_t<double, py::array::c_style | py::array::forcecast> arr(obj);
        auto info = arr.request();

        Vec out(info.size);
        const double* ptr = static_cast<const double*>(info.ptr);
        for (py::ssize_t i = 0; i < info.size; ++i) {
            out[i] = ptr[i];
        }
        return out;
    }

    Vec squash(const Vec& q) const {
        Vec out(q.size());
        for (Eigen::Index i = 0; i < q.size(); ++i) {
            double qi = q[i];
            out[i] = 0.5 * (upper[i] + lower[i])
                   + 0.5 * (upper[i] - lower[i]) * qi / std::sqrt(qi * qi + 1.0);
        }
        return out;
    }

    Vec squash_jacobian_diag(const Vec& q) const {
        Vec d(q.size());
        for (Eigen::Index i = 0; i < q.size(); ++i) {
            double qi = q[i];
            d[i] = 0.5 * (upper[i] - lower[i]) / std::pow(qi * qi + 1.0, 1.5);
        }
        return d;
    }

    Vec grad_logdet_squash_jacobian(const Vec& q) const {
        Vec out(q.size());
        for (Eigen::Index i = 0; i < q.size(); ++i) {
            out[i] = -3.0 * q[i] / (1.0 + q[i] * q[i]);
        }
        return out;
    }

    Vec constr(const Vec& q) const {
        return S * squash(q) - c;
    }

    Mat jacob_constr(const Vec& q) const {
        Vec d = squash_jacobian_diag(q);
        return S * d.asDiagonal();
    }

    // Gradient of the Hausdorff (Gram) correction
    //     g(q) = 1/2 log det( J M^-1 J^T ),   J = S diag(d(q)).
    // With A = J M^-1 J^T and d'_k / d_k = -3 q_k / (1 + q_k^2):
    //     dg/dq_k = (d'_k / d_k) * [ M^-1 J^T A^-1 J ]_kk.
    Vec grad_gram(const Vec& q) const {
        Mat J = jacob_constr(q);                    // m x n
        Mat MiJt = metric_inv * J.transpose();      // n x m
        Mat A = J * MiJt;                           // m x m
        Mat AinvJ = A.ldlt().solve(J);              // m x n

        Vec diag(q.size());
        for (Eigen::Index k = 0; k < q.size(); ++k) {
            diag[k] = MiJt.row(k).dot(AinvJ.col(k));
        }
        return grad_logdet_squash_jacobian(q).cwiseProduct(diag);
    }

    // Gradient of
    //     h1(q) = -log pi(x(q)) - [sum_i log d_i(q)] + [1/2 logdet(J M^-1 J^T)]
    // The two bracketed terms are individually switchable, for diagnostic runs
    // only.  Both belong in the Hamiltonian; switching either off changes the
    // invariant distribution.
    Vec dh1_dpos(const Vec& q) const {
        Vec theta = squash(q);
        Vec x = proj_matr * (theta - particular_solution);

        py::object grad_obj = model.attr("log_gradient")(x);
        Vec grad_x = as_vector(grad_obj);

        Vec lifted_grad = proj_matr.transpose() * grad_x;
        Vec d = squash_jacobian_diag(q);

        Vec out = -d.cwiseProduct(lifted_grad);
        if (include_squash_logdet) {
            out -= grad_logdet_squash_jacobian(q);
        }
        if (include_gram_term) {
            out += grad_gram(q);
        }
        return out;
    }

    Vec dh2_dmom(const Vec& mom) const {
        return metric_inv * mom;
    }

    void h1_flow(Vec& q, Vec& mom, double factor) const {
        mom -= factor * stepsize * dh1_dpos(q);
    }

    void h2_flow(Vec& q, const Vec& mom, double factor) const {
        q += factor * stepsize * dh2_dmom(mom);
    }

    void project_onto_cotangent_space(const Vec& q, Vec& mom) const {
        Mat J = jacob_constr(q);
        Mat G = J * metric_inv * J.transpose();
        Vec rhs = J * (metric_inv * mom);
        Vec lambda = G.partialPivLu().solve(rhs);
        mom -= J.transpose() * lambda;
    }

    void newton_solver(
        Vec& q,
        Vec& mom,
        const Vec& q_prev,
        double factor,
        double constraint_tol = 1e-9,
        double position_tol = 1e-8,
        double divergence_tol = 1e10,
        int max_iters = 50,
        int line_search_max_iters = 10,
        double line_search_decrease = 0.5
    ) const {
        Vec mu = Vec::Zero(q.size());
        Mat J_prev = jacob_constr(q_prev);
        Mat dh2_flow_q_dmom = std::abs(factor) * stepsize * metric_inv;
        Mat dh2_flow_mom_dmom = Mat::Identity(metric_inv.rows(), metric_inv.cols());

        for (int i = 0; i < max_iters; ++i) {
            Mat J = jacob_constr(q);
            Vec con = constr(q);
            double error = con.cwiseAbs().maxCoeff();

            Mat G = J * dh2_flow_q_dmom * J_prev.transpose();
            Vec delta_lambda = G.partialPivLu().solve(con);
            Vec delta_mu = J_prev.transpose() * delta_lambda;
            Vec delta_q = dh2_flow_q_dmom * delta_mu;

            double q_norm = q.cwiseAbs().maxCoeff();
            double delta_q_norm = delta_q.cwiseAbs().maxCoeff();

            if (
                error > divergence_tol ||
                !std::isfinite(error) ||
                !std::isfinite(q_norm) ||
                !std::isfinite(delta_q_norm) ||
                q_norm > 1e4 ||
                delta_q_norm > 1e4
            ) {
                throw std::runtime_error("Newton solver diverged.");
            }

            if (error < constraint_tol && delta_q_norm < position_tol) {
                mom -= (factor >= 0.0 ? 1.0 : -1.0) * (dh2_flow_mom_dmom * mu);
                return;
            }

            double step_size = 1.0;
            bool accepted = false;
            Vec best_q = q - delta_q;
            Vec best_mu = mu + delta_mu;
            double best_error = std::numeric_limits<double>::infinity();

            for (int ls = 0; ls < line_search_max_iters; ++ls) {
                Vec q_candidate = q - step_size * delta_q;
                double candidate_error = constr(q_candidate).cwiseAbs().maxCoeff();

                if (std::isfinite(candidate_error) && candidate_error < best_error) {
                    best_error = candidate_error;
                    best_q = q_candidate;
                    best_mu = mu + step_size * delta_mu;
                }

                if (std::isfinite(candidate_error) && candidate_error <= error) {
                    q = q_candidate;
                    mu += step_size * delta_mu;
                    accepted = true;
                    break;
                }

                step_size *= line_search_decrease;
            }

            if (!accepted) {
                if (!std::isfinite(best_error) || best_error >= error) {
                    throw std::runtime_error("Newton line search failed.");
                }
                q = best_q;
                mu = best_mu;
            }
        }

        throw std::runtime_error("Newton solver did not converge.");
    }

    void h2_flow_with_retraction_onto_mfd(Vec& q, Vec& mom, const Vec& q_prev, double factor) const {
        h2_flow(q, mom, factor);
        newton_solver(q, mom, q_prev, factor);
    }

    // q_start and mom_start are captured ONCE, before the retry loop.  Capturing
    // them inside meant every retry restarted from the state left behind by the
    // failed attempt, which is generally off the manifold.
    void step_b(Vec& q, Vec& mom, double factor = 1.0) const {
        const Vec q_start = q;
        const Vec mom_start = mom;

        const int max_inner = adaptive_inner_steps ? 1024 : n_inner_step;
        int current = n_inner_step;
        std::string last_error;

        while (current <= max_inner) {
            q = q_start;
            mom = mom_start;

            double factor_inner = factor / static_cast<double>(current);

            try {
                for (int i = 0; i < current; ++i) {
                    Vec q_prev = q;
                    h2_flow_with_retraction_onto_mfd(q, mom, q_prev, factor_inner);
                    project_onto_cotangent_space(q, mom);
                }
                return;
            } catch (const std::runtime_error& exc) {
                last_error = exc.what();
                current *= 2;
            }
        }

        q = q_start;
        mom = mom_start;
        throw std::runtime_error("step_b failed. Last error: " + last_error);
    }

    void step_a(Vec& q, Vec& mom, double factor) const {
        h1_flow(q, mom, factor);
        project_onto_cotangent_space(q, mom);
    }

    void rattle_step(Vec& q, Vec& mom) const {
        step_a(q, mom, 0.5);
        step_b(q, mom, 1.0);
        step_a(q, mom, 0.5);
    }

    std::pair<Vec, Vec> integrate(Vec q, Vec mom) const {
        for (int i = 0; i < integration_length; ++i) {
            rattle_step(q, mom);
        }
        return {q, mom};
    }
};

py::tuple integrate(
    py::object model,
    const Vec& q,
    const Vec& mom,
    const Mat& metric_inv,
    const Mat& S,
    const Vec& c,
    const Vec& lower,
    const Vec& upper,
    const Mat& proj_matr,
    double stepsize,
    int integration_length,
    const Vec& particular_solution,
    bool include_gram_term,
    int n_inner_step,
    bool adaptive_inner_steps,
    bool include_squash_logdet
) {
    Vec p_sol = particular_solution.size() == 0
        ? Vec::Zero(S.cols())
        : particular_solution;

    RattleIntegrator integrator{
        model,
        metric_inv,
        S,
        c,
        lower,
        upper,
        proj_matr,
        p_sol,
        stepsize,
        integration_length,
        include_gram_term,
        n_inner_step,
        adaptive_inner_steps,
        include_squash_logdet
    };

    auto result = integrator.integrate(q, mom);
    return py::make_tuple(result.first, result.second);
}

PYBIND11_MODULE(cpp_rattle_2, m) {
    m.def(
        "integrate",
        &integrate,
        py::arg("model"),
        py::arg("q"),
        py::arg("mom"),
        py::arg("metric_inv"),
        py::arg("S"),
        py::arg("c"),
        py::arg("lower"),
        py::arg("upper"),
        py::arg("proj_matr"),
        py::arg("stepsize"),
        py::arg("integration_length"),
        py::arg("particular_solution") = Vec(),
        py::arg("include_gram_term") = true,
        py::arg("n_inner_step") = 1,
        py::arg("adaptive_inner_steps") = false,
        py::arg("include_squash_logdet") = true
    );
}
