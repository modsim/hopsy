import hopsy
import numpy as np
import numpy.typing as npt
from typing import Callable
import arviz as az
from scipy.stats import dirichlet
import pandas as pd
import warnings

import numpy as np

import cpp_rattle_2 as cpp_rattle


class CustomProposal:
    """RATTLE proposal on {q : S sigma(q) = c}, algebraic squashing.

        sigma(q) = c0 + r q/sqrt(q^2+1),  d = r/(q^2+1)^{3/2},  dlog d/dq = -3q/(1+q^2)

        h1(q) = -log pi(x(q)) - sum_i log d_i(q) + 1/2 log det(J M^-1 J^T),  J = S diag(d)

    Die beiden Klammerterme sind einzeln abschaltbar, aber nur fuer Diagnoselaeufe;
    beide gehoeren in den Hamiltonian.
    """

    def __init__(
        self,
        problem,
        starting_point,
        fisher,
        map_guess,
        param_matr,
        proj_matr,
        beta,
        equality_constraints,
        box_constraints,
        particular_solution=None,
        momentum=None,
        stepsize=1.0,
        integration_length=1,
        nuts=False,
        random_int_length=False,
        pullback_momentum=False,
        include_gram_term=True,
        include_squash_logdet=True,
        reversibility_check=True,
        reversibility_tol=1e-5,
        constraint_tol=1e-8,
        n_inner_step=1,
        adaptive_inner_steps=False,
        squash_eps=1e-12,
        feasibility_tol=0.0,
        seed=None,
    ):
        self.problem = problem
        self.stepsize = stepsize
        self.integration_length = integration_length
        self.random_int_length = random_int_length
        self.nuts = nuts
        if nuts:
            warnings.warn("nuts=True is stored but not implemented.", RuntimeWarning)

        S, c = equality_constraints
        self.S = np.asarray(S, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)

        lower, upper = box_constraints
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

        self.param_matr = np.asarray(param_matr, dtype=np.float64)
        self.proj_matr = np.asarray(proj_matr, dtype=np.float64)
        self.beta = beta
        self.particular_solution = (
            np.zeros(self.S.shape[1], dtype=np.float64)
            if particular_solution is None
            else np.asarray(particular_solution, dtype=np.float64)
        )

        self.squash_eps = float(squash_eps)
        self.feasibility_tol = float(feasibility_tol)
        self.include_gram_term = bool(include_gram_term)
        self.include_squash_logdet = bool(include_squash_logdet)
        self.reversibility_check = bool(reversibility_check)
        # gegen das Residuenhistogramm kalibrieren, nicht raten
        self.reversibility_tol = float(reversibility_tol)
        self.constraint_tol = float(constraint_tol)
        self.n_inner_step = int(n_inner_step)
        self.adaptive_inner_steps = bool(adaptive_inner_steps)

        self.n_proposals = 0
        self.n_squash_clips = 0
        self.n_integrator_failures = 0
        self.n_constraint_failures = 0
        self.n_reversibility_failures = 0
        self.n_nonfinite_energy = 0
        self.last_integrator_error = None
        self.last_reversibility_residual = np.nan
        self.max_reversibility_residual = 0.0
        self.last_constraint_residual = np.nan
        self.max_constraint_residual = 0.0

        self.map_guess = np.asarray(map_guess, dtype=np.float64)
        self.map_guess_full = self.affine_transformation(self.map_guess)

        self.base_metric = np.asarray(fisher, dtype=np.float64)
        self.base_metric_sqrt = np.linalg.cholesky(self.base_metric)
        self.pullback_momentum = pullback_momentum
        if pullback_momentum:
            warnings.warn(
                "pullback_momentum=True is inconsistent with the constant metric "
                "in h2; stationarity is lost.",
                RuntimeWarning,
            )

        self.metric = self.pullback_metric()
        self.metric_inv = np.linalg.inv(self.metric)
        self.metric_sqrt = np.linalg.cholesky(self.metric)

        ineq_A = np.asarray(getattr(self.problem, "A", np.empty((0, 0))), dtype=np.float64)
        ineq_b = np.asarray(getattr(self.problem, "b", np.empty(0)), dtype=np.float64).ravel()
        if ineq_A.size == 0:
            ineq_A = np.empty((0, 0))
            ineq_b = np.empty(0)

        self.rattle = cpp_rattle.Rattle(
            model=self.problem.model,
            metric_inv=self.metric_inv,
            S=self.S,
            c=self.c,
            lower=self.lower,
            upper=self.upper,
            proj_matr=self.proj_matr,
            particular_solution=self.particular_solution,
            ineq_A=ineq_A,
            ineq_b=ineq_b,
            stepsize=self._stepsize,
            include_gram_term=self.include_gram_term,
            n_inner_step=self.n_inner_step,
            adaptive_inner_steps=self.adaptive_inner_steps,
            include_squash_logdet=self.include_squash_logdet,
            reversibility_check=False,  # MH prueft die ganze Trajektorie
            reversibility_tol=self.reversibility_tol,
            feasibility_tol=self.feasibility_tol,
        )

        self.rng = np.random.default_rng(seed)

        self.state_transformed = None
        self.state = np.asarray(starting_point, dtype=np.float64)

        self.proposal = self._state.copy()
        self.proposal_q = self.state_transformed.copy()
        self.momentum = momentum
        self.proposal_mom = momentum
        self._proposal_valid = False

    # ------------------------------------------------------------------ nuts
    @property
    def stepsize(self):
        return self._stepsize

    @stepsize.setter
    def stepsize(self, value):
        self._stepsize = float(value)
        r = getattr(self, "rattle", None)
        if r is not None:
            r.stepsize = self._stepsize
    # ------------------------------------------------------------------ state


    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = np.asarray(value, dtype=np.float64)
        self.state_transformed = self.squash_inv(
            self.affine_transformation(self._state)
        )

    # ------------------------------------------------------------------- maps

    def affine_transformation(self, x):
        return self.param_matr @ x + self.particular_solution

    def projection(self, x):
        return self.proj_matr @ (x - self.particular_solution)

    def squash(self, q):
        q = np.asarray(q, dtype=np.float64)
        return (
            0.5 * (self.upper + self.lower)
            + 0.5 * (self.upper - self.lower) * (q / np.sqrt(q**2 + 1.0))
        )

    def squash_inv(self, x):
        x = np.asarray(x, dtype=np.float64)
        hh = (2.0 * x - (self.upper + self.lower)) / (self.upper - self.lower)
        hh_clipped = np.clip(hh, -1.0 + self.squash_eps, 1.0 - self.squash_eps)
        if np.any(hh_clipped != hh):
            self.n_squash_clips += 1
        return hh_clipped / np.sqrt(1.0 - hh_clipped**2)

    def squash_jacobian_diag(self, q):
        q = np.asarray(q, dtype=np.float64)
        return 0.5 * (self.upper - self.lower) / ((q**2 + 1.0) ** 1.5)

    def squash_jacobian(self, q):
        return np.diag(self.squash_jacobian_diag(q))

    def squash_inv_jacobian(self, x):
        x = np.asarray(x, dtype=np.float64)
        hh = (2.0 * x - (self.upper + self.lower)) / (self.upper - self.lower)
        # d/dx [hh / sqrt(1 - hh^2)] = (dhh/dx) * 1 / (1 - hh^2)^(3/2)
        dhh_dx = 2.0 / (self.upper - self.lower)
        diag = dhh_dx / (1.0 - hh ** 2) ** 1.5
        return np.diag(diag)

    # ----------------------------------------------------------------- metric

    def pullback_metric(self):
        q_map = self.squash_inv(self.map_guess_full)
        d = self.squash_jacobian_diag(q_map)
        m_reg = (
            self.proj_matr.T @ self.base_metric @ self.proj_matr
            + self.beta * np.eye(self.param_matr.shape[0])
        )
        return d[:, None] * m_reg * d[None, :]

    def jacob_constr(self, q):
        d = self.squash_jacobian_diag(q)
        return self.S * d[None, :]

    def project_onto_cotangent_space(self, q, mom):
        J = self.jacob_constr(q)
        G = J @ self.metric_inv @ J.T
        lam = np.linalg.solve(G, J @ (self.metric_inv @ mom))
        return mom - J.T @ lam

    def constraint_residual(self, q):
        return float(np.linalg.norm(self.S @ self.squash(q) - self.c, ord=np.inf))

    # --------------------------------------------------------------- momentum

    def _resolve_rng(self, rng):
        if rng is not None and hasattr(rng, "standard_normal"):
            return rng
        return self.rng

    def sample_momentum(self, q, rng=None):
        rng = self._resolve_rng(rng)
        if self.pullback_momentum:
            try:
                with np.errstate(over='raise', invalid='raise'):
                    #if self._affine_transformation is not None:
                    v_red = np.linalg.inv(self.base_metric_sqrt) @ self.rng.standard_normal(self.base_metric_sqrt.shape[0])
                    #print("mom_red: ", mom_red.shape)
                    v = self.squash_inv_jacobian(self.squash(q)) @ self.param_matr @ v_red
                    mom = self.metric @ v
                    #print(mom)
                    #mom = self.project_onto_cotangent_space(q, mom)
                    return mom
            except FloatingPointError:
                pass
        mom = self.metric_sqrt @ rng.standard_normal(self.metric_sqrt.shape[0])
        return self.project_onto_cotangent_space(q, mom)

    # ------------------------------------------------------------- Hamiltonian

    def logdet_squash_jacobian(self, q):
        return float(np.sum(np.log(self.squash_jacobian_diag(q))))

    def log_gram_term(self, q):
        """0.5 * log det(J M^-1 J^T). Reweight old draws with exp(-this)."""
        J = self.jacob_constr(q)
        sign, logdet = np.linalg.slogdet(J @ self.metric_inv @ J.T)
        if sign <= 0.0 or not np.isfinite(logdet):
            return np.inf
        return 0.5 * logdet

    def h1(self, q):
        theta = self.squash(q)
        x = self.projection(theta)
        log_pi = self.problem.model.log_density(x)
        if not np.isfinite(log_pi):
            return np.inf
        value = -log_pi
        if self.include_squash_logdet:
            value = value - self.logdet_squash_jacobian(q)
        if self.include_gram_term:
            value = value + self.log_gram_term(q)
        return value

    def dh2_dmom(self, mom):
        return self.metric_inv @ mom

    def h2(self, mom):
        return 0.5 * mom @ self.dh2_dmom(mom)

    def h(self, q, mom):
        return self.h1(q) + self.h2(mom)

    # ------------------------------------------------------------- integration

    def _integrate(self, q, mom, n_steps):
        try:
            q_out, mom_out = self.rattle.integrate(q, mom, n_steps)
        except Exception as exc:
            self.last_integrator_error = f"{type(exc).__name__}: {exc}"
            return None
        q_out = np.asarray(q_out, dtype=np.float64)
        mom_out = np.asarray(mom_out, dtype=np.float64)
        if not (np.all(np.isfinite(q_out)) and np.all(np.isfinite(mom_out))):
            self.last_integrator_error = "non-finite integrator output"
            return None
        return q_out, mom_out

    def _check_reversibility(self, q0, mom0, q1, mom1, n_steps):
        back = self._integrate(q1, -mom1, n_steps)
        if back is None:
            self.last_reversibility_residual = np.inf
            return False
        q_back, mom_back = back
        scale_q = max(1.0, float(np.linalg.norm(q0)))
        scale_p = max(1.0, float(np.linalg.norm(mom0)))
        residual = max(
            float(np.linalg.norm(q_back - q0)) / scale_q,
            float(np.linalg.norm(mom_back + mom0)) / scale_p,
        )
        self.last_reversibility_residual = residual
        self.max_reversibility_residual = max(self.max_reversibility_residual, residual)
        return residual <= self.reversibility_tol

    # --------------------------------------------------------------- proposal

    def _invalidate(self):
        self._proposal_valid = False
        self.proposal_q = self.state_transformed.copy()
        self.proposal_mom = None if self.momentum is None else self.momentum.copy()
        self.proposal = self._state.copy()
        return self.proposal

    def propose(self, rng=None):
        rng = self._resolve_rng(rng)
        self.n_proposals += 1
        self._proposal_valid = False

        self.momentum = self.sample_momentum(self.state_transformed, rng)

        if self.random_int_length:
            n_steps = int(rng.integers(1, self.integration_length + 1))
        else:
            n_steps = int(self.integration_length)

        forward = self._integrate(self.state_transformed, self.momentum, n_steps)
        if forward is None:
            self.n_integrator_failures += 1
            return self._invalidate()
        proposal_q, proposal_mom = forward

        residual = self.constraint_residual(proposal_q)
        self.last_constraint_residual = residual
        self.max_constraint_residual = max(self.max_constraint_residual, residual)
        if not (residual <= self.constraint_tol):
            self.n_constraint_failures += 1
            return self._invalidate()

        if self.reversibility_check and not self._check_reversibility(
            self.state_transformed, self.momentum, proposal_q, proposal_mom, n_steps
        ):
            self.n_reversibility_failures += 1
            return self._invalidate()

        self.proposal_q = proposal_q
        self.proposal_mom = proposal_mom
        self.proposal = self.projection(self.squash(proposal_q))
        self._proposal_valid = True
        return self.proposal

    def log_acceptance_probability(self):
        if not self._proposal_valid:
            return -np.inf

        theta = self.affine_transformation(self.proposal)
        if (self.problem.A @ theta - self.problem.b > self.feasibility_tol).any():
            return -np.inf

        h_proposal = self.h(self.proposal_q, self.proposal_mom)
        if not np.isfinite(h_proposal):
            return -np.inf

        h_state = self.h(self.state_transformed, self.momentum)
        if not np.isfinite(h_state):
            self.n_nonfinite_energy += 1
            return 0.0

        return -(h_proposal - h_state)

    def accept_proposal(self):
        self._state = self.proposal.copy()
        self.state_transformed = self.proposal_q.copy()
        return self._state

    # ------------------------------------------------------------ diagnostics

    def diagnostics(self):
        n = max(1, self.n_proposals)
        return {
            "n_proposals": self.n_proposals,
            "squash_clips": self.n_squash_clips,
            "integrator_failures": self.n_integrator_failures,
            "constraint_failures": self.n_constraint_failures,
            "reversibility_failures": self.n_reversibility_failures,
            "reversibility_failure_rate": self.n_reversibility_failures / n,
            "max_reversibility_residual": self.max_reversibility_residual,
            "max_constraint_residual": self.max_constraint_residual,
            "nonfinite_energy": self.n_nonfinite_energy,
            "last_integrator_error": self.last_integrator_error,
        }


