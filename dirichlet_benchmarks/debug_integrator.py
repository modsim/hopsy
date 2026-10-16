import hopsy
import numpy as np

class CustomProposal:
    def __init__(self,
                 problem: hopsy.Problem,
                 starting_point: np.array,
                 fisher: np.ndarray,
                 map_guess: np.ndarray,
                 param_matr: np.ndarray,
                 proj_matr: np.ndarray,
                 beta: float,
                 equality_constraints: tuple[np.ndarray, np.ndarray],
                 box_constraints: tuple[npt.ArrayLike, npt.ArrayLike],
                 particular_solution: np.ndarray = None,
                 momentum: np.ndarray = None,
                 stepsize: float = 1.,
                 integration_length: int = 1,
                 nuts: bool = False,
                 #seed: int = 0
                ):
        self.problem = problem
        self.stepsize = stepsize
        self.integration_length = integration_length
        self.nuts = nuts

        # Stoichiometric equality constraints
        S, c = equality_constraints
        S = np.asarray(S)
        c = np.asarray(c)
        if S.ndim != 2:
            raise ValueError("Equality constraints: Matrix must be 2-dimensional.")
        if c.ndim != 1:
            raise ValueError("Equality constraints: Vector must be 1-dimensional.")
        if S.shape[0] != c.shape[0]:
            raise ValueError("Equality constraints: Matrix and vector dimensions are incompatible.")
        self.S = S
        self.c = c
        self.particular_solution = np.zeros(S.shape[1]) if particular_solution is None else particular_solution

        # Box constraints
        lower, upper = box_constraints
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        #print(lower)
        #print(upper)
        if lower.ndim != 1 or upper.ndim != 1:
            raise ValueError("Expected a 1D array-like object.")
        if lower.shape[0] != self.S.shape[1] or upper.shape[0] != self.S.shape[1]:
            raise ValueError("Box constraints do not match equality constraints.")
        self.lower = lower
        self.upper = upper

        # Transformation between reduced and full parameter space
        self.param_matr = None if param_matr is None else np.asarray(param_matr)
        self.proj_matr = None if proj_matr is None else np.asarray(proj_matr)
        self.beta = beta
        self.map_guess = np.asarray(map_guess)
        if self.param_matr is not None:
            self.map_guess = self.affine_transformation(map_guess)

        # Mass matrix
        self.base_metric = np.asarray(fisher)
        if self.base_metric.shape[0] != self.base_metric.shape[1]:
            raise ValueError("Metric must be a square matrix.")
        self.metric = self.pullback_metric()
        #print(self.metric)
        #print(self.metric) # Debugger des kleinen Mannes
        self.metric_inv = np.linalg.inv(self.metric)
        self.metric_sqrt = np.linalg.cholesky(self.metric)
        self.metric_sqrt_red = np.linalg.cholesky(self.base_metric)
        self.dh2_dpos = np.zeros_like(self.lower)

        # State and momentum
        self.state = self.proposal = np.asarray(starting_point)
        self.state_transformed = self.squash_inv(self.affine_transformation(self.state))
        #self.seed = seed
        #self.rng = np.random.default_rng(self.seed)
        self.rng = np.random.default_rng()
        self.momentum = self.proposal_mom = momentum
        '''
        if momentum is not None:
            #print(self.proj_matr.T.shape)
            self.momentum = self.proposal_mom = self.squash_jacobian(self.state_transformed).T @ self.proj_matr.T @ momentum
            #print(self.momentum.shape)
        else:
            self.momentum = self.proposal_mom = self.sample_momentum(self.state_transformed)
        '''
#####################################################################################
    # Affine transformations for reduced and full parameter space
    def affine_transformation(self, x_):
        #print("affine transformation")
        #print("param_matrix: ", self.param_matr.shape)
        #print("x_: ", x_.shape)
        #print("x_: ", x_)
        res = self.param_matr@x_ + self.particular_solution
        #print("x: ", x_)
        #print("trafo: ", res)
        return res

    def projection(self, x):
        res = self.proj_matr@(x-self.particular_solution)
        #print("x: ", x)
        #print("projection: ", res)
        return res

    #####################################################################################
    # Transformation of the problem

    # squashing function (aka 'sigmoid')
    def squash(self, q):
        upper = self.upper
        lower = self.lower
        q = np.asarray(q)
        sig = 0.5 * (upper + lower) + 0.5 * (upper - lower) * (q / np.sqrt(q**2 + np.ones_like(q)))
        return sig

    def squash_inv(self, x):
        x = np.asarray(x)
        lower = self.lower
        upper = self.upper
        hh = (2 * x - (upper + lower)) / (upper - lower)
        logit = hh / np.sqrt(np.ones_like(x) - hh**2)
        return logit

    def squash_jacobian(self, q):
        q = np.asarray(q)
        lower = self.lower
        upper = self.upper
        d = 0.5 * (upper - lower) / ((q**2 + np.ones_like(q))**1.5)
        return np.diag(d)

    def squash_inv_jacobian(self, x):
        x = np.asarray(x)
        lower = self.lower
        upper = self.upper
        hh = (2 * x - (upper + lower)) / (upper - lower)
        diag = (2 / (upper - lower)) * (np.ones_like(x) - hh**2)**(-1.5)
        return np.diag(diag)

    def logdet_squash_jacobian(self, q):
        q = np.asarray(q)
        return np.log(np.linalg.det(self.squash_jacobian(q)))

    def grad_logdet_squash_jacobian(self, q):
        q = np.asarray(q)
        return -3.0 * q / (1.0 + q * q)

    #####################################################################################
    # Constraints related functions
    def constr(self, q):
        return self.S @ self.squash(q) - self.c

    def jacob_constr(self, q):
        j_sigma = self.squash_jacobian(q)
        return self.S @ j_sigma

    def jacob_constr_inner_product(
            self,
            jacob_constr_1,
            inner_product_matrix,
            jacob_constr_2=None
    ):
        if jacob_constr_2 is None:
            return jacob_constr_1 @ (inner_product_matrix @ jacob_constr_1.T)
        return jacob_constr_1 @ (inner_product_matrix @ jacob_constr_2.T)

    def gram(self, q):
        return self.jacob_constr_inner_product(self.jacob_constr(q), self.metric_inv, None)

    def inv_gram(self, q):
        identity = np.eye(self.c.shape[0])
        return np.linalg.solve(self.gram(q), identity)

    #####################################################################################
    # Dynamics

    # Potential
    def h1(self, q):
        q = np.asarray(q)

        if self.proj_matr is None:
            x = self.squash(q)
        else:
            x = self.proj_matr @ self.squash(q)

        return -1 * self.problem.model.log_density(x) - self.logdet_squash_jacobian(q)

    # Gradient des Potentials
    def dh1_dpos(self, q):
        q = np.asarray(q)

        if self.proj_matr is None:
            x = self.squash(q)
            #print(self.squash_jacobian(q).shape)
            #print(self.problem.model.log_gradient(x).shape)
            #print((self.squash_jacobian(q).T @ self.problem.model.log_gradient(x)).shape)
            #print(self.grad_logdet_squash_jacobian(q).shape)
            return -1*self.squash_jacobian(q).T @ self.problem.model.log_gradient(x) - self.grad_logdet_squash_jacobian(q)

        else:
            # möglicher Bug: Nicht 100% sicher ob + oder - grad_logdet
            x = self.proj_matr @ self.squash(q)
            return -1*self.squash_jacobian(q).T @ self.proj_matr.T @ self.problem.model.log_gradient(x) - self.grad_logdet_squash_jacobian(q)

        #return -1*self.squash_jacobian(q).T @ self.problem.model.log_gradient(x) + self.grad_logdet_squash_jacobian(q)


    # Function is only used once for definition of the pullback metric
    def pullback_metric(self):
        jac = self.squash_jacobian(self.squash_inv(self.map_guess))
        #print("jac: ", jac)
        #print(jac.shape)
        if self.param_matr is None:  # full rank case
            m = jac.T @ self.base_metric @ jac
            #print("pullback metric: ", m)
            return m
        m_reg = self.proj_matr.T @ self.base_metric @ self.proj_matr + self.beta * np.eye(self.param_matr.shape[0])
        #print(jac)
        #print(m_reg)
        m = jac.T @ m_reg @ jac
        #print("m_reg:", m_reg)
        #print("pullback metric: ", m)
        return m

    # Separate the Hamiltonian
    def h1_flow(self, q, mom, factor=1):
        return q, mom - factor*self.stepsize * self.dh1_dpos(q)

    def h2(self, mom):
        return 0.5 * mom.T @ self.dh2_dmom(mom)

    def dh2_dmom(self, mom):
        #print("metric_inv: ", self.metric_inv.shape)
        #print("mom: ", mom.shape)
        return self.metric_inv @ mom

    def h2_flow(self, q, mom, factor=1):
        return q + factor * self.stepsize * self.dh2_dmom(mom), mom

    def dh2_flow_dmom(self, factor=1):
        return factor * self.stepsize * self.metric_inv, np.eye(self.metric.shape[0])

    def project_onto_cotangent_space(self, q, mom):
        diff = self.jacob_constr(q).T @ (
            self.inv_gram(q) @ (self.jacob_constr(q) @ (self.metric_inv @ mom))
        )
        return mom - diff

    def sample_momentum(self, q):
        '''
        try:
            with np.errstate(over='raise', invalid='raise'):
                if self.param_matr is not None:
                    mom_red = self.metric_sqrt_red @ self.rng.standard_normal(self.metric_sqrt_red.shape[0])
                    #print("mom_red: ", mom_red.shape)
                    mom = self.squash_jacobian(q).T @ self.proj_matr.T @ mom_red
                    #print(q)
                    #print(mom)
                    mom = self.project_onto_cotangent_space(q, mom)
                    return mom
        except FloatingPointError:
            pass
        '''
        mom = self.metric_sqrt @ self.rng.standard_normal(self.metric_sqrt.shape[0])
        #print("shape of mom: ", mom)
        return self.project_onto_cotangent_space(q, mom)

    def h(self, q, mom):
        return self.h1(q) + self.h2(mom)

    #####################################################################################
    # Integrator related functions
    def newton_solver(self,
                      q,
                      mom,
                      q_prev,
                      mom_prev,
                      factor=1,
                      constraint_tol=1e-9,
                      position_tol=1e-8,
                      divergence_tol=1e10,
                      max_iters=50):
        #if self.integration_length>100:
        #    max_iters = 200
        mu = np.zeros_like(q)
        jacob_constr_prev = self.jacob_constr(q_prev)
        dh2_flow_q_dmom, dh2_flow_mom_dmom = self.dh2_flow_dmom(factor=abs(factor))
        try:
            for i in range(max_iters):
                #print("Solver: ", i)
                jacob_constr = self.jacob_constr(q)
                constr = self.constr(q)
                error = np.max(np.abs(constr))
                #delta_mu = jacob_constr_prev.T @ np.linalg.inv(self.jacob_constr_inner_product(jacob_constr, dh2_flow_q_dmom, jacob_constr_prev)) @ constr
                G = (self.jacob_constr_inner_product(jacob_constr, dh2_flow_q_dmom, jacob_constr_prev))
                #G_inv = np.linalg.solve(G, np.eye(G.shape[0]))
                #delta_mu = jacob_constr_prev.T @ G_inv @ constr
                delta_lambda = np.linalg.solve(G, constr)
                delta_mu = jacob_constr_prev.T @ delta_lambda
                delta_q = dh2_flow_q_dmom @ delta_mu

                q_norm = np.max(np.abs(q))
                delta_q_norm = np.max(np.abs(delta_q))
                '''
                print(
                    "it", i,
                    "error", f"{error:.3e}",
                    "max|q|", f"{np.max(np.abs(q)):.3e}",
                    "max|delta_q|", f"{np.max(np.abs(delta_q)):.3e}",
                    "rankG", np.linalg.matrix_rank(G),
                    "condG", f"{np.linalg.cond(G):.3e}",
                    "constr", constr,
                )
                '''
                #print("delta_q", delta_q)
                if (
                    error > divergence_tol
                    or np.isnan(error)
                    or not np.isfinite(q_norm)
                    or not np.isfinite(delta_q_norm)
                    or q_norm > 1e4
                    or delta_q_norm > 1e4
                ):
                    raise RuntimeError(
                        f"Newton solver diverged at iteration {i}. "
                        f"|constr|={error:.1e}, |q|={q_norm:.1e}, "
                        f"|delta_q|={delta_q_norm:.1e}."
                    )

                if error < constraint_tol and np.max(np.abs(delta_q)) < position_tol:
                    mom -= np.sign(factor) * dh2_flow_mom_dmom @ mu
                    return q, mom
                mu += delta_mu
                q -= delta_q
            #print("solver:", self.squash(q))
            #return q, mom
        except np.linalg.LinAlgError as exc:
                msg = (
                    f"Newton solver failed at iteration {i}: singular constraint Jacobian system. "
                    f"Last |constr|={error:.1e}."
                )
                raise RuntimeError(msg) from exc
        msg = (
            f"Newton solver did not converge in {max_iters} iterations. "
            f"Last |constr|={error:.1e}, |delta_q|={np.max(np.abs(delta_q)):.1e}."
        )
        raise RuntimeError(msg)


    def h2_flow_with_retraction_onto_mfd(self, q, mom, q_prev, mom_prev, factor=1):
        q_before = q.copy()
        mom_before = mom.copy()
        q, mom = self.h2_flow(q, mom, factor=factor)
        '''
        print(
            "before Newton:",
            "factor", factor,
            "max|q_prev|", f"{np.max(np.abs(q_prev)):.3e}",
            "max|mom|", f"{np.max(np.abs(mom_before)):.3e}",
            "max|M^-1 mom|", f"{np.max(np.abs(self.metric_inv @ mom_before)):.3e}",
            "max|q_free-q_prev|", f"{np.max(np.abs(q - q_prev)):.3e}",
            "|constr(q_free)|", f"{np.max(np.abs(self.constr(q))):.3e}",
            "|constr(q_prev)|", f"{np.max(np.abs(self.constr(q_prev))):.3e}",
        )
        '''
        q, mom = self.newton_solver(q, mom, q_prev, mom_prev, factor=factor)
        #print("retract: ", self.squash(q))
        return q, mom


    def step_a(self, q, mom, factor):
        q, mom = self.h1_flow(q, mom, factor)
        mom = self.project_onto_cotangent_space(q, mom)
        return q, mom

    def step_b(self, q, mom, factor=1.0, n_inner_step=1, reverse_check_tol=2e-8):
        #factor_inner = self.stepsize/n_inner_step
        factor_inner = factor/n_inner_step
        for i in range(n_inner_step):
            q_prev = q.copy()
            mom_prev = mom.copy()
            q, mom = self.h2_flow_with_retraction_onto_mfd(q, mom, q_prev, mom_prev, factor=factor_inner)
            # TODO: hier später evtl. caching vorteil einbauen
            mom = self.project_onto_cotangent_space(q, mom)

            q_back = q.copy()
            mom_back = mom.copy()
            q_back, mom_back = self.h2_flow_with_retraction_onto_mfd(q_back, mom_back, q, mom, factor=-1*factor_inner)

            rev_diff = np.max(np.abs(q_prev-q_back)) # Maximumsnorm der Differenz
            if rev_diff > reverse_check_tol:
                msg = (
                f"Non-reversible trajectory. Distance between initial and "
                f"forward-backward integrated positions = {rev_diff:.1e}."
                )
                raise RuntimeError(msg)
        return q, mom

    def rattle_step(self, q, mom):
        q, mom = self.step_a(q, mom, factor=0.5)
        q, mom = self.step_b(q, mom)
        q, mom = self.step_a(q, mom, factor=0.5)
        #print(q,mom)
        return q, mom

    def fixed_length_integrator(self, q, mom):
        #mom = self.momentum.copy()

        if self.nuts:
            pass
        else:
            for i in range(self.integration_length):
                #print(i)
                q, mom = self.rattle_step(q, mom)

        return q, mom




    #####################################################################################
    # Proposal logic
    def propose(self, rng):
        # if self.momentum is None:
        self.momentum = self.sample_momentum(self.state_transformed)
        #print("momentum: ", self.momentum)
        proposal_q, proposal_mom = self.fixed_length_integrator(self.state_transformed, self.momentum)
        theta = self.squash(proposal_q)
        #self.proposal_mom = self.param_matr.T @ self.squash_inv_jacobian(theta).T @ self.proposal_mom
        self.proposal_mom = proposal_mom
        #proposal = self.proj_matr@self.squash(proposal_q)
        proposal = self.projection(theta)
        #print("proposal: ", proposal)
        self.proposal = proposal
        return self.proposal

    def log_acceptance_probability(self) -> float:
        theta = self.affine_transformation(self.proposal)
        if (self.problem.A @ theta - self.problem.b >= 0).any():
            print("rejected due to violated constraints")
            return -np.inf
        #print("mom proposal: ", self.proposal_mom.shape)
        #print("")
        proposal_transformed = self.squash_inv(theta)
        h_state = self.h(self.state_transformed, self.momentum)
        h_proposal = self.h(proposal_transformed, self.proposal_mom)
        h_diff = h_proposal - h_state
        #print("Energy difference: ", h_diff)
        return -h_diff

    def accept_proposal(self):
        self.state = self.proposal
        self.state_transformed = self.squash_inv(self.affine_transformation(self.state))
        #self.momentum = self.proposal_mom
        return self.state


if __name__ == "__main__":
    # Example usage
    print("Hello World")
    i = 1
    problem = hopsy.Problem(...)  # Define your problem here
    starting_point = np.array([...])  # Define your starting point here
    fisher = np.array([...])  # Define your Fisher information matrix here
    map_guess = np.array([...])  # Define your MAP guess here
    param_matr = np.array([...])  # Define your parameter matrix here
    proj_matr = np.array([...])  # Define your projection matrix here
    beta = 1.0  # Define your beta value here
    equality_constraints = (np.array([...]), np.array([...]))  # Define your equality constraints here
    box_constraints = (np.array([...]), np.array([...]))  # Define your box constraints here

    proposal_generator = CustomProposal(
        problem=problem,
        starting_point=starting_point,
        fisher=fisher,
        map_guess=map_guess,
        param_matr=param_matr,
        proj_matr=proj_matr,
        beta=beta,
        equality_constraints=equality_constraints,
        box_constraints=box_constraints,
        stepsize=0.1,
        integration_length=10,
        nuts=False
    )

    rng = np.random.default_rng()
    proposal = proposal_generator.propose(rng)
    log_accept_prob = proposal_generator.log_acceptance_probability()
    print("Proposed state:", proposal)
    print("Log acceptance probability:", log_accept_prob)