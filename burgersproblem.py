import numpy as np
import os
import scipy.io

import utilities

try:
    from scipy.integrate import cumtrapz
except:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize._numdiff import approx_derivative
from scipy import sparse

class MakeConfig():
    def __init__(
            self,   
            ode_solver='LSODA',
            ocp_solver='indirect',
            atol=1e-06,
            rtol=1e-03,
            fp_tol=5e-03,
            indirect_tol=1e-04,
            direct_tol=1e-06,
            direct_tol_scale=0.1,
            indirect_max_nodes=1500,
            direct_n_init_nodes=16,
            direct_n_add_nodes=16,
            direct_max_nodes=64,
            direct_max_slsqp_iter=500,
            t1_sim=30,
            t1_scale = 6/5,
            t1_max=150.,
            n_trajectories_train_fixed=40,
            n_trajectories_initial_adaptive=5,
            n_trajectories_val=100,
            batch_ratio=0.025,
            n_trajectories_test=100,
            n_hidden=5,
            n_neurons=32,
            activation='tanh',
            callback_epoch=1,
            optimizer='AdamOptimizer',
            optimizer_opts={},
            sampling='adaptive', # 'adaptive' or 'fixed'
            architecture='GradientQRnet', # 'GradientQRnet' or 'GradientNN'
            C=0.001, # 0.002 GradientQRnet, 0.001 GradientNN
            M=1.2,
            r=10,
            Nc=1000,
            gradient_loss_weight=10,
            n_epochs=10             
        ):
        '''
        Class defining (default) configuration options for setting up how ODES
        and BVPs are integrated, how many trajectories are generated and over
        what time horizons, NN architecture parameters, and training options.

        Parameters
        ----------
        ode_solver : string, default='RK45'
            ODE solver for closed loop integration. See
            scipy.integrate.solve_ivp for options.
        ocp_solver : {'indirect', 'direct'}, default='indirect'
            Whether to use an indirect method (Pontryagin's principle + boundary
            value problem solver) or direct method (Pseudospectral collocation)
            to solve the open loop OCP.
        atol : float, default=1e-06
            Absolute accuracy tolerance for the ODE solver
        rtol : float, default=1e-03
            Relative accuracy tolerance for the ODE solver
        fp_tol : float, default=1e-05
            Maximum value of the vector field allowed for a trajectory to be
            considered as convergence to an equilibrium
        indirect_tol : float, default=1e-05
            Accuracy tolerance for the indirect BVP solver.
        direct_tol : float, default=1e-06
            Accuracy tolerance for the direct OCP solver.
        direct_tol_scale : float, default=0.1
            Number to multiply the accuracy tolerance for the direct OCP solver
            at each solution iteration.
        indirect_max_nodes : int, default=5000
            Maximum number of collocation points used by the indirect BVP solver.
        direct_n_init_nodes : int, default=16
            Initial number of collocation points used by the direct OCP solver.
        direct_n_add_nodes : int, default=16
            How many additional nodes to add when refining the grid used by the
            direct OCP solver.
        direct_max_nodes : int, default=64
            Maximum number of collocation points used by the direct OCP solver.
        direct_max_slsqp_iter : int, default=500
            Maximum number of iterations for the SLSQP optimization routine used
            by the direct OCP solver.
        t1_sim : float, default=60.
            Default time to integrate the ODE over
        t1_max : float, default=300.
            Maximum time horizon to integrate for.
        t1_scale : float, default=3/2
            Amount to multiply the time horizon by if need to integrate the ODE
            or BVP for longer to achieve convergence.
        n_trajectories_train : int, default=100
            Number of trajectories used for the training data set
        n_trajectories_test : int, default=100
            Number of trajectories used for the test data set
        n_trajectories_MC : int, default=100
            Number of trajectories integrated for Monte Carlo tests
        n_hidden : int, default=5
            Number of hidden layers to use
        n_neurons : int, default=32
            Number of neurons per layer
        activation : str, default='tanh'
            Activation function to use. Current only 'tanh' is implemented
        value_loss_weight : float, default=1.
            How much to weight the value function MSE term in the loss function
        gradient_loss_weight : float, default=1.
            How much to weight the value gradient MSE term in the loss function
        batch_ratio : int, optional
            Maximum number of data points (not trajectories) to use for
            training. If set to None (default), use the entire data set. Useful
            for controlling the data set size to speed up optimization.
        n_epochs : int, default=1
            How many times to iterate through the dataset (for SGD optimizers).
        callback_epoch : int, default=1
            Specifies after how many epochs to print loss functions (for SGD
            optimizers).
        optimizer : str, default='Adam'
            Which optimizer to use. Options are 'Adam' and any optimizer
            implemented in tensorflow.train.
        optimizer_opts : dict, optional
            Options to pass to the NN optimizer.
        '''
        

        self.ode_solver=ode_solver
        self.ocp_solver=ocp_solver
        self.atol=atol
        self.rtol=rtol
        self.fp_tol=fp_tol
        self.indirect_tol=indirect_tol
        self.direct_tol=direct_tol
        self.direct_tol_scale=direct_tol_scale
        self.indirect_max_nodes=indirect_max_nodes
        self.direct_n_init_nodes=direct_n_init_nodes
        self.direct_n_add_nodes=direct_n_add_nodes
        self.direct_max_nodes=direct_max_nodes
        self.direct_max_slsqp_iter=direct_max_slsqp_iter
        self.t1_sim=t1_sim
        self.t1_scale = t1_scale
        self.t1_max=t1_max
        self.n_trajectories_train_fixed=n_trajectories_train_fixed
        self.n_trajectories_initial_adaptive=n_trajectories_initial_adaptive
        self.n_trajectories_val=n_trajectories_val
        self.batch_ratio=batch_ratio
        self.n_trajectories_test=n_trajectories_test
        self.n_hidden=n_hidden
        self.n_neurons=n_neurons
        self.activation=activation
        self.callback_epoch=callback_epoch
        self.optimizer=optimizer
        self.optimizer_opts=optimizer_opts
        self.sampling=sampling
        self.architecture=architecture
        self.C=C
        self.M=M
        self.r=r
        self.Nc=Nc
        self.gradient_loss_weight=gradient_loss_weight
        self.n_epochs=n_epochs




def cheb(N): #tf2ready
    '''
    Build Chebyshev differentiation matrix.
    Uses algorithm on page 54 of Spectral Methods in MATLAB by Trefethen.
    '''
    theta = np.pi / N * np.arange(0, N+1)
    X_nodes = np.cos(theta)

    X = np.tile(X_nodes, (N+1, 1))
    X = X.T - X

    C = np.concatenate(([2.], np.ones(N-1), [2.]))
    C[1::2] = -C[1::2]
    C = np.outer(C, 1./C)

    D = C / (X + np.identity(N+1))
    D = D - np.diag(D.sum(axis=1))

    # Clenshaw-Curtis weights
    # Uses algorithm on page 128 of Spectral Methods in MATLAB
    w = np.empty_like(X_nodes)
    v = np.ones(N-1)
    for k in range(2, N, 2):
        v -= 2.*np.cos(k * theta[1:-1]) / (k**2 - 1)

    if N % 2 == 0:
        w[0] = 1./(N**2 - 1)
        v -= np.cos(N*theta[1:-1]) / (N**2 - 1)
    else:
        w[0] = 1./N**2

    w[-1] = w[0]
    w[1:-1] = 2.*v/N

    return X_nodes, D, w

class MakeOCP():
    '''Defines an optimal control problem (OCP).

    Template super class defining an optimal control problem (OCP) including
    dynamics, running cost, and optimal control as a function of costate.

    Parameters
    ----------
    X_bar : (n_states, 1) array
        Goal state, nominal linearization point.
    U_bar : (n_controls, 1) array
        Control values at nominal linearization point.
    A : (n_states, n_states) array or None
        State Jacobian matrix at nominal equilibrium. If None, approximates
        this with central differences.
    B : (n_states, n_controls) array or None
        Control Jacobian matrix at nominal equilibrium. If None, approximates
        this with central differences.
    Q : (n_states, n_states) array
        Hessian of running cost with respect to states. Must be positive
        semi-definite.
    R : (n_controls, n_controls) array
        Hessian of running cost with respect to controls. Must be positive
        definite.
    P : (n_states, n_states) array, optional
        Pre-computed Riccati matrix, if available.
    U_lb : (n_controls, 1) array, optional
        Lower control saturation bounds.
    U_ub : (n_controls, 1) array, optional
        Upper control saturation bounds.
    X0_lb : (n_states, 1) array, optional
        Lower bounds for (uniform) initial condition samples.
    X0_ub : (n_states, 1) array, optional
        Upper bounds for (uniform) initial condition samples.
    '''
    '''Defines an optimal control problem (OCP).

    Template super class defining an optimal control problem (OCP) including
    dynamics, running cost, and optimal control as a function of costate.

    Parameters
    ----------
    X_bar : (n_states, 1) array
        Goal state, nominal linearization point.
    U_bar : (n_controls, 1) array
        Control values at nominal linearization point.
    A : (n_states, n_states) array or None
        State Jacobian matrix at nominal equilibrium. If None, approximates
        this with central differences.
    B : (n_states, n_controls) array or None
        Control Jacobian matrix at nominal equilibrium. If None, approximates
        this with central differences.
    Q : (n_states, n_states) array
        Hessian of running cost with respect to states. Must be positive
        semi-definite.
    R : (n_controls, n_controls) array
        Hessian of running cost with respect to controls. Must be positive
        definite.
    P : (n_states, n_states) array, optional
        Pre-computed Riccati matrix, if available.
    U_lb : (n_controls, 1) array, optional
        Lower control saturation bounds.
    U_ub : (n_controls, 1) array, optional
        Upper control saturation bounds.
    X0_lb : (n_states, 1) array, optional
        Lower bounds for (uniform) initial condition samples.
    X0_ub : (n_states, 1) array, optional
        Upper bounds for (uniform) initial condition samples.
    '''
    def __init__(
            self,   
            n_states=64,
            n_controls=2,
            nu=0.02,
            beta=0.1,
            A=None, 
            B=None, 
            Q=None, 
            R=0.5, 
            P=None, 
            X0_lb=None, 
            X0_ub=None
        ):
        
        
        self.n_states = n_states
        self.n_controls =n_controls
        

        self.nu = nu
        self.beta = beta
        self.R = R
        kappa = 25.

        
        # Default number of sin functions for initial conditions
        self.n_X0_terms = 10
        
        U_lb, U_ub = None, None
        
        
        # Chebyshev nodes, differentiation matrices, and Clenshaw-Curtis weights
        self.xi, self.D, self.w_flat = cheb(self.n_states+1)
        self.D2 = np.matmul(self.D, self.D)
        
        # Truncates system to account for zero boundary conditions
        self.xi = self.xi[1:-1].reshape(-1,1)
        self.w_flat = self.w_flat[1:-1]
        self.w = self.w_flat.reshape(-1,1)
        self.D = self.D[1:-1, 1:-1]
        self.D2 = self.D2[1:-1, 1:-1]
        
        # Control multiplier
        self.B = np.hstack((
            (-4/5 <= self.xi) & (self.xi <= -2/5),
            (2/5 <= self.xi) & (self.xi <= 4/5)
        ))
        self.B = -kappa * self.B * np.hstack((
            (self.xi + 4/5)*(self.xi + 2/5),
            (self.xi - 2/5)*(self.xi - 4/5)
        ))
        self.B = np.abs(self.B)

        # Forcing term coefficient
        self.alpha = np.abs(self.xi) <= 1/5
        self.alpha = - kappa * self.alpha * (self.xi + 1/5)*(self.xi - 1/5)
        self.alpha = np.abs(self.alpha)
        self.alpha_flat = self.alpha.flatten()
        

        
        self.RBT = - self.B.T / (2.*self.R)
        

        
        ##### Makes LQR controller #####
        
        # Linearization point
        X_bar = np.zeros((self.n_states,1))
        U_bar = np.zeros((self.n_controls,1))
        
        # Dynamics linearized around origin (dxdt ~= Ax + Bu)
        A = self.nu*self.D2 + np.diag(self.alpha_flat)
        
        # Cost matrices
        Q = np.diag(self.w_flat)
        R = np.diag([self.R]*self.n_controls)
        

        self.X_bar = np.reshape(X_bar, (-1,1))
        self.U_bar = np.reshape(U_bar, (-1,1))

        # self.n_states = self.X_bar.shape[0]
        # self.n_controls = self.U_bar.shape[0]

        # Approximate state matrices numerically if not given
        if A is None or B is None:
            _A, _B = self.jacobians(X_bar, U_bar, F0=np.zeros_like(self.X_bar))

        if A is None:
            A = _A
            A[np.abs(A) < 1e-10] = 0.

        if B is None:
            B = _B
            B[np.abs(B) < 1e-10] = 0.

        self._A = np.reshape(A, (self.n_states, self.n_states))
        self._B = np.reshape(B, (self.n_states, self.n_controls))

        self._Q = np.reshape(Q, (self.n_states, self.n_states))
        self._R = np.reshape(R, (self.n_controls, self.n_controls))

        from controller import LQR
        self.LQR = LQR(
            X_bar, U_bar, self._A, self._B, self._Q, self._R,
            P=P, U_lb=U_lb, U_ub=U_ub
        )

        self.U_lb, self.U_ub = self.LQR.U_lb, self.LQR.U_ub

        self.X0_lb, self.X0_ub = X0_lb, X0_ub

        if self.X0_lb is not None:
            self.X0_lb = np.reshape(self.X0_lb, (-1,1))
        if self.X0_ub is not None:
            self.X0_ub = np.reshape(self.X0_ub, (-1,1))




    def get_params(self, **params):
        '''
        Function to return a dict of parameters which might be needed by matlab
        scripts.

        Parameters
        ----------
        params : keyword arguments
            Additional parameters to return, usually called by subclass.

        Returns
        -------
        params_dict : dict
            Dict of name-value pairs including
            'n_states' : int
            'n_controls' : int
            'X_bar' : (n_states, 1) array
            'U_bar' : (n_controls, 1) array
            'U_lb' : (n_controls, 1) array or None
            'U_ub' : (n_controls, 1) array or None
            'P' : (n_states, n_states) array
            'K' : (n_controls, n_states) array
            **params
        '''
        params_dict = {
            'n_states': self.n_states,
            'n_controls': self.n_controls,
            'X_bar': self.X_bar,
            'U_bar': self.U_bar,
            'U_lb': self.U_lb if self.U_lb is not None else np.nan,
            'U_ub': self.U_ub if self.U_ub is not None else np.nan,
            'A': self._A,
            'B': self._B,
            'Q': self._Q,
            'R': self._R,
            'P': self.LQR.P,
            'K': self.LQR.K,
            'xi': self.xi,
            'w': self.w,
            **params
        }
        
        
        return params_dict


    def norm(self, X, center_X_bar=True): #tf2ready
        '''
        Calculate the distance of a batch of spatial points from X_bar or zero.
        Uses the Clenshaw Curtis quadrature weights to compute a weighted norm.

        Arguments
        ----------
        X : (n_states, n_data) array
            Points to compute distances for
        center_X_bar : not used
            For API consistency only

        Returns
        ----------
        X_norm : (n_data,) array
            Norm for each point in X
        '''
        X = X.reshape(self.n_states, -1)
        return np.sqrt(np.sum(X**2 * self.w, axis=0))

    def sample_X0(self, Ns, dist=None, K=None): #tf2ready
        '''Sampling from sum of sine functions.'''

        if K is None:
            K = self.n_X0_terms

        xi = np.pi * self.xi
        X0 = np.zeros((self.n_states, Ns))

        for k in range(1,K+1):
            ak = (2.*np.random.rand(1,Ns) - 1.)/k
            X0 += ak * np.sin(k * xi)

        if dist is not None:
            X0_norm = self.norm(X0).reshape(1, -1)
            X0 *= dist / X0_norm

        if Ns == 1:
            X0 = X0.flatten()
        return X0

    def make_bc(self, X0):
        '''
        Generates a function to evaluate the boundary conditions for a given
        initial condition. Terminal cost is zero so final condition on lambda is
        zero.

        Parameters
        ----------
        X0 : (n_states, 1) array
            Initial condition.

        Returns
        -------
        bc : callable
            Function of X_aug_0 (augmented states at initial time) and X_aug_T
            (augmented states at final time), returning a function which
            evaluates to zero if the boundary conditions are satisfied.
        '''
        X0 = X0.flatten()
        def bc(X_aug_0, X_aug_T):
            return np.concatenate((
                X_aug_0[:self.n_states] - X0, X_aug_T[self.n_states:]
            ))
        return bc

    def running_cost(self, X, U, wX=None): #tf2ready
        '''
        Evaluate the running cost L(X,U) at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        wX : (n_states,) or (n_states, n_points) array, optional
            States(s) multiplied by the Chebyshev quadrature weights.

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(X,U) evaluated at pair(s) (X,U).
        '''
        if wX is None:
            if X.ndim == 1:
                wX = self.w_flat * X
            else:
                wX = self.w * X

        return np.sum(wX * X, axis=0) + self.R * np.sum(U**2, axis=0)

    def running_cost_gradient(self, X, U, return_dLdX=True, return_dLdU=True): #tf2ready
        '''
        Evaluate the gradients of the running cost, dL/dX (X,U) and dL/dU (X,U),
        at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        return_dLdX : bool, default=True
            Set to True to compute the gradient with respect to states, dL/dX.
        return_dLdU : bool, default=True
            Set to True to compute the gradient with respect to controls, dL/dU.

        Returns
        -------
        dLdX : (n_states,) or (n_states, n_points) array
            Gradient dL/dX (X,U) evaluated at pair(s) (X,U).
        dLdU : (n_states,) or (n_states, n_points) array
            Gradient dL/dU (X,U) evaluated at pair(s) (X,U).
        '''
        if return_dLdX:
            if X.ndim == 1:
                dLdX = 2. * self.w_flat * X
            else:
                dLdX = 2. * self.w * X
            if not return_dLdU:
                return dLdX

        if return_dLdU:
            dLdU = 2. * self.R * U
            if not return_dLdX:
                return dLdU

        return dLdX, dLdU

    def Hamiltonian(self, X, U, dVdX):
        '''
        Evaluate the Pontryagin Hamiltonian,
        H(X,U,dVdX) = L(X,U) + <dVdX, F(X,U)>
        where L(X,U) is the running cost, dVdX is the costate or value gradient,
        and F(X,U) is the dynamics. A necessary condition for optimality is that
        H(X,U,dVdX) ~ 0 for the whole trajectory.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Value gradient dV/dX (X,U) evaluated at pair(s) (X,U).

        Returns
        -------
        H : (1,) or (n_points,) array
            Pontryagin Hamiltonian each each point in time.
        '''
        L = self.running_cost(X, U)
        F = self.dynamics(X, U)
        return L + np.sum(dVdX * F, axis=0)

    def compute_cost(self, t, X, U):
        '''Computes the accumulated cost J(t) of a state-control trajectory.'''
        L = self.running_cost(X, U)
        J = cumtrapz(L.flatten(), t)
        return np.concatenate((J, J[-1:]))


    def dynamics(self, X, U): #tf2ready
        '''
        Evaluate the closed-loop dynamics at single or multiple time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current state.
        U : (n_controls,) or (n_controls, n_points)  array
            Feedback control U=U(X).

        Returns
        -------
        dXdt : (n_states,) or (n_states, n_points) array
            Dynamics dXdt = F(X,U).
        '''
        flat_out = X.ndim < 2
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)

        dXdt = (
            - 0.5*np.matmul(self.D, X**2)
            + np.matmul(self.nu*self.D2, X)
            + X * self.alpha * np.exp(-self.beta * X)
            + np.matmul(self.B, U)
        )

        if flat_out:
            dXdt = dXdt.flatten()

        return dXdt

    def jacobians(self, X, U, F0=None): #tf2ready
        '''
        Evaluate the Jacobians of the dynamics with respect to states and
        controls at single or multiple time instances. Default implementation
        approximates the Jacobians with central differences.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current states.
        U : (n_controls,) or (n_controls, n_points)  array
            Control inputs.
        F0 : ignored
            For API consistency only.

        Returns
        -------
        dFdX : (n_states, n_states) or (n_states, n_states, n_points) array
            Jacobian with respect to states, dF/dX.
        dFdU : (n_states, n_controls) or (n_states, n_controls, n_points) array
            Jacobian with respect to controls, dF/dX.
        '''
        X = X.reshape(self.n_states, -1)

        beta_X = -self.beta * X
        beta_X = (1. + beta_X) * self.alpha * np.exp(beta_X)

        dFdX = (
            - X * np.expand_dims(self.D, -1)
            + np.expand_dims(self.nu * self.D2, -1)
        )

        diag_idx = np.diag_indices(self.n_states)
        for k in range(X.shape[1]):
            dFdX[diag_idx[0],diag_idx[1],k] += beta_X[:,k]

        dFdU = np.expand_dims(self.B, -1)
        dFdU = np.tile(dFdU, (1,1,X.shape[-1]))

        return dFdX, dFdU


    def closed_loop_jacobian(self, X, controller):
        '''
        Evaluate the Jacobian of the closed-loop dynamics at single or multiple
        time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current states.
        controller : object
            Controller instance implementing eval_U and eval_dUdX methods.

        Returns
        -------
        dFdX : (n_states, n_states) or (n_states, n_states, n_points) array
            Closed-loop Jacobian dF/dX + dF/dU * dU/dX.
        '''
        dFdX, dFdU = self.jacobians(X, controller.eval_U(X))
        dUdX = controller.eval_dUdX(X)

        while dFdU.ndim < 3:
            dFdU = dFdU[...,None]
        while dUdX.ndim < 3:
            dUdX = dUdX[...,None]

        dFdX += np.einsum('ijk,jhk->ihk', dFdU, dUdX)

        if X.ndim < 2:
            dFdX = np.squeeze(dFdX)

        return dFdX

    def U_star(self, X, dVdX): #tf2ready
        '''
        Evaluate the optimal control as a function of state and costate.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        '''
        U = np.matmul(self.RBT, dVdX)
        return utilities.saturate_np(U, self.U_lb, self.U_ub)

    def jac_U_star(self, X, dVdX, U0=None): #tf2ready
        '''
        Evaluate the Jacobian of the optimal control with respect to the state,
        leaving the costate fixed.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        U0 : ignored
            For API consistency only.

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        '''
        dVdX = dVdX.reshape(self.n_states, -1)
        return np.zeros((self.n_controls, self.n_states, dVdX.shape[-1]))

    def make_U_NN(self, X, dVdX):
        '''Makes TensorFlow graph of optimal control with NN value gradient.'''
        from tensorflow.compat.v1 import matmul

        U = matmul(self.RBT.astype(np.float32), dVdX)

        return utilities.saturate_tf(U, self.U_lb, self.U_ub)
    
    def bvp_dynamics(self, t, X_aug): #tf2ready
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.
        Default implementation uses finite differences for the costate dynamics.

        Parameters
        ----------
        X_aug : (2*n_states+1, n_points) array
            Current state, costate, and running cost.

        Returns
        -------
        dX_aug_dt : (2*n_states+1, n_points) array
            Concatenation of dynamics dXdt = F(X,U^*), costate dynamics,
            dAdt = -dH/dX(X,U^*,dVdX), and change in cost dVdt = -L(X,U*),
            where U^* is the optimal control.
        '''
        X = X_aug[:self.n_states].reshape(self.n_states, -1)
        A = X_aug[self.n_states:2*self.n_states].reshape(self.n_states, -1)

        # Control as a function of the costate
        U = self.U_star(X, A)

        wX = self.w * X
        aeX = self.alpha * np.exp(-self.beta * X)

        dXdt = (
            - 0.5*np.matmul(self.D, X**2)
            + np.matmul(self.nu*self.D2, X)
            + X * aeX
            + np.matmul(self.B, U)
        )

        dAdt = (
            - 2.*wX
            + X * np.matmul(self.D.T, A)
            - np.matmul(self.nu * self.D2.T, A)
            - aeX * (1. - self.beta*X) * A
        )

        L = np.atleast_2d(self.running_cost(X, U, wX))

        return np.vstack((dXdt, dAdt, -L))

    def apply_state_constraints(self, X):
        '''
        Manually update states to satisfy some state constraints. At present
        time, the OCP format only supports constraints which are intrinsic to
        the dynamics (such as quaternions or periodicity), not dynamic
        constraints which need to be satisfied by admissible controls.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states with constrained values.
        '''
        return X

    def constraint_fun(self, X):
        '''
        A (vector-valued) function which is zero when the state constraints are
        satisfied. At present time, the OCP format only supports constraints
        which are intrinsic to the dynamics (such as quaternions or
        periodicity), not dynamic constraints which need to be satisfied by
        admissible controls.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        ----------
        C : (n_constraints,) or (n_constraints, n_data) array or None
            Algebraic equation such that C(X)=0 means that X satisfies the state
            constraints.
        '''
        return

    def constraint_jacobian(self, X):
        '''
        Constraint function Jacobian dC/dX of self.constraint_fun. Default
        implementation approximates this with central differences.

        Parameters
        ----------
        X : (n_states,) array
            Current state.

        Returns
        -------
        dCdX : (n_constraints, n_states) array or None
            dC/dX evaluated at the point X, where C(X)=self.constraint_fun(X).
        '''
        C0 = self.constraint_fun(X)
        if C0 is None:
            return

        return approx_derivative(self.constraint_fun, X, f0=C0)

    def make_integration_events(self):
        '''
        Construct a (list of) callables that are tracked during integration for
        times at which they cross zero. Such events can terminate integration
        early.

        Returns
        -------
        events : None, callable, or list of callables
            Each callable has a function signature e = event(t, X). If the ODE
            integrator finds a sign change in e then it searches for the time t
            at which this occurs. If event.terminal = True then integration
            stops.
        '''
        return
    

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
