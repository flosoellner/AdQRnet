import numpy as np
import scipy.io
from scipy.interpolate import interp1d
import os
import time
import utilities
from simulate.integrate import solve_ivp
from simulate.openloop import solve_ocp


def sim_closed_loop(
        dynamics, jacobian, controller, tspan, X0, t_eval=None, events=None,
        solver=None, atol=None, rtol=None
    ):
    '''
    Simulate the closed-loop system for a fixed time interval.

    Parameters
    ----------
        OCP: instance of TemplateOCP defining dynamics, Jacobian, etc.
        tspan: integration time, list of two floats
        X0: initial condition, (n,) numpy array
        controller: instance of a trained QRnet
        solver: ODE solver to use, str
        atol: absolute integration tolerance, float
        rtol: relative integration tolerance, float
        t_eval: optional (Nt,) numpy array of time instances to evaluate solution at

    Returns
    -------
        t: time vector, (Nt,) numpy array
        X: state time series, (n,Nt) numpy array
    '''
    def dynamics_wrapper(t, X):
        U = controller.eval_U(X)
        return dynamics(X, U)

    def jac_wrapper(t, X):
        return jacobian(X, controller)

    ode_sol = solve_ivp(
        dynamics_wrapper, tspan, X0, t_eval=t_eval, jac=jac_wrapper,
        events=events, vectorized=True, method=solver, rtol=rtol, atol=atol
    )

    return ode_sol.t, ode_sol.y, ode_sol.status

def sim_to_converge(
        dynamics, jacobian, controller, X0, config, events=None
    ):
    '''
    Simulate the closed-loop system until reach t_max or the dX/dt = 0.

    Parameters
    ----------
        OCP: instance of a setupProblem class defining dynamics, Jacobian, etc.
        config: a configuration dict defined in problem_def.py
        X0: initial condition, (n,) numpy array
        controller: instance of a trained QRnet

    Returns
    -------
        t: time vector, (Nt,) numpy array
        X: state time series, (n,Nt) numpy array
        converged: whether or not equilibrium was reached, bool
    '''

    t = np.zeros(1)
    
    
    X = X0.reshape(-1,1)

    converged = False

    # Solves over an extended time interval if needed to make ||f(x,u)|| -> 0
    while not converged and t[-1] < config.t1_max:
        t1 = np.maximum(config.t1_sim, t[-1] * config.t1_scale)
        # Simulate the closed-loop system
        t_new, X_new, status = sim_closed_loop(
            dynamics,
            jacobian,
            controller,
            [t[-1], t1],
            X[:,-1],
            events=events,
            solver=config.ode_solver,
            atol=config.atol,
            rtol=config.rtol
        )

        t = np.concatenate((t, t_new[1:]))
        X = np.hstack((X, X_new[:,1:]))

        if status == 1:
            break

        U = controller.eval_U(X[:,-1])
        converged = np.linalg.norm(dynamics(X[:,-1], U)) < config.fp_tol

    return t, X, converged

def sim_controller(OCP, config, X0, timestamp, _model_registry, model_dir, results_dir):
    controller, _ = utilities.load_NN(_model_registry, model_dir, timestamp) 
    # Integrates the closed-loop system (LQR controller and NN controllers)
    print('Integrating closed loop system (LQR)...')
    start_time = time.time()
    
    t_LQR, X_LQR, LQR_converged = sim_to_converge(
        OCP.dynamics, OCP.closed_loop_jacobian, OCP.LQR, X0, config,
        events=OCP.make_integration_events()
    )
    
    print('Integration time (LQR) %.4f s' % (time.time() - start_time))
    
    print('Integrating closed loop system (%s)...' % controller.architecture())
    
    start_time = time.time()
    
    t_NN, X_NN, NN_converged = sim_to_converge(
        OCP.dynamics, OCP.closed_loop_jacobian, controller, X0, config,
        events=OCP.make_integration_events()
    )
    
    print(
        'Integration time (%s) %.4f s'
        % (controller.architecture(), time.time() - start_time)
    )
    
    # Compute costates, controls, and cost
    V_LQR, dVdX_LQR, U_LQR = OCP.LQR.bvp_guess(X_LQR)
    J_LQR = OCP.compute_cost(t_LQR, X_LQR, U_LQR)
    
    V_NN, dVdX_NN, U_NN = controller.bvp_guess(X_NN)
    J_NN = OCP.compute_cost(t_NN, X_NN, U_NN)
    
    save_dict = {
        'architecture': controller.architecture(),
        'timestamp': timestamp,
        'LQR_converged': LQR_converged,
        'NN_converged': NN_converged,
        't_LQR': t_LQR, 'X_LQR': X_LQR, 'U_LQR': U_LQR, 'J_LQR': J_LQR,
        'V_LQR': V_LQR, 'dVdX_LQR': dVdX_LQR,
        't_NN': t_NN, 'X_NN': X_NN, 'U_NN': U_NN, 'J_NN': J_NN,
        'V_NN': V_NN, 'dVdX_NN': dVdX_NN
    }
    
    # ---------------------------------------------------------------------------- #
    
    # Solves the two-point BVP with LQR and NN initial guesses
    def _linear_guess(t, Y0, Y1):
        Y = np.hstack((Y0.reshape(-1,1), Y1.reshape(-1,1)))
        Y = interp1d([0., config.t1_sim], Y)
        return Y(t)
    
    def _solve_ocp_from_guess(t, X, U, dVdX, V):
        if OCP.running_cost(X[:,-1], U[:,-1]) > config.fp_tol:
            print('Initial guess failed to converge. Using linear interpolation.')
            t = np.linspace(0., config.t1_sim)
            X = _linear_guess(t, X0, OCP.X_bar)
            U = _linear_guess(t, U[:,:1], OCP.U_bar)
            dVdX = np.zeros_like(X)
            V = _linear_guess(t, V[:,:1], np.zeros((1,1)))
    
        return solve_ocp(
            OCP, config,
            t_guess=t, X_guess=X, U_guess=U, dVdX_guess=dVdX, V_guess=V,
            solve_to_converge=True, verbose=2-(config.ocp_solver=='direct')
        )
    
    
    start_time = time.time()
    
    _, ocp_sol_LQR, LQR_ocp_converged = _solve_ocp_from_guess(
        t_LQR, X_LQR, U_LQR, dVdX_LQR, V_LQR
    )
    ocp_sol_LQR = ocp_sol_LQR(t_LQR)
    ocp_sol_LQR['t'] = t_LQR
    
    print('OCP solution time: %.2f s' % (time.time() - start_time))
    start_time = time.time()
    
    _, ocp_sol_NN, NN_ocp_converged = _solve_ocp_from_guess(
        t_NN, X_NN, U_NN, dVdX_NN, V_NN
    )
    ocp_sol_NN = ocp_sol_NN(t_NN)
    ocp_sol_NN['t'] = t_NN
    
    print('OCP solution time: %.2f s' % (time.time() - start_time))
    
    J_opt_LQR = ocp_sol_LQR['V'].flatten()[::-1]
    J_opt_NN = ocp_sol_NN['V'].flatten()[::-1]
    
    # Uses the better BVP solution in case of multiple local minima
    if LQR_ocp_converged and NN_ocp_converged:
        if J_opt_LQR[-1] < J_opt_NN[-1]:
            J_opt, ocp_sol = J_opt_LQR, ocp_sol_LQR
        else:
            J_opt, ocp_sol = J_opt_NN, ocp_sol_NN
    elif LQR_ocp_converged:
        J_opt, ocp_sol = J_opt_LQR, ocp_sol_LQR
    elif NN_ocp_converged:
        J_opt, ocp_sol = J_opt_NN, ocp_sol_NN
    elif J_opt_LQR[-1] < J_opt_NN[-1]:
        J_opt, ocp_sol = J_opt_LQR, ocp_sol_LQR
    else:
        J_opt, ocp_sol = J_opt_NN, ocp_sol_NN
    
    
    for key, val in ocp_sol.items():
        save_dict[key + '_opt'] = val
    save_dict['H_opt'] = OCP.Hamiltonian(
        ocp_sol['X'], ocp_sol['U'], ocp_sol['dVdX']
    )
    save_dict['J_opt'] = J_opt
    
    # -----------------------------------------------------------------------------#
    
    if LQR_converged:
        print('LQR cost: %.2f' % J_LQR[-1])
    else:
        print('LQR cost: infinite (%.2f)' % J_LQR[-1])
    if NN_converged:
        print('NN cost: %.2f' % J_NN[-1])
    else:
        print('NN cost: infinite (%.2f)' % J_NN[-1])
    
    if J_opt[-1] < np.infty:
        print('Optimal cost: %.2f \n' % J_opt[-1])
        if LQR_converged:
            print('LQR sub-optimality: {J:.2f} %'.format(
                J=np.maximum(0., 100.*(J_LQR[-1]/J_opt[-1] - 1.))
            ))
        else:
            print('LQR sub-optimality: infinite ({J:.2f}) %'.format(
                J=np.maximum(0., 100.*(J_LQR[-1]/J_opt[-1] - 1.))
            ))
        if NN_converged:
            print('NN sub-optimality: {J:.2f} %'.format(
                J=np.maximum(0., 100.*(J_NN[-1]/J_opt[-1] - 1.))
            ))
        else:
            print('NN sub-optimality: infinite ({J:.2f}) %'.format(
                J=np.maximum(0., 100.*(J_NN[-1]/J_opt[-1] - 1.))
            ))
            
    return save_dict