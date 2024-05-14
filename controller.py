import os
import time
import dill
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.linalg import solve_continuous_are as care
from scipy.optimize._numdiff import approx_derivative
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import utilities


from tqdm import tqdm


class NoControl:
    def __init__(self, U_bar):
        self.U_bar = np.reshape(U_bar, (-1,1))

    def eval_U(self, X):
        if X.ndim == 1:
            return np.squeeze(self.U_bar)
        return np.tile(self.U_bar, (1,X.shape[1]))

    def eval_dUdX(self, X):
        zeros = np.zeros((self.U_bar.shape[0], X.shape[0]))
        if X.ndim < 2:
            return zeros

        dUdX = np.expand_dims(zeros, -1)
        dUdX = np.tile(dUdX, (1,1,X.shape[1]))
        return dUdX

class BaseController:
    '''
    Base class for implementing a state feedback controller.
    '''
    def __init__(self, *args, **kwargs):
        pass

    def architecture(self):
        return type(self).__name__

    def _get_learned_params(self):
        '''
        Get numpy values for the trainable tensorflow variables such as NN
        weights. By default returns the NN weights but should be overwritten to
        return other variables as needed.

        Returns
        -------
        parameters : dict
            Dictionary of (lists of) arrays giving trained parameter values
        '''
        return {}

    def eval_V(self, X):
        '''
        Predicts the value function, V(X), for each sample state in X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value function for.

        Returns
        -------
        dVdX : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        '''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        XPX = X_err * np.matmul(self.P, X_err)
        XPX = np.sum(XPX, axis=0, keepdims=True)

        if X.ndim < 2:
            XPX = XPX.flatten()

        return XPX

    def eval_dVdX(self, X):
        '''
        Predicts the value function gradient, dV/dX(X), for each sample state in
        X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value gradient for.

        Returns
        -------
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        '''
        raise NotImplementedError

    def eval_U(self, X):
        '''
        Evaluates the NN feedback control, U(X), for each sample state in X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        -------
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        raise NotImplementedError

    def eval_dUdX(self, X):
        '''
        Evaluates the Jacobian of the NN feedback control, [dU/dX](X), for each
        sample state in X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        -------
        dUdX : (n_controls, n_states, n_data) or (n_controls, n_states) array
            Jacobian of NN feedback control for each column in X.
        '''
        raise NotImplementedError

    def bvp_guess(self, X, eval_U=False):
        '''
        Predicts the value function V(X), its gradient dVdX(X), and the optimal
        control U(X) for each sample state in X. If the network does not make
        predictions for a quantity, then return the LQR approximation.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to make predictions for.
        eval_U : bool, default=False
            Specify as True to return the control prediction U(X), otherwise
            just computes the value function and gradient predictions

        Returns
        -------
        V : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X, only returned if
            eval_U=True.
        '''
        raise NotImplementedError

    def train(
            self, data,
            optimizer='AdamOptimizer', optimizer_opts={},
            batch_size=None, n_epochs=1,
            gradient_loss_weight=1., value_loss_weight=1., **kwargs
        ):
        '''
        Optimize the training loss using limited memory BFGS.

        Parameters
        ----------
        data : dict
            Dict of open loop optimal control data containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data
            V : (1, n_data) array
                Value function data
        optimizer : str, default='L-BFGS-B'
            Which optimizer to use. Options are 'L-BFGS-B' and any optimizer
            implemented in tensorflow.train.
        optimizer_opts : dict
            Dict of options to pass to the optimizer.
        batch_size : int, optional
            Number of data points (not trajectories) to use for SGD optimizers.
            If used in conjunction with 'L-BFGS-B', sets the maximum number of
            data to use.
        n_epochs : int, default=1
            How many times to iterate through the dataset (for SGD optimizers).
        gradient_loss_weight : float, default=1.
            Scalar multiplier in front of the value gradient mean square loss
            term. Not used by control_networks.
        value_loss_weight : float, default=1.
            Scalar multiplier in front of the value function mean square loss
            term. Not used by control_networks and gradient_networks.
        '''
        raise NotImplementedError

    def save(self, model_dir, error_dict, random_seed, sampling, architecture):
        '''
        Saves the model parameters to a pickle file. Also stores error data and
        important configuration parameters in a .csv file.

        Parameters
        ----------
        model_dir : path-like
            Which folder to save model parameters and information in
        error_dict : dict
            Dictionary containing error metrics to save
        random_seed : int or None
            Random seed set prior to training
        '''
        timestamp = int(time.time())

        self._save_parameters(timestamp, model_dir, sampling, architecture)

    def _save_parameters(self, timestamp, model_dir, sampling, architecture):
        '''
        Saves the model parameters to a pickle file in the specified directory.

        Parameters
        ----------
        timestamp : int
            UTC time at which the model was saved, without milliseconds
        model_dir : path-like
            Which folder to save model parameters and information in
        '''

        model_dict = {
            'architecture': self.architecture(),
            'parameters': self._get_learned_params()
        }
        for param in ['LQR', 'U_star_fun', 'activation', 'scaling']:
            if hasattr(self, param):
                model_dict[param] = getattr(self, param)
        if self.architecture() == 'LQR':
            model_dict['LQR'] = self



        if architecture=='GradientQRnet':
            if sampling=='adaptive':
                timestamp = 'AdQRnet'
            else:
                timestamp = 'FixQRnet'
        else:
            if sampling=='fixed':
                timestamp = 'FixNN'
            else:
                timestamp = 'AdNN'
                
        model_path = os.path.join(model_dir, timestamp + '.pkl')

        with open(model_path, 'wb') as model_file:
            dill.dump(model_dict, model_file)

        print('Model saved as ' + timestamp)

        
        
class LQR(BaseController):
    '''
    Implements a linear quadratic regulator (LQR) control with saturation
    constraints.

    Parameters
    ----------
    X_bar : (n_states, 1) array
        Goal state, nominal linearization point.
    U_bar : (n_controls, 1) array
        Control values at nominal linearization point.
    A : (n_states, n_states) array
        State Jacobian matrix at nominal equilibrium.
    B : (n_states, n_controls) array
        Control Jacobian matrix at nominal equilibrium.
    Q : (n_states, n_states) array
        Hessian of running cost with respect to states. Must be positive
        semi-definite.
    R : (n_controls, n_controls) array
        Hessian of running cost with respect to controls. Must be positive
        definite.
    U_lb : (n_controls, 1) array, optional
        Lower control saturation bounds.
    U_ub : (n_controls, 1) array, optional
        Upper control saturation bounds.
    '''
    def __init__(self, X_bar, U_bar, A, B, Q, R, P=None, U_lb=None, U_ub=None):
        self.X_bar = np.reshape(X_bar, (-1,1))
        self.U_bar = np.reshape(U_bar, (-1,1))

        self.n_states = self.X_bar.shape[0]
        self.n_controls = self.U_bar.shape[0]

        self.U_lb, self.U_ub = U_lb, U_ub

        if self.U_lb is not None:
            self.U_lb = np.reshape(self.U_lb, (-1,1))
        if self.U_ub is not None:
            self.U_ub = np.reshape(self.U_ub, (-1,1))

        # Make Riccati matrix and LQR control gain matrix
        if P is not None:
            self.P = np.asarray(P)
        else:
            self.P = care(A, B, Q, R)
        self.RB = np.linalg.solve(R, np.transpose(B))
        self.K = np.matmul(self.RB, self.P)

    def eval_V(self, X):
        '''
        Predicts the value function, V(X), for each sample state in X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value function for.

        Returns
        -------
        dVdX : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        '''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        XPX = X_err * np.matmul(self.P, X_err)
        XPX = np.sum(XPX, axis=0, keepdims=True)

        if X.ndim < 2:
            XPX = XPX.flatten()

        return XPX

    def eval_dVdX(self, X):
        '''
        Predicts the value function gradient, dV/dX(X), for each sample state in
        X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value gradient for.

        Returns
        ----------
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        '''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        PX = 2. * np.matmul(self.P, X_err)
        return PX.reshape(X.shape)

    def eval_U(self, X):
        '''
        Evaluates the NN feedback control, U(X), for each sample state in X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        ----------
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        U = self.U_bar - np.matmul(self.K, X_err)
        U = utilities.saturate_np(U, self.U_lb, self.U_ub)

        if X.ndim < 2:
            U = U.flatten()

        return U

    def eval_dUdX(self, X):
        '''
        Evaluates the Jacobian of the NN feedback control, [dU/dX](X), for each
        sample state in X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        ----------
        dUdX : (n_controls, n_states, n_data) or (n_controls, n_states) array
            Jacobian of NN feedback control for each column in X.
        '''
        if X.ndim < 2:
            return - self.K

        dUdX = np.expand_dims(- self.K, -1)
        dUdX = np.tile(dUdX, (1,1,X.shape[1]))
        return dUdX

    def bvp_guess(self, X):
        '''
        Predicts the value function V(X), its gradient dVdX(X), and the optimal
        control U(X) for each sample state in X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to make predictions for.

        Returns
        -------
        V : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar

        PX = np.matmul(self.P, X_err)

        XPX = np.sum(X_err * PX, axis=0, keepdims=True)
        U = self.U_bar - np.matmul(self.RB, PX)
        U = utilities.saturate_np(U, self.U_lb, self.U_ub)
        PX = 2. * PX.reshape(X.shape)

        if X.ndim < 2:
            XPX = XPX.flatten()
            U = U.flatten()

        return XPX, PX, U

    def train(self, data, **kwargs):
        '''
        Dummy train method for the LQR, which requires no training.

        Parameters
        ----------
        data : ignored
            For API consistency only.
        kwargs : ignored
            For API consistency only.
        '''
        pass
    
    def heuristic(self, data):
        pass

class BaseNN(BaseController):
    '''
    Base class for implementing a NN for approximating the optimal feedback
    control of an infinite horizon OCP.
    '''
    def __init__(
            self, LQR, n_hidden=None, n_neurons=None, n_out=None,
            activation='tanh', U_star_fun=None, scaling={}, parameters={}
        ):
        '''
        Build the computational graph for the NN and its loss functions. If
        scaling has been pre-computed elsewhere and supplied, uses this.
        Otherwise if data is supplied, computes scaling parameters
        appropriately. Initializes NN parameters.

        Parameters
        ----------
        LQR : object
            Instance of controllers.linear_quadratic_regulator.LQR.
        n_hidden : int
            Number of hidden layers. Required if weights is None.
        n_neurons : int
            Number of neurons per hidden layer. Required if weights is None.
        n_out : int
            Number of output neurons. Required if weights is None.
        activation : str, default='tanh'
            Activation function to use for hidden layers.
        U_star_fun : callable
            Function which evaluates the optimal control based on the state and
            value gradient. Takes two tensor arguments and outputs a tensor.
            Required for value_networks and gradient_networks.
        scaling : dict, optional
            Dict specifying scaling for inputs and outputs containing
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
            dVdX_lb : (n_states, 1) array
                Lower bound of gradient/costate data
            dVdX_ub : (n_states, 1) array
                Upper bound of gradient/costate data
            V_ub : float
                Upper bound of value function data
        parameters : dict, optional
            Dict containing a list of pre-trained weights and biases for each
            layer under the key 'weights', arranged as
            [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
        '''
        self.LQR = LQR
        self.activation = activation
        self.U_star_fun = U_star_fun

        self.weights = utilities.initialize_dense(
            n_in=LQR.n_states,
            n_out=n_out,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            weights=parameters.get('weights')
        )

        self.initialized_graph = False
        self._build(scaling=scaling)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build(self, scaling={}, data={}):
        '''
        Build the computational graph for the NN and its loss functions. If
        scaling has been pre-computed elsewhere and supplied, uses this.
        Otherwise if data is supplied, computes scaling parameters
        appropriately.

        Parameters
        ----------
        scaling : dict, optional
            Dict of arrays specifying scaling for inputs and outputs containing
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
            dVdX_lb : (n_states, 1) array
                Lower bound of gradient/costate data
            dVdX_ub : (n_states, 1) array
                Upper bound of gradient/costate data
            V_ub : float
                Upper bound of value function data
        data : dict, optional
            Dict of open loop optimal control data containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data
            V : (1, n_data) array
                Value function data
        '''
        if not self.initialized_graph:
            self.scaling, success_flag = self._setup_scaling(
                scaling=scaling, data=data
            )
            if success_flag:
                self._build_graph()
                self.initialized_graph = True

    def _build_graph(self):
        '''
        Build the computational graph for the NN and its loss functions. Should
        be implemented by all subclasses.
        '''
        raise NotImplementedError

    def _setup_scaling(self, var_names, scaling={}, data={}):
        '''
        Setup input and output scaling parameters for the network. If scaling
        has been pre-computed elsewhere and supplied, uses this. Otherwise if
        data is supplied, computes scaling parameters appropriately.

        Parameters
        ----------
        var_names : list of strings
            Which variables to compute scaling parameters for.
        scaling : dict, optional
            Dict of arrays specifying scaling for inputs and outputs containing
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
            dVdX_lb : (n_states, 1) array
                Lower bound of gradient/costate data
            dVdX_ub : (n_states, 1) array
                Upper bound of gradient/costate data
            V_ub : float
                Upper bound of value function data
        data : dict, optional
            Dict of open loop optimal control data containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data
            V : (1, n_data) array
                Value function data

        Returns
        -------
        scaling : dict
            Dictionary of arrays specifying scaling for inputs and outputs. May
            not have meaningful contents if success_flag is False
        success_flag : bool
            True if scaling was initialized successfully
        '''
        for bound in ['U_lb', 'U_ub']:
            if bound not in scaling and getattr(self.LQR, bound) is not None:
                scaling[bound] = getattr(self.LQR, bound)

        scaling['V_lb'] = 0.

        scaling_funs = {'lb': np.min, 'ub': np.max}

        # Loop over variables in data
        for var in var_names:
            # Loop over scaling functions to compute
            for fun_name, fun in scaling_funs.items():
                key = var + '_' + fun_name
                # If haven't already computed this parameter, compute it now
                if key not in scaling and var in data:
                    scaling[key] = fun(data[var], axis=1, keepdims=True)

        # Check to make sure all combinations of variables and scaling functions
        # were computed
        success_flag = all([
            '_'.join(pair) in scaling
            for pair in itertools.product(var_names, scaling_funs)
        ])

        if success_flag:
            if 'X_lb' in scaling and 'X_ub' in scaling:
                self.X_scale = 2./(scaling['X_ub'] - scaling['X_lb'])

            if 'U_lb' in scaling and 'U_ub' in scaling:
                self.U_lb = scaling['U_lb']
                self.U_ub = scaling['U_ub']
                self.U_scale = 2./(self.U_ub - self.U_lb)

            if 'dVdX_lb' in scaling and 'dVdX_ub' in scaling:
                self.dVdX_lb = scaling['dVdX_lb']
                self.dVdX_ub = scaling['dVdX_ub']
                self.dVdX_scale = 2./(self.dVdX_ub - self.dVdX_lb)

            if 'V_ub' in scaling:
                self.V_ub = scaling['V_ub']

        return scaling, success_flag

    def _get_learned_params(self):
        '''
        Get numpy values for the trainable tensorflow variables such as NN
        weights. By default returns the NN weights but should be overwritten to
        return other variables as needed.

        Returns
        ----------
        parameters : dict
            Dictionary of (lists of) arrays giving trained parameter values
        '''
        return {'weights': self.sess.run(self.weights)}

    def eval_V(self, X):
        '''
        Predicts the value function, V(X), for each sample state in X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value function for.

        Returns
        ----------
        dVdX : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        '''
        if not hasattr(self, 'V_pred'):
            raise NotImplementedError

        V = self.sess.run(self.V_pred, {self.X_tf: X.reshape(X.shape[0], -1)})
        if X.ndim < 2:
            V = V.flatten()
        return V

    def eval_dVdX(self, X):
        '''
        Predicts the value function gradient, dV/dX(X), for each sample state in
        X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value gradient for.

        Returns
        ----------
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        '''
        if not hasattr(self, 'dVdX_pred'):
            raise NotImplementedError

        dVdX = self.sess.run(
            self.dVdX_pred, {self.X_tf: X.reshape(X.shape[0], -1)}
        )
        return dVdX.reshape(X.shape)

    def eval_U(self, X):
        '''
        Evaluates the NN feedback control, U(X), for each sample state in X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        ----------
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        U = self.sess.run(self.U_pred, {self.X_tf: X.reshape(X.shape[0], -1)})
        if X.ndim < 2:
            U = U.flatten()
        return U.astype(np.float64)

    def eval_dUdX(self, X):
        '''
        Evaluates the Jacobian of the NN feedback control, [dU/dX](X), for each
        sample state in X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        ----------
        dUdX : (n_controls, n_states, n_data) or (n_controls, n_states) array
            Jacobian of NN feedback control for each column in X.
        '''
        dUdX = self.sess.run(self.dUdX, {self.X_tf: X.reshape(X.shape[0], -1)})
        if X.ndim < 2:
            dUdX = dUdX[:,:,0]
        return dUdX.astype(np.float64)
    


class GradientNN(BaseNN):
    '''
    A NN for approximating the gradient of the value function of an infinite
    horizon OCP.
    '''
    def __init__(
            self, LQR, n_hidden=None, n_neurons=None, n_out=None,
            activation='tanh', U_star_fun=None, scaling={}, parameters={}
        ):
        '''
        Build the computational graph for the NN and its loss functions. If
        scaling has been pre-computed elsewhere and supplied, uses this.
        Otherwise if data is supplied, computes scaling parameters
        appropriately. Initializes NN parameters.

        Parameters
        ----------
        LQR : object
            Instance of controllers.linear_quadratic_regulator.LQR.
        n_hidden : int
            Number of hidden layers. Required if weights is None.
        n_neurons : int
            Number of neurons per hidden layer. Required if weights is None.
        n_out : int
            Number of output neurons. Required if weights is None.
        activation : str, default='tanh'
            Activation function to use for hidden layers.
        U_star_fun : callable
            Function which evaluates the optimal control based on the state and
            value gradient. Takes two tensor arguments and outputs a tensor.
            Required for value_networks and gradient_networks.
        scaling : dict, optional
            Dict of arrays specifying scaling for inputs and outputs containing
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
            dVdX_lb : (n_states, 1) array
                Lower bound of gradient/costate data
            dVdX_ub : (n_states, 1) array
                Upper bound of gradient/costate data
        parameters : dict, optional
            Dict containing a list of pre-trained weights and biases for each
            layer under the key 'weights', arranged as
            [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
        '''
        if n_out is None:
            n_out = LQR.n_states

        super().__init__(
            LQR,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            n_out=n_out,
            activation=activation,
            U_star_fun=U_star_fun,
            scaling=scaling,
            parameters=parameters
        )

    def _setup_scaling(self, scaling=None, data=None):
        '''
        Setup input and output scaling parameters for the network. If scaling
        has been pre-computed elsewhere and supplied, uses this. Otherwise if
        data is supplied, computes scaling parameters appropriately.

        Parameters
        ----------
        var_names : list of strings
            Which variables to compute scaling parameters for.
        scaling : dict, optional, containing
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
            dVdX_lb : (n_states, 1) array
                Lower bound of gradient/costate data
            dVdX_ub : (n_states, 1) array
                Upper bound of gradient/costate data
        data : dict, optional, containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data

        Returns
        ----------
        scaling : dict
            Dictionary of arrays specifying scaling for inputs and outputs. May
            not have meaningful contents if success_flag is False
        success_flag : bool
            True if scaling was initialized successfully
        '''
        return super()._setup_scaling(
            ['X', 'U', 'dVdX'], scaling=scaling, data=data
        )

    def _build_graph(self):
        '''
        Build the computational graph for the NN and its loss functions.
        '''
        n_states = self.LQR.n_states
        n_controls = self.LQR.n_controls

        # Builds computational graph
        self.X_tf = tf.placeholder(tf.float32, shape=(n_states, None))
        self.dVdX_tf = tf.placeholder(tf.float32, shape=(n_states, None))
        self.V_tf = tf.placeholder(tf.float32, shape=(1, None))
        self.U_tf = tf.placeholder(tf.float32, shape=(n_controls, None))

        self.V_scaled_tf = tf.placeholder(tf.float32, shape=(1, None))
        self.dVdX_scaled_tf = tf.placeholder(tf.float32, shape=(n_states, None))
        self.U_scaled_tf = tf.placeholder(tf.float32, shape=(n_controls, None))

        dVdX_scaled_pred, self.dVdX_pred = self._make_eval_graph(self.X_tf)

        self.U_pred = self.U_star_fun(self.X_tf, self.dVdX_pred)
        self.dUdX = utilities.tf_jacobian(self.U_pred, self.X_tf)
        U_scaled_pred = self.U_scale*(self.U_pred - self.U_lb) - 1.
        

        # Value gradient loss using scaled data
        self.loss_dVdX = tf.reduce_mean(
            tf.reduce_sum((dVdX_scaled_pred - self.dVdX_scaled_tf)**2, axis=0)
        )

        # Control loss using scaled data
        self.loss_U = tf.reduce_mean(
            tf.reduce_sum((U_scaled_pred - self.U_scaled_tf)**2, axis=0)
        )

        self.gradient_loss_weight = tf.placeholder(tf.float32)
        
        # original loss function, however U_loss not needed
        # self.loss = self.loss_U + self.gradient_loss_weight * self.loss_dVdX 


        self.loss = self.loss_U + self.gradient_loss_weight * self.loss_dVdX

    def _make_eval_graph(self, X):
        '''
        Helper function which builds a dense NN and transforms the output to
        make the prediction tensor operations.

        Arguments
        ----------
        X : (n_states, n_data) tensor
            State locations to make predictions for

        Returns
        ----------
        dVdX_scaled : (n_states, n_data) tensor
            Linearly scaled value gradient predictions for each state
        dVdX : (n_states, n_data) tensor
            Value gradient predictions for each state in original domain
        '''

        # Raw NN prediction in the scaled domain
        dVdX_scaled = utilities.make_dense_graph(
            X - self.LQR.X_bar,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        dVdX = (dVdX_scaled + 1.)/self.dVdX_scale + self.dVdX_lb

        return dVdX_scaled, dVdX

    def bvp_guess(self, X):
        '''
        Predicts the value function V(X), its gradient dVdX(X), and the optimal
        control U(X) for each sample state in X. If the network does not make
        predictions for a quantity, then return the LQR approximation.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to make predictions for.

        Returns
        ----------
        V : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        V, _, _ = self.LQR.bvp_guess(X)

        dVdX, U = self.sess.run(
            (self.dVdX_pred, self.U_pred),
            {self.X_tf: X.reshape(X.shape[0], -1)}
        )
        if X.ndim < 2:
            U = U.flatten()
        return V, dVdX.reshape(X.shape), U.astype(np.float64)


    
                    
    def heuristic(self, data, C):  
        '''
        Heuristic for updating the size of the data set in the adaptive sampling algorithm.

        Arguments
        ----------
        data : (n_states, n_data) tensor
            State locations to make predictions for

        Returns
        ----------
        heuristic : (n_states, n_data) tensor
            Linearly scaled value gradient predictions for each state
        D : (n_states, n_data) tensor
            Value gradient predictions for each state in original domain
        '''
        data_available = list(data['X'])
        D = data_available[0].shape[-1]
        
        # Compute L1 norm and variance over the current gradient array
        sum_l1_norm = np.sum(np.abs(np.mean(self.all_gradients, axis=1)))
        variances = [np.var(component_gradients) for component_gradients in self.all_gradients]
        total_variance = np.sum(variances)

        # Calculate your quotients and return the appropriate value
        heuristic = total_variance // ((C**2) * (sum_l1_norm**2))

        
        return heuristic, D


# ---------------------------------------------------------------------------- #

class GradientQRnet(GradientNN):
    def _make_eval_graph(self, X):
        '''
        Helper function which builds a dense NN and transforms the output to
        make the prediction tensor operations.

        Arguments
        ----------
        X : (n_states, n_data) tensor
            State locations to make predictions for

        Returns
        ----------
        dVdX_scaled : (n_states, n_data) tensor
            Linearly scaled value gradient predictions for each state
        dVdX : (n_states, n_data) tensor
            Value gradient predictions for each state in original domain
        '''

        X_err = X - self.LQR.X_bar

        # Raw NN prediction in the scaled domain
        dVdX_scaled = utilities.make_dense_graph(
            X_err,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        # Subtract NN contribution at zero
        dVdX_scaled_0 = utilities.make_dense_graph(
            tf.zeros((self.LQR.n_states, 1), dtype=tf.float32),
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        dVdX_scaled = dVdX_scaled - dVdX_scaled_0

        # LQR component
        PX = tf.matmul(2. * self.LQR.P.astype(np.float32), X_err)

        dVdX = PX + dVdX_scaled / self.dVdX_scale

        dVdX_scaled = dVdX_scaled + self.dVdX_scale*(PX - self.dVdX_lb) - 1.

        return dVdX_scaled, dVdX

