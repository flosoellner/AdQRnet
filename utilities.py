import numpy as np
from scipy.optimize import root
import os
import dill
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from IPython.display import display, Latex

def cross_product_matrix(w):
    zeros = np.zeros_like(w[0])
    wx = np.array([
        [zeros, -w[2], w[1]],
        [w[2], zeros, -w[0]],
        [-w[1], w[0], zeros]]
    )
    return wx

def cheb(N):
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

def get_batches(n_data, batch_size, force_batch_size=False):
    '''
    Generates slices for taking subsets of arrays.

    Parameters
    ----------
    n_data : int
        Number of data points in the arrays of interest.
    batch_size : int
        The size of each slice to take. The last slice generated may be smaller
        than this.
    force_batch_size : bool, default=False
        If True and batch_size doesn't evenly divide n_data, the last batch will
        be omitted so that all batches have the same batch size.

    Yields
    ------
    batch_slice : slice
        The ith iterate is a slice from batch_size*i to batch_size*(i+1). If
        batch_size doesn't evenly divide n_data and force_batch_size=False, then
        the last batch will be larger to include all data.
    '''
    n_batches = max(1, n_data // batch_size)

    if force_batch_size or n_batches*batch_size==n_data:
        combine_last = False
    else:
        combine_last = True

    for i in range(n_batches):
        # If reached the end make the last batch bigger to include all data
        if combine_last and batch_size*(i+2)>n_data:
            yield slice(batch_size*i, n_data)
        else:
            yield slice(batch_size*i, batch_size*(i+1))


def shuffle_data(dataset, n_data):
    '''
    Shuffles a list of arrays in place with a common reindexing. Note that the
    list is modified in place but the original arrays are copied. Shuffles only
    those arrays whose last index has n_data elements.

    Parameters
    ----------
    dataset : list of arrays and/or floats
        Data to shuffle, for example dataset=[X, Y, c], where X, Y are
        (n_x, n_data) and (n_y, n_data) arrays and c is a float.
    n_data : int
        The number of data points in the dataset. Each entry X of dataset will
        be shuffled X.shape[-1] == n_data.
    '''
    shuffle_idx = np.random.permutation(n_data)

    for i, data in enumerate(dataset):
        if np.ndim(data) >= 1 and np.shape(data)[-1] == n_data:
            dataset[i] = data[...,shuffle_idx]

def saturate_np(U, U_lb, U_ub):
    '''
    Hard saturation of control for numpy arrays.

    Parameters
    ----------
    U : (n_controls, n_data) or (n_controls,) array
        Control(s) to saturate.
    U_lb : (n_controls, 1) array
        Lower control bounds.
    U_ub : (n_controls, 1) array
        Upper control bounds.

    Returns
    -------
    U : array with same shape as input
        Control(s) saturated between U_lb and U_ub
    '''
    if U_lb is not None or U_ub is not None:
        if U.ndim < 2:
            U = np.clip(U, U_lb.flatten(), U_ub.flatten())
        else:
            U = np.clip(U, U_lb, U_ub)

    return U

def saturate_tf(U, U_lb, U_ub):
    '''
    Hard saturation of control for tensorflow variables.

    Parameters
    ----------
    U : (n_controls, None) tensor
        Control tensor to saturate.
    U_lb : (n_controls, 1) array
        Lower control bounds.
    U_ub : (n_controls, 1) array
        Upper control bounds.

    Returns
    -------
    U : (n_controls, None) tensor
        Controls saturated between U_lb and U_ub
    '''
    from tensorflow import clip_by_value

    if U_lb is not None and U_ub is not None:
        U = clip_by_value(U, U_lb, U_ub)
    elif U_lb is not None:
        U = clip_by_value(U, U_lb, np.inf)
    elif U_ub is not None:
        U = clip_by_value(U, -np.inf, U_ub)

    return U


def xavier_init(n_in, n_out):
    '''
    Xavier normal initialization for dense network weights.

    Parameters
    ----------
    n_in : int
        Number of input neurons
    n_out : int
        Number of output neurons

    Returns
    -------
    W : (n_out, n_in) tensor
        Tensorflow variable initialized from a normal distribution
    '''
    std = np.sqrt(2. / (n_in + n_out))
    init = std * np.random.randn(n_out, n_in)
    return tf.Variable(init, dtype=tf.float32)

def initialize_dense(n_in, n_out, n_hidden, n_neurons, weights=None):
    '''
    Initializes tensorflow variables corresponding to weights and biases for a
    standard dense feedforward neural network. Weights and biases are stored in
    a list arranged as
        [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]

    Parameters
    ----------
    n_in : int
        Number of input variables to the NN
    n_out : int
        Number of output variables from the NN
    n_hidden : int
        Number of hidden layers
    n_neurons : int
        Number of neurons per hidden layer
    weights : list of numpy arrays, optional
        List of pre-trained weights and biases for each layer

    Returns
    -------
    tf_weights : list of tensors
        Initialized tensorflow variables for the NN weights and biases
    '''
    tf_weights = []

    if weights is None or len(weights) < 2:
        layers = [n_in] + n_hidden * [n_neurons] + [n_out]

        for l in range(n_hidden + 1):
            tf_weights.append(
                xavier_init(n_in=layers[l], n_out=layers[l+1])
            )
            tf_weights.append(
                tf.Variable(tf.zeros((layers[l+1], 1), dtype=tf.float32))
            )
    else:
        for l in range(len(weights)):
            tf_weights.append(tf.Variable(weights[l], dtype=tf.float32))

    return tf_weights

def make_dense_graph(X, X_scale, weights, activation='tanh'):
    '''
    Makes a tensorflow computational graph for a standard dense feedforward
    neural network.

    Parameters
    ----------
    X : (n_in, n_data) tensor
        Inputs to the network
    X_scale : (n_in, 1) tensor or array
        Scale vector to multiply the inputs by
    weights : list of tensors
        List of weights and biases for each layer, arranged as
        [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
    activation : str or list of strs, default='tanh'
        Activation function to use. Can also be a list, in which case the
        activation functions are specified per hidden layer.

    Returns
    -------
    Y : (n_out, n_data) tensor
        Neural network predictions for each X
    '''
    n_hidden = int(len(weights) / 2) - 1

    if not isinstance(activation, list):
        activation = n_hidden * [activation]

    Y = X_scale * X

    for l in range(n_hidden + 1):
        W = weights[2*l]
        b = weights[2*l+1]
        Y = tf.matmul(W, Y) + b

        if l < len(activation):
            # TODO: choice of activation functions
            Y = tf.tanh(Y)

    return Y

def tf_jacobian(Y, X, stop_gradients=None):
    '''
    Compute the Jacobian, dYdX, of a vector-valued tensorflow graph Y=Y(X).

    Parameters
    ----------
    Y : (n_out, ?) tensor
        Dependent variables
    X : (n_in, ?) tensor
        Independent variables. X.shape[1] must be compatible with Y.shape[1]
    stop_gradients : tensor or list of tensors, optional
        Variables to be held constant during differentiation

    Returns
    -------
    dYdX : (n_out, n_in, ?) tensor
        The Jacobian dYdX of Y=Y(X) at each input point
    '''
    dYdX = [
        tf.gradients(Y[d], X, stop_gradients=stop_gradients)
        for d in range(Y.shape[0])
    ]
    return tf.concat(dYdX, axis=0)


def create_NN(_model_registry, architecture, LQR, **kwargs):
    '''
    Convenience function to initialize an NN with the architecture specified in
    the config file. Can be a new NN or an existing one.

    Parameters
    ----------
    architecture : str
        NN architecture to instantiate. See QRnet.controllers for options.
    LQR : object
        Instance of controllers.linear_quadratic_regulator.LQR.
    kwargs : name-value pairs
        Keyword arguments to pass to the controller.

    Returns
    -------
    controller : object
        Instantiated BaseNN subclass.
    '''

    if architecture == 'LQR':
        return LQR

    NN = get_model_class(_model_registry, architecture)

    return NN(LQR, **kwargs)

def load_NN(_model_registry, model_dir, timestamp, verbose=True):
    '''
    Load a previously trained NN model from <model_dir>/<timestamp>.pkl.

    Parameters
    ----------
    model_dir : path-like
        Which folder to load model parameters from
    timestamp : int or str, optional
        UTC time at which the model was saved, without milliseconds. If
        timestamp is not in the list of timestamps in model_info.csv, then sorts
        this chronologically and treats the timestamp argument as a python list
        index to select from the timestamps list. If this fails, picks the most
        recently trained model.

    Returns
    -------
    controller : instantiated BaseNN subclass
        Model ready to use for control
    timestamp : int
        UTC time at which the model was saved, without milliseconds
    '''
    if timestamp is None:
        timestamp = -1

    timestamps_list = [
        fn[:-4] for fn in os.listdir(model_dir) if fn[-4:] == '.pkl'
    ]

    if timestamp not in timestamps_list:
        try:
            timestamp = timestamps_list[timestamp]
        except IndexError:
            timestamp = timestamps_list[-1]

    model_path = os.path.join(model_dir, timestamp + '.pkl')

    with open(model_path, 'rb') as model_file:
        model_dict = dill.load(model_file)

    controller = create_NN(_model_registry, **model_dict)

    return controller, timestamp

# 4096
def eval_errors(NN, data, batch_size, return_predictions=False):
    '''
    Evaluate a set of error metrics for a given test (or train) data set.

    Parameters
    ----------
    controller : instantiated BaseNN subclass
        Model to evaluate error metrics for
    data : dict containing
        X : (n_states, n_data) array
            Input state data
        U : (n_controls, n_data) array
            Optimal control data
        dVdX : (n_states, n_data) array
            Value function gradient data
        V : (1, n_data) array
            Value function data
    batch_size : int, default=4096
        Number of data points to pass to the controller at once. Set smaller or
        larger to adjust memory footprint.
    return_predictions : bool, default=False
        If return_predictions=True then output will also contain raw NN
        predictions.

    Returns
    -------
    error_dict : dict containing (a subset of)
        U_maxL2 : maximum L2 error in control over test data set
        U_ML2 : mean L2 error in control over test data set
        U_RML2 : mean L2 error in control over test data set, scaled
            by the maximum control L2 norm
        U_pred : NN predictions of the control
        dVdX_ML2 : mean L2 error in value gradient over test data set
        dVdX_RML2 : mean L2 error in value gradient over test data set,
            scaled by the maximum value gradient L2 norm
        dVdX_pred : NN predictions of the value gradient
        V_MAE : mean absolute error in the value function over test data
        V_RMAE : mean absolute error in the value function over test
            data set, scaled by the maximum value of the value function data
        V_pred : NN predictions of the value function
    '''
    error_dict = {}

    def batch_predict(pred_fun):
        batches = get_batches(data['X'].shape[-1], batch_size)
        pred = [pred_fun(data['X'][:,batch_idx]) for batch_idx in batches]
        return np.hstack(pred)

    try:
        U_pred = batch_predict(NN.eval_U)
        U_err = np.linalg.norm(U_pred - data['U'], axis=0)
        data_norm = np.linalg.norm(data['U'], axis=0)
        #U_max = np.max(np.linalg.norm(data['U'], axis=0))

        #error_dict['U_ML2'] = np.mean(U_err)
        error_dict['U_RML2'] = np.mean(U_err) / np.mean(data_norm)
        #error_dict['U_maxL2'] = np.max(U_err)
        if return_predictions:
            error_dict['U_pred'] = U_pred
    except (NotImplementedError, KeyError):
        pass

    try:
        dVdX_pred = batch_predict(NN.eval_dVdX)
        dVdX_err = np.linalg.norm(dVdX_pred - data['dVdX'], axis=0)
        data_norm = np.linalg.norm(data['dVdX'], axis=0)
        #dVdX_max = np.max(np.linalg.norm(data['dVdX'], axis=0))

        #error_dict['dVdX_ML2'] = np.mean(dVdX_err)
        error_dict['dVdX_RML2'] = np.mean(dVdX_err) / np.mean(data_norm)
        #error_dict['dVdX_maxL2'] = np.max(dVdX_err)
        if return_predictions:
            error_dict['dVdX_pred'] = dVdX_pred
    except (NotImplementedError, KeyError):
        pass

    try:
        V_pred = batch_predict(NN.eval_V)
        V_err = np.abs(V_pred - data['V'])
        V_max = np.max(np.abs(data['V']))
        error_dict['V_MAE'] = np.mean(V_err)
        error_dict['V_RMAE'] = error_dict['V_MAE'] / V_max
        if return_predictions:
            error_dict['V_pred'] = V_pred
    except NotImplementedError:
        pass

    return error_dict, error_dict['U_RML2'], error_dict['dVdX_RML2']

def print_errors(train_errs, test_errs):
    '''
    Prints out a table of error metrics.

    Parameters
    ----------
    train_errs : dict
        Dictionary of error metrics produced by eval_errors for training data
    test_errs : dict
        Dictionary of error metrics produced by eval_errors for test data
    '''
    col_width = np.max([len(key) for key in test_errs] + [len('error metric')])
    col_width = col_width.astype(str)
    header = ' {metric:<' + col_width + 's} |  train   |  test    '
    header = header.format(metric='error metric')
    row = ' {metric:<' + col_width + 's} | {train_err:1.2e} | {test_err:1.2e}'

    print('-'*len(header))
    print(header)
    print('-'*len(header))

    for key in np.sort(list(test_errs.keys())):
        print(
            row.format(
                metric=key, train_err=train_errs[key], test_err=test_errs[key]
            )
        )

    print('-'*len(header) + '\n')


def find_fixed_point(OCP, controller, tol, X0=None, verbose=True):
    '''
    Use root-finding to find a fixed point (equilibrium) of the closed-loop
    dynamics near the desired goal state OCP.X_bar. ALso computes the
    closed-loop Jacobian and its eigenvalues.

    Parameters
    ----------
    OCP : instance of QRnet.problem_template.TemplateOCP
    config : instance of QRnet.problem_template.MakeConfig
    tol : float
        Maximum value of the vector field allowed for a trajectory to be
        considered as convergence to an equilibrium
    X0 : array, optional
        Initial guess for the fixed point. If X0=None, use OCP.X_bar
    verbose : bool, default=True
        Set to True to print out the deviation of the fixed point from OCP.X_bar
        and the Jacobian eigenvalue

    Returns
    -------
    X_star : (n_states, 1) array
        Closed-loop equilibrium
    X_star_err : float
        ||X_star - OCP.X_bar||
    F_star : (n_states, 1) array
        Vector field evaluated at X_star. If successful should have F_star ~ 0
    Jac : (n_states, n_states) array
        Close-loop Jacobian at X_star
    eigs : (n_states, 1) complex array
        Eigenvalues of the closed-loop Jacobian
    max_eig : complex scalar
        Largest eigenvalue of the closed-loop Jacobian
    '''
    if X0 is None:
        X0 = OCP.X_bar
    X0 = np.reshape(X0, (OCP.n_states,))

    def dynamics_wrapper(X):
        U = controller.eval_U(X)
        F = OCP.dynamics(X, U)
        C = OCP.constraint_fun(X)
        if C is not None:
            F = np.concatenate((F.flatten(), C.flatten()))
        return F

    def Jacobian_wrapper(X):
        J = OCP.closed_loop_jacobian(X, controller)
        JC = OCP.constraint_jacobian(X)
        if JC is not None:
            J = np.vstack((
                J.reshape(-1,X.shape[0]), JC.reshape(-1,X.shape[0])
            ))
        return J

    sol = root(dynamics_wrapper, X0, jac=Jacobian_wrapper, method='lm')

    sol.x = OCP.apply_state_constraints(sol.x)

    X_star = sol.x.reshape(-1,1)
    U_star = controller.eval_U(X_star)
    # V_star = controller.eval_V(X_star)
    F_star = OCP.dynamics(X_star, U_star).reshape(-1,1)
    Jac = OCP.closed_loop_jacobian(sol.x, controller)

    X_star_err = OCP.norm(X_star)[0]

    eigs = np.linalg.eigvals(Jac)
    idx = np.argsort(eigs.real)
    eigs = eigs[idx].reshape(-1,1)
    max_eig = np.squeeze(eigs[-1])

    # Some linearized systems always have one or more zero eigenvalues.
    # Handle this situation by taking the next largest.
    if np.abs(max_eig.real) < tol**2:
        Jac0 = np.squeeze(OCP.closed_loop_jacobian(OCP.X_bar, OCP.LQR))
        eigs0 = np.linalg.eigvals(Jac0)
        idx = np.argsort(eigs0.real)
        eigs0 = eigs0[idx].reshape(-1,1)
        max_eig0 = np.squeeze(eigs0[-1])

        i = 2
        while all([
                i <= OCP.n_states,
                np.abs(max_eig.real) < tol**2,
                np.abs(max_eig0.real) < tol**2
            ]):
            max_eig = np.squeeze(eigs[OCP.n_states - i])
            max_eig0 = np.squeeze(eigs0[OCP.n_states - i])
            i += 1

    return X_star, X_star_err, F_star, Jac, eigs, max_eig

def yn_input(message):
    while True:
        try:
            user_input = input(message + ' Enter (y/n):\n').lower()
            if user_input == 'y':
                return True
            elif user_input == 'n':
                return False
            else:
                raise ValueError
        except ValueError:
            print(user_input, "is not a valid input. Must enter 'y' or 'n'...")



def summary(sampling, n_initial_data, n_data_available, train_time, loss_train, val_errs_U, val_errs_dVdX):
      
    info = [sampling, n_initial_data, n_data_available, "{}".format(int(train_time)), "{:.1e}".format(loss_train).replace('e', '\\times 10^{') + '}', "{:.1e}".format(val_errs_U).replace('e', '\\times 10^{') + '}', "{:.1e}".format(val_errs_dVdX).replace('e', '\\times 10^{') + '}' ]
    
    return info

def print_summary(model_info):
    """
    Generates a LaTeX table from a list of neural network metrics, with a predefined set of headers.
    
    Parameters:
    - metric_lists: A list of lists, where each inner list contains metrics for a neural network.
    
    Returns:
    - A string containing the LaTeX code for the table.
    """
    # Predefined headers
    headers = ["Sampling", "$|\mathcal{D}_{train}^1|$", "$|\mathcal{D}_{train}^r|$", "Time (sec)", "Loss",  "$RML^2_{u,val}$", "$RML^2_{V_x,val}$"]
    
    # Start of the table
    latex_str = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{" + "c" * len(headers) + "}\n\\hline\n"
    
    # Add the header
    latex_str += " & ".join(headers) + " \\\\\n\\hline\n"
    
    # Add the rows for each set of metrics
    for metrics in model_info:
        row = "$ & $".join(str(metric) for metric in metrics) + "\\\\\n"
        latex_str += row
    
    # Close the table
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Your caption here.}\n\\label{tab:your_label}\n\\end{table}"
    
    # Print the LaTeX table
    print(latex_str)
    return latex_str





def register(_model_registry, model_name, model_class):
    '''
    Add a model class to the factory dictionary.

    Parameters
    ----------
    model_name : str
        Key to associate with model_class
    model_class : class reference
        Class definition to add
    '''
    _model_registry[model_name] = model_class

def available_models(_model_registry):
    """
    Returns a list of currently registered NN names.
    """
    return list(_model_registry.keys())

def get_model_class(_model_registry, model_name):
    '''
    Get a model class by name lookup.

    Arguments
    ----------
    model_name : str
        Name of the model class reference to look up

    Returns
    ----------
    model_class : class reference
        Class definition associated with model_name
    '''
    model_class = _model_registry.get(model_name)

    if model_class is None:
        err_str = model_name + ' is not a registered model.'
        err_str += ' Available model names are:\n'
        err_str += '\n'.join(_model_registry.keys())
        raise ValueError(err_str)

    return model_class






