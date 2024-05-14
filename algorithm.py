import os
import time
import numpy as np
import scipy.io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import generate, utilities




def train(_model_registry, config, OCP, data_dir, model_dir, architecture, sampling, seed):
    tf.random.set_random_seed(seed)
    
    start_time = time.time()

    # validation Data Generation ########################################################
    if sampling == 'fixed':
        n_trajectories_val = config.n_trajectories_val
        q = config.r
    else: 
        n_trajectories_val = config.n_trajectories_initial_adaptive
        q = 1
    # Check if validation data already exists, if not, generate new validation data.
    if not os.path.exists(os.path.join(data_dir, 'val'+sampling+'.mat')):
        n_trajectories = n_trajectories_val
        np.random.seed(2*seed)
        X0_pool = OCP.sample_X0(n_trajectories).reshape(OCP.n_states, -1)

        val_data, val_initial_conditions = generate.generate(
            OCP, config, n_trajectories, X0_pool, OCP.LQR,
            resolve_failed=True
        )
    else:
        val_data = scipy.io.loadmat(os.path.join(data_dir, 'val'+sampling+'.mat'))


        # val Data Generation ########################################################

    train_errors_U_list = []
    train_errors_dVdX_list = []
    val_errors_U_list = []
    val_errors_dVdX_list = []


    # Define a modified callback function to append losses
    def callback(feed_dict, session, epoch, train_errors_U, train_errors_dVdX):
        loss_train = session.run((controller.loss), feed_dict)
        #print('\nEpoch = %d' % epoch)


    # Training Data Generation ##################################################

    # Configure the number of trajectories and epochs based on the sampling strategy.
    if sampling == 'adaptive':
        n_trajectories = config.n_trajectories_initial_adaptive
        r = config.r
        
    # Generate initial training data.
    
        np.random.seed(seed+2)
        X0_pool = OCP.sample_X0(n_trajectories).reshape(OCP.n_states, -1)
        data, train_ic = generate.generate(
            OCP, config, n_trajectories, X0_pool, OCP.LQR,
            resolve_failed=True
        )
    
    else:
        if os.path.exists(os.path.join(data_dir, 'train'+sampling+'.mat')):
            data = scipy.io.loadmat(os.path.join(data_dir, 'train'+sampling+'.mat'))
            r = 1
    
        else:
            n_trajectories = config.n_trajectories_train_fixed
            r = 1
            
        # Generate initial training data.
        
            np.random.seed(seed+2)
            X0_pool = OCP.sample_X0(n_trajectories).reshape(OCP.n_states, -1)
            data, train_ic = generate.generate(
                OCP, config, n_trajectories, X0_pool, OCP.LQR,
                resolve_failed=True
            )


    data_available = list(data['X'])
    n_initial_data = data_available[0].shape[-1]
    #print(n_initial_data)
        
    # Training Data Generation ##################################################

    # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2 # 2

    # create a neural network controller.

    controller = utilities.create_NN(_model_registry,
        architecture,
        OCP.LQR,
        n_hidden=config.n_hidden,
        n_neurons=config.n_neurons,
        activation=config.activation,
        U_star_fun=OCP.make_U_NN
    )
        
    # Begin the training process.
    #print('\nTraining ' + controller.architecture() + '...')
    controller._build(data=data)
    
    parameter_keys = [controller.gradient_loss_weight] 
    parameter_vals = [config.gradient_loss_weight]

    options = {}
    # Set up the optimizer and training steps.
    optimizer = getattr(tf.train, 'AdamOptimizer')(**options)
    train_step = optimizer.minimize(controller.loss)

        
    controller.sess.run(tf.global_variables_initializer())
        
        
    # convergence criterion


    generation_round = 1
    while generation_round <= r:
        generation_round +=1
        


        
        
    # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3 # 3

    # Training loop



        # Scale the control input and derivative of the value function.
        U_scaled = controller.U_scale * (data['U'] - controller.U_lb) - 1.
        dVdX_scaled = controller.dVdX_scale * (data['dVdX'] - controller.dVdX_lb) - 1.

        # Prepare data for training.
        data_keys = [controller.X_tf, controller.U_scaled_tf, controller.dVdX_scaled_tf]
        data_vals = [data['X'], U_scaled, dVdX_scaled]

        callback = callback
        callback_epoch = 1


        # Initialize TensorFlow gradients.
        raw_gradients = tf.gradients(controller.loss, tf.trainable_variables())
        gradients = [grad for grad in raw_gradients if grad is not None]
        
        n_data_available = data_vals[0].shape[-1]


        batch_size = np.int(np.round(config.batch_ratio*n_data_available))
        
        # Determine the batch size and initialize the training loop.


        if not batch_size or batch_size > n_data_available:
            batch_size = n_data_available  

        # print('\nBatch size = %d\n' % batch_size)
        print('\nData available = %d\n' % n_data_available)
        feed_dict = dict(zip(parameter_keys, parameter_vals))

        controller.all_gradients = []

            

        epoch = 1
        while epoch <= q*config.n_epochs:

            
            utilities.shuffle_data(data_vals, n_data_available)
            batches = utilities.get_batches(
                n_data_available, batch_size, force_batch_size=True
            )

         
            for batch_idx in batches:
                feed_dict.update(dict(zip(
                    data_keys, [data[...,batch_idx] for data in data_vals]
                )))
                controller.sess.run(train_step, feed_dict=feed_dict)


            _, train_errs_U, train_errs_dVdX = utilities.eval_errors(controller, data, batch_size)
            train_errors_U_list.append(train_errs_U)
            train_errors_dVdX_list.append(train_errs_dVdX)
            
            _, val_errs_U, val_errs_dVdX = utilities.eval_errors(controller, val_data, 5**batch_size)
            val_errors_U_list.append(val_errs_U)
            val_errors_dVdX_list.append(val_errs_dVdX)


            
            if callable(callback) and callback_epoch:
                if epoch % callback_epoch == 0:
                    callback(feed_dict, controller.sess, epoch, train_errs_U, train_errs_dVdX)

            epoch +=1
            
            ##### Training Loop #########




    # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4 # 4
        if sampling=='adaptive' and generation_round < r:
            # Calculate and store gradients for each data point.
            for i in range(n_data_available):
                feed_dict_individual = dict(zip(data_keys, [data_val[:, i:i+1] for data_val in data_vals]))
                feed_dict_individual.update(zip(parameter_keys, parameter_vals))
                individual_gradient = controller.sess.run(gradients, feed_dict=feed_dict_individual)
                individual_gradient_flat = np.concatenate([grad.flatten() for grad in individual_gradient if grad is not None])
                while len(controller.all_gradients) < len(individual_gradient_flat):
                    controller.all_gradients.append([])
                for component_index, grad_component in enumerate(individual_gradient_flat):
                    controller.all_gradients[component_index].append(grad_component)
        
            initial_condition_norms = []
            # Additional training iterations based on heuristic evaluation.
            #for n in range(r):
            heuristic, D = controller.heuristic(data, config.C)
            if architecture != 'LQR':
                    
        # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5 # 5
        # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6 # 6
        
                if heuristic > D:
        
        
        
        # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7 # 7
        # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8 # 8
                    D0 = D
                    while D < np.min([config.M*D0, heuristic]):
        
                            
        # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9 # 9
                        
                        new_initial_conditions = np.zeros((OCP.n_states, 0))
                        for n in range(config.Nc):
                            np.random.seed(seed+3+n)
                            X0 = OCP.sample_X0(1).reshape(OCP.n_states, -1)
                            new_initial_conditions = np.hstack((X0.reshape(-1,1), new_initial_conditions))
        
        # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10 # 10
        
                        l1_norms = []
                        for col in range(new_initial_conditions.shape[1]):
                            transformed_column = np.apply_along_axis(controller.eval_dVdX, 0, new_initial_conditions[:, col])
                            l1_norm = np.sum(np.abs(transformed_column))
                            l1_norms.append(l1_norm)
                                
        # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11 # 11
            
                        sorted_indices = np.argsort(l1_norms)[::-1]
                        X0_pool_train = new_initial_conditions[:, sorted_indices[::2]]
                        X0_pool_val = new_initial_conditions[:, sorted_indices[1::2]]
                        trajectory_id = 0
        
        # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12 # 12
                    
                        if architecture=='GradientQRnet':
                            new_data, new_ic = generate.generate(
                                OCP, config, 1, X0_pool_train[:, trajectory_id], controller,
                                resolve_failed=True
                            )
                            new_val_data, new_val_ic = generate.generate(
                                OCP, config, 1, X0_pool_val[:, trajectory_id], controller,
                                resolve_failed=True
                            )
                        else:
                            new_data, new_ic = generate.generate(
                                OCP, config, 1, X0_pool_train[:, trajectory_id], OCP.LQR,
                                resolve_failed=True
                            )
                            new_val_data, new_ic = generate.generate(
                                OCP, config, 1, X0_pool_val[:, trajectory_id], OCP.LQR,
                                resolve_failed=True
                            )
                        for key in ['t', 'X', 'dVdX', 'V', 'U']:
                        # for key in data.keys():
                            if key in data:
                                data[key] = np.hstack((data[key], new_data[key]))
                                data['n_trajectories'] = data['n_trajectories'] + new_data['n_trajectories']
                        for key in ['t', 'X', 'dVdX', 'V', 'U']:  
                            if key in val_data:
                                val_data[key] = np.hstack((val_data[key], new_val_data[key]))
                                val_data['n_trajectories'] = val_data['n_trajectories'] + new_val_data['n_trajectories']




                        _, D = controller.heuristic(data, config.C)
                        trajectory_id+=1
                        norm = np.sum(np.abs(X0_pool[:, trajectory_id]))
                        initial_condition_norms.append(norm)
        
                else:
                    pass
                



    # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13 # 13
    # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14 # 14
    # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15 # 15
    # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16 # 16

    train_time = time.time() - start_time

    print('\nTraining time: %.0f sec\n' % train_time)

    # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17 # 17






    # # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18 # 18

    # Record the training time and evaluate the errors.

    # Record the training time and evaluate the errors.


    train_errs, _, _ = utilities.eval_errors(controller, data, batch_size)
    val_errs, _, _ = utilities.eval_errors(controller, val_data, batch_size)

    # Find fixed points and save the results.
    X_star, X_star_deviation, F_star, Jac, eigs, max_eig = utilities.find_fixed_point(
        OCP, controller, config.fp_tol
    )
    error_dict = {
        'train_time': train_time,
        'fixed_point_deviation': X_star_deviation,
        'max_eig_real': max_eig.real,
        'max_eig_imag': max_eig.imag,
        **dict((key + '_val', val) for key, val in val_errs.items())
    }
    for config_key in [
            'n_hidden', 'n_neurons',
            'gradient_loss_weight',
            'n_trajectories_test', 'n_trajectories_train_fixed',
            'n_epochs', 'optimizer', 'optimizer_opts'
        ]:
        error_dict[config_key] = getattr(config, config_key)

    error_dict = {}

    controller.save(model_dir, error_dict, seed, sampling, architecture)

    # Save the final training data.
    save_path = os.path.join(data_dir, 'train'+sampling+'.mat')
    scipy.io.savemat(save_path, data)
    
    
    # Save the final validation data.
    save_path = os.path.join(data_dir, 'val'+sampling+'.mat')
    scipy.io.savemat(save_path, val_data)
    


    
    if sampling=='adaptive':
        data = {'IC': X0_pool_val}
        save_path = os.path.join(data_dir, 'X0' + sampling + '_' + architecture + '.mat')
        scipy.io.savemat(save_path, data)


    #test_errors = [test_errs[key] for key in np.sort(list(test_errs.keys()))]
    loss_train = controller.sess.run((controller.loss), feed_dict)
    model_stats = utilities.summary(sampling, n_initial_data, n_data_available, train_time, loss_train, val_errs_U, val_errs_dVdX)
    
    return train_errors_U_list, train_errors_dVdX_list, val_errors_U_list, val_errors_dVdX_list, model_stats, max_eig



