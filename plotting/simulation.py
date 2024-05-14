import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat



def plot(plot_path, sim_data, params):
    # Example data unpacking, adjust according to your .mat file structure
    X_NN = sim_data['X_NN']
    t_NN = sim_data['t_NN'].flatten()
    t_index = np.where(t_NN <= 35)[0]
    t_NN = t_NN[t_index]
    xi = np.hstack([1, params['xi'].flatten(), -1])
    xi_grid, t_grid = np.meshgrid(xi, t_NN)  # Creating grids for 3D plotting
    X_padded = np.vstack([np.zeros((1, X_NN.shape[1])), X_NN, np.zeros((1, X_NN.shape[1]))])
    
    X_padded = X_padded[:, t_index]
    
    # Create a single 3D plot
    fig = plt.figure(figsize=(8, 8))  # Wide figure to accommodate z-label
    ax = fig.add_subplot(((111)), projection='3d')  # Single plot
    
    # Plot the surface
    surf = ax.plot_surface(xi_grid, t_grid, X_padded.T, cmap='winter', edgecolor='none')
    
    # Set labels with increased padding
    ax.set_xlabel('\u03BE', fontsize=12, labelpad=5)
    ax.set_ylabel('t', fontsize=12, labelpad=5)
    ax.set_zlabel('X', fontsize=12, labelpad=2)
    
    # Set title and adjust view for better label visibility
    
    ax.view_init(elev=20, azim=20)  # Adjust elevation angle
    
    
    # Show the plot
    plt.show() 

    plt.savefig(plot_path, dpi=300)

    


