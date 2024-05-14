import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter



def plot(plot_path, data): 
    # Prepare data for plotting
    max_eig_fix_qrnet = np.array(data['max_eig_fix_qrnet'], dtype=float)
    max_eig_ad_qrnet = np.array(data['max_eig_ad_qrnet'], dtype=float)
    max_eig_fix_nn = np.array(data['max_eig_fix_nn'], dtype=float)
    max_eig_ad_nn = np.array(data['max_eig_ad_nn'], dtype=float)
    
    # Extract validation errors and eigenvalues
    val_errors_U_fix_qrnet = max_eig_fix_qrnet[:, 1]
    max_eig_fix_qrnet = np.real(max_eig_fix_qrnet[:, 0])
    val_errors_U_ad_qrnet = max_eig_ad_qrnet[:, 1]
    max_eig_ad_qrnet = np.real(max_eig_ad_qrnet[:, 0])
    val_errors_U_fix_nn = max_eig_fix_nn[:, 1]
    max_eig_fix_nn = np.real(max_eig_fix_nn[:, 0])
    val_errors_U_ad_nn = max_eig_ad_nn[:, 1]
    max_eig_ad_nn = np.real(max_eig_ad_nn[:, 0])
    
    plt.figure(figsize=(4, 2))
    ax = plt.subplot(1, 1, 1)
    
    # Setting the light grey color you preferred
    ax.set_facecolor('#f8f8f8')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#606060')
    ax.spines['left'].set_color('#606060')
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Scatter plots
    plt.scatter(val_errors_U_fix_qrnet, max_eig_fix_qrnet, color='#002147', marker='^', label='Fixed $\lambda$-QRnet')
    plt.scatter(val_errors_U_ad_qrnet, max_eig_ad_qrnet, color='#002147', marker='x', label='Adaptive $\lambda$-QRnet')
    plt.scatter(val_errors_U_fix_nn, max_eig_fix_nn, color='#00883A', marker='^', label='Fixed NN')
    plt.scatter(val_errors_U_ad_nn, max_eig_ad_nn, color='#00883A', marker='x', label='Adaptive NN')
    
    # Labels and legend
    plt.xlabel('$RML^2_{u,val}$', fontsize=18)
    plt.ylabel('Maximum Eigenvalue of $\mathcal{A}$', fontsize=18)
    plt.legend(fontsize='large', markerscale=1.5)
    
    # Adding a very fine grid
    ax.grid(which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    
    # Scalar Formatter for axes
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    
    plt.savefig(plot_path, dpi=300)
    
    plt.show()