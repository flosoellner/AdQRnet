import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter



def plot(plot_path, QRnet_adaptive, QRnet_fixed, NN_adaptive):
    QRnet_adaptive_costs = QRnet_adaptive['NN_costs'].squeeze()
    QRnet_fixed_costs = QRnet_fixed['NN_costs'].squeeze()
    NN_adaptive_costs = NN_adaptive['NN_costs'].squeeze()
    
    QRnet_adaptive_times = QRnet_adaptive['NN_final_times'].squeeze()
    QRnet_fixed_times = QRnet_fixed['NN_final_times'].squeeze()
    NN_adaptive_times = NN_adaptive['NN_final_times'].squeeze()
    
    plt.figure(figsize=(12, 12))  # Increased figure size for better readability
    
    # QRnet adaptive cost vs QRnet fixed cost
    plt.subplot(2, 2, 1)
    plt.scatter(QRnet_adaptive_costs, QRnet_fixed_costs, color='#002147', s=10)  # Increased marker size
    plt.axline((0.3, 0.3), (1, 1), linewidth=1, color='black')  # Thicker axline
    
    plt.ylabel('QRnet Fixed', fontsize=18)  # Increased font size
    plt.title('Final Costs', fontsize=24)  # Increased title font size
    
    # QRnet adaptive time vs QRnet fixed time
    plt.subplot(2, 2, 2)
    plt.scatter(QRnet_adaptive_times, QRnet_fixed_times, color='#002147', s=10)  # Increased marker size
    plt.axline((8, 8), (10, 10), linewidth=1, color='black')  # Thicker axline
    
    
    plt.title('Final Times', fontsize=24)  # Increased title font size
    
    # QRnet adaptive time vs NN adaptive time
    plt.subplot(2, 2, 3)
    plt.scatter(QRnet_adaptive_costs, NN_adaptive_costs, color='#002147', s=10)  # Increased marker size
    plt.axline((0, 0), (1, 1), linewidth=1, color='black')  # Thicker axline
    plt.xlabel('QRnet Adaptive', fontsize=18)  # Increased font size
    plt.ylabel('NN Adaptive', fontsize=18)  # Increased font size
    
    
    # QRnet adaptive cost vs NN adaptive cost
    plt.subplot(2, 2, 4)
    plt.scatter(QRnet_adaptive_times, NN_adaptive_times, color='#002147', s=10)  # Increased marker size
    plt.axline((8, 8), (10, 10), linewidth=1, color='black')  # Thicker axline
    plt.xlabel('QRnet Adaptive', fontsize=18)  # Increased font size
    
    
    
    # Apply ScalarFormatter for real number formatting and adjust aesthetics uniformly
    for i in range(1, 5):
        ax = plt.subplot(2, 2, i)
        ax.set_facecolor('#f8f8f8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#606060')
        ax.spines['left'].set_color('#606060')
        ax.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()

    
    plt.savefig(plot_path, dpi=300)
    
    plt.show()

