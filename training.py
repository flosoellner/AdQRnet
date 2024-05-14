import matplotlib.pyplot as plt

def plot(train_errors_dVdX_fix, train_errors_dVdX_ad, train_errors_U_ad, train_errors_U_fix, val_errors_dVdX_fix, val_errors_dVdX_ad, val_errors_U_fix, val_errors_U_ad):
    
    epochs = range(1, len(train_errors_U_ad) + 1)
    
    plt.figure(figsize=(8, 2))
    
    # Using Oxford Blue color '#002147', and different line styles
    plt.plot(epochs, train_errors_dVdX_fix, color='#002147', linestyle='--', linewidth=1, label='fixed')
    plt.plot(epochs, train_errors_dVdX_ad, color='#002147', linestyle='-', linewidth=1, label='adaptive')
    
    
    plt.xlabel('Epoch')
    plt.ylabel('$RML^2_{V_x,train}$')
    plt.yscale('log')
    
    # Customizing the legend to make it more delicate
    plt.legend(frameon=False)
    
    # Setting a slightly visible grey background for the axes
    ax = plt.gca()  # Get current axes
    ax.set_facecolor('#f8f8f8')  # Light grey background
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#606060')  # Lighter spines
    ax.spines['left'].set_color('#606060')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.show()
    
    
    
    
    plt.figure(figsize=(8, 2))
    
    # Using Oxford Blue color '#002147', and different line styles
    plt.plot(epochs, train_errors_U_ad, color='#002147', linestyle='-', linewidth=1, label='adaptive')
    plt.plot(epochs, train_errors_U_fix, color='#002147', linestyle='--', linewidth=1, label='fixed')
    
    
    
    
    
    plt.xlabel('Epoch')
    plt.ylabel('$RML^2_{u,train}$')
    plt.yscale('log')
    
    # Customizing the legend to make it more delicate
    plt.legend(frameon=False)
    
    # Setting a slightly visible grey background for the axes
    ax = plt.gca()  # Get current axes
    ax.set_facecolor('#f8f8f8')  # Light grey background
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#606060')  # Lighter spines
    ax.spines['left'].set_color('#606060')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.show()
    
    
    plt.figure(figsize=(8, 2))
    
    # Using Oxford Blue color '#002147', and different line styles
    plt.plot(epochs, val_errors_dVdX_fix, color='#00883A', linestyle='--', linewidth=1, label='fixed')
    plt.plot(epochs, val_errors_dVdX_ad, color='#00883A', linestyle='-', linewidth=1, label='adaptive')
    
    
    
    plt.xlabel('Epoch')
    plt.ylabel('$RML^2_{V_x,val}$')
    plt.yscale('log')
    
    # Customizing the legend to make it more delicate
    plt.legend(frameon=False)
    
    # Setting a slightly visible grey background for the axes
    ax = plt.gca()  # Get current axes
    ax.set_facecolor('#f8f8f8')  # Light grey background
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#606060')  # Lighter spines
    ax.spines['left'].set_color('#606060')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.show()
    
    
    
    
    
    
    
    plt.figure(figsize=(8, 2))
    
    # Using Oxford Blue color '#002147', and different line styles
    plt.plot(epochs, val_errors_U_fix, color='#00883A', linestyle='--', linewidth=1, label='fixed')
    plt.plot(epochs, val_errors_U_ad, color='#00883A', linestyle='-', linewidth=1, label='adaptive')
    
    
    
    plt.xlabel('Epoch')
    plt.ylabel('$RML^2_{u,val}$')
    plt.yscale('log')
    
    # Customizing the legend to make it more delicate
    plt.legend(frameon=False)
    
    # Setting a slightly visible grey background for the axes
    ax = plt.gca()  # Get current axes
    ax.set_facecolor('#f8f8f8')  # Light grey background
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#606060')  # Lighter spines
    ax.spines['left'].set_color('#606060')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.show()
    
    