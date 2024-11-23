import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load eigenvectors from the file
e_vec = np.loadtxt('eigenvectors.txt')

# Define the grid size (M, N), assuming you know it
M, N = 100, 100  # Example dimensions, change as needed

# Plot eigenvectors for all states (from input_data[0] to input_data[1])
for i in range(int(9)):
    figi, axi = plt.subplots(1, 1)
    
    # Reshape the eigenvector for plotting
    eig_vector = np.abs(e_vec[:, i].reshape(M, N)) ** 2  # Probability density
    eig_vector = np.transpose(eig_vector) 
    plot = plt.imshow(np.transpose(eig_vector), cmap='magma', interpolation='gaussian')

    plt.setp(axi, xticks=[], yticks=[])
    divider = make_axes_locatable(axi)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = figi.colorbar(plot, ax=axi, extend='both', cax=cax)
    cbar.minorticks_on()
    cbar.ax.tick_params(labelsize=5, pad=0.1)
    
    # Set title based on eigenstate
    if i == 0:
        axi.set_title('The ground excited state',fontsize=12)
        
    elif i == 1:
        axi.set_title('The 1$^{st}$ excited state',fontsize=12)
        
    elif i == 2:
        axi.set_title('The 2$^{nd}$ excited state',fontsize=12)
        
    elif i == 3:
        axi.set_title('The 3$^{rd}$ excited state',fontsize=12)
        
    else:
        axi.set_title(str(i)+'$^{th}$ excited state',fontsize=12)   
    plt.savefig( str(i)+'.pdf') 
