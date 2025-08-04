import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import argparse

def load_and_visualize_data(filepath, error_type):
    """
    Load data and create contour plot with epsilon on z-axis, timestep on y-axis, and MSE as color.
    """
    # Load the data (assuming CSV format, adjust as needed)
    try:
        data = pd.read_csv(filepath)
    except:
        # Try other common formats
        try:
            data = pd.read_json(filepath)
        except:
            data = np.load(filepath, allow_pickle=True)
            if isinstance(data, dict):
                data = pd.DataFrame(data)
    
    # Filter out rows where epsilon is 0
    if 'epsilon' in data.columns:
        data = data[data['epsilon'] != 0]
    else:
        data = data[data.iloc[:, 0] != 0]
    
    # Extract the three dimensions
    # Adjust column names as needed based on your data format
    epsilon = data['epsilon'].values if 'epsilon' in data.columns else data.iloc[:, 0].values
    timestep = data['timestep'].values if 'timestep' in data.columns else data.iloc[:, 1].values  
    mse = data[error_type].values if error_type in data.columns else data.iloc[:, 2].values
    
    # Create grid for contour plot
    eps_min, eps_max = epsilon.min(), epsilon.max()
    time_min, time_max = timestep.min(), timestep.max()
    
    eps_grid = np.linspace(eps_min, eps_max, 100)
    time_grid = np.linspace(time_min, time_max, 100)
    eps_mesh, time_mesh = np.meshgrid(eps_grid, time_grid)
    
    # Interpolate MSE values onto the grid
    mse_grid = griddata((epsilon, timestep), mse, (eps_mesh, time_mesh), method='cubic', fill_value=np.nan)
    
    # Create the contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(eps_mesh, time_mesh, mse_grid, levels=100, cmap='viridis')
    plt.colorbar(contour, label=error_type)
    
    # Add contour lines for better readability
    plt.contour(eps_mesh, time_mesh, mse_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    plt.xlabel('Epsilon')
    plt.ylabel('Timestep') 
    plt.title(f'{error_type} Analysis: Epsilon vs Timestep')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis/analysis-images/map-{error_type}.png")

if __name__ == "__main__":
    load_and_visualize_data("/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis/data/average/30_2025-07-22_10-56-24.csv", 'mse')
    load_and_visualize_data("/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis/data/average/30_2025-07-22_10-56-24.csv", 'cosine_similarity')