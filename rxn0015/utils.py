import torch
import numpy as np
import matplotlib.pyplot as plt
from rxn0015.potential_energy_function import PotentialEnergyFunction
from saddlefinder import NNSaddleFinder

def plot_potential_heatmap(pot_func, fig=None, ax=None, 
                          num_points=200, cmap='viridis', add_contour=True):
    """
    Custom function to plot potential energy heatmap using matplotlib directly.
    
    Args:
        pot_func: PotentialEnergyFunction object
        fig: matplotlib figure (optional)
        ax: matplotlib axis (optional)
        num_points: resolution of the heatmap
        cmap: colormap name
        add_contour: whether to add contour lines
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a fine grid for visualization
    oh_min, oh_max = pot_func.oh_grid.min(), pot_func.oh_grid.max()
    coh_min, coh_max = pot_func.coh_grid.min(), pot_func.coh_grid.max()
    
    oh_fine = np.linspace(oh_min, oh_max, num_points)
    coh_fine = np.linspace(coh_min, coh_max, num_points)
    OH, COH = np.meshgrid(oh_fine, coh_fine)
    
    # Evaluate the function on the grid
    Z = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            Z[i, j] = pot_func(oh_fine[j], coh_fine[i])
    
    # Plot the heatmap
    contourf = ax.contourf(OH, COH, Z, levels=50, cmap=cmap, alpha=0.9)
    
    # Add contour lines if requested
    if add_contour:
        contour_lines = ax.contour(OH, COH, Z, levels=15, colors='white', 
                                  alpha=0.5, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%0.1f')
    
    # Add colorbar
    cbar = fig.colorbar(contourf, ax=ax)
    cbar.set_label('Potential Energy (eV)')
    
    # Add labels
    ax.set_xlabel('OH Bond Length (Å)', fontsize=12)
    ax.set_ylabel('COH Angle (degrees)', fontsize=12)
    
    return fig, ax

def run_saddle_search(pot_func, initial_point, iterations=100, step_size=0.01, momentum=0.8):
    """
    Run saddle point search and return the results.
    
    Args:
        pot_func: PotentialEnergyFunction object
        initial_point: Starting point as tensor or list/tuple
        iterations: Number of iterations
        step_size: Step size for optimization
        momentum: Momentum parameter
        
    Returns:
        dict: Results with keys 'path', 'final_point', 'finder'
    """
    # Convert initial point to tensor if needed
    if not isinstance(initial_point, torch.Tensor):
        initial_point = torch.tensor(initial_point, dtype=torch.float32)
    
    # Create the saddle finder
    finder = NNSaddleFinder(
        potential=pot_func.torch_forward,
        initial_x=initial_point,
        hessian_fn=pot_func.compute_hessian,
        step_size=step_size,
        momentum=momentum,
        device='cpu',
        eigsolver='AD'
    )
    
    # Run saddle search and collect the path
    path = [initial_point.clone().detach().numpy()]
    
    # Find the saddle point
    finder.find_saddle(iterations=iterations, verbose=False)
    
    # Collect the path
    for state in finder.path:
        path.append(state.clone().detach().numpy())
    
    # Get the final point
    final_point = finder.x.clone().detach().numpy()
    
    return {
        'path': np.array(path),
        'final_point': final_point,
        'finder': finder
    }

def plot_saddle_search_results(pot_func, results, fig=None, ax=None, show_path=True, 
                              cmap='viridis', add_contour=True, title=None):
    """
    Plot saddle search results on a heatmap.
    
    Args:
        pot_func: PotentialEnergyFunction object
        results: Results dict from run_saddle_search
        fig, ax: Optional figure and axis objects
        show_path: Whether to show the search path
        cmap: Colormap to use
        add_contour: Whether to add contour lines
        title: Optional custom title
        
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    # Create figure if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the heatmap
    fig, ax = plot_potential_heatmap(
        pot_func=pot_func, 
        fig=fig, 
        ax=ax, 
        cmap=cmap, 
        add_contour=add_contour
    )
    
    path = results['path']
    final_point = results['final_point']
    
    # Plot the saddle search path
    if show_path and len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], 'w-', linewidth=2, alpha=0.7)
        ax.plot(path[:, 0], path[:, 1], 'ko', markersize=3, alpha=0.7)
    
    # Mark the starting and final points
    ax.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
    ax.plot(final_point[0], final_point[1], 'ro', markersize=10, label='End')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add title
    if title is None:
        iterations = len(path) - 1
        ax.set_title(f'Saddle Point Search (iterations={iterations})')
    else:
        ax.set_title(title)
    
    return fig, ax
