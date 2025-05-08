import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from simple_pot import SimplePotential
from saddlefinder import NNSaddleFinder

def plot_potential_contour(potential_fn, x_range=(-2, 2), y_range=(-2, 2), resolution=200):
    """
    Plot a 2D contour map of the potential energy landscape.
    
    Args:
        potential_fn: The potential energy function to plot
        x_range: Tuple of (min, max) for x-axis
        y_range: Tuple of (min, max) for y-axis
        resolution: Number of points along each axis
    """
    # Create a grid of x, y points
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate potential at each point
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
            with torch.no_grad():
                Z[i, j] = potential_fn(point).item()
    
    # Create the 2D contour plot
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Plot filled contours with a colormap
    contourf = ax.contourf(X, Y, Z, 50, cmap=cm.viridis, alpha=0.8)
    
    # Add contour lines
    contour = ax.contour(X, Y, Z, 15, colors='white', linewidths=0.5, alpha=0.7)
    
    # Add contour labels
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    # Add a color bar
    cbar = fig.colorbar(contourf, ax=ax)
    cbar.set_label('Potential Energy')
    
    # Mark critical points (minima at x = ±1, y = 0)
    ax.plot(1, 0, 'ro', markersize=8, label='Minimum')
    ax.plot(-1, 0, 'ro', markersize=8)
    ax.plot(0, 0, 'yo', markersize=8, label='Saddle Point')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Potential Energy Contours')
    ax.legend()
    
    plt.tight_layout()
    return fig, ax, X, Y, Z

def plot_saddle_search_path(potential_fn, search_path, x_range=(-2, 2), y_range=(-2, 2), resolution=200):
    """
    Plot the saddle point search path with color gradient to show order.
    
    Args:
        potential_fn: The potential energy function
        search_path: List of points from the saddle point search
        x_range, y_range: Ranges for the plot
        resolution: Number of points along each axis
    """
    # First create the contour plot
    fig, ax, X, Y, Z = plot_potential_contour(potential_fn, x_range, y_range, resolution)
    
    # Extract x and y coordinates from the search path
    path_x = [point[0].item() for point in search_path]
    path_y = [point[1].item() for point in search_path]
    
    # Create a color map for the path points
    norm = Normalize(vmin=0, vmax=len(path_x)-1)
    
    # Plot the path with a color gradient
    for i in range(len(path_x)-1):
        ax.plot([path_x[i], path_x[i+1]], [path_y[i], path_y[i+1]], '-', 
                color=plt.cm.cool(norm(i)), linewidth=1.5)
    
    # Plot the points with a color gradient
    scatter = ax.scatter(path_x, path_y, c=range(len(path_x)), cmap='cool', 
                         s=30, zorder=5, edgecolors='black', linewidths=0.5)
    
    # Add a colorbar for the path
    # path_cbar = fig.colorbar(scatter, ax=ax, location='right', pad=0.01)
    # path_cbar.set_label('Iteration')
    
    # Mark the starting and ending points
    ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start', zorder=10)
    ax.plot(path_x[-1], path_y[-1], 'mo', markersize=10, label='End', zorder=10)
    
    # Update the title and legend
    ax.set_title('Saddle Point Search Path')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig, ax

def plot_eigenvalues_and_gradient(eigenvalues_history, gradient_norms):
    """
    Plot the eigenvalues and gradient norm over iterations.
    
    Args:
        eigenvalues_history: List of eigenvalue tensors for each iteration
        gradient_norms: List of gradient norm values for each iteration
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    
    # Plot eigenvalues
    iterations = range(len(eigenvalues_history))
    
    # Extract the first two eigenvalues from each iteration
    eig1 = [eig[0].item() for eig in eigenvalues_history]
    eig2 = [eig[1].item() for eig in eigenvalues_history]
    
    ax1.plot(iterations, eig1, 'b-', label='First eigenvalue')
    ax1.plot(iterations, eig2, 'r-', label='Second eigenvalue')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Eigenvalues over Iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot gradient norm
    ax2.plot(iterations, gradient_norms, 'g-')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm over Iterations')
    ax2.set_yscale('log')  # Log scale for better visualization
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def run_and_plot_saddle_search(potential_fn, initial_x, iterations=50, saddle_index=1, 
                              step_size=0.01, momentum=0.8, save_path=None):
    """
    Run the saddle finder and plot the search path.
    
    Args:
        potential_fn: The potential energy function
        initial_x: Initial guess for the saddle point
        iterations: Number of iterations to run
        saddle_index, step_size, momentum: Parameters for the saddle finder
        save_path: Path to save the plot, if None, the plot is not saved
    """
    # Create the saddle finder
    saddle_finder = NNSaddleFinder(
        potential=potential_fn,
        initial_x=initial_x,
        saddle_index=saddle_index,
        step_size=step_size,
        momentum=momentum,
    )
    
    # Collect all points during the search
    search_path = [initial_x.detach().clone()]
    eigenvalues_history = []
    gradient_norms = []
    
    # Run the search
    for i in range(iterations):
        # Compute eigenvalues and gradient norm before step
        with torch.enable_grad():
            hessian = saddle_finder.hessian(saddle_finder.x)
            eigvals, _ = saddle_finder.eigen_vals_vecs(hessian)
            grad = saddle_finder.grad_potential(saddle_finder.x)
            grad_norm = torch.norm(grad).item()
        
        # Store values
        eigenvalues_history.append(eigvals.detach().clone())
        gradient_norms.append(grad_norm)
        
        # Take a step
        saddle_finder.step()
        search_path.append(saddle_finder.x.detach().clone())
    
    # Plot the search path
    path_fig, path_ax = plot_saddle_search_path(potential_fn, search_path)
    
    # Plot eigenvalues and gradient norm
    eig_grad_fig, eig_grad_axes = plot_eigenvalues_and_gradient(eigenvalues_history, gradient_norms)
    
    # Save the plots if requested
    if save_path:
        path_fig.savefig(f"{save_path}_path.png", dpi=300, bbox_inches='tight')
        eig_grad_fig.savefig(f"{save_path}_eig_grad.png", dpi=300, bbox_inches='tight')
    
    return search_path, eigenvalues_history, gradient_norms, (path_fig, eig_grad_fig)

def plot_convergence_analysis(potential_fn, x1_range_end=1/np.sqrt(3) + 0.15, num_points=50, 
                            max_iterations=1000, convergence_threshold=1e-7, y2=3.0):
    """
    Analyze and plot the number of iterations needed for convergence as a function of initial x₁.
    
    Args:
        potential_fn: The potential energy function
        x1_range_end: End point for x₁ range (start is 0)
        num_points: Number of initial points to test
        max_iterations: Maximum number of iterations before declaring non-convergence
        convergence_threshold: Gradient norm threshold for convergence
        y2: Fixed y-coordinate for initial points
    """
    # Create array of initial x₁ values
    x1_values = np.linspace(0, x1_range_end, num_points)
    iterations_to_converge = []
    
    # Test each initial point
    for x1 in x1_values:
        initial_x = torch.tensor([x1, y2], dtype=torch.float32, requires_grad=True)
        
        # Create saddle finder
        saddle_finder = NNSaddleFinder(
            potential=potential_fn,
            initial_x=initial_x,
            saddle_index=1,
            step_size=0.01,
            momentum=0.8,
        )
        
        # Run until convergence or max iterations
        converged = False
        for i in range(max_iterations):
            with torch.enable_grad():
                grad = saddle_finder.grad_potential(saddle_finder.x)
                grad_norm = torch.norm(grad).item()
                
                if grad_norm < convergence_threshold:
                    iterations_to_converge.append(i)
                    converged = True
                    break
                
            saddle_finder.step()
        
        if not converged:
            iterations_to_converge.append(max_iterations)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Plot iterations vs x₁
    ax.plot(x1_values, iterations_to_converge, 'b-', linewidth=2)
    ax.set_yscale('log')
    
    # Add vertical line at x₁ = 1/sqrt(3)
    critical_x1 = 1/np.sqrt(3)
    ax.axvline(x=critical_x1, color='r', linestyle='--', 
               label=f'x₁ = 1/√3 ≈ {critical_x1:.3f}')
    
    # Customize the plot
    ax.set_xlabel('Initial x₁')
    ax.set_ylabel('Iterations to Convergence')
    ax.set_title('Convergence Analysis')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    # ax.set_ylim(1, 100)
    ax.legend()
    
    # Add text box with parameters
    param_text = f'Convergence threshold: {convergence_threshold}\n' \
                 f'Max iterations: {max_iterations}\n' \
                 f'Initial x_2: {y2}'
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"convergence_analysis_{y2}.png", dpi=300, bbox_inches='tight')
    return fig, ax, x1_values, iterations_to_converge

if __name__ == "__main__":
    # Create the potential function
    potential = SimplePotential()
    
    # Set the initial point with some randomness
    torch.manual_seed(6)  # For reproducibility
    # epsilon = 0.12
    # initial_x = torch.tensor([1/np.sqrt(3) + epsilon, 1.5], dtype=torch.float32, requires_grad=True) #+ 2*torch.randn(2)
    
    # # Run the saddle finder and plot the search path
    # search_path, eigenvalues_history, gradient_norms, figs = run_and_plot_saddle_search(
    #     potential_fn=potential,
    #     initial_x=initial_x,
    #     iterations=100,
    #     saddle_index=1,
    #     step_size=0.01,
    #     momentum=0.8,
    #     save_path="2D_potential_search"
    # )
    
    # # Print the final point
    # print(f"Final saddle point: {search_path[-1]}")
    
    # Run convergence analysis
    conv_fig, conv_ax, x1_vals, iter_counts = plot_convergence_analysis(
        potential_fn=potential,
        convergence_threshold=1e-4,
        max_iterations=1000
    )
    plt.show() 