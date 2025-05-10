#!/usr/bin/env python
"""
Script to demonstrate the use of the smooth potential energy function.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from potential_energy_function import PotentialEnergyFunction, create_potential_function_from_heatmap
from rxn0015.potential_nequip import plot_potential_heatmap

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

def generate_and_save_function(grid_size=30, smooth_factor=0.1):
    """Generate the heatmap data and create a smooth potential energy function."""
    print(f"Generating heatmap data with grid size {grid_size}x{grid_size}...")
    heatmap_data = plot_potential_heatmap(
        grid_size=grid_size, 
        batch_size=5,
        output_file="figures/potential_energy_surface.png"
    )
    
    print(f"Creating smooth potential energy function (smooth factor: {smooth_factor})...")
    pot_func = create_potential_function_from_heatmap(
        heatmap_data, 
        smooth_factor=smooth_factor,
        method='spline'
    )
    
    # Save the function
    pot_func.save('models/potential_energy_function.pkl')
    print("Function saved to models/potential_energy_function.pkl")
    
    return pot_func

def load_function():
    """Load a previously saved potential energy function."""
    try:
        pot_func = PotentialEnergyFunction.load('models/potential_energy_function.pkl')
        print("Loaded existing potential energy function")
        return pot_func
    except FileNotFoundError:
        print("No saved function found. Generating a new one...")
        return generate_and_save_function()

def test_energy_calculation(pot_func, oh_length=1.0, coh_angle=120.0):
    """Test energy calculation at a specific point."""
    energy = pot_func(oh_length, coh_angle)
    print(f"Potential energy at OH={oh_length}Å, COH={coh_angle}°: {energy:.4f} eV")
    return energy

def plot_2d_slice(pot_func, slice_type='oh', fixed_value=None, num_points=200):
    """
    Plot a 1D slice through the potential energy surface.
    
    Args:
        pot_func: The potential energy function
        slice_type: 'oh' for OH bond length slice, 'coh' for COH angle slice
        fixed_value: The fixed value for the other coordinate
        num_points: Number of points to sample
    """
    # Determine min and max values for the variable coordinate
    if slice_type == 'oh':
        var_min = pot_func.oh_grid.min()
        var_max = pot_func.oh_grid.max()
        fixed_name = 'COH angle'
        var_name = 'OH bond length'
        var_unit = 'Å'
        fixed_unit = '°'
        
        # Use middle value if not specified
        if fixed_value is None:
            fixed_value = np.median(pot_func.coh_grid)
        
        # Generate coordinate values
        var_vals = np.linspace(var_min, var_max, num_points)
        energies = [pot_func(oh, fixed_value) for oh in var_vals]
        
    else:  # slice_type == 'coh'
        var_min = pot_func.coh_grid.min()
        var_max = pot_func.coh_grid.max()
        fixed_name = 'OH bond length'
        var_name = 'COH angle'
        var_unit = '°'
        fixed_unit = 'Å'
        
        # Use middle value if not specified
        if fixed_value is None:
            fixed_value = np.median(pot_func.oh_grid)
        
        # Generate coordinate values
        var_vals = np.linspace(var_min, var_max, num_points)
        energies = [pot_func(fixed_value, coh) for coh in var_vals]
    
    # Plot the slice
    plt.figure(figsize=(10, 6))
    plt.plot(var_vals, energies, 'b-', linewidth=2)
    plt.xlabel(f'{var_name} ({var_unit})')
    plt.ylabel('Potential Energy (eV)')
    plt.title(f'Potential Energy vs {var_name} at {fixed_name}={fixed_value:.2f}{fixed_unit}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    filename = f'figures/potential_slice_{slice_type}_{fixed_value:.2f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Slice plot saved to {filename}")
    
    plt.show()

def demo_gradient_calculation(pot_func, oh_length=1.0, coh_angle=120.0, delta=1e-5):
    """
    Demonstrate numerical gradient calculation of the potential energy function.
    
    Args:
        pot_func: The potential energy function
        oh_length: OH bond length value
        coh_angle: COH angle value
        delta: Small increment for numerical differentiation
    """
    # Calculate numerical gradient
    e_center = pot_func(oh_length, coh_angle)
    e_oh_plus = pot_func(oh_length + delta, coh_angle)
    e_oh_minus = pot_func(oh_length - delta, coh_angle)
    e_coh_plus = pot_func(oh_length, coh_angle + delta)
    e_coh_minus = pot_func(oh_length, coh_angle - delta)
    
    # Central difference approximation
    grad_oh = (e_oh_plus - e_oh_minus) / (2 * delta)
    grad_coh = (e_coh_plus - e_coh_minus) / (2 * delta)
    
    print("\nNumerical Gradient Calculation:")
    print(f"At OH={oh_length}Å, COH={coh_angle}°:")
    print(f"  Energy: {e_center:.6f} eV")
    print(f"  dE/dOH: {grad_oh:.6f} eV/Å")
    print(f"  dE/dCOH: {grad_coh:.6f} eV/°")
    
    # Visualize the gradient as a vector field
    plot_gradient_field(pot_func, num_points=15)

def plot_gradient_field(pot_func, num_points=15):
    """
    Plot the gradient vector field of the potential energy surface.
    
    Args:
        pot_func: The potential energy function
        num_points: Number of points along each dimension for the vector field
    """
    # Create a grid for visualization
    oh_vals = np.linspace(pot_func.oh_grid.min(), pot_func.oh_grid.max(), num_points)
    coh_vals = np.linspace(pot_func.coh_grid.min(), pot_func.coh_grid.max(), num_points)
    OH, COH = np.meshgrid(oh_vals, coh_vals)
    
    # Calculate energy on the grid
    Z = np.zeros((num_points, num_points))
    grad_oh = np.zeros((num_points, num_points))
    grad_coh = np.zeros((num_points, num_points))
    
    delta = 1e-5  # Small increment for numerical differentiation
    
    # Calculate energy and numerical gradients
    for i in range(num_points):
        for j in range(num_points):
            oh = oh_vals[j]
            coh = coh_vals[i]
            
            # Energy at center point
            Z[i, j] = pot_func(oh, coh)
            
            # Calculate numerical gradients
            e_oh_plus = pot_func(oh + delta, coh)
            e_oh_minus = pot_func(oh - delta, coh)
            e_coh_plus = pot_func(oh, coh + delta)
            e_coh_minus = pot_func(oh, coh - delta)
            
            grad_oh[i, j] = (e_oh_plus - e_oh_minus) / (2 * delta)
            grad_coh[i, j] = (e_coh_plus - e_coh_minus) / (2 * delta)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot the energy contours
    contour = plt.contourf(OH, COH, Z, levels=50, cmap='viridis', alpha=0.8)
    
    # Scale the gradients for better visualization
    # Normalize the gradient vectors for better visualization
    grad_magnitude = np.sqrt(grad_oh**2 + grad_coh**2)
    max_mag = np.percentile(grad_magnitude, 90)  # Use 90th percentile to avoid outliers
    
    # Scale factors for visualization
    scale_factor = 0.3 * (oh_vals.max() - oh_vals.min()) / max_mag
    
    # Plot the gradient vector field
    plt.quiver(OH, COH, grad_oh, grad_coh, 
               scale=1.0/scale_factor, width=0.002, 
               color='w', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label('Potential Energy (eV)')
    
    # Add labels and title
    plt.xlabel('OH Bond Length (Å)', fontsize=12)
    plt.ylabel('COH Angle (degrees)', fontsize=12)
    plt.title('Potential Energy Surface with Gradient Field', fontsize=14)
    
    # Save the figure
    plt.savefig('figures/potential_gradient_field.png', dpi=300, bbox_inches='tight')
    print("Gradient field plot saved to figures/potential_gradient_field.png")
    
    plt.show()

if __name__ == "__main__":
    # Either load an existing function or generate a new one
    pot_func = load_function()
    
    # Visualize the 3D surface
    pot_func.plot_surface(save_path='figures/potential_energy_surface_3d.png')
    
    # Test energy calculation at a specific point
    test_energy_calculation(pot_func, oh_length=1.0, coh_angle=110.0)
    
    # Plot 1D slices through the potential energy surface
    plot_2d_slice(pot_func, slice_type='oh', fixed_value=120.0)  # Fix COH angle
    plot_2d_slice(pot_func, slice_type='coh', fixed_value=1.0)   # Fix OH bond length
    
    # Demonstrate gradient calculation
    demo_gradient_calculation(pot_func) 