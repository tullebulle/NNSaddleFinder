import numpy as np
import matplotlib.pyplot as plt
import torch
from function_wrapper_rxn0015 import nequip_potential_rxn0015, extract_reaction_coordinates
import time
from tqdm import tqdm  # For progress bar
from matplotlib.patches import FancyArrowPatch

def plot_potential_heatmap(grid_size=20, batch_size=10, output_file="figures/potential_energy_surface.png"):
    """
    Plot a heatmap of the potential energy surface using NEquIP model,
    with overlaid trajectory points.
    
    Args:
        grid_size (int): Number of points along each axis for the grid
        batch_size (int): Number of positions to process in a batch for efficiency
        output_file (str): Filename for saving the plot
    """
    # Load the dataset
    print("Loading dataset...")
    full_data = np.load('nequip_data/benchmark_data/rxn0015-full.npz', allow_pickle=True)
    positions = full_data['R']  # All positions in the dataset
    atomic_numbers = full_data['z'].astype(int)
    
    # Reference position for creating grid samples
    reference_pos = positions[0].copy()  # Use first position as reference
    
    # Extract reaction coordinates for all positions in the trajectory
    print("Extracting reaction coordinates for trajectory...")
    oh_lengths = []
    coh_angles = []
    for pos in positions:
        flat_pos = pos.flatten()
        oh_length, coh_angle = extract_reaction_coordinates(flat_pos)
        oh_lengths.append(oh_length)
        coh_angles.append(coh_angle)
    
    # Find min/max with buffer
    oh_min, oh_max = min(oh_lengths), max(oh_lengths)
    coh_min, coh_max = min(coh_angles), max(coh_angles)
    
    # Add buffer (10% on each side)
    oh_buffer = (oh_max - oh_min) * 0.1
    coh_buffer = (coh_max - coh_min) * 0.1
    
    oh_min -= oh_buffer
    oh_max += oh_buffer
    coh_min -= coh_buffer
    coh_max += coh_buffer
    
    print(f"OH bond length range: {oh_min:.3f} to {oh_max:.3f} Å")
    print(f"COH angle range: {coh_min:.3f} to {coh_max:.3f} degrees")
    
    # Get energies for the actual trajectory points
    print("Calculating energies for trajectory points...")
    energies = []
    for pos in positions:
        flat_pos = torch.tensor(pos.flatten(), dtype=torch.float32, requires_grad=True)
        energy = nequip_potential_rxn0015(flat_pos).detach().numpy()
        energies.append(energy)
    energies = np.array(energies)
    
    # Create a grid for the heatmap
    print(f"Creating potential energy heatmap grid ({grid_size}x{grid_size})...")
    oh_grid = np.linspace(oh_min, oh_max, grid_size)
    coh_grid = np.linspace(coh_min, coh_max, grid_size)
    OH, COH = np.meshgrid(oh_grid, coh_grid)
    
    # Calculate potential energy for each grid point
    Z = np.zeros((grid_size, grid_size))
    
    # Function to modify a position to have the specified OH length and COH angle
    def modify_position(pos, oh_length, coh_angle):
        # Make a copy of the position
        new_pos = pos.reshape(-1, 3).copy()
        
        # Get the relevant atom positions
        o_pos = new_pos[0]  # O atom at index 0
        c_pos = new_pos[1]  # C atom at index 1
        h_pos = new_pos[6]  # H atom at index 6
        
        # Create a coordinate system centered at the O atom
        # Z-axis is along O-C direction
        z_axis = c_pos - o_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Find a perpendicular vector to create X-axis
        # Start with a guess
        guess = np.array([1, 0, 0])
        if np.abs(np.dot(z_axis, guess)) > 0.9:
            guess = np.array([0, 1, 0])
        
        x_axis = np.cross(z_axis, guess)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y-axis completes the orthogonal system
        y_axis = np.cross(z_axis, x_axis)
        
        # Convert angle from degrees to radians
        angle_rad = np.radians(coh_angle)
        
        # Calculate new H position
        new_h_pos = o_pos + oh_length * (
            np.cos(angle_rad) * z_axis + 
            np.sin(angle_rad) * x_axis
        )
        
        # Update H position
        new_pos[6] = new_h_pos
        
        return new_pos.flatten()
    
    # Process grid points in batches for efficiency
    print("Calculating potential energy for grid points (batch processing)...")
    total_points = grid_size * grid_size
    
    try:
        # Try to import tqdm for progress bar
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    # Prepare all grid positions at once
    all_grid_positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            modified_pos = modify_position(reference_pos, OH[i, j], COH[i, j])
            all_grid_positions.append((i, j, modified_pos))
    
    # Process in batches
    batch_iterator = range(0, len(all_grid_positions), batch_size)
    if use_tqdm:
        batch_iterator = tqdm(batch_iterator, desc="Processing grid batches")
    
    for start_idx in batch_iterator:
        end_idx = min(start_idx + batch_size, len(all_grid_positions))
        batch = all_grid_positions[start_idx:end_idx]
        
        # Process each position in the batch
        for i, j, modified_pos in batch:
            flat_pos = torch.tensor(modified_pos, dtype=torch.float32, requires_grad=True)
            try:
                energy = nequip_potential_rxn0015(flat_pos).detach().numpy()
                Z[i, j] = energy
            except Exception as e:
                print(f"Error at grid point ({i},{j}): {e}")
                Z[i, j] = np.nan
    
    # Create the plot
    print("Creating plot...")
    plt.figure(figsize=(8, 5))
    
    # Normalize data for better visualization
    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)
    
    # Plot heatmap of the potential energy surface
    contour = plt.contourf(OH, COH, Z, levels=50, cmap='viridis', alpha=0.8, 
                           vmin=vmin, vmax=vmax)
    
    # Add contour lines
    contour_lines = plt.contour(OH, COH, Z, levels=15, colors='white', alpha=0.5, linewidths=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%0.1f')
    
    # Plot scatter points for the trajectory
    scatter = plt.scatter(oh_lengths, coh_angles, c=energies, cmap='plasma', 
                         s=50, edgecolor='black', zorder=10)
    
    # Add lines connecting the trajectory points
    plt.plot(oh_lengths, coh_angles, 'k--', alpha=0.7, zorder=5, linewidth=1.5)
    
    # Add arrows along the trajectory to show direction
    arrow_frequency = max(1, len(oh_lengths) // 10)  # Show about 10 arrows
    ax = plt.gca()
    
    for i in range(0, len(oh_lengths) - 1, arrow_frequency):
        # Get the current and next point
        x1, y1 = oh_lengths[i], coh_angles[i]
        x2, y2 = oh_lengths[i+1], coh_angles[i+1]
        
        # Calculate the midpoint for arrow placement
        midx = (x1 + x2) / 2
        midy = (y1 + y2) / 2
        
        # Calculate the direction vector
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize to get a unit vector
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx = dx / length
            dy = dy / length
        
        # Create an arrow
        arrow = FancyArrowPatch(
            (midx - 0.3*dx, midy - 0.3*dy),  # Start slightly before midpoint
            (midx + 0.3*dx, midy + 0.3*dy),  # End slightly after midpoint
            arrowstyle='->',
            mutation_scale=15,  # Arrow size
            color='black',
            linewidth=1.5,
            zorder=20  # Above the line and points
        )
        ax.add_patch(arrow)
    
    # Add colorbar for the heatmap
    cbar = plt.colorbar(contour)
    cbar.set_label('Potential Energy (eV)')
    
    # Add labels and title
    plt.xlabel('O-H Bond Length (Å)', fontsize=12)
    plt.ylabel('C-O-H Angle (degrees)', fontsize=12)
    plt.title('Potential Energy Surface and Reaction Trajectory', fontsize=14)
    
    # Highlight starting and ending points
    plt.scatter([oh_lengths[0]], [coh_angles[0]], color='red', s=100, 
                edgecolor='black', label='Reactant', zorder=15)
    plt.scatter([oh_lengths[-1]], [coh_angles[-1]], color='blue', s=100, 
                edgecolor='black', label='Product', zorder=15)
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")
    
    plt.show()
    
    # Return the data for further analysis if needed
    return {
        'oh_lengths': oh_lengths, 
        'coh_angles': coh_angles,
        'energies': energies,
        'grid_oh': OH,
        'grid_coh': COH, 
        'grid_energies': Z
    }

if __name__ == "__main__":
    start_time = time.time()
    # Use a smaller grid size for faster calculation, increase for higher resolution
    plot_potential_heatmap(grid_size=20, batch_size=5)  
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds") 