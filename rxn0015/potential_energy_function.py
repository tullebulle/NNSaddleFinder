import numpy as np
from scipy.interpolate import RectBivariateSpline, Rbf
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

class PotentialEnergyFunction:
    """
    A class that provides a smooth 2D function approximating the potential energy
    based on OH bond length and COH angle coordinates.
    
    Uses scipy's interpolation methods to create a continuous smooth function.
    """
    
    def __init__(self, oh_grid=None, coh_grid=None, energy_values=None, smooth_factor=0.1, method='spline'):
        """
        Initialize the potential energy function.
        
        Args:
            oh_grid (np.ndarray): 1D array of OH bond length values
            coh_grid (np.ndarray): 1D array of COH angle values
            energy_values (np.ndarray): 2D array of energy values at grid points
            smooth_factor (float): Smoothing factor for interpolation
            method (str): Interpolation method ('spline' or 'rbf')
        """
        self.oh_grid = oh_grid
        self.coh_grid = coh_grid
        self.energy_values = energy_values
        self.smooth_factor = smooth_factor
        self.method = method
        self.interp_func = None
        self._torch_nn = None  # PyTorch neural network for gradients
        
        if oh_grid is not None and coh_grid is not None and energy_values is not None:
            self.create_interpolation_function(oh_grid, coh_grid, energy_values, smooth_factor, method)
    
    def create_interpolation_function(self, oh_grid, coh_grid, energy_values, smooth_factor=0.1, method='spline', train_nn=True):
        """
        Create a smooth interpolation function from the grid data.
        
        Args:
            oh_grid (np.ndarray): 1D array of OH bond length values
            coh_grid (np.ndarray): 1D array of COH angle values
            energy_values (np.ndarray): 2D array of energy values at grid points
            smooth_factor (float): Smoothing factor for interpolation
            method (str): Interpolation method ('spline' or 'rbf')
        """
        self.oh_grid = oh_grid
        self.coh_grid = coh_grid
        self.energy_values = energy_values
        self.smooth_factor = smooth_factor
        self.method = method
        
        if method == 'spline':
            # Use RectBivariateSpline for a smooth 2D spline interpolation
            self.interp_func = RectBivariateSpline(
                oh_grid, coh_grid, energy_values.T, 
                kx=3, ky=3,  # Cubic spline
                s=smooth_factor  # Smoothing factor
            )
        
        elif method == 'rbf':
            # Create a grid of all coordinates
            xx, yy = np.meshgrid(oh_grid, coh_grid)
            points = np.vstack((xx.flatten(), yy.flatten())).T
            values = energy_values.flatten()
            
            # Use Radial Basis Function interpolation
            self.rbf_interp = Rbf(
                points[:, 0], points[:, 1], values,
                function='thin_plate',  # Thin plate spline for smoothness
                smooth=smooth_factor
            )
            
            # Create a wrapper function to match the spline API
            def rbf_wrapper(x, y):
                if np.isscalar(x) and np.isscalar(y):
                    return self.rbf_interp(np.array([x]), np.array([y]))[0]
                elif np.isscalar(x):
                    x = np.full_like(y, x)
                elif np.isscalar(y):
                    y = np.full_like(x, y)
                
                return self.rbf_interp(x, y)
            
            self.interp_func = rbf_wrapper
        
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        # Create PyTorch neural network approximation for gradient computation
        if train_nn:
            self._create_torch_nn()
    
    def _create_torch_nn(self, hidden_size=128, num_layers=3):
        """
        Create a PyTorch neural network that approximates the potential energy surface
        for gradient and Hessian computation.
        
        Args:
            hidden_size (int): Size of hidden layers
            num_layers (int): Number of hidden layers
        """
        class PotentialNN(torch.nn.Module):
            def __init__(self, hidden_size, num_layers):
                super().__init__()
                self.layers = torch.nn.ModuleList()
                self.layers.append(torch.nn.Linear(2, hidden_size))
                
                for _ in range(num_layers - 1):
                    self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
                
                self.layers.append(torch.nn.Linear(hidden_size, 1))
                self.activation = torch.nn.Tanh()
            
            def forward(self, x):
                for i, layer in enumerate(self.layers[:-1]):
                    x = self.activation(layer(x))
                return self.layers[-1](x).squeeze(-1)
        
        # Create the neural network
        self._torch_nn = PotentialNN(hidden_size, num_layers)
        
        # Sample points from the potential energy surface
        num_samples = 5000
        oh_samples = np.random.uniform(self.oh_grid.min(), self.oh_grid.max(), num_samples)
        coh_samples = np.random.uniform(self.coh_grid.min(), self.coh_grid.max(), num_samples)
        
        # Evaluate the interpolation function at these points
        energies = np.array([self(oh, coh) for oh, coh in zip(oh_samples, coh_samples)])
        
        # Normalize data for better training
        self._oh_mean = oh_samples.mean()
        self._oh_std = oh_samples.std()
        self._coh_mean = coh_samples.mean()
        self._coh_std = coh_samples.std()
        self._energy_mean = energies.mean()
        self._energy_std = energies.std()
        
        # Convert to PyTorch tensors
        inputs = torch.tensor(np.column_stack([
            (oh_samples - self._oh_mean) / self._oh_std,
            (coh_samples - self._coh_mean) / self._coh_std
        ]), dtype=torch.float32)
        targets = torch.tensor((energies - self._energy_mean) / self._energy_std, dtype=torch.float32)
        
        # Train the neural network
        self._torch_nn.train()
        optimizer = torch.optim.Adam(self._torch_nn.parameters(), lr=0.001)
        
        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = self._torch_nn(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/1000, Loss: {loss.item():.6f}")
        
        # Set to evaluation mode
        self._torch_nn.eval()
        print("PyTorch neural network approximation trained successfully.")
    
    def __call__(self, oh_length, coh_angle):
        """
        Evaluate the potential energy at the given OH bond length and COH angle.
        
        Args:
            oh_length (float or np.ndarray): OH bond length value(s)
            coh_angle (float or np.ndarray): COH angle value(s)
            
        Returns:
            float or np.ndarray: Interpolated potential energy value(s)
        """
        if self.interp_func is None:
            raise ValueError("Interpolation function not initialized. Call create_interpolation_function first.")
        
        if self.method == 'spline':
            return self.interp_func(oh_length, coh_angle, grid=False)
        else:
            return self.interp_func(oh_length, coh_angle)
        


    
    def torch_forward(self, coords, requires_grad=True):
        """
        PyTorch-compatible forward function for gradient computation.
        
        Args:
            coords (torch.Tensor): Tensor of shape [batch_size, 2] containing 
                                   OH bond length and COH angle coordinates
            requires_grad (bool): Whether to enable autograd
            
        Returns:
            torch.Tensor: Potential energy values
        """
        if self._torch_nn is None:
            raise ValueError("PyTorch neural network not initialized.")
        
        # Normalize inputs
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32)
        
        if requires_grad and not coords.requires_grad:
            coords.requires_grad_(True)
            
        # Normalize inputs
        if len(coords.shape) == 1:
            normalized_coords = torch.zeros_like(coords)
            normalized_coords[0] = (coords[0] - self._oh_mean) / self._oh_std
            normalized_coords[1] = (coords[1] - self._coh_mean) / self._coh_std
        else:
            normalized_coords = torch.zeros_like(coords)
            normalized_coords[:, 0] = (coords[:, 0] - self._oh_mean) / self._oh_std
            normalized_coords[:, 1] = (coords[:, 1] - self._coh_mean) / self._coh_std
        
        # Forward pass
        energies = self._torch_nn(normalized_coords)
        
        # Denormalize outputs
        return energies * self._energy_std + self._energy_mean
    
    def compute_gradient(self, *args):
        """
        Compute the gradient of the potential energy with respect to 
        OH bond length and COH angle using PyTorch autograd.
        
        Args:
            Either:
                - coords: torch.Tensor of shape [batch_size, 2]
                - oh_length, coh_angle: Two separate arguments for the coordinates
            
        Returns:
            np.ndarray: Gradient [dE/d(oh_length), dE/d(coh_angle)]
        """
        if self._torch_nn is None:
            raise ValueError("PyTorch neural network not initialized. Make sure the model is properly loaded.")
        
        # Handle different input formats
        if len(args) == 1:
            if isinstance(args[0], torch.Tensor):
                coords = args[0]
            else:
                # Could be a tuple/list or a tensor that needs conversion
                coords = torch.tensor([[args[0][0], args[0][1]]], dtype=torch.float32)
        elif len(args) == 2:
            # Two separate arguments (oh_length, coh_angle)
            oh_length, coh_angle = args
            coords = torch.tensor([[oh_length, coh_angle]], dtype=torch.float32)
        else:
            raise ValueError("Invalid arguments. Expected either a tensor or oh_length and coh_angle values.")
        
        # Ensure coords requires gradient
        if not coords.requires_grad:
            coords = coords.detach().requires_grad_(True)
            
        # Compute gradient
        energy = self.torch_forward(coords)
        energy.backward()
        grad = coords.grad.detach().numpy()[0]
        
        return grad
    
    def compute_hessian(self, coords):
        """
        Compute the Hessian matrix of the potential energy with respect to
        OH bond length and COH angle using PyTorch autograd.
        
        Args:
            coords: Either a PyTorch tensor of shape [1, 2] or a tuple/list of (oh_length, coh_angle)
            
        Returns:
            torch.Tensor or np.ndarray: 2x2 Hessian matrix
                [[d²E/d(oh_length)², d²E/d(oh_length)d(coh_angle)]
                 [d²E/d(oh_length)d(coh_angle), d²E/d(coh_angle)²]]
        """
        if self._torch_nn is None:
            raise ValueError("PyTorch neural network not initialized. Make sure the model is properly loaded.")
        
        # Handle different input formats
        if not isinstance(coords, torch.Tensor):
            if isinstance(coords, (list, tuple)) and len(coords) == 2:
                # If coordinates are provided as (oh_length, coh_angle)
                oh_length, coh_angle = coords
                coords = torch.tensor([[oh_length, coh_angle]], dtype=torch.float32, requires_grad=True)
            else:
                # Assume it's a single coordinate value (scalar)
                coords = torch.tensor([[coords, 0.0]], dtype=torch.float32, requires_grad=True)
        
        # Ensure coords requires gradient
        if not coords.requires_grad:
            coords = coords.detach().requires_grad_(True)
        
        def compute_gradient_for_hessian(coords):
            energy = self.torch_forward(coords)
            grad_tensors = torch.ones_like(energy)
            grad = torch.autograd.grad(energy, coords, grad_outputs=grad_tensors, 
                                       create_graph=True, retain_graph=True)[0]
            return grad
        
        # First compute gradient
        grad = compute_gradient_for_hessian(coords)
        
        # Now compute Hessian row by row
        hessian = torch.zeros((2, 2), dtype=torch.float32)
        
        for i in range(2):
            grad_i = grad[i]
            second_derivs = torch.autograd.grad(grad_i, coords, create_graph=True, 
                                              retain_graph=True)[0]
            hessian[i] = second_derivs[0]
        
        return hessian.detach()  # Return as tensor, caller can convert to numpy if needed
    
    def save(self, filename='potential_energy_function.pkl'):
        """
        Save the potential energy function to a file.
        
        Args:
            filename (str): Path to save the function
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # For RBF, we need to save the raw data since the function itself isn't easily serializable
        data = {
            'oh_grid': self.oh_grid,
            'coh_grid': self.coh_grid,
            'energy_values': self.energy_values,
            'smooth_factor': self.smooth_factor,
            'method': self.method
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        # Also save the torch model if it exists
        if self._torch_nn is not None:
            torch_filename = filename.replace('.pkl', '_torch.pt')
            torch.save({
                'model': self._torch_nn.state_dict(),
                'oh_mean': self._oh_mean,
                'oh_std': self._oh_std,
                'coh_mean': self._coh_mean,
                'coh_std': self._coh_std,
                'energy_mean': self._energy_mean,
                'energy_std': self._energy_std
            }, torch_filename)
    
    @classmethod
    def load(cls, filename='models/potential_energy_function_spline.pkl'):
        """
        Load a potential energy function from a file.
        
        Args:
            filename (str): Path to the saved function
            
        Returns:
            PotentialEnergyFunction: Loaded function
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.create_interpolation_function(
            data['oh_grid'],
            data['coh_grid'],
            data['energy_values'],
            data['smooth_factor'],
            data['method'],
            train_nn=False
        )
        
        # Initialize the neural network structure without training
        class PotentialNN(torch.nn.Module):
            def __init__(self, hidden_size=128, num_layers=3):
                super().__init__()
                self.layers = torch.nn.ModuleList()
                self.layers.append(torch.nn.Linear(2, hidden_size))
                
                for _ in range(num_layers - 1):
                    self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
                
                self.layers.append(torch.nn.Linear(hidden_size, 1))
                self.activation = torch.nn.Tanh()
            
            def forward(self, x):
                for i, layer in enumerate(self.layers[:-1]):
                    x = self.activation(layer(x))
                return self.layers[-1](x).squeeze(-1)
        
        # Create the neural network without training
        instance._torch_nn = PotentialNN()
        
        # Try to load the torch model if available
        torch_filename = filename.replace('.pkl', '_torch.pt')
        if os.path.exists(torch_filename):
            print("Loading PyTorch model...")
            try:
                checkpoint = torch.load(torch_filename, map_location=torch.device('cpu'), weights_only=False)
                instance._torch_nn.load_state_dict(checkpoint['model'])
                instance._oh_mean = checkpoint['oh_mean']
                instance._oh_std = checkpoint['oh_std']
                instance._coh_mean = checkpoint['coh_mean']
                instance._coh_std = checkpoint['coh_std']
                instance._energy_mean = checkpoint['energy_mean']
                instance._energy_std = checkpoint['energy_std']
                print("PyTorch model loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load PyTorch model: {e}")
                # Reset if loading fails
                instance._torch_nn = None
        else:
            print(f"Warning: Could not find PyTorch model at {torch_filename}")
            print("The function will work for evaluation but not for gradient/Hessian computation.")
            instance._torch_nn = None
        
        return instance
    
    def plot_heatmap(self,fig=None, ax=None, fig_size=(10, 8), num_points=200, show=True, save_path=None, 
                     add_contour=True, cmap='viridis', trajectory_points=None, plot_gradients=False):
        """
        Plot a 2D heatmap of the potential energy function.
        
        Args:
            fig_size (tuple): Figure size (width, height)
            num_points (int): Number of points to sample along each dimension
            show (bool): Whether to display the plot
            save_path (str): Path to save the figure, if provided
            add_contour (bool): Whether to add contour lines
            cmap (str): Colormap to use
            trajectory_points (tuple): Optional list of trajectory points as (oh_lengths, coh_angles, energies)
            plot_gradients (bool): Whether to plot gradient vectors
        """
        # Create a fine grid for smooth visualization
        oh_fine = np.linspace(self.oh_grid.min(), self.oh_grid.max(), num_points)
        coh_fine = np.linspace(self.coh_grid.min(), self.coh_grid.max(), num_points)
        OH, COH = np.meshgrid(oh_fine, coh_fine)
        
        # Evaluate the function on the fine grid
        Z = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                Z[i, j] = self(oh_fine[j], coh_fine[i])
        
        # Create 2D plot
        plt.figure(figsize=fig_size)
        
        # Plot the heatmap
        contourf = plt.contourf(OH, COH, Z, levels=50, cmap=cmap, alpha=0.9)
        
        # Add contour lines if requested
        if add_contour:
            contour_lines = plt.contour(OH, COH, Z, levels=15, colors='white', alpha=0.5, linewidths=0.5)
            plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%0.1f')
        
        # Add gradient vectors if requested
        if plot_gradients and self._torch_nn is not None:
            # Sample fewer points for gradient visualization
            grad_points = 15
            oh_grad = np.linspace(self.oh_grid.min(), self.oh_grid.max(), grad_points)
            coh_grad = np.linspace(self.coh_grid.min(), self.coh_grid.max(), grad_points)
            OH_grad, COH_grad = np.meshgrid(oh_grad, coh_grad)
            
            # Calculate gradients
            grad_x = np.zeros((grad_points, grad_points))
            grad_y = np.zeros((grad_points, grad_points))
            
            for i in range(grad_points):
                for j in range(grad_points):
                    oh = oh_grad[j]
                    coh = coh_grad[i]
                    grad = self.compute_gradient(oh, coh)
                    grad_x[i, j] = -grad[0]  # Negative gradient for energy minimization
                    grad_y[i, j] = -grad[1]  # Negative gradient for energy minimization
            
            # Normalize gradient vectors for better visualization
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            max_mag = np.percentile(grad_magnitude, 95)  # Use 95th percentile to avoid outliers
            scale_factor = 0.4 * (oh_fine.max() - oh_fine.min()) / max_mag
            
            # Plot gradient vectors
            plt.quiver(OH_grad, COH_grad, grad_x, grad_y, color='yellow', alpha=0.8, 
                       scale=1.0/scale_factor, width=0.003, zorder=8)
        
        # Add trajectory points if provided
        if trajectory_points is not None:
            oh_lengths, coh_angles, energies = trajectory_points
            
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
                from matplotlib.patches import FancyArrowPatch
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
            
            # Highlight starting and ending points
            plt.scatter([oh_lengths[0]], [coh_angles[0]], color='red', s=100, 
                        edgecolor='black', label='Reactant', zorder=15)
            plt.scatter([oh_lengths[-1]], [coh_angles[-1]], color='blue', s=100, 
                        edgecolor='black', label='Product', zorder=15)
            
            plt.legend(fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(contourf)
        cbar.set_label('Potential Energy (eV)')
        
        # Add labels and title
        plt.xlabel('OH Bond Length (Å)', fontsize=12)
        plt.ylabel('COH Angle (degrees)', fontsize=12)
        plt.title('Potential Energy Surface', fontsize=14)
        
        # Save if path provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_hessian_eigenvectors(self, oh_length, coh_angle, scale=0.1, fig_size=(10, 8), 
                                 show=True, save_path=None):
        """
        Plot the eigenvectors of the Hessian matrix at a specific point.
        
        Args:
            oh_length (float): OH bond length value
            coh_angle (float): COH angle value
            scale (float): Scale factor for eigenvectors
            fig_size (tuple): Figure size (width, height)
            show (bool): Whether to display the plot
            save_path (str): Path to save the figure, if provided
        """
        if self._torch_nn is None:
            raise ValueError("PyTorch neural network not initialized.")
        
        # Compute Hessian at the given point
        torch_coords = torch.tensor([[oh_length, coh_angle]], dtype=torch.float32, requires_grad=True)
        hessian = self.compute_hessian(torch_coords)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        
        # Create plot with the potential energy surface as background
        self.plot_heatmap(fig_size=fig_size, show=False, add_contour=True)
        
        # Draw eigenvectors
        for i in range(2):
            eigenvector = eigenvectors[:, i]
            eigenvalue = eigenvalues[i]
            
            # Scale eigenvector for visualization
            scaled_eigenvector = scale * eigenvector / np.linalg.norm(eigenvector)
            
            # Draw the eigenvector
            plt.arrow(oh_length, coh_angle, 
                     scaled_eigenvector[0], scaled_eigenvector[1],
                     width=0.01, head_width=0.03, head_length=0.05,
                     fc='r' if eigenvalue < 0 else 'g', ec='k', 
                     length_includes_head=True, zorder=20)
            
            # Add text label with eigenvalue
            plt.text(oh_length + 1.2 * scaled_eigenvector[0], 
                    coh_angle + 1.2 * scaled_eigenvector[1],
                    f"λ{i+1}={eigenvalue:.2f}",
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                    zorder=25)
        
        # Highlight the point
        plt.scatter([oh_length], [coh_angle], c='yellow', s=100, 
                   edgecolor='black', zorder=15)
        
        # Add title
        plt.title(f'Hessian Eigenvectors at OH={oh_length:.2f}, COH={coh_angle:.2f}', fontsize=14)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_surface(self, fig_size=(10, 8), num_points=100, show=True, save_path=None):
        """
        Plot a 3D surface of the potential energy function.
        
        Args:
            fig_size (tuple): Figure size (width, height)
            num_points (int): Number of points to sample along each dimension
            show (bool): Whether to display the plot
            save_path (str): Path to save the figure, if provided
        """
        # Create a fine grid for smooth visualization
        oh_fine = np.linspace(self.oh_grid.min(), self.oh_grid.max(), num_points)
        coh_fine = np.linspace(self.coh_grid.min(), self.coh_grid.max(), num_points)
        OH, COH = np.meshgrid(oh_fine, coh_fine)
        
        # Evaluate the function on the fine grid
        Z = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                Z[i, j] = self(oh_fine[j], coh_fine[i])
        
        # Create 3D plot
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(OH, COH, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Add labels and colorbar
        ax.set_xlabel('OH Bond Length (Å)')
        ax.set_ylabel('COH Angle (degrees)')
        ax.set_zlabel('Potential Energy (eV)')
        ax.set_title('Smooth Potential Energy Surface')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Potential Energy (eV)')
        
        # Adjust view angle
        ax.view_init(elev=30, azim=45)
        
        # Save if path provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()

    def create_hessian_fn(self):
        """
        Create a Hessian function compatible with the saddle finder.
        
        Returns:
            callable: A function that takes a tensor and returns its Hessian matrix as a NumPy array
        """
        def hessian_fn(x):
            # Ensure x is a tensor with requires_grad
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
                
            # Compute the Hessian
            hessian = self.compute_hessian(x)
            
            # Convert to NumPy array for compatibility
            return hessian.numpy()
            
        return hessian_fn

def create_potential_function_from_heatmap(heatmap_data, smooth_factor=0.1, method='spline'):
    """
    Create a smooth potential energy function from the heatmap data.
    
    Args:
        heatmap_data (dict): Dictionary containing 'grid_oh', 'grid_coh', and 'grid_energies'
        smooth_factor (float): Smoothing factor for interpolation
        method (str): Interpolation method ('spline' or 'rbf')
        
    Returns:
        PotentialEnergyFunction: A callable object that evaluates the potential energy function
    """
    # Extract the grid values from the heatmap data
    oh_grid = np.unique(heatmap_data['grid_oh'])
    coh_grid = np.unique(heatmap_data['grid_coh'])
    energy_values = heatmap_data['grid_energies']

    try:
        pot_func = PotentialEnergyFunction.load()
    except FileNotFoundError:
        pot_func = PotentialEnergyFunction(oh_grid, coh_grid, energy_values, smooth_factor, method)
    
    # Create and return the potential energy function
    return pot_func

# Example usage (will run if this script is executed directly)
if __name__ == "__main__":
    from rxn0015.potential_nequip import plot_potential_heatmap
    
    print("Generating heatmap data...")
    # Use a smaller grid for quick demonstration
    heatmap_data = plot_potential_heatmap(grid_size=20, batch_size=5, output_file="figures/potential_energy_surface.png")
    
    print("Creating smooth potential energy function...")
    # Create the potential energy function (try both methods)
    pot_func_spline = create_potential_function_from_heatmap(heatmap_data, smooth_factor=0.1, method='spline')
    
    # Save the function for later use
    pot_func_spline.save('models/potential_energy_function_spline.pkl')
    
    # Plot a 3D surface to visualize the function
    # pot_func_spline.plot_surface(save_path='figures/potential_energy_surface_3d.png')
    pot_func_spline.plot_heatmap(save_path='figures/potential_energy_surface_2d.png')
    
    print("Smooth potential energy function created and saved!")
    
    # Example of function evaluation at a specific point
    oh_test = 1.0  # Example OH bond length
    coh_test = 120.0  # Example COH angle
    energy = pot_func_spline(oh_test, coh_test)
    print(f"Potential energy at OH={oh_test}Å, COH={coh_test}°: {energy:.4f} eV") 