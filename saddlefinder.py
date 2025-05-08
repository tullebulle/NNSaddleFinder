import torch
from torch.autograd.functional import hessian as torch_hessian
from nequip_saddle.sirqit import sirqit

class NNSaddleFinder:
    def __init__(self, potential, initial_x, grad_fn = None, saddle_index=None, step_size=0.01, momentum=0.8, device='cpu', eigsolver='AD'):
        """
        Initialize the saddle finder.

        Args:
            potential (callable): The energy potential E(x), implemented as a PyTorch model.
            initial_x (torch.Tensor): Initial guess for the saddle point.
            saddle_index (int): Index of saddle to find (number of negative eigen-directions).
            step_size (float): Step size β.
            momentum (float): Momentum coefficient γ for heavy-ball acceleration.
            device (str): 'cpu' or 'cuda' for GPU acceleration.
        """
        self.potential = potential
        self.grad_fn = grad_fn
        self.x = initial_x.clone().detach().to(device).requires_grad_(True)
        self.prev_x = self.x.clone().detach()
        self.saddle_index = saddle_index # dynamic if None
        self.step_size = step_size
        self.momentum = momentum
        self.device = device
        assert eigsolver in ['AD', 'SIRQIT'], "eigsolver must be either 'AD' or 'SIRQIT'"
        self.eigsolver = eigsolver

    def gradient(self, x):
        """ Compute the gradient using PyTorch autograd. """
        if self.grad_fn is None:
            E = self.potential(x)
            grad, = torch.autograd.grad(E, x, create_graph=True)
        else:
            grad = self.grad_fn(x)
        return grad

    def hessian(self, x):
        """
        Computes the Hessian matrix at x using PyTorch's built-in hessian function.
        """
        # Use PyTorch's built-in hessian function
        if self.grad_fn:
            print("Using gradient function")
            H = torch.autograd.functional.jacobian(self.grad_fn, x)
        else:
            # We need to define a scalar function that takes a tensor and returns a scalar
            # def scalar_potential(input_x):
            #     return self.potential(input_x)
            
            # Compute the Hessian using PyTorch's function
            H = torch_hessian(self.potential, x)
        
        return H

    def grad_potential(self, x):
        """Computes gradient using AD."""
        E = self.potential(x)
        grad, = torch.autograd.grad(E, x, create_graph=True)
        return grad

    def eigen_vals_vecs(self, hessian): # AD method
        """Computes eigenvectors corresponding to smallest eigenvalues."""
        eigvals, eigvecs = torch.linalg.eigh(hessian)
        return eigvals, eigvecs
    
    def sirqit_eigen_vals_vecs(self, prev_eigvecs=None): # SIRQIT method
        """Computes eigenvectors corresponding to smallest eigenvalues."""
        
        if prev_eigvecs is not None:
            V0 = torch.cat([prev_eigvecs, V0], dim=1)
        else:
            V0 = torch.randn(self.x.shape[0], self.saddle_index, dtype=torch.float32)
        eigvals, eigvecs = sirqit(self.grad_potential, self.x, V0, self.saddle_index)
        return eigvals, eigvecs

    def step(self, eigenvalue_threshold=1e-3):
        """
        Perform one iteration of the accelerated saddle search method.
        Heavy-ball (momentum) acceleration is used here.
        
        Args:
            eigenvalue_threshold (float): Threshold below which eigenvalue directions are ignored.
        """
        with torch.enable_grad():
            if self.eigsolver == 'AD':
                hessian = self.hessian(self.x)
                eigvals, eigvecs = self.eigen_vals_vecs(hessian)
            elif self.eigsolver == 'SIRQIT':
                try:
                    eigvals, eigvecs = self.sirqit_eigen_vals_vecs(hessian, prev_eigvecs = eigvecs)
                except:
                    eigvals, eigvecs = self.sirqit_eigen_vals_vecs(hessian, prev_eigvecs = None)
            
            # Filter eigenvectors based on the eigenvalue threshold
            significant_indices = torch.where(eigvals.abs() >= eigenvalue_threshold)[0]
            if self.saddle_index is None:
                relevant_eigvecs = eigvecs[:, significant_indices[:sum(eigvals[significant_indices] < 0)]]
            else:
                relevant_eigvecs = eigvecs[:, significant_indices[:self.saddle_index]]

            grad = self.grad_potential(self.x)
            
            # Calculate projection onto the significant negative eigenvalue directions
            projection = sum([torch.dot(grad, v) * v for v in relevant_eigvecs.T])
            
            # Reverse the gradient in these directions (this is the key fix)
            effective_grad = grad - 2 * projection
            
            # Print diagnostic information
            print(f"Gradient norm: {torch.norm(grad).item()}")
            print(f"Number of significant negative eigenvalues in current iteration: {sum(eigvals[significant_indices] < 0)}")
            
            # Update position with momentum
            new_x = (
                self.x
                - self.step_size * effective_grad  # Note the negative sign here
                + self.momentum * (self.x - self.prev_x)
            )

            # Update previous and current positions
            self.prev_x = self.x.detach()
            self.x = new_x.clone().detach().requires_grad_(True)

    def find_saddle(self, iterations=100, verbose=False):
        """
        Run multiple iterations to find the saddle.

        Returns:
            torch.Tensor: Approximate saddle point after iterations.
            list: List of gradient norms at each iteration.
        """
        grad_norms = []  # List to store gradient norms
        for i in range(iterations):
            self.step()
            grad_norm = torch.norm(self.grad_potential(self.x)).item()
            grad_norms.append(grad_norm)  # Store the gradient norm
            if verbose and i % 10 == 0:
                print(f'Iteration {i}, Gradient norm: {grad_norm:.4e}')
        return self.x.detach(), grad_norms

    def grad_potential(self, x):
        """Compute gradient of the potential."""
        E = self.potential(x)
        grad, = torch.autograd.grad(E, x, create_graph=True)
        return grad


