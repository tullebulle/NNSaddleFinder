# Setup
Requirements:
- You need NequIP installed, see https://github.com/mir-group/nequip.
- NumPy, Torch, Maptlotlib, SciPy.

## TRAINING AND RUNNING AN EXAMPLE

The NequIP models are already trained, and will be loaded through the `function_wrapper` files. The examples are given in the `full_run` files.

## NNSaddleFinder

The `NNSaddleFinder` class is designed to locate saddle points in a potential energy landscape using a neural network approach. It leverages PyTorch for automatic differentiation and Hessian computation. The class is initialized with a potential function, an initial guess for the saddle point, and optional parameters such as a gradient function, saddle index, step size, momentum, and device type (CPU or GPU).

### Key Methods:
- `gradient(x)`: Computes the gradient of the potential at a given point `x`.
- `hessian(x)`: Computes the Hessian matrix at `x` using PyTorch's built-in functions.
- `eigen_vals_vecs(hessian)`: Computes eigenvectors corresponding to the smallest eigenvalues of the Hessian.
- `step(eigenvalue_threshold=1e-3)`: Performs one iteration of the accelerated saddle search method, using heavy-ball momentum.
- `find_saddle(iterations=100, verbose=False)`: Runs multiple iterations to find the saddle point, returning the approximate saddle point and a list of gradient norms.

The class is particularly useful for finding saddle points in high-dimensional spaces, where traditional methods may struggle. It dynamically adjusts the search direction based on the eigenvalues of the Hessian, focusing on directions of negative curvature to efficiently locate saddle points.
