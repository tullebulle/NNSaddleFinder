# Setup
Requirements:
- You need NequIP installed, see https://github.com/mir-group/nequip.
- NumPy, Torch, Maptlotlib, SciPy.

## TRAINING AND RUNNING AN EXAMPLE

The NequIP models are already trained, and will be loaded through the `function_wrapper` files. The examples are given in the `full_run` files.

## NNSaddleFinder

The `NNSaddleFinder` class is designed to locate saddle points in a potential energy landscape using a neural network approach. It leverages PyTorch for automatic differentiation, and can optionally use SIRQIT for finding $k$ smallest eigenpairs. The class is initialized with a potential function, an initial guess for the saddle point, and optional parameters such as a gradient function, saddle index, step size, momentum, device type (CPU or GPU) and Eigensolver.
