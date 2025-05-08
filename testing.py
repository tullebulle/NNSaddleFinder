from nequip_saddle.sirqit import sirqit
import torch
from torch.autograd.functional import jacobian
import pickle
from function_wrapper import nequip_potential, grad_potential, get_model
from scipy.sparse.linalg import eigsh  # 'h' = Hermitian (symmetric)
from saddlefinder import NNSaddleFinder
import numpy as np
import time

grad_pot = grad_potential

with open('nequip_saddle/saved_data/energy_minima_positions.pkl', 'rb') as f:
    data = pickle.load(f)

# print(torch.min(data['energies'][0]), torch.argmin(data['energies'][0]))
# #compute a bunch of jacobians using torch.autograd.functional.jacobian

x_init = data['positions'][0]
# x_init = torch.randn(45)
jac = jacobian(grad_pot, x_init)

# compare speed of eigsh and torch.linalg.eigh
start_time = time.time()
eigen_vals, eigen_vecs =  torch.linalg.eigh(jac) 
end_time = time.time()




# eigen_vals, eigen_vecs = torch.linalg.eigh(jac)
# # k is number of negative eigenvalues
# k = torch.sum(eigen_vals < 0)


# #We now want to search for an index k saddle point
# print("k: ", k)
# saddle_finder = NNSaddleFinder(nequip_potential, x_init, grad_fn = grad_pot, saddle_index=k, step_size=0.001, momentum=0.8, device='cpu')

# saddle_finder.find_saddle(iterations=100, verbose=True)





