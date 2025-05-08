import torch
from saddlefinder import NNSaddleFinder
from function_wrapper_aspirin import nequip_potential_aspirin, grad_potential_aspirin
import matplotlib.pyplot as plt

SAVE_FILE = "nequip_saddle/saved_data/best_minimizer_aspirin.pt"

checkpoint = torch.load(SAVE_FILE)
# Use the loaded positions as the starting point
initial_positions = checkpoint['best_positions'].clone().detach()
evals, evecs = torch.linalg.eigh(torch.autograd.functional.hessian(nequip_potential_aspirin, initial_positions))
saddle_index_k = sum(evals < 0) #number of negative eigenvalues of
# print(f"Number of negative eigenvalues: {saddle_index_k}")
# print(evals)

saddle_finder = NNSaddleFinder(nequip_potential_aspirin, initial_positions, step_size=0.001, momentum=0.4, saddle_index=saddle_index_k, grad_fn=grad_potential_aspirin)
saddle_point, grad_norm_history = saddle_finder.find_saddle(iterations=100)


plt.plot(grad_norm_history)
plt.title("Gradient norm history - NN-HiSD")
plt.xlabel("Iteration")
plt.ylabel(r"$\| \nabla f \|$")
plt.savefig("grad_norm_history_aspirin.png")
plt.show()

# random initial positions
# initial_positions = torch.randn(45)
# evals, evecs = torch.linalg.eigh(torch.autograd.functional.hessian(nequip_potential, initial_positions))
# saddle_index_k = sum(evals < 0) #number of negative eigenvalues of

# saddle_finder = NNSaddleFinder(nequip_potential, initial_positions, step_size=0.1, momentum=0.8, saddle_index=saddle_index_k, grad_fn=grad_potential)
# saddle_finder.find_saddle(iterations=10)











