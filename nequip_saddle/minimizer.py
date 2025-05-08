from function_wrapper import nequip_potential, flat_positions as initial_flat_positions
import torch
import numpy as np
import pickle
## This method searches through the training and test data which built the model to look for energy minima, 
# as they are allegedly more likely to occur in DFT simulations.

test_data = np.load('./benchmark_data/toluene_ccsd_t-test.npz', allow_pickle=True)
training_data = np.load('./benchmark_data/toluene_ccsd_t-train.npz', allow_pickle=True)
test_data = test_data['R'].reshape(test_data['R'].shape[0], -1) # shape (n_train, 15, 3) # want to flatten the last two dimensions
training_data = training_data['R'].reshape(training_data['R'].shape[0], -1)
data = torch.from_numpy(np.concatenate((training_data, test_data), axis=0)).clone().detach().requires_grad_(True)


class Minimizer:
    def __init__(self, model, data, filename='nequip_saddle/saved_data/energy_minima_positions.pkl'): # training/test has shape (n_train, flattened_positions)
        self.model = model
        self.data = data
        self.k = 3
        self.filename = filename

    def search_for_minima(self):
        #training set
        smallest_gradient_norm = float('inf')
        # print([pos for pos in self.data])
        chopped_data = self.data[:5]
        gradient_norm_list = [torch.norm(torch.autograd.grad(self.model(pos), pos)[0]) for pos in chopped_data]
        lowest_gradient_norms = torch.topk(torch.tensor(gradient_norm_list), k=self.k, largest=False)

        # minimizing the energy with LBFGS
        pos_energy_dict = {"positions": [], "energies": [], "gradient_norms": [], "eigenvalues": []}
        for idx in lowest_gradient_norms.indices:
            position = self.data[idx].clone().detach().requires_grad_(True)
            optimized_pos = self.minimize_with_lbfgs(position)
            energy = self.model(optimized_pos)
            gradient_norm = torch.norm(torch.autograd.grad(energy, optimized_pos)[0])
            evals, evecs = torch.linalg.eigh(torch.autograd.functional.hessian(nequip_potential, optimized_pos))
            pos_energy_dict["positions"].append(optimized_pos)
            pos_energy_dict["energies"].append(energy)
            pos_energy_dict["gradient_norms"].append(gradient_norm)
            pos_energy_dict["eigenvalues"].append(evals)

        
        #save pos_energy_dict to a file
        with open(self.filename, 'wb') as f:
            pickle.dump(pos_energy_dict, f)
        return pos_energy_dict
    
    def load_energy_minima_positions(self):
        with open(self.filename, 'rb') as f:
            return pickle.load(f)
    
    def minimize_with_lbfgs(self, initial_position, max_iter=100):
        """
        Minimize the energy using LBFGS optimizer.
        
        Args:
            initial_position: Starting position for optimization
            max_iter: Maximum number of iterations
            
        Returns:
            Optimized position
        """
        # Create a copy of the initial position that requires gradients
        position = initial_position.clone().detach().requires_grad_(True)
        
        # Initialize the optimizer
        optimizer = torch.optim.LBFGS([position], 
                                      lr=0.1,
                                      max_iter=20,
                                      line_search_fn="strong_wolfe")
        
        # Keep track of the loss history
        loss_history = []
        
        # Define the closure function required by LBFGS
        def closure():
            optimizer.zero_grad()
            energy = self.model(position)
            energy.backward()
            loss_history.append(energy.item())
            return energy
        
        # Run the optimization
        for i in range(max_iter):
            optimizer.step(closure)
            
            # Check for convergence
            if len(loss_history) >= 2 and abs(loss_history[-1] - loss_history[-2]) < 1e-6:
                print(f"Converged after {i+1} iterations")
                break
        
        return position

#FAILURE -- smallest gradient norm is about 3.0


minimizer = Minimizer(nequip_potential, data)
pos_energy_dict = minimizer.search_for_minima()
# pos_energy_dict = minimizer.load_energy_minima_positions()
print(pos_energy_dict["energies"])
print(pos_energy_dict["gradient_norms"])
print(pos_energy_dict["eigenvalues"])


