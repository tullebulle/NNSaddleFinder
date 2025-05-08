best_min_file = "nequip_saddle/saved_data/best_minimizer.pt"
norm_history_file = "nequip_saddle/saved_data/norm_history.pt"

norm_history_file_aspirin = "nequip_saddle/saved_data/norm_history_aspirin.pt"

# plot the norm history
import torch
import matplotlib.pyplot as plt

# Load the norm history

norm_history = torch.load(norm_history_file)
norm_history_aspirin = torch.load(norm_history_file_aspirin)
print("norm history:", norm_history['norm_history'])
print("norm history aspirin:", norm_history_aspirin['norm_history'])

# number of negative eigenvalues


# plot the norm history
# plt.plot(norm_history['norm_history'])
plt.plot(norm_history_aspirin['norm_history'])
plt.xlabel('Iteration')
plt.ylabel(r'$\| \nabla f \|$')
plt.title('Gradient Norm over iterations')
plt.savefig('gradient_norm_history_aspirin.png')
plt.show()