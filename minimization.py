from saddlefinder import NNSaddleFinder
from function_wrapper import nequip_potential, flat_positions as initial_flat_positions
import torch
import psutil
import os

# --- Configuration for saving/loading best result ---
SAVE_FILE = "nequip_saddle/saved_data/best_minimizer.pt"

# NEW ── file that will store the history of gradient norms
NORM_HISTORY_FILE = os.path.join(os.path.dirname(SAVE_FILE), "norm_history.pt")

min_grad_norm_so_far = float('inf')
best_positions = None

# --- Load previous best result if available ---
if os.path.exists(SAVE_FILE):
    try:
        checkpoint = torch.load(SAVE_FILE)
        # Use the loaded positions as the starting point
        flat_positions = checkpoint['best_positions'].clone().detach()
        min_grad_norm_so_far = checkpoint['min_grad_norm']
        best_positions = flat_positions.clone() # Keep track of the loaded best
        print(f"Loaded previous best result from {SAVE_FILE}.")
        print(f"Starting with positions yielding gradient norm: {min_grad_norm_so_far:.4e}")
    except Exception as e:
        print(f"Error loading {SAVE_FILE}: {e}. Starting from initial positions.")
        # Fallback to initial positions if loading fails
        flat_positions = initial_flat_positions.clone().detach()
else:
    print(f"No previous result found at {SAVE_FILE}. Starting from initial positions.")
    flat_positions = torch.from_numpy(initial_flat_positions)

# print("Initial positions:", flat_positions)
# Ensure flat_positions requires gradients
if not flat_positions.requires_grad:
    flat_positions.requires_grad_(True)



##### EIGENVALS OF HESSIAN #######
print("min grad norm so far:", min_grad_norm_so_far)
evals, evecs = torch.linalg.eigh(torch.autograd.functional.hessian(nequip_potential, flat_positions))
print("Number of negative eigenvalues:", torch.sum(evals < 0).item())
##############################


optimize = False

######## OPTIMIZATION ########

if optimize:
    # NEW ── list for keeping the gradient-norm history
    norm_history = []
    # k_history    = []          # number of negative eigen-values at each step
    # Define the optimizer
    optimizer = torch.optim.Adam([flat_positions], lr=1e-3) # Adjust learning rate (lr) as needed

    print("Optimizing to minimize gradient norm...")
    n_steps = 10000 # Number of optimization steps
    for step in range(n_steps):
        optimizer.zero_grad()

        # Calculate potential and gradient
        if not flat_positions.requires_grad:
            flat_positions.requires_grad_(True)
        out = nequip_potential(flat_positions)
        gradient = torch.autograd.grad(out, flat_positions, create_graph=True)[0]

        # Define the loss as the squared L2 norm of the gradient
        loss = torch.sum(gradient**2)
        current_grad_norm = torch.sqrt(loss).item() # Calculate current norm

        # NEW ── store this step's gradient norm
        norm_history.append(current_grad_norm)
        # if step % 10 == 0:
        #     evals, evecs = torch.linalg.eigh(torch.autograd.functional.hessian(nequip_potential, flat_positions))
        #     k_history.append(torch.sum(evals < 0).item())

        # --- Check if this is the best result so far ---
        if current_grad_norm < min_grad_norm_so_far:
            min_grad_norm_so_far = current_grad_norm
            best_positions = flat_positions.clone().detach() # Save a detached copy
            # Save the new best result to the file
            torch.save({
                'best_positions': best_positions,
                'min_grad_norm': min_grad_norm_so_far
            }, SAVE_FILE)
            print(f"*** New best found at step {step}! Norm: {min_grad_norm_so_far:.4e} ***")

        # Backpropagate the loss to get gradients w.r.t. flat_positions
        loss.backward()

        # Update the positions
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            print(f"Step: {step}, Loss (Gradient Norm^2): {loss.item():.4e}, Current Gradient Norm: {current_grad_norm:.4e}")

    # NEW ── save the gradient-norm history to disk
    try:
        torch.save({'norm_history': torch.tensor(norm_history)}, NORM_HISTORY_FILE)
        print(f"Saved gradient-norm history ({len(norm_history)} points) to {NORM_HISTORY_FILE}.")
    except Exception as e:
        print(f"Failed to save norm history: {e}")

    print("\nOptimization finished.")
    print("Final positions (last step):", flat_positions.detach())
    # Calculate final gradient at the last step
    out_final = nequip_potential(flat_positions)
    gradient_final = torch.autograd.grad(out_final, flat_positions)[0]
    final_grad_norm = torch.linalg.norm(gradient_final)
    print("Final gradient norm (last step):", final_grad_norm.item())

    # --- Report the overall best result found ---
    if best_positions is not None:
        print("\n--- Overall Best Result Found ---")
        print("Best positions found:", best_positions)
        print(f"Minimum gradient norm achieved: {min_grad_norm_so_far:.4e}")
        # Verify the gradient norm for the best positions
        best_positions.requires_grad_(True)
        out_best = nequip_potential(best_positions)
        gradient_best = torch.autograd.grad(out_best, best_positions)[0]
        verified_grad_norm = torch.linalg.norm(gradient_best)
        print(f"Verified gradient norm for best positions: {verified_grad_norm.item():.4e}")
    else:
        print("\nNo improvement found during this run.")


# #load the norm history
# norm_history = torch.load(NORM_HISTORY_FILE)
# print("norm history:", norm_history)








# initial_x = torch.randn(size=)
# saddle_finder = NNSaddleFinder(nequip_potential, initial_x, saddle_index=1, step_size=0.01, momentum=0.8, device='cpu')