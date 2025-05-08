import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------- helpers ----------------------------------------------------
def gram_schmidt(B, V):
    """B-inner-product Gram-Schmidt; columns of V → B-orthonormal basis."""
    Q = []
    for v in V.T:
        w = v.clone()
        for q in Q:
            w -= (q.T @ B @ w) * q
        nrm = torch.sqrt(w.T @ B @ w)
        if nrm > 1e-14:
            Q.append(w / nrm)
    return torch.stack(Q, dim=1) if Q else torch.zeros((V.shape[0], 0))


def hess_vec(force, x, v, eps=1e-3):
    """Dimer Hessian–vector product:  (F(x-eps v) − F(x+eps v)) / (2 eps)."""
    result = (force(x - eps * v) - force(x + eps * v)) / (2.0 * eps)
    # Convert result to same dtype as input
    return result.to(dtype=x.dtype)


def compute_analytical_eigenpairs(A, p):
    """
    Compute the p smallest eigenvalues and corresponding eigenvectors of matrix A.
    
    Parameters
    ----------
    A : ndarray (n,n) - symmetric matrix
    p : int - number of eigenpairs to return
    
    Returns
    -------
    eigvals : ndarray (p,) - p smallest eigenvalues
    eigvecs : ndarray (n,p) - corresponding eigenvectors (columns)
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvals[:p], eigvecs[:, :p]


# ---------------- SIRQIT core ------------------------------------------------
def sirqit(force, x, V0, k, 
           eps=1e-3, max_iter=100, tol=1e-8, verbose=False):
    """
    Compute the k most-negative Hessian eigenpairs at point x by SIRQIT.

    Parameters
    ----------
    force     : f(x)  →  F(x) = −∇E(x)
    x         : ndarray (n,)   – point where Hessian is evaluated
    V0        : ndarray (n,k)  – initial guesses (columns)
    k         : int            – number of eigenpairs requested
    B         : ndarray (n,n)  – metric matrix
    eps       : float          – dimer finite-difference step
    max_iter  : int            – maximum number of iterations
    tol       : float          – ||gradient||_2 stopping criterion
    verbose   : bool           – print convergence information
    
    Returns
    -------
    eigvals   : ndarray (p,)   – eigenvalues (ascending order)
    eigvecs   : ndarray (n,p)  – corresponding eigenvectors (columns)
    """
    n = len(x)
    B = torch.eye(n, dtype=x.dtype, device=x.device)  # Match input dtype
    
    # Initialize X(0) with B-orthonormal columns
    X = gram_schmidt(B, V0)
    
    if X.shape[1] < k:
        raise ValueError("Initial vectors are linearly dependent.")
    
    # Keep track of best solution so far
    best_eigvals = None
    best_eigvecs = None
    best_g_norm = float('inf')
    
    for i in range(max_iter):
        # Step 1: Ritz projection onto span{X}
        # Compute A*X using finite differences
        AX = torch.zeros_like(X)
        for j in range(X.shape[1]):
            AX[:, j] = hess_vec(force, x, X[:, j], eps)
        
        # Solve reduced eigenproblem (X^T A X) w = d w
        Hred = X.T @ AX  # k×k
        d, Q = torch.linalg.eigh(Hred)
        
        # Sort eigenvalues in ascending order
        idx = torch.argsort(d)
        d = d[idx]
        Q = Q[:, idx]
        
        # Set Y = X Q (now Y^T B Y = I_p)
        Y = X @ Q
        
        # Step 2: Form gradient columns g(y_l) = A y_l - d_l B y_l
        G = torch.zeros_like(Y)
        for l in range(k):
            Ay_l = hess_vec(force, x, Y[:, l], eps)
            G[:, l] = Ay_l - d[l] * (B @ Y[:, l])
        
        # Check convergence
        g_norms = torch.sqrt(torch.sum(G * G, dim=0))
        max_g_norm = torch.max(g_norms)
        
        # Update best solution if this is better
        if max_g_norm < best_g_norm:
            best_g_norm = max_g_norm
            best_eigvals = d[:k].clone()
            best_eigvecs = Y[:, :k].clone()
        
        if verbose:
            print(f"Iteration {i:3d}: max‖g‖ = {max_g_norm:.3e}, eigenvalues: {d[:k]}")
        
        if max_g_norm < tol:
            return d[:k], Y[:, :k]  # Converged
        
        # Step 3: Construct step matrix S (diagonal)
        Z = torch.empty_like(Y)
        
        for l in range(k):
            y_l = Y[:, l]
            g_l = G[:, l]
            
            # Step 3.a: Orthogonalize g_l against z_1,...,z_{k-1} in B-inner product
            g_hat_l = g_l.clone()
            for j in range(l):  # Changed from k to l to avoid using uninitialized Z columns
                z_j = Z[:, j]
                g_hat_l -= (z_j.T @ B @ g_hat_l) / (z_j.T @ B @ z_j) * z_j
            
            # Step 3.b: Compute optimal step size using line search
            best_R = float('inf')
            best_s = 0.0
            
            # Try a range of step sizes with exponential spacing
            for s in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
                if torch.norm(g_hat_l) < 1e-14:
                    continue
                    
                z_trial = y_l - s * g_hat_l
                z_norm = torch.sqrt(z_trial.T @ B @ z_trial)
                if z_norm < 1e-14:
                    continue
                
                z_trial = z_trial / z_norm
                Az_trial = hess_vec(force, x, z_trial, eps)
                
                # Debug prints
                
                R_trial = z_trial.T @ Az_trial
                
                # Update best if this is better (smaller Rayleigh quotient)
                if R_trial < best_R:
                    best_R = R_trial
                    best_s = s
            
            # Step 3.c: z_l = y_l - s_l * g_hat_l
            if torch.norm(g_hat_l) > 1e-14:
                z_l = y_l - best_s * g_hat_l
                
                # Normalize
                z_norm = torch.sqrt(z_l.T @ B @ z_l)
                if z_norm > 1e-14:
                    Z[:, l] = z_l / z_norm
                else:
                    Z[:, l] = y_l  # Fallback if normalization fails
            else:
                Z[:, l] = y_l  # No step if gradient is zero
        
        # Step 4: B-orthonormalize new vectors
        X = gram_schmidt(B, Z)
        
        # Add some randomization if we're stuck
        if i > 10 and i % 10 == 0 and max_g_norm > 0.1:
            # Add small random perturbation to break symmetry
            X_perturb = X + 0.1 * torch.randn_like(X)
            X = gram_schmidt(B, X_perturb)
    
    # Return best solution found if we didn't converge
    if best_eigvals is not None:
        print(f"Warning: SIRQIT did not fully converge after {max_iter} iterations. Final max‖g‖ = {best_g_norm:.3e}")
        return best_eigvals, best_eigvecs
    
    raise RuntimeError(f"SIRQIT did not converge after {max_iter} iterations. Final max‖g‖ = {max_g_norm:.3e}")


def benchmark_comparison():
    """
    Compare performance of SIRQIT vs direct AD methods for different matrix sizes.
    Tests both computation time and accuracy.
    """
    import time
    from torch.autograd.functional import jacobian
    
    # Test different matrix sizes
    sizes = [10, 50, 100, 200, 500, 1000, 2000, 2500]
    k = 3  # number of eigenvalues to find
    n_runs = 3  # number of runs for averaging
    
    # Storage for plotting
    avg_times_sirqit = []
    avg_times_ad = []
    avg_errors = []
    
    print("\nBenchmarking SIRQIT vs Direct AD Methods")
    print("----------------------------------------")
    print(f"Finding {k} lowest eigenvalues")
    print(f"Averaging over {n_runs} runs")
    print("\nMatrix Size | Method      | Time (s)    | Memory (MB)  | Error")
    print("-" * 65)
    
    for n in sizes:
        times_sirqit = []
        times_ad = []
        errors_sirqit = []
        memory_sirqit = []
        memory_ad = []
        
        for run in range(n_runs):
            # Create random test point
            torch.manual_seed(42 + run)  # for reproducibility
            x = torch.randn(n, dtype=torch.float32).requires_grad_(True)
            
            # Create test matrix and force function
            A = torch.randn(n, n, dtype=torch.float32)
            A = (A + A.T) / 2  # make symmetric
            A = A / torch.norm(A)  # normalize
            
            def force(x):
                return -torch.mv(A, x)
            
            # Time SIRQIT method
            torch.cuda.empty_cache()  # clear GPU memory
            start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            start = time.time()
            V0 = torch.randn(n, k, dtype=torch.float32)
            V0, _ = torch.linalg.qr(V0)
            sirqit_vals, _ = sirqit(force, x, V0, k, eps=1e-3, max_iter=100, tol=1e-3, verbose=False)
            end = time.time()
            end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            times_sirqit.append(end - start)
            memory_sirqit.append((end_mem - start_mem) / 1024**2)  # Convert to MB
            
            # Time AD method
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            start = time.time()
            H = -jacobian(force, x)
            true_vals, _ = torch.linalg.eigh(H)
            end = time.time()
            end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            times_ad.append(end - start)
            memory_ad.append((end_mem - start_mem) / 1024**2)
            
            # Compute error
            error = torch.norm(sirqit_vals - true_vals[:k]) / torch.norm(true_vals[:k])
            errors_sirqit.append(error.item())
        
        # Average results
        avg_time_sirqit = sum(times_sirqit) / n_runs
        avg_time_ad = sum(times_ad) / n_runs
        avg_mem_sirqit = sum(memory_sirqit) / n_runs
        avg_mem_ad = sum(memory_ad) / n_runs
        avg_error = sum(errors_sirqit) / n_runs
        
        # Store for plotting
        avg_times_sirqit.append(avg_time_sirqit)
        avg_times_ad.append(avg_time_ad)
        avg_errors.append(avg_error)
        
        # Print results
        print(f"{n:10d} | SIRQIT     | {avg_time_sirqit:10.3f} | {avg_mem_sirqit:10.1f} | {avg_error:.2e}")
        print(f"{' ':10} | AD Jacobian| {avg_time_ad:10.3f} | {avg_mem_ad:10.1f} | Reference")
        print("-" * 65)
    
    # Plotting
    plt.figure(figsize=(6, 4))
    
    # Time comparison plot
    # plt.subplot(2, 1, 1)
    plt.plot(sizes, avg_times_sirqit, 'bo-', label='SIRQIT')
    plt.plot(sizes, avg_times_ad, 'ro-', label='AD Jacobian')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (s)')
    plt.title('Computation Time vs Matrix Size')
    plt.legend()
    plt.grid(True)
    
    # Log-scale plot
    # plt.subplot(2, 1, 2)
    # plt.loglog(sizes, avg_times_sirqit, 'bo-', label='SIRQIT')
    # plt.loglog(sizes, avg_times_ad, 'ro-', label='AD Jacobian')
    # plt.xlabel('Matrix Size')
    # plt.ylabel('Time (s)')
    # plt.title('Computation Time vs Matrix Size (Log-Log Scale)')
    # plt.legend()
    # plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('nequip_saddle/figures/benchmark_results.png')
    print("\nPlot saved as 'benchmark_results.png'")
    
    print("\nNotes:")
    print("- Time: Average computation time in seconds")
    print("- Memory: Peak additional memory usage in MB")
    print("- Error: Relative L2 error of eigenvalues compared to AD method")
    print("- AD Jacobian computes full Hessian, SIRQIT only requested eigenvalues")
    
    # Print scaling analysis
    print("\nScaling Analysis:")
    # Compute approximate scaling for larger matrices
    large_idx = [i for i, s in enumerate(sizes) if s >= 500]
    if len(large_idx) >= 2:
        # Take log of both sizes and times
        log_sizes = np.log(np.array(sizes)[large_idx])
        log_times_sirqit = np.log(np.array(avg_times_sirqit)[large_idx])
        log_times_ad = np.log(np.array(avg_times_ad)[large_idx])
        
        # Linear fit in log-log space gives power law exponent
        b_sirqit, m_sirqit = np.polyfit(log_sizes, log_times_sirqit, 1)
        b_ad, m_ad = np.polyfit(log_sizes, log_times_ad, 1)
        
        print(f"Approximate scaling for n ≥ 500:")
        print(f"SIRQIT:      O(n^{b_sirqit:.2f})")
        print(f"AD Jacobian: O(n^{b_ad:.2f})")

if __name__ == "__main__":
    benchmark_comparison()