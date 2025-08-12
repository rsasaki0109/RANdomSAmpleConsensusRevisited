import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth, svd

# --- 1. Data Generation ---
def generate_data(n, d, r_star, epsilon, noise_sigma=0.01):
    """
    Function to generate data with noise and adversarial outliers.

    Args:
        n (int): Total number of samples
        d (int): Ambient dimension
        r_star (int): Dimension of the true subspace
        epsilon (float): Fraction of outliers (0 to 1)
        noise_sigma (float): Standard deviation of Gaussian noise added to inliers

    Returns:
        X (np.ndarray): Generated data (d x n)
        U_star (np.ndarray): Basis of the true subspace (d x r_star)
    """
    # Randomly generate the true subspace U*
    U_star = orth(np.random.randn(d, r_star))

    # Calculate the number of inliers and outliers
    n_inliers = int(n * (1 - epsilon))
    n_outliers = n - n_inliers

    # Generate inliers
    # Clean data is represented as a linear combination of U*
    clean_inliers = U_star @ np.random.randn(r_star, n_inliers)
    # Add Gaussian noise
    noise = noise_sigma * np.random.randn(d, n_inliers)
    inliers = clean_inliers + noise

    # Generate outliers
    # Outliers are generated from a subspace orthogonal to U*
    U_outlier = orth(np.random.randn(d, r_star))
    # Adjust to be orthogonal to U_star
    U_outlier -= U_star @ (U_star.T @ U_outlier)
    U_outlier = orth(U_outlier)
    
    # Make outliers have a larger variance than inliers
    outliers = U_outlier @ np.random.randn(r_star, n_outliers) * 10
    
    # Concatenate the datasets
    X = np.hstack((inliers, outliers))
    
    # Shuffle the data
    perm = np.random.permutation(n)
    X = X[:, perm]
    
    return X, U_star

# --- 2. Distance Calculation ---
def subspace_distance(U1, U2):
    """Calculate the distance between two subspaces."""
    # Project U2 onto U1 and calculate the norm of the residual
    residual = U2 - U1 @ (U1.T @ U2)
    return np.linalg.norm(residual, 'fro')

# --- 3. Algorithm Implementation ---

def pca(X, r_star):
    """Subspace recovery using Principal Component Analysis (PCA)."""
    U, _, _ = svd(X, full_matrices=False)
    return U[:, :r_star]

def ransac_plus(X, r_star_true):
    """Implementation of RANSAC+ (Algorithm 1)."""
    d, n = X.shape

    # --- Algorithm 2: Coarse-grained Estimation ---
    B = 2
    V = None
    # The paper defines a threshold, but for simplicity here,
    # we search until the batch size is a constant multiple of the true dimension.
    while B < 2 * r_star_true:
        # Randomly select B samples
        indices = np.random.choice(n, B, replace=False)
        X_B = X[:, indices]
        
        # Calculate the candidate subspace V
        V_candidate, _, _ = svd(X_B, full_matrices=False)
        V = V_candidate[:, :B]
        
        # Calculate the median of distances (omitted for simplicity).
        # This median would normally be used for the termination condition.
        B *= 2
    
    if V is None: # If V was not found
        return pca(X, r_star_true)
        
    r_hat = V.shape[1]

    # --- Algorithm 3: Fine-grained Estimation ---
    # Project the data onto the coarse subspace V
    X_hat = V.T @ X
    
    T = 50  # Number of trials
    B_fine = int(2.5 * r_hat) # Number of samples
    
    # Adjust B_fine so it does not exceed n
    if B_fine >= n:
        B_fine = n - 1
        
    singular_values_storage = []
    best_batch_idx = -1
    min_r_plus_1_sv = float('inf')

    for j in range(T):
        indices = np.random.choice(n, B_fine, replace=False)
        X_j_hat = X_hat[:, indices]
        
        _, s, _ = svd(X_j_hat, full_matrices=False)
        
        # Find the batch with the smallest (r_star_true + 1)-th singular value.
        # The paper uses the estimated dimension, but we substitute the true dimension here.
        if len(s) > r_star_true and s[r_star_true] < min_r_plus_1_sv:
            min_r_plus_1_sv = s[r_star_true]
            best_batch_idx = j
            
        singular_values_storage.append(s)

    # Calculate the final subspace from the best batch
    if best_batch_idx != -1:
      indices = np.random.choice(n, B_fine, replace=False)
      X_k_hat = X_hat[:, indices]
    else: # If not found, use a random batch
      indices = np.random.choice(n, B_fine, replace=False)
      X_k_hat = X_hat[:, indices]

    U_k_hat, _, _ = svd(X_k_hat, full_matrices=False)
    
    # The final subspace
    U_final = V @ U_k_hat[:, :r_star_true]
    
    return U_final


# --- 4. Experiment and Visualization ---
# Parameter settings
n = 500       # Number of samples
d = 100       # Ambient dimension
r_star = 10     # Dimension of the true subspace
noise_sigma = 0.01

# Range of corruption parameter epsilon
epsilons = np.linspace(0.0, 0.4, 9)

# Dictionary to store results
results = {
    'PCA': [],
    'RANSAC+': []
}

# Run experiment for each epsilon
for epsilon in epsilons:
    print(f"Running experiment for epsilon = {epsilon:.2f}")
    
    # Average over multiple trials
    num_trials = 5
    dist_pca_trials = []
    dist_ransac_plus_trials = []
    
    for _ in range(num_trials):
        # Data generation
        X, U_star = generate_data(n, d, r_star, epsilon, noise_sigma)
        
        # PCA
        U_pca = pca(X, r_star)
        dist_pca_trials.append(subspace_distance(U_star, U_pca))

        # RANSAC+
        U_ransac_plus = ransac_plus(X, r_star)
        dist_ransac_plus_trials.append(subspace_distance(U_star, U_ransac_plus))

    results['PCA'].append(np.mean(dist_pca_trials))
    results['RANSAC+'].append(np.mean(dist_ransac_plus_trials))

# Visualization
plt.figure(figsize=(10, 7))
plt.plot(epsilons, results['PCA'], 'o-', label='PCA', linewidth=2, markersize=8)
plt.plot(epsilons, results['RANSAC+'], 's-r', label='RANSAC+', linewidth=2, markersize=8)

plt.xlabel('Adversarial Corruption Parameter ε', fontsize=14)
plt.ylabel('Distance from the True Subspace', fontsize=14)
plt.title('Performance of RSR Methods', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--")
plt.yscale('log') # Set y-axis to log scale
plt.xticks(epsilons)
plt.tight_layout()

# Save the figure as an image
plt.savefig('image/rsr_performance_comparison.png')

print("\nVisualization graph has been saved as 'rsr_performance_comparison.png'.")
