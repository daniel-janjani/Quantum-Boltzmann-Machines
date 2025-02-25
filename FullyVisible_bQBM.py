import numpy as np
from scipy.linalg import expm
import pandas as pd
import matplotlib.pyplot as plt

# Define Parameters
N = 8  # Number of visible qubits
M = 8  # Number of modes for data distribution
p = 0.9  # Spin alignment probability with mode centers
eta = 0.7 # Learning rate (increased)
iterations = 35  # Number of optimization steps
Gamma = 2 # Fixed transverse field strength

# Pauli Matrices
I = np.array([[1, 0], [0, 1]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])

# Generate M random center points s^k in {+1, -1}^N
centers = np.random.randint(low=0, high=2, size=(M, N)) # in {0,1}
centers = 2*centers - 1  # map to {+1,-1}

def mixture_data_distribution(all_states, centers, p):  
    """Generate training data as a mixture of M modes using 
        Bernouilli distribution: p^(N-d_kv)*(1-p)^d_kv """
    num_modes = centers.shape[0]  # The number of modes (M=8) is the centers' number of rows
    N_ = centers.shape[1]         # The number of bits (N=10) is the centers' number of columns 
    N_states = all_states.shape[0] # (2^N)
    probs = np.zeros(N_states, dtype=np.float32)
    for s in range(N_states):  
        mode_sum = 0.0
        for k in range(num_modes): 
            d_ks = 0.5 * np.sum(1 - all_states[s, :] * centers[k, :])  # Hamming distance between state s and center k
            mode_sum += p**(N_ - d_ks) * (1 - p)**d_ks  # mixture of Bernoulli distribution
        probs[s] = mode_sum / num_modes   # Generating P_data for each state
    # normalitation
    probs /= probs.sum()
    return probs

def tensor_product(ops):
    """Compute the tensor product of multiple operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

# Compute sigma_z(a), sigma_x(a) and sigma_z(a,b) matrices for each a,b = 1,...,N
b_sigma = np.zeros(N, dtype=object)
gamma_sigma = np.zeros((2**N, 2**N))
W_sigma = np.zeros((N, N),dtype=object)
for a in range(N):
    gamma_sigma += tensor_product([I] * a + [sigma_x] + [I] * (N - a - 1))
    b_sigma[a] = tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1)) 
    for b in range(a + 1, N): 
        W_sigma[a, b] = tensor_product([I] * a + [sigma_z] + [I] * (b - a - 1) + [sigma_z] + [I] * (N - b - 1))

def build_states(N):
    all_states = np.zeros((2**N, N))
    for s in range(N):
        all_states[:, s] = np.diag(b_sigma[s])  # each state is a diagonal element of the sigma_z(a) matrices 
    return all_states

def build_hamiltonian(N, Gamma, b, W):
    """Construct the Fully Visible QBM Hamiltonian with a transverse field."""
    H = np.zeros((2**N, 2**N), dtype=complex) # Size (2^N, 2^N)
    H = -Gamma * gamma_sigma  # Transverse field
    H -= np.dot(b, b_sigma)  
    H -= np.sum(W * W_sigma, axis=None)  
    return H

def compute_density_matrix(H):
    """Compute the density matrix rho = exp(-H) / Z."""
    exp_H = expm(-H)
    Z = np.trace(exp_H)
    rho = exp_H / Z
    return rho, Z

def compute_full_probability_distribution(rho):
    """Compute the full probability distribution P_v from diagonal elements of rho."""
    return np.real(np.diag(rho))  # Extract diagonal elements as probabilities

# Kullback-Leibler (KL) divergence: KL = Likelihood - Likelihood_min
def compute_kl_upper_bound(P_data, P_model):
    """Compute the KL divergence upper bound using P_model: diagonal elements of rho."""
    return np.sum(P_data * np.log((P_data + 1e-12)/(P_model + 1e-12)))

# Compute "positive" and "negative phase" averages: <sigma_z_a>, <sigma_z_a sigma_z_b> for each a,b = 1,2,...,N
def compute_gradient_update(P_data, rho, all_states, N, eta):
    """Compute gradient updates for b and w."""

    z_model_avg = np.zeros(N)
    zz_model_avg = np.zeros((N, N))
    z_data_avg = np.zeros(N)
    zz_data_avg = np.zeros((N, N))

    # Negative phase
    N_states = all_states.shape[0]
    for a in range(N):
      z_model_avg[a] = np.trace(rho @ b_sigma[a]).real
      for b in range(a + 1, N):
         zz_model_avg[a, b] = np.trace(rho @ W_sigma[a, b]).real
         #zz_model_avg[b, a] = zz_model_avg[a, b]
    
    # Positive phase
    N_states = all_states.shape[0]
    for i in range(N_states):
        z_data_avg += P_data[i] * all_states[i, :]
        zz_data_avg += P_data[i] * np.outer(all_states[i, :], all_states[i, :])

    # Compute gradient steps as difference between positive and negative phases
    delta_b = eta * (z_data_avg - z_model_avg)
    delta_W = eta * (zz_data_avg - zz_model_avg)
    return delta_b, delta_W

def optimize_qbm(P_data, all_states, N, Gamma, b, W, eta, iterations):
    """Optimize the Fully Visible Bound-Based QBM."""

    kl_upper_bounds = []
    for it in range(iterations):
        H = build_hamiltonian(N, Gamma, b, W)
        rho, _ = compute_density_matrix(H)

        # Compute model distribution
        P_model = compute_full_probability_distribution(rho)
        
        # Compute and save KL value
        KL_bound = compute_kl_upper_bound(P_data, P_model)
        kl_upper_bounds.append(KL_bound)
        
        delta_b, delta_W = compute_gradient_update(P_data, rho, all_states, N, eta)
        b += delta_b
        W += delta_W

        print(f"Iteration {it+1}/{iterations}, KL Upper Bound: {KL_bound:.6f}, Δb={np.linalg.norm(delta_b):.6f}, Δw={np.linalg.norm(delta_W):.6f}")
    return kl_upper_bounds

# Initialize parameters (b, W) using 'random.seed'
np.random.seed(42)
b = 0.01 * np.random.randn(N)
W = 0.01 * np.random.randn(N, N)

all_states = build_states(N)
P_data = mixture_data_distribution(all_states, centers, p)
print("Check sum of P_data:", P_data.sum().item())  # ~1.0
print("Check dimension of P_data:", P_data.shape)  # ~2^10 = 1024
print(type(P_data))

# Optimize the Fully Visible Bound-Based QBM
kl_upper_bounds = optimize_qbm(P_data, all_states, N, Gamma, b, W, eta, iterations)

# Saving Data frame in CSV
df = pd.DataFrame({"iteration": range(1, iterations + 1), "kl_upper_bounds": kl_upper_bounds})
df.to_csv("FullyVisible_bQBM.csv", index=False)
print("Dati salvati in FullyVisible_bQBM.csv")
