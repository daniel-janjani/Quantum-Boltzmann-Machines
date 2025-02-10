import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import itertools
import pandas as pd

# Constants
N = 10  # Number of visible qubits
M = 8  # Number of modes for data distribution
p = 0.9  # Probability of alignment
eta = 0.2  # Learning rate (increased)
iterations = 35  # Number of optimization steps
Gamma = 2  # Fixed transverse field strength

# Pauli Matrices
I = np.array([[1, 0], [0, 1]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])

# Generate M random center points s^k in {+1, -1}^N
centers = np.random.randint(low=0, high=2, size=(M, N)) # in {0,1}
centers = 2*centers - 1  # map to {+1,-1}

def mixture_data_distribution(all_states, centers, p):  
    num_modes = centers.shape[0]  # The number of modes (M=8) is the centers' number of rows
    N_ = centers.shape[1]         # The number of bits (N=10) is the centers' number of columns 
    N_states = all_states.shape[0] 
    probs = np.zeros(N_states, dtype=np.float32)
    for s in range(N_states):  # Per ogni stato possibile
        mode_sum = 0.0
        for k in range(num_modes):  # Per ogni centro
            d_ks = 0.5 * np.sum(1 - all_states[s, :] * centers[k, :])  
            mode_sum += p**(N_ - d_ks) * (1 - p)**d_ks 
        probs[s] = mode_sum / num_modes  
    # normalitation
    probs /= probs.sum()
    return probs

def tensor_product(ops):
    """Compute the tensor product of multiple operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def build_hamiltonian(N, Gamma, b, W):
    """Construct the Fully Visible QBM Hamiltonian with a transverse field."""
    H = np.zeros((2**N, 2**N), dtype=complex)
    all_states = np.zeros((2**N, N))

    for a in range(N):
        H -= Gamma * tensor_product([I] * a + [sigma_x] + [I] * (N - a - 1))  # Transverse field
        H -= b[a] * tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1))  
        all_states[:, a] = np.diag(tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1)))
        for i in range(a + 1, N):
            H -= W[a, i] * tensor_product([I] * a + [sigma_z] + [I] * (i - a - 1) + [sigma_z] + [I] * (N - i - 1))
    return H, all_states

def compute_density_matrix(H):
    """Compute the density matrix rho = exp(-H) / Z."""
    exp_H = expm(-H)
    Z = np.trace(exp_H)
    rho = exp_H / Z
    return rho, Z

def compute_full_probability_distribution(rho):
    """Compute the full probability distribution P_v from diagonal elements of rho."""
    return np.real(np.diag(rho))  # Extract diagonal elements as probabilities

def compute_kl_upper_bound(P_data, P_model):
    """Compute the KL divergence upper bound using P_model: diagonal elements of rho."""
    return np.sum(P_data * np.log((P_data + 1e-12)/(P_model + 1e-12)))

def compute_gradient_update(P_data, rho, all_states, N, eta):
    """Compute gradient updates for b and w."""
    z_model_avg = np.zeros(N)
    zz_model_avg = np.zeros((N, N))
    z_data_avg = np.zeros(N)
    zz_data_avg = np.zeros((N, N))

    N_states = all_states.shape[0]
    for a in range(N):
      z_model_avg[a] = np.trace(rho @ tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1))).real
      for b in range(a + 1, N):
         zz_model_avg[a, b] = np.trace(rho @ tensor_product([I] * a + [sigma_z] + [I] * (b - a - 1) + [sigma_z] + [I] * (N - b - 1))).real
         zz_model_avg[b, a] = zz_model_avg[a, b]
     
    N_states = all_states.shape[0]
    for i in range(N_states):
        z_data_avg += P_data[i] * all_states[i, :]
        zz_data_avg += P_data[i] * np.outer(all_states[i, :], all_states[i, :])

    delta_b = eta * (z_data_avg - z_model_avg)
    delta_W = eta * (zz_data_avg - zz_model_avg)
    return delta_b, delta_W

def optimize_qbm(P_data, all_states, N, Gamma, b, W, eta, iterations):
    """Optimize the Fully Visible Bound-Based QBM."""

    kl_upper_bounds = []
    for it in range(iterations):
        H, _ = build_hamiltonian(N, Gamma, b, W)
        rho, _ = compute_density_matrix(H)
        P_model = compute_full_probability_distribution(rho)

        KL_bound = compute_kl_upper_bound(P_data, P_model)
        kl_upper_bounds.append(KL_bound)

        delta_b, delta_W = compute_gradient_update(P_data, rho, all_states, N, eta)
        b += delta_b
        W += delta_W

        print(f"Iteration {it+1}/{iterations}, KL Upper Bound: {KL_bound:.6f}, Δb={np.linalg.norm(delta_b):.6f}, Δw={np.linalg.norm(delta_W):.6f}")
    return kl_upper_bounds

# Initialize parameters
b = 0.01 * np.random.randn(N)
W = 0.01 * np.random.randn(N, N)

_, all_states = build_hamiltonian(N, Gamma, b, W)
P_data = mixture_data_distribution(all_states, centers, p)
print("Check sum of P_data:", P_data.sum().item())  # ~1.0
print("Check dimension of P_data:", P_data.shape)  # ~2^10 = 1024
print(type(P_data))

# Optimize the Fully Visible Bound-Based QBM
kl_upper_bounds = optimize_qbm(P_data, all_states, N, Gamma, b, W, eta, iterations)


df = pd.DataFrame({"iteration": range(1, iterations + 1), "kl_upper_bounds": kl_upper_bounds})

# Saving Data frame in CSV
df.to_csv("FullyVisible_bQBM.csv", index=False)
print("Dati salvati in FullyVisible_bQBM.csv")

df = pd.read_csv("FullyVisible_bQBM.csv")

# Plot KL divergence upper bound over iterations
plt.figure(figsize=(8, 6))
plt.plot(df['iteration'], df['kl_upper_bounds'], marker='o', label='KL Upper Bound over Iterations')
plt.xlabel("Iteration")
plt.ylabel("KL Upper Bound")
plt.title("KL Upper Bound Over Iterations (FV-bQBM)")
plt.grid()
plt.show()
