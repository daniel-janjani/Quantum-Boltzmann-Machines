import numpy as np
from scipy.linalg import expm
from itertools import product
import matplotlib.pyplot as plt

# Define constants
N = 3  # Number of qubits
M = 8  # Number of modes
p = 0.9  # Probability of alignment
eta = 0.1  # Learning rate
iterations = 40  # Number of optimization steps

def generate_data_distribution(N, M, p):
    """Generate P_v^data based on M modes and parameter p."""
    modes = [np.random.choice([-1, 1], N) for _ in range(M)]
    data_distribution = {}

    for state in product([-1, 1], repeat=N):
        state = np.array(state)
        P_v = 0
        for mode in modes:
            d_v = np.sum(state != mode)  # Hamming distance
            P_v += p**(N - d_v) * (1 - p)**d_v
        data_distribution[tuple(state)] = P_v / M
    
    # Normalize the distribution
    total = sum(data_distribution.values())
    return {k: v / total for k, v in data_distribution.items()}

# Define Pauli matrices and identity matrix
I = np.array([[1, 0], [0, 1]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])

def tensor_product(operators):
    """Compute the tensor product of a list of operators."""
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result

def build_hamiltonian(N, Gamma, b, w):
    """Construct the Hamiltonian H."""
    H = np.zeros((2**N, 2**N), dtype=complex)
    for a in range(N):
        H -= Gamma * tensor_product([I] * a + [sigma_x] + [I] * (N - a - 1))
        H -= b[a] * tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1))
        
    for a, b in product(range(N), repeat=2):
        if a < b:
            H -= w[a, b] * tensor_product(
                [I] * a + [sigma_z] + [I] * (b - a - 1) + [sigma_z] + [I] * (N - b - 1)
            )
    return H

def compute_density_matrix(H):
    """Compute the density matrix rho = Z^-1 * exp(-H)."""
    exp_H = expm(-H)
    Z = np.trace(exp_H)
    return exp_H / Z, Z

def compute_clamped_hamiltonian(H, Lambda_v):
    """Compute the clamped Hamiltonian H_v = H - ln(Lambda_v)."""
    clamped_H = H.copy()  # Clone original Hamiltonian
    Lambda_v = np.maximum(Lambda_v, 1e-12)  # Avoid log(0)
    H_v = clamped_H - np.log(Lambda_v)
    
    # Ensure clamping: Set large energy penalty for states different from v
    np.fill_diagonal(H_v, np.where(np.diag(Lambda_v) > 0, np.diag(H_v), 1e6))
    
    return H_v


def compute_kl_upper_bound(data_distribution, H, N):
    """Compute the upper bound on the KL divergence with numerical stability fixes."""
    rho, Z = compute_density_matrix(H)
    kl_upper_bound = 0.0
    
    for visible_state, P_data in data_distribution.items():
        Lambda_v = compute_projection_operator(visible_state, N)
        H_v = compute_clamped_hamiltonian(H, Lambda_v)
        rho_v, Z_v = compute_density_matrix(H_v)
        
        if P_data > 0 and Z_v > 1e-12:
            # Use absolute value to ensure positivity
            kl_upper_bound += P_data * np.abs(np.log((Z_v + 1e-12) / (Z + 1e-12)))  

    return kl_upper_bound


def compute_projection_operator(visible_state, N):
    """Compute the projection operator Lambda_v correctly."""
    Lambda_v = np.eye(2**N, dtype=complex)  # Identity matrix
    for idx, v in enumerate(visible_state):
        op = (I + v * sigma_z) / 2
        Lambda_v = Lambda_v @ tensor_product([op if i == idx else I for i in range(N)])
    
    return np.maximum(Lambda_v, 1e-12)  # Avoid zero elements

def compute_gradient_update(data_distribution, H, N, b, w, eta):
    """Compute parameter updates for b and w."""
    rho, _ = compute_density_matrix(H)
    grad_b, grad_w = np.zeros(N), np.zeros((N, N))
    
    for visible_state, P_data in data_distribution.items():
        Lambda_v = compute_projection_operator(visible_state, N)
        H_v = compute_clamped_hamiltonian(H, Lambda_v)
        rho_v, _ = compute_density_matrix(H_v)
        
        avg_sigma_z_v = np.array([np.trace(rho_v @ tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1))).real for a in range(N)])
        avg_sigma_z = np.array([np.trace(rho @ tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1))).real for a in range(N)])
        grad_b += P_data * (avg_sigma_z_v - avg_sigma_z)
        
        for a in range(N):
            for b in range(a + 1, N):
                avg_sigma_z_z_v = np.trace(rho_v @ tensor_product([I] * a + [sigma_z] + [I] * (b - a - 1) + [sigma_z] + [I] * (N - b - 1))).real
                avg_sigma_z_z = np.trace(rho @ tensor_product([I] * a + [sigma_z] + [I] * (b - a - 1) + [sigma_z] + [I] * (N - b - 1))).real
                grad_w[a, b] += P_data * (avg_sigma_z_z_v - avg_sigma_z_z)
                grad_w[b, a] = grad_w[a, b]  # Symmetry
    
    delta_b = eta * grad_b
    delta_w = eta * grad_w
    return delta_b, delta_w

def optimize_qbm(data_distribution, N, Gamma, b, w, eta, iterations):
    """Optimize the bound-based Quantum Boltzmann Machine."""
    kl_upper_bounds = []
    
    for it in range(iterations):
        H = build_hamiltonian(N, Gamma, b, w)
        KL_bound = compute_kl_upper_bound(data_distribution, H, N)
        kl_upper_bounds.append(KL_bound)
        
        delta_b, delta_w = compute_gradient_update(data_distribution, H, N, b, w, eta)
        b -= delta_b
        w -= delta_w
        
        print(f"Iteration {it+1}/{iterations}, KL Upper Bound: {KL_bound:.6f}")
    
    return kl_upper_bounds

# Initialize parameters
Gamma = 2.5
b = np.random.normal(0, 0.5, N)  # Larger values for b
w = np.random.normal(0, 0.5, (N, N))  # Larger initial values
w = (w + w.T) / 2  # Ensure symmetry


data_distribution = generate_data_distribution(N, M, p)
kl_upper_bounds = optimize_qbm(data_distribution, N, Gamma, b, w, eta, iterations)

plt.figure(figsize=(8, 6))
plt.plot(range(1, iterations + 1), kl_upper_bounds, marker='o')
plt.xlabel("Iteration")
plt.ylabel("KL Upper Bound")
plt.title("KL Upper Bound over Iterations")
plt.grid()
plt.show()