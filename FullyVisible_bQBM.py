import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from itertools import product

# Constants
N = 7  # Number of visible qubits
M = 8  # Number of modes for data distribution
p = 0.9  # Probability of alignment
eta =1  # Learning rate (increased)
iterations = 40  # Number of optimization steps
Gamma = 2  # Fixed transverse field strength

# Pauli Matrices
I = np.array([[1, 0], [0, 1]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])

def generate_data_distribution(N, M, p):
    """Generate a fully visible data distribution P_v^data."""
    modes = [np.random.choice([-1, 1], N) for _ in range(M)]
    data_distribution = {}

    for state in product([-1, 1], repeat=N):
        state = np.array(state)
        P_v = sum(p**(N - np.sum(state != mode)) * (1 - p)**np.sum(state != mode) for mode in modes) / M
        data_distribution[tuple(state)] = P_v

    # Normalize the distribution
    total = sum(data_distribution.values())
    return {k: v / total for k, v in data_distribution.items()}

def tensor_product(ops):
    """Compute the tensor product of multiple operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def build_hamiltonian(N, Gamma, b, w):
    """Construct the Fully Visible QBM Hamiltonian with a transverse field."""
    H = np.zeros((2**N, 2**N), dtype=complex)

    for a in range(N):
        H -= Gamma * tensor_product([I] * a + [sigma_x] + [I] * (N - a - 1))  # Transverse field
        H -= b[a] * tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1))  # Local bias

    for a in range(N):
        for b in range(a + 1, N):
            H -= w[a, b] * tensor_product([I] * a + [sigma_z] + [I] * (b - a - 1) + [sigma_z] + [I] * (N - b - 1))

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

def compute_kl_upper_bound(data_distribution, rho):
    """Compute the KL divergence upper bound using diagonal elements of rho."""
    P_model = compute_full_probability_distribution(rho)
    kl_upper_bound = 0.0

    for idx, (visible_state, P_data) in enumerate(data_distribution.items()):
        P_v = P_model[idx]  # Directly read diagonal probability

        if P_data > 0 and P_v > 1e-6:
            kl_upper_bound += P_data * np.abs(np.log(P_data / P_v))

    return kl_upper_bound

def compute_gradient_update(data_distribution, H, N, b, w, eta):
    """Compute gradient updates for b and w."""
    rho, _ = compute_density_matrix(H)
    grad_b = np.zeros(N)
    grad_w = np.zeros((N, N))

    P_model = compute_full_probability_distribution(rho)

    for idx, (visible_state, P_data) in enumerate(data_distribution.items()):
        P_v = P_model[idx]  # Directly use diagonal elements of rho

        if P_v > 0:
            avg_sigma_z = np.array([
                np.trace(rho @ tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1))).real for a in range(N)
            ])
            grad_b += P_data * avg_sigma_z

            for a in range(N):
                for b in range(a + 1, N):
                    avg_sigma_z_z = np.trace(
                        rho @ tensor_product([I] * a + [sigma_z] + [I] * (b - a - 1) + [sigma_z] + [I] * (N - b - 1))
                    ).real
                    grad_w[a, b] += P_data * avg_sigma_z_z
                    grad_w[b, a] = grad_w[a, b]  # Ensure symmetry

    print(f"Gradient norms: ||grad_b||={np.linalg.norm(grad_b):.6f}, ||grad_w||={np.linalg.norm(grad_w):.6f}")
    
    delta_b = eta * grad_b
    delta_w = eta * grad_w
    return delta_b, delta_w

def optimize_qbm(data_distribution, N, Gamma, b, w, eta, iterations):
    """Optimize the Fully Visible Bound-Based QBM."""
    kl_upper_bounds = []

    for it in range(iterations):
        H = build_hamiltonian(N, Gamma, b, w)
        rho, _ = compute_density_matrix(H)

        KL_bound = compute_kl_upper_bound(data_distribution, rho)
        kl_upper_bounds.append(KL_bound)

        delta_b, delta_w = compute_gradient_update(data_distribution, H, N, b, w, eta)
        b -= delta_b
        w -= delta_w

        print(f"Iteration {it+1}/{iterations}, KL Upper Bound: {KL_bound:.6f}, Δb={np.linalg.norm(delta_b):.6f}, Δw={np.linalg.norm(delta_w):.6f}")

    return kl_upper_bounds

# Initialize parameters
b = np.random.normal(0, 0.5, N)
w = np.random.normal(0, 0.5, (N, N)) #increased since the KL would go down very slowly
w = (w + w.T) / 2  # Ensure symmetry

# Generate synthetic data distribution
data_distribution = generate_data_distribution(N, M, p)

# Optimize the Fully Visible Bound-Based QBM
kl_upper_bounds = optimize_qbm(data_distribution, N, Gamma, b, w, eta, iterations)

# Plot KL divergence upper bound over iterations
plt.figure(figsize=(8, 6))
plt.plot(range(1, iterations + 1), kl_upper_bounds, marker='o')
plt.xlabel("Iteration")
plt.ylabel("KL Upper Bound")
plt.title("KL Upper Bound Over Iterations (FV-bQBM)")
plt.grid()
plt.show()

