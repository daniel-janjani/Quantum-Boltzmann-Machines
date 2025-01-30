import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from itertools import product
import matplotlib.pyplot as plt

# Constants
N = 3  # Number of qubits
M = 8  # Number of modes
p = 0.9  # Probability of alignment (Poissonian)

def generate_data_distribution(N, M, p):
    """Generate probability distribution for Fully Visible QBM."""
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
        result = np.kron(result, op)  # Kronecker product
    return result

def build_hamiltonian(N, Gamma, b, w):
    """Construct the Fully Visible QBM Hamiltonian."""
    H = np.zeros((2**N, 2**N), dtype=complex)
    
    for a in range(N):
        H -= Gamma * tensor_product([I] * a + [sigma_x] + [I] * (N - a - 1))
        H -= b[a] * tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1))
        
    for a in range(N):
        for b in range(a + 1, N):  # Only compute for unique pairs (a,b)
            H -= w[a, b] * tensor_product(
                [I] * a + [sigma_z] + [I] * (b - a - 1) + [sigma_z] + [I] * (N - b - 1)
            )
    return H

steps = 5  # Trotter steps
def trotter_decomposition(H):
    """Approximate e^(-H) using a Trotter decomposition."""
    delta_tau = 1.0 / steps
    exp_H_approx = np.eye(H.shape[0], dtype=complex)

    for _ in range(steps):
        exp_H_approx = expm_multiply(-delta_tau * H, exp_H_approx)

    return exp_H_approx

def compute_partial_H(H, partial_H, n=100):
    """Compute derivative of e^(-H) using numerical integration with Trotterization."""
    delta_tau = 1.0 / n
    partial_eH = np.zeros_like(H, dtype=complex)

    for m in range(1, n + 1):
        tau = m * delta_tau
        exp1 = trotter_decomposition(-tau * H)  # First part of exponentiation
        exp2 = trotter_decomposition(-(1 - tau) * H)  # Second part
        partial_eH += exp1 @ partial_H @ exp2 * delta_tau  # Numerical integration

    return -partial_eH


def compute_density_matrix(H):
    """Compute the density matrix using Trotter decomposition for e^(-H)."""
    exp_H_approx = trotter_decomposition(H)
    Z = np.trace(exp_H_approx)
    return exp_H_approx / Z, Z

def compute_full_probability_distribution(rho):
    """Compute the full probability distribution P(v) = diag(rho)."""
    return np.real(np.diag(rho))

def compute_kl_divergence(data_distribution, rho):
    """Compute KL divergence for Fully Visible QBM."""
    P_model = compute_full_probability_distribution(rho)
    kl_divergence = 0.0
    for idx, (visible_state, P_data) in enumerate(data_distribution.items()):
        P_v = P_model[idx]
        if P_data > 0 and P_v > 0:  # Avoid log(0) or division by zero
            kl_divergence += P_data * np.log(P_data / P_v)
            print(f"State {visible_state}: P_data={P_data:.6f}, P_model={P_v:.6f}, log-ratio={np.log(P_data / P_v):.6f}")

    return kl_divergence

def compute_gradient_update(data_distribution, H, N, Gamma, b, w, eta, n=100):
    """Compute parameter updates for Fully Visible QBM."""
    rho, _ = compute_density_matrix(H)
    grad_Gamma, grad_b, grad_w = 0.0, np.zeros(N), np.zeros((N, N))

    # Define partial_H matrices
    partial_H_Gamma = -sum(tensor_product([I] * a + [sigma_x] + [I] * (N - a - 1)) for a in range(N))
    partial_H_b = [
        -tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1)) for a in range(N)
    ]
    partial_H_w = [
        [
            -tensor_product(
                [I] * a + [sigma_z] + [I] * (b - a - 1) + [sigma_z] + [I] * (N - b - 1)
            )
            if a < b
            else np.zeros_like(H)
            for b in range(N)
        ]
        for a in range(N)
    ]

    P_model = compute_full_probability_distribution(rho)

    for visible_state, P_data in data_distribution.items():
        P_v = P_model[list(data_distribution.keys()).index(visible_state)]  # Get diagonal entry

        if P_v > 0:
            partial_eH_Gamma = compute_partial_H(H, partial_H_Gamma, n)
            grad_Gamma += P_data * np.trace(partial_H_Gamma @ rho).real

            for a in range(N):
                partial_eH_b = compute_partial_H(H, partial_H_b[a], n)
                grad_b[a] += P_data * np.trace(partial_H_b[a] @ rho).real

            for a in range(N):
                for b in range(a + 1, N):
                    partial_eH_w = compute_partial_H(H, partial_H_w[a][b], n)
                    grad_w[a, b] += P_data * np.trace(partial_H_w[a][b] @ rho).real
                    grad_w[b, a] = grad_w[a, b]  # Ensure symmetry

    delta_Gamma = -eta * grad_Gamma
    delta_b = -eta * grad_b
    delta_w = -eta * grad_w

    return delta_Gamma, delta_b, delta_w

def optimize_qbm(data_distribution, N, Gamma, b, w, eta, iterations):
    """Optimize Fully Visible QBM using gradient descent."""
    kl_divergences = []

    for it in range(iterations):
        H = build_hamiltonian(N, Gamma, b, w)
        rho, _ = compute_density_matrix(H)

        KL = compute_kl_divergence(data_distribution, rho)
        kl_divergences.append(KL)

        delta_Gamma, delta_b, delta_w = compute_gradient_update(data_distribution, H, N, Gamma, b, w, eta)

        Gamma -= delta_Gamma
        b -= delta_b
        w -= delta_w

        print(f"Iteration {it+1}/{iterations}, KL Divergence: {KL:.6f}")

    return kl_divergences

# Run Optimization
Gamma = 1.0
eta = 0.005
b = np.random.normal(0, 0.01, N) 
w = np.random.normal(0, 0.01, (N, N))
iterations = 30

data_distribution = generate_data_distribution(N, M, p)
kl_divergences = optimize_qbm(data_distribution, N, Gamma, b, w, eta, iterations)

plt.plot(kl_divergences)
plt.xlabel("Iteration")
plt.ylabel("KL Divergence")
plt.show()


