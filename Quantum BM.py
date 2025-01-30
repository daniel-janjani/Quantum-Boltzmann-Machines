import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from itertools import product
import matplotlib.pyplot as plt


# Constants
N = 5  # Number of qubits (let's keep it simple)

M = 8  # Number of modes
p = 0.9  # Probability of alignment (Poissonian)

def generate_data_distribution(N, M, p):
    modes = [np.random.choice([-1, 1], N) for _ in range(M)]
    data_distribution = {}

    for state in product([-1, 1], repeat=N):
        state = np.array(state)
        P_v = 0
        for mode in modes:
            d_v = np.sum(state != mode)  # Hamming distance; fascinating!
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
        result = np.kron(result, op) #Kronecker product 
    return result

def build_hamiltonian(N, Gamma, b, w):
    """Construct the Hamiltonian H."""
    H = np.zeros((2**N, 2**N), dtype=complex)
    for a in range(N):
        # Transverse field term: -Gamma * sigma_a^x
        H -= Gamma * tensor_product([I] * a + [sigma_x] + [I] * (N - a - 1))
        
        H -= b[a] * tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1))
        
    for a, b in product(range(N), repeat=2):
        if a < b:  # Avoid double-counting pairs
            H -= w[a, b] * tensor_product(
                [I] * a + [sigma_z] + [I] * (b - a - 1) + [sigma_z] + [I] * (N - b - 1)
            )
    return H

steps=5 #otherwise it takes too much
def trotter_decomposition(H):
    """Approximate e^(-H) using a Trotter decomposition."""
    delta_tau = 1.0 / steps
    exp_H_approx = np.eye(H.shape[0], dtype=complex) #identity matrix of the same size as H

    for _ in range(steps):
        exp_H_approx = expm_multiply(-delta_tau * H, exp_H_approx)

    return exp_H_approx


def compute_density_matrix(H):
    """Compute the density matrix using Trotter decomposition for e^(-H)."""
    exp_H_approx = trotter_decomposition(H)
    Z = np.trace(exp_H_approx)
    return exp_H_approx / Z, Z

def compute_projection_operator(visible_state, N):
    """Compute the projection operator Lambda_v."""
    Lambda_v = np.eye(2**N, dtype=complex)
    for idx, v in enumerate(visible_state):
        op = (I + v * sigma_z) / 2
        Lambda_v = Lambda_v @ tensor_product([op if i == idx else I for i in range(N)])
    return Lambda_v

def compute_marginal_probability(visible_state, rho, N):
    """Compute P_v = Tr[Lambda_v * rho]."""
    Lambda_v = compute_projection_operator(visible_state, N)
    return np.trace(Lambda_v @ rho).real

def compute_partial_H(H, partial_H, n=100):
    """Compute derivative of e^(-H) using numerical integration with Trotterization."""
    delta_tau = 1.0 / n
    partial_eH = np.zeros_like(H, dtype=complex)

    for m in range(1, n + 1):
        tau = m * delta_tau
        exp1 = trotter_decomposition(-tau * H)  # Use Trotterized exponentiation
        exp2 = trotter_decomposition(-(1 - tau) * H)  
        partial_eH += exp1 @ partial_H @ exp2 * delta_tau

    return -partial_eH


def compute_gradient_update(data_distribution, H, N, Gamma, b, w, eta, n=100):
    """
    Compute parameter updates (delta_Gamma, delta_b, delta_w) using the given gradients.
    """
    rho, _ = compute_density_matrix(H)  # Compute the density matrix
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

    for visible_state, P_data in data_distribution.items():
        Lambda_v = compute_projection_operator(visible_state, N)
        P_v = compute_marginal_probability(visible_state, rho, N)

        if P_v > 0:
            # manual update for Gamma
            partial_eH_Gamma = compute_partial_H(H, partial_H_Gamma, n)
            trace_term_Gamma = np.trace(Lambda_v @ partial_eH_Gamma) / np.trace(Lambda_v @ expm(-H))
            boltzmann_avg_Gamma = np.trace(partial_H_Gamma @ rho).real
            grad_Gamma += P_data * (trace_term_Gamma + boltzmann_avg_Gamma)

            # manual update for b
            for a in range(N):
                partial_eH_b = compute_partial_H(H, partial_H_b[a], n)
                trace_term_b = np.trace(Lambda_v @ partial_eH_b) / np.trace(Lambda_v @ expm(-H))
                boltzmann_avg_b = np.trace(partial_H_b[a] @ rho).real
                grad_b[a] += P_data * (trace_term_b + boltzmann_avg_b)

            # manual update for w
            for a in range(N):
                for b in range(a + 1, N):
                    partial_eH_w = compute_partial_H(H, partial_H_w[a][b], n)
                    trace_term_w = np.trace(Lambda_v @ partial_eH_w) / np.trace(Lambda_v @ expm(-H))
                    boltzmann_avg_w = np.trace(partial_H_w[a][b] @ rho).real
                    grad_w[a, b] += P_data * (trace_term_w + boltzmann_avg_w)
                    grad_w[b, a] = grad_w[a, b]  # Symmetry

    # Parameter updates
    delta_Gamma = -eta * grad_Gamma
    delta_b = -eta * grad_b
    delta_w = -eta * grad_w

    return delta_Gamma, delta_b, delta_w

def compute_kl_divergence(data_distribution, H, N):
    """
    Compute the KL divergence:
    KL = sum(P_v^data * log(P_v^data / P_v)).
    """
    rho, _ = compute_density_matrix(H)  # Compute the density matrix
    kl_divergence = 0.0

    for visible_state, P_data in data_distribution.items():
        P_v = compute_marginal_probability(visible_state, rho, N)
        if P_data > 0 and P_v > 0:  # Avoid log(0) or division by zero
            kl_divergence += P_data * np.log(P_data / P_v)

    return kl_divergence

def optimize_qbm(data_distribution, N, Gamma, b, w, eta, iterations, n_integration_steps=100):
    """
    Minimize the loss function by updating parameters via gradient descent.
    """
    kl_divergences = []

    for it in range(iterations):
        # Build the Hamiltonian
        H = build_hamiltonian(N, Gamma, b, w)

        # Compute KL divergence
        KL = compute_kl_divergence(data_distribution, H, N)
        kl_divergences.append(KL)

        # Compute parameter updates
        delta_Gamma, delta_b, delta_w = compute_gradient_update(
            data_distribution, H, N, Gamma, b, w, eta, n=n_integration_steps
        )

        # Update parameters
        Gamma -= delta_Gamma
        b -= delta_b
        w -= delta_w

        # Print progress
        print(f"Iteration {it+1}/{iterations}")
        print(f"KL Divergence: {KL:.6f}")

    return kl_divergences

# Initialize parameters
Gamma = 1.0
eta = 0.001  # Reduce learning rate for stability
b = np.random.normal(0, 0.1, N)  # Smaller initial values
w = np.random.normal(0, 0.1, (N, N))

iterations = 30  # Number of optimization steps

# Generate data distribution
data_distribution = generate_data_distribution(N, M, p)

kl_divergences = optimize_qbm(data_distribution, N, Gamma, b, w, eta, iterations)

# Plot KL divergence over iterations
plt.figure(figsize=(8, 6))
plt.plot(range(1, iterations + 1), kl_divergences, marker='o')
plt.xlabel("Iteration")
plt.ylabel("KL Divergence")
plt.title("KL Divergence over Iterations")
plt.grid()
plt.show()