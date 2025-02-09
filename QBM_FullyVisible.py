import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from itertools import product
import matplotlib.pyplot as plt

# ============================
# PARAMETERS & DATA GENERATION
# ============================

N = 7   # Number of qubits
M = 8   # Number of modes
p = 0.9 # Probability of alignment
np.random.seed(42)  # For reproducibility

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

# ====================================
# DEFINE PAULI MATRICES & TENSOR PRODUCT
# ====================================

I = np.array([[1, 0], [0, 1]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])

def tensor_product(operators):
    """Compute the tensor (Kronecker) product of a list of operators."""
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result

# ================================
# BUILD THE (QUANTUM) HAMILTONIAN
# ================================

def build_hamiltonian(N, Gamma, b, w):
    """Construct the Fully Visible QBM Hamiltonian."""
    H = np.zeros((2**N, 2**N), dtype=complex)
    for a in range(N):
        H -= Gamma * tensor_product([I] * a + [sigma_x] + [I] * (N - a - 1))
        H -= b[a] * tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1))
    for a in range(N):
        for c in range(a + 1, N):
            H -= w[a, c] * tensor_product(
                [I] * a + [sigma_z] + [I] * (c - a - 1) + [sigma_z] + [I] * (N - c - 1)
            )
    return H

# ===========================
# TROTTER DECOMPOSITION
# ===========================

steps = 5  # Number of Trotter steps

def trotter_decomposition(H):
    """
    Approximate e^(-H) by breaking the evolution into small time slices.
    Given H, it returns exp(-H) ≈ (exp(-H/steps))^(steps).
    """
    delta_tau = 1.0 / steps
    exp_H_approx = np.eye(H.shape[0], dtype=complex)
    for _ in range(steps):
        exp_H_approx = expm_multiply(-delta_tau * H, exp_H_approx)
    return exp_H_approx

def compute_density_matrix(H):
    """Compute the density matrix ρ = e^(-H)/Z using Trotterized exponentiation."""
    exp_H = trotter_decomposition(H)  # exp(-H)
    Z = np.trace(exp_H)
    return exp_H / Z, Z

def compute_partial_expH(H, partial_H, n=100):
    """
    Compute the derivative of e^(-H) with respect to a parameter,
    i.e. ∂θ e^(-H) = -∫₀¹ dτ e^(-τH) (∂θH) e^{-(1-τ)H}.
    Here we approximate the integral by a Riemann sum.
    """
    delta_tau = 1.0 / n
    partial_eH = np.zeros_like(H, dtype=complex)
    for m in range(1, n + 1):
        tau = m * delta_tau
        exp1 = trotter_decomposition(tau * H)          # approximates e^(-τH)
        exp2 = trotter_decomposition((1 - tau) * H)      # approximates e^{-(1-τ)H}
        partial_eH += exp1 @ partial_H @ exp2 * delta_tau
    return -partial_eH

def compute_full_probability_distribution(rho):
    """Return the diagonal elements of ρ as the model probability distribution."""
    return np.real(np.diag(rho))

# ====================================================
# HELPER: CONVERT VISIBLE STATE (TUPLE) TO BASIS INDEX
# ====================================================

def state_to_index(visible_state):
    """
    Convert a visible state (tuple of -1 and 1) to an integer index.
    Here we define -1 -> 0 and 1 -> 1, and use the usual binary expansion.
    """
    return sum((1 if bit == 1 else 0) * 2**i for i, bit in enumerate(reversed(visible_state)))

# ============================
# KL DIVERGENCE & GRADIENTS
# ============================

def compute_kl_divergence(data_distribution, rho):
    """Compute the Kullback-Leibler divergence between data and model distributions."""
    P_model = compute_full_probability_distribution(rho)
    kl_divergence = 0.0
    # Order the data states according to the standard basis ordering.
    items_sorted = sorted(data_distribution.items(), key=lambda x: state_to_index(x[0]))
    for (visible_state, P_data) in items_sorted:
        idx = state_to_index(visible_state)
        P_v = P_model[idx]
        kl_divergence += P_data * np.log(P_data / (P_v + 1e-10))  # prevent log(0)
    return kl_divergence

# ----------------------------
# Precompute Partial Hamiltonians
# ----------------------------

# For parameter Gamma (transverse field)
partial_H_Gamma = -sum(tensor_product([I] * a + [sigma_x] + [I] * (N - a - 1)) for a in range(N))
# For biases b
partial_H_b = [-tensor_product([I] * a + [sigma_z] + [I] * (N - a - 1)) for a in range(N)]
# For couplings w (only define for a < b; fill the rest with zeros)
partial_H_w = [
    [
        -tensor_product([I] * a + [sigma_z] + [I] * (b - a - 1) + [sigma_z] + [I] * (N - b - 1))
        if a < b else np.zeros((2**N, 2**N), dtype=complex)
        for b in range(N)
    ]
    for a in range(N)
]

def compute_gradient_update(data_distribution, H, N, Gamma, b, w, eta):
    """Compute the gradient updates for the QBM parameters."""
    n = 100
    rho, _ = compute_density_matrix(H)
    P_model = compute_full_probability_distribution(rho)
    
    # Initialize gradients for Gamma, b, and w.
    grad_Gamma = 0.0
    grad_b = np.zeros(N)
    grad_w = np.zeros((N, N))
    
    # Compute global (model) expectations.
    global_grad_Gamma = np.trace(partial_H_Gamma @ rho).real
    global_grad_b = np.array([np.trace(partial_H_b[i] @ rho).real for i in range(N)])
    global_grad_w = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            global_grad_w[i, j] = np.trace(partial_H_w[i][j] @ rho).real
            global_grad_w[j, i] = global_grad_w[i, j]
    
    # Compute the parameter–dependent derivatives of e^(-H).
    gamma1 = compute_partial_expH(H, partial_H_Gamma, n)
    b1 = np.zeros((N, 2**N, 2**N), dtype=complex)
    for i in range(N):
        b1[i] = compute_partial_expH(H, partial_H_b[i], n)
    w1 = np.zeros((N, N, 2**N, 2**N), dtype=complex)
    for i in range(N):
        for j in range(i + 1, N):
            w1[i][j] = compute_partial_expH(H, partial_H_w[i][j], n)
    
    # Loop over the data states (ordered in the standard basis order).
    items_sorted = sorted(data_distribution.items(), key=lambda x: state_to_index(x[0]))
    for (visible_state, P_data) in items_sorted:
        idx = state_to_index(visible_state)
        P_v = P_model[idx]
        if P_v > 0:
            # Build the projection operator |v⟩⟨v| (a diagonal matrix with a 1 at position idx).
            state_proj = np.zeros((2**N, 2**N), dtype=complex)
            state_proj[idx, idx] = 1
            # Compute the expectation (conditional on the visible state v) for each parameter.
            exp_Gamma = np.trace(state_proj @ gamma1 @ rho).real / np.trace(state_proj @ rho).real
            grad_Gamma += P_data * (exp_Gamma - global_grad_Gamma)
            for i in range(N):
                exp_b = np.trace(state_proj @ b1[i] @ rho).real / np.trace(state_proj @ rho).real
                grad_b[i] += P_data * (exp_b - global_grad_b[i])
            for i in range(N):
                for j in range(i + 1, N):
                    exp_w = np.trace(state_proj @ w1[i][j] @ rho).real / np.trace(state_proj @ rho).real
                    grad_w[i, j] += P_data * (exp_w - global_grad_w[i, j])
                    grad_w[j, i] = grad_w[i, j]  # Ensure symmetry
    # Return parameter updates (with learning rate η).
    return eta * grad_Gamma, eta * grad_b, eta * grad_w

def optimize_qbm(data_distribution, N, Gamma, b, w, eta, iterations):
    """Run the parameter optimization loop and record the KL divergence."""
    kl_divergences = []
    for it in range(iterations):
        H = build_hamiltonian(N, Gamma, b, w)
        rho, _ = compute_density_matrix(H)
        KL = compute_kl_divergence(data_distribution, rho)
        kl_divergences.append(KL)
        delta_Gamma, delta_b, delta_w = compute_gradient_update(data_distribution, H, N, Gamma, b, w, eta)
        Gamma += delta_Gamma
        b += delta_b
        w += delta_w
        print(f"∆Gamma={delta_Gamma}, ∆b_mean={np.mean(delta_b):.4f}, ∆w_mean={np.mean(delta_w):.4f}")
        print(f"Iteration {it+1}/{iterations}, KL Divergence: {KL:.6f}")
    return kl_divergences

# ====================
# RUN THE OPTIMIZATION
# ====================

Gamma = 0.0
eta = 0.01
b = np.random.normal(0, 0.01, N)      # biases (1D array)
w = np.random.normal(0, 0.01, (N, N))   # couplings (2D array)
w = (w + w.T) / 2  # Ensure symmetry of w.
iterations = 35

data_distribution = generate_data_distribution(N, M, p)
kl_divergences = optimize_qbm(data_distribution, N, Gamma, b, w, eta, iterations)

# Plot KL divergence vs iterations.
plt.plot(kl_divergences, marker='o')
plt.xlabel("Iteration")
plt.ylabel("KL Divergence")
plt.title("KL Divergence vs Iterations")
plt.show()

# ======================================================
# Save the KL divergence data to a CSV file for future use.
# ======================================================
output_filename = "qbm_n7.csv"
iterations_arr = np.arange(1, len(kl_divergences) + 1)
data_to_save = np.column_stack((iterations_arr, kl_divergences))
np.savetxt(output_filename, data_to_save, delimiter=",", header="Iteration,KL Divergence", comments="")
print(f"Saved KL divergence data to {output_filename}")