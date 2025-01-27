# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 02:15:36 2025

@author: danie

"""


# -*- coding: utf-8 -*-
import pennylane as qml
from pennylane import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

N = 3  # Number of qubits
dev = qml.device("default.qubit", wires=N)

def all_configurations(N):
    """Return an array of shape (2^N, N) containing all +/-1 configurations."""
    """Will need it to compute the training data probabilities"""
    configs = []
    for val in range(2**N):
        bits = []
        for i in range(N):
            bit = (val >> i) & 1
            spin = +1 if bit == 0 else -1
            bits.append(spin)
        configs.append(bits[::-1])
    return np.array(configs, dtype=int)

def data_distribution(N, M, p, seed):
    """
    Construct the data distribution P^data over all configurations in {+/-1}^N,
    as a mixture of M "centers" with some poissonian probability p.
    """
    np.random.seed(seed)

    # Randomly pick M centers s^k in {±1}^N
    centers = [np.random.choice([+1, -1], size=N) for _ in range(M)]

    configs = all_configurations(N)  # shape (2^N, N)
    P_data = np.zeros(len(configs), dtype=float)

    # For each configuration v, compute average probability from M centers
    for i, v in enumerate(configs):
        val = 0.0
        for s in centers:
            d_v_s = np.sum(v != s)  # number of spin mismatches=Hamming distance
            val += p**(N - d_v_s) * (1 - p)**(d_v_s)
        P_data[i] = val / M

    return configs, P_data

def config_to_index(v):
    """
    Convert spin config v (±1) into integer in [0, 2^N-1],
    assuming qubit 0 is the leftmost bit.
    """
    idx = 0
    for spin in v:
        bit = 0 if spin == +1 else 1
        idx = (idx << 1) | bit
    return idx

def make_hamiltonian(Gamma, b, w):
    """Construct the Hamiltonian H."""
    coeffs = []
    ops = []

    # Transverse field: -Gamma * X_a
    for a in range(N):
        coeffs.append(np.float64(-Gamma))  # Explicit cast
        ops.append(qml.PauliX(a))

    # Local fields: - b[a] * Z_a
    for a in range(N):
        coeffs.append(np.float64(-b[a]))  # Explicit cast
        ops.append(qml.PauliZ(a))

    # Pairwise interactions: - W[a,b] * Z_a Z_b
    for a_ in range(N):
        for b_ in range(a_ + 1, N):
            coeffs.append(np.float64(-w[a_, b_]))  # Explicit cast
            ops.append(qml.PauliZ(a_) @ qml.PauliZ(b_))
    

    return qml.Hamiltonian(coeffs, ops)

def P_v(Gamma, b, w, v):
    """
    Probability P_v = <v| rho |v>, for rho = e^{-H}/Tr[e^{-H}], evaluated
    at classical config v (±1).
    """
    H = make_hamiltonian(Gamma, b, w)

    # Use the dense matrix representation
    H_mat = qml.matrix(H, wire_order=list(range(N)))

    expH = la.expm(-H_mat)
    Z = np.trace(expH)  

    rho = expH / Z

    # Find the diagonal element corresponding to the configuration v
    # we can do this since the model is fully visible
    idx_v = config_to_index(v)
    return np.real(rho[idx_v, idx_v])


def model_prob_fn(Gamma, b, w):
    """
    Return the distribution P_model(v) for all v in {±1}^N, in the same order
    as all_configurations(N).
    """
    configs_ = all_configurations(N)
    Pvals = [P_v(Gamma, b, w, v) for v in configs_]
    return np.array(Pvals, dtype=float)


def cost(params):
    Gamma, b, w = params
    P_model = model_prob_fn(Gamma, b, w)

    eps = 1e-12 #we do this to avoid numerical errors
    return -np.sum(P_data * np.log(P_model + eps))


def kl_divergence(P_data, P_model):

    eps = 1e-12  
    return np.sum(P_data * np.log((P_data + eps) / (P_model + eps)))


# Optimization parameters
max_steps = 100
configs, P_data = data_distribution(N=3, M=8, p=0.9, seed=42)

Gamma_init = 0.01 * qml.numpy.random.randn()
b_init = 0.01 * qml.numpy.random.randn(N)
w_init = 0.01 * qml.numpy.random.randn(N, N)
params = [qml.numpy.array(Gamma_init), qml.numpy.array(b_init), qml.numpy.array(w_init)]

# This is PennyLane's GradientDescentOptimizer
opt = qml.AdamOptimizer(stepsize=0.05)


kl_values = []
cost_values = []


for step in range(max_steps):
    params = opt.step(cost, params)
    # Define a gradient function for the cost function
    grad_fn = qml.grad(cost)
    grads = grad_fn(params)
    print(f"Step {step}: Gradients = {grads}")
    if grads==():
        print("nope qualcosa non funzia di nuovo")
        exit()


    # Unpack the parameters
    Gamma, b, w = params[0], params[1], params[2]

    # Compute cost and KL divergence
    current_cost = cost(params)
    current_model = model_prob_fn(Gamma, b, w)
    current_kl = kl_divergence(P_data, current_model)
    
    # Store metrics
    cost_values.append(current_cost)
    kl_values.append(current_kl)

    # Print progress 
    if step % 10 == 0:

        print(f"Step {step}: Cost = {current_cost:.6f}, KL = {current_kl:.6f}")

plt.figure(figsize=(12, 6))
plt.plot(range(max_steps), cost_values, label="Cost (Cross-Entropy)")
plt.plot(range(max_steps), kl_values, label="KL Divergence")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.title("Cost and KL Divergence Over Optimization")
plt.show()

# Final evaluation
P_trained = model_prob_fn(Gamma, b, w)

print("\nTrained distribution:", P_trained)
print("Data distribution:   ", P_data)
print("Sum of P_trained:", np.sum(P_trained), "(should be ~1)")




