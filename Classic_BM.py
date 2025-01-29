# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:13:50 2025

@author: danie
"""

import torch
import itertools
import matplotlib.pyplot as plt

N = 10  # Number of visible spins z in {+1, -1}


def energy(z, b, W):
    """
    E(z) = - sum_a b_a z_a - sum_{a,b} w_{a,b} z_a z_b
    for a single configuration z in {+1, -1}^N.
    """
    # We'll compute this directly:
    #   E(z) = - (b·z + z^T W z)
    bz = torch.dot(b, z)
    zWz = torch.dot(z, torch.matmul(W, z))
    return -(bz + zWz)

def boltzmann_distribution(b, W, all_states):
    """
    Enumerate all states z in {+1, -1}^N, compute
        P_model(z) = exp[-E(z)] / Z
    and return a (2^N,) probability vector.
    """
    energies = []
    for z in all_states:
        E_z = energy(z, b, W)
        energies.append(E_z)
    energies = torch.stack(energies)  # shape (2^N,)

    # exponentiate -E(z) already appears in 'energies' as negative
    # but we have E_z = -(...) so actually we want exp(-E_z) = exp(+ ...).
    # Let's just do exp(-E_z) carefully:

    negE = -energies
    max_negE = negE.max()
    exp_shifted = torch.exp(negE - max_negE)
    Z = exp_shifted.sum()
    return exp_shifted / Z

def kl_divergence(P_data, P_model):

    return torch.sum(P_data * torch.log((P_data + 1e-12)/(P_model + 1e-12)))


M = 8  # number of modes
p = 0.9 # spin alignment probability with its mode center

# Generate M random center points s^k in {+1, -1}^N
centers = torch.randint(low=0, high=2, size=(M, N))  # in {0,1}
centers = 2*centers - 1  # map to {+1,-1}

# Enumerate all states z in {+1, -1}^N
all_configs = list(itertools.product([-1, +1], repeat=N))
all_z = torch.tensor(all_configs, dtype=torch.float32)  # (2^N, N)

def mixture_data_distribution(all_states, centers, p):

    num_modes = centers.shape[0]
    N_ = all_states.shape[1]
    probs = []
    for z in all_states:
        mode_sum = 0.0
        for s in centers:

            d_zs = 0.5 * torch.sum(1 - z*s)
            mode_sum += p**(N_-d_zs) * (1-p)**d_zs
        probs.append(mode_sum / num_modes)
    probs = torch.tensor(probs, dtype=torch.float32)
    
    # normalize
    probs /= probs.sum()
    return probs

P_data = mixture_data_distribution(all_z, centers, p)
print("Check sum of P_data:", P_data.sum().item())  # ~1.0

# Compute "positive phase" averages once: <z_a>_data, <z_a z_b>_data
z_data_avg = torch.zeros(N)
zz_data_avg = torch.zeros(N, N)
for i, z in enumerate(all_z):
    z_data_avg += P_data[i]*z
    zz_data_avg += P_data[i]*torch.ger(z, z)  # outer product

# Manual Gradient Updates Using exact formulas
# Initialize parameters (b, W).  We won't use autograd.
b = torch.zeros(N)
W = torch.zeros(N, N)

eta = 0.1  # learning rate
num_steps = 30

kl_history = []

for step in range(num_steps):
    # 1) Compute model distribution
    P_model = boltzmann_distribution(b, W, all_z)

    # 2) Negative phase averages: <z_a>_model, <z_a z_b>_model
    z_model_avg = torch.zeros(N)
    zz_model_avg = torch.zeros(N, N)
    for i, z in enumerate(all_z):
        z_model_avg += P_model[i]*z
        zz_model_avg += P_model[i]*torch.ger(z, z)

    db = eta * (z_data_avg - z_model_avg)
    dw = eta * (zz_data_avg - zz_model_avg)

    b += db
    W += dw

    # (Optional chatGPT advice) Symmetrize W if desired: W = 0.5*(W + W.t())

    this_kl = kl_divergence(P_data, P_model)
    kl_history.append(this_kl.item())

    if step % 5 == 0:
        print(f"Iter {step}: KL = {this_kl.item():.4f}")

plt.figure(figsize=(6,4))
plt.plot(kl_history, marker='o')
plt.xlabel("Iteration")
plt.ylabel("KL Divergence")
plt.title("BM Training (Exact) ")
plt.grid(True)
plt.show()
