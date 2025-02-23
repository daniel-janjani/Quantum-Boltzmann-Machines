import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def energy(z, b, W):
    """ 
    E(z) = - sum_a (b_a z_a) - sum_{a,b} (w_{a,b} z_a z_b)
    for a single configuration z in {+1, -1}^N. 
    """
    # We'll compute this directly:
    #   E(z) = - (b·z + z^T W z)
    bz = np.dot(b, z)
    zWz = np.dot(z, np.matmul(W, z))
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
    energies = np.stack(energies)  # shape (2^N)

    # exponentiate -E(z) already appears in 'energies' as negative
    # but we have E_z = -(...) so actually we want exp(-E_z) = exp(+ ...).
    # Let's just do exp(-E_z):
    negE = -energies
    exp_shifted = np.exp(negE)
    Z = exp_shifted.sum()
    return exp_shifted / Z

# Kullback-Leibler (KL) divergence: KL = Likelyhood - Likelyhood_min
def kl_divergence(P_data, P_model):
    return np.sum(P_data * np.log((P_data + 1e-12)/(P_model + 1e-12)))

# Define Parameters
N = 8  # Number of visible spins z in {+1, -1}
M = 8  # Number of modes for data distribution
p = 0.9 # Spin alignment probability with mode centers
eta = 0.2  # Learning rate
num_steps = 35 # Number of optimization steps

# Generate M random center points s^k in {+1, -1}^N
centers = np.random.randint(low=0, high=2, size=(M, N))  # in {0,1}
centers = 2*centers - 1  # map to {+1,-1}

# Enumerate all states z in {+1, -1}^N
all_configs = list(itertools.product([-1, +1], repeat=N))
all_z = np.array(all_configs, dtype=np.float32)  # (2^N, N)

def mixture_data_distribution(all_states, centers, p):  
    """ 
    Generate training data as a mixture of M modes using 
    Bernouilli distribution: p^(N-d_kv)*(1-p)^d_kv
    """
    num_modes = centers.shape[0]  # The number of modes (M=8) is the centers' number of rows
    N_ = centers.shape[1]         # The number of bits (N=10) is the centers' number of columns 
    N_states = all_states.shape[0]  # (2^N)
    probs = np.zeros(N_states, dtype=np.float32) 
    for s in range(N_states): 
        mode_sum = 0.0
        for k in range(num_modes):
            d_ks = 0.5 * np.sum(1 - all_states[s, :] * centers[k, :])  # Hamming distance between state s and center k
            mode_sum += p**(N_ - d_ks) * (1 - p)**d_ks  # mixture of Bernoulli distribution
        probs[s] = mode_sum / num_modes  # Generating P_data for each state
    # normalitation
    probs /= probs.sum()
    return probs

P_data = mixture_data_distribution(all_z, centers, p)
print("Check sum of P_data:", P_data.sum().item())  # ~1.0
print("Check dimension of P_data:", P_data.shape)  # ~2^10 = 1024

# Compute "positive phase" averages once: <z_a>_data, <z_a z_b>_data for each a,b = 1,2,...,N
z_data_avg = np.zeros(N)
zz_data_avg = np.zeros((N, N))
N_states = all_z.shape[0]
for i in range(N_states):
    z_data_avg += P_data[i] * all_z[i, :] 
    zz_data_avg += P_data[i] * np.outer(all_z[i, :], all_z[i, :])

# Manual Gradient Updates Using exact formulas
# Initialize parameters (b, W) using 'random.seed'
np.random.seed(42)
b = 0.01 * np.random.randn(N)
W = 0.01 * np.random.randn(N, N)

kl_history = []
for step in range(num_steps):
    # 1) Compute model distribution
    P_model = boltzmann_distribution(b, W, all_z)
    # 2) 'Negative phase' averages: <z_a>_model, <z_a z_b>_model for each a,b = 1,2,...,N
    z_model_avg = np.zeros(N)
    zz_model_avg = np.zeros((N, N))
    for i in range(N_states):
      z_model_avg += P_model[i] * all_z[i, :]
      zz_model_avg += P_model[i] * np.outer(all_z[i, :], all_z[i, :])
    
    # Compute gradient steps as difference between positive and negative phases
    db = eta * (z_data_avg - z_model_avg) 
    dW = eta * (zz_data_avg - zz_model_avg)

    b += db
    W += dW
    
    # Compute and save KL value
    this_kl = kl_divergence(P_data, P_model)
    kl_history.append(this_kl.item())

    if step % 5 == 0:
        print(f"Iter {step}: KL = {this_kl.item():.4f}")

# Saving Data frame in CSV
df = pd.DataFrame({"iteration": list(range(num_steps)), "kl_history": kl_history})
df.to_csv("BM.csv", index=False)
print("Dati salvati in BM.csv")

df = pd.read_csv("BM.csv")

plt.figure(figsize=(6,4))
plt.plot(df['iteration'], df['kl_history'], marker='o', label='KL Divergence')
plt.xlabel("Iteration")
plt.ylabel("KL Divergence")
plt.title("BM Training (Exact)")
plt.grid(True)
plt.show()
