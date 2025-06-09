import numpy as np
import pandas as pd
from HMM_functions import *

# Toy example with 2 clusters (N = 2), 3 positions (T = 3)
# Each cluster has A, T, C, G rows (4 per cluster)
# Columns: dummy id, allele letter, then 3 positions
ref = np.array([
    [1, 'A', 1.0, 0.0, 0.0],  # Cluster 1, A
    [1, 'T', 0.0, 1.0, 0.0],  # Cluster 1, T
    [1, 'C', 0.0, 0.0, 1.0],  # Cluster 1, C
    [1, 'G', 0.0, 0.0, 0.0],  # Cluster 1, G

    [2, 'A', 0.5, 0.0, 0.2],  # Cluster 2, A
    [2, 'T', 0.5, 0.5, 0.3],  # Cluster 2, T
    [2, 'C', 0.0, 0.1, 0.5],  # Cluster 2, C
    [2, 'G', 0.0, 0.4, 0.0],  # Cluster 2, G
], dtype=object)

target = [0, 1, 2]

e = 0.01  # small smoothing
s = 0.9   # mostly stay in same cluster

g, alpha_hat = forward_scaled_matrix(target, ref, e, s)
beta_hat = backward_scaled_matrix(target, ref, e, s, g)
mmpp = compute_mmpp(alpha_hat, beta_hat)

print("g (scaling factors):", g)
print("alpha_hat:\n", np.round(alpha_hat, 4))
print("beta_hat:\n", np.round(beta_hat, 4))
print("MMPP:", round(mmpp, 4))

print("checking forward probabilities and g:")
a00 = 1.01/1.04*0.5
a01 = 0.51/1.04*0.5

g0 = a00+a01

a00 = a00/g0
a01 = a01/g0

a10 = a00*0.9*1.01/1.04+a01*0.1*1.01/1.04
a11 = a01*0.9*0.51/1.04+a00*0.1*0.51/1.04

g1 = a10+a11

a10 = a10/g1
a11 = a11/g1

a20 = a10*0.9*1.01/1.04+a11*0.1*1.01/1.04
a21 = a11*0.9*0.51/1.04+a10*0.1*0.51/1.04

g2 = a20+a21

a20 = a20/g2
a21 = a21/g2

# Put into a table
df = pd.DataFrame([
    ["t=0", a00, a01, g0],
    ["t=1", a10, a11, g1],
    ["t=2", a20, a21, g2],
], columns=["Time", "alpha_0", "alpha_1", "g"])
print(df.round(4))

print("checking nll:")
print(nll([e,s], target, ref))
print(-sum(np.log([g0, g1, g2])))

print("checking backward probabilities:")
b10 = (0.9*1.01/1.04*1) + (0.1*0.51/1.04*1)
b11 = (0.1*1.01/1.04*1) + (0.9*0.51/1.04*1)

b10 = b10/g1
b11 = b11/g1

b00 = b10*0.9*1.01/1.04+b11*0.1*0.51/1.04
b01 = b10*0.1*1.01/1.04+b11*0.9*0.51/1.04

b00 = b00/g0
b01 = b01/g0

# Create a DataFrame for display
data = [
    ["t=1", b10, b11, g1],
    ["t=0", b00, b01, g0]
]
df = pd.DataFrame(data, columns=["Time", "beta_0", "beta_1", "g"])

# Round for readability
print(df.round(4))

print("checking MMPP:")
print(a00*b00/(a00*b00 + a01*b01))
print(a01*b01/(a00*b00 + a01*b01))

print(a10*b10/(a10*b10 + a11*b11))
print(a11*b11/(a10*b10 + a11*b11))

print(a20*1/(a20*1 + a21*1))
print(a21*1/(a20*1 + a21*1))

print(np.mean([a00*b00/(a00*b00 + a01*b01), a10*b10/(a10*b10 + a11*b11), a20*1/(a20*1 + a21*1)]))

