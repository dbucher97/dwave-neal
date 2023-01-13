from neal import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel, BINARY, SPIN
import time

import numpy as np

def random_qubo(size=100):
    x = np.random.randn(size, size)
    return (x + x.T) / 2

def random_bqm(size=100):
    x = np.random.randn(size * (size + 1) // 2)
    h = {i: x[i] for i in range(size)}
    idxs = [(i, j) for i in range(size) for j in range(i + 1, size)]
    J = {idx: x[n + size] for n, idx in enumerate(idxs)}
    return BinaryQuadraticModel(h, J, vartype=SPIN)

def random_ising(size=100):
    x = np.random.randn(size * (size + 1) // 2)
    h = {i: x[i] for i in range(size)}
    idxs = [(i, j) for i in range(size) for j in range(i + 1, size)]
    J = np.zeros((size, size))
    J = {idx: x[n + size] for n, idx in enumerate(idxs)}
    return h, J

print("CREATION ========================================")

s = 4000
q = random_qubo(s)
b = random_bqm(s)
h, J = random_ising(s)

h2 = np.zeros(s)
J2 = np.zeros((s, s))
h2[list(h.keys())] = np.array(list(h.values()))
idxs = np.array(list(J.keys())).T
J2[idxs[0], idxs[1]] = np.array(list(J.values()))

beta_range = [0.1, 100]

print("RUN QUBO ========================================")

a = time.time()
res = SimulatedAnnealingSampler().sample_qubo(q, beta_range=beta_range)
print("Total", time.time() - a, "s")

print(res)

print("RUN BQM =========================================")

a = time.time()
res = SimulatedAnnealingSampler().sample(b, beta_range=beta_range)
print("Total", time.time() - a, "s")

print(res)

print("RUN ISING =======================================")

a = time.time()
res = SimulatedAnnealingSampler().sample_ising(h, J, beta_range=beta_range)
print("Total", time.time() - a, "s")

print(res)

print("RUN ISING MATRIX =================================")

a = time.time()
res = SimulatedAnnealingSampler().sample_ising(h2, J2, beta_range=beta_range)
print("Total", time.time() - a, "s")

print(res)
