from neal import SimulatedAnnealingSampler
from neal.simulated_annealing import simulated_annealing
from neal.timer import Timer
from dimod import BinaryQuadraticModel, BINARY, SPIN
import time
import pandas as pd

import numpy as np


def random_raw(size=100, filling=0.7):
    h = np.random.randn(size)
    x = np.random.randn(int(filling * size * (size - 1) // 2))
    x1 = np.random.choice(np.arange(size * (size - 1) // 2), len(x), replace=False)
    r, c = np.triu_indices(size, 1)
    return h, r[x1], c[x1], x

def random_qubo(size=100, filling=0.7):
    h, r, c, x = random_raw(size, filling)
    q = np.zeros((size, size))
    q[r, c] = x
    q[c, r] = x
    q += np.diag(h)
    return q

def random_bqm(size=100, filling=0.7):
    h, r, c, x = random_raw(size, filling)
    J = {(ri, ci): xi for ri, ci, xi in zip(r, c, x)}
    h = {i: hi for i, hi in enumerate(h)}
    return BinaryQuadraticModel(h, J, vartype=SPIN)


def random_ising(size=100, filling=0.7):
    h, r, c, x = random_raw(size, filling)
    J = {(ri, ci): xi for ri, ci, xi in zip(r, c, x)}
    h = {i: hi for i, hi in enumerate(h)}
    return h, J


def run(func):
    t1 = Timer()
    a = time.time()
    func(t1)
    t1.full_call += time.time() - a
    return t1


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

raw = random_raw(s)

beta_range = [0.1, 100]

num_sweeps_per_beta = 1
num_betas = 1000

states = np.random.randint(2, size=(1, s), dtype=int)

states = (states - 0.5) * 2

states = np.round(states)
states = states.astype(np.int8)

beta_schedule = np.geomspace(*beta_range, num=num_betas)

timers = []
labels = []

for i in range(10):
    print(f"Run {i}")
    timers.append(
        run(
            lambda t: SimulatedAnnealingSampler().sample_qubo(
                q, timer=t, beta_range=beta_range
            )
        )
    )
    labels.append("qubo")

    timers.append(
        run(lambda t: SimulatedAnnealingSampler().sample(b, timer=t, beta_range=beta_range))
    )
    labels.append("bqm")

    timers.append(
        run(
            lambda t: SimulatedAnnealingSampler().sample_ising(
                h, J, timer=t, beta_range=beta_range
            )
        )
    )
    labels.append("ising")

    timers.append(
        run(
            lambda t: SimulatedAnnealingSampler().sample_ising(
                h2, J2, timer=t, beta_range=beta_range
            )
        )
    )
    labels.append("ising2")

    timers.append(
        run(lambda t: simulated_annealing(1, *raw, 1, beta_schedule, 213745893, states, t))
    )
    labels.append("pyx")

data = [[l, *list(x.__dict__.values())] for l, x in zip(labels, timers)]
columns = ["kind", *list(timers[0].__dict__.keys())]
df = pd.DataFrame(data, columns=columns)
dfx = df.groupby("kind").mean()
dfx.to_csv("res.csv")
print(dfx)
