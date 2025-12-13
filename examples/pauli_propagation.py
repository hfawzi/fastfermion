from common import read_circuit # To set the path for fastfermion
from fastfermion import poly, propagate
import time

circuit = read_circuit('data/su4circuit-16.txt')
obs = poly("Z15")
maxdegree = 4
t0 = time.perf_counter()
obs_final = propagate(circuit, obs, maxdegree)
print("Total time = ", time.perf_counter() - t0)
print("Number of terms in evolved observable = ", len(obs_final))
assert abs(obs_final.overlapwithzero() - 0.008664) <= 1e-6