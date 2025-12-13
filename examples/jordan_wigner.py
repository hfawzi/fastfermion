from common import read_fermipolynomial
from fastfermion import jw
import time

H = read_fermipolynomial('data/CrO-38.txt')
t0 = time.perf_counter()
Hjw = jw(H)
print("Total time = ", time.perf_counter() - t0)
print("Number of terms in Fermi polynomial = ", len(H))
print("Number of terms in Pauli polynomial = ", len(Hjw))