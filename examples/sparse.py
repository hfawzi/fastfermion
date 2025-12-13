from hamiltonians import fermihubbard2d_creann, heisenberg1d
import time
from scipy.special import binom

# Sparse matrix representation of the 3x3 Fermi-Hubbard model
Nx = 3
Ny = 3
N = Nx*Ny
H_hubbard = fermihubbard2d_creann(Nx=Nx,Ny=Ny,t=1,U=4)
t0 = time.perf_counter()
H_hubbard_sp = H_hubbard.sparse()
total_time = time.perf_counter() - t0
assert H_hubbard_sp.shape == (2**(2*N),2**(2*N))
print("total time Hubbard = ", total_time)

# Fix number of occupied orbitals = Nx*Ny
H_hubbard_sp_2 = H_hubbard.sparse(nocc=N)
assert H_hubbard_sp_2.shape == (binom(2*N,N),binom(2*N,N))

# Sparse matrix representation of the Heisenberg model on a chain of length 20
N = 20
H_heisenberg = heisenberg1d(N)
t0 = time.perf_counter()
H_heisenberg_sp = H_heisenberg.sparse()
total_time = time.perf_counter() - t0
assert H_heisenberg_sp.shape == (2**N,2**N)
print("total time Heisenberg = ", total_time)

# Fix number of up spins = N/2
H_heisenberg_sp = H_heisenberg.sparse(nup=N//2)
assert H_heisenberg_sp.shape == (binom(N,N//2),binom(N,N//2))
print("total time Heisenberg subspace = ", total_time)
