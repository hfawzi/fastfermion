import fastfermion
from fastfermion import MajoranaPolynomial, FockState, majoranas, MROT, propagate
from hamiltonians import fermihubbard2d_majorana
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
from scipy.integrate import solve_ivp

def trotterize(H: MajoranaPolynomial, dt: float) -> list[MROT]:
    circuit = []
    # Loop through the terms of H
    H_terms = H.terms
    for ms, coeff in H_terms.items():
        circuit.append(MROT(ms,coeff,2*dt))
    return circuit

print(f"fastfermion version = {fastfermion.__version__}")

# system size
Nx,Ny = 3,3
N = Nx * Ny
Nmodes = 2*N
# hopping term
t = 1.0
# coloumb terms
U = 4.0
# boundary conditions
pbc = [False,False]
# time-step
dt = 0.02
# number of steps
steps = 50
# truncation parameters
maxdegree = 6
mincoeff = 1e-5
# get majorana strings
m = majoranas(2*Nmodes)

#######
# state and observable
#######

# antiferromagnetic fock state with hole in central site
hole = 4
state = FockState(occ=[2*i+((i>hole)^(i%2)) for i in range(0,N) if i != hole])
state_vec = np.array(state.vec(2*N),dtype=complex) # Vector of size 2^Nmodes

# observable - probability of empty site at `hole`
O = 0.25*(1-1j*m[4*hole]*m[4*hole+1])*(1-1j*m[4*hole+2]*m[4*hole+3])
O_sparse = O.sparse(Nmodes) # Sparse matrix of size 2^Nmodes x 2^Nmodes

# Hamiltonian
H = fermihubbard2d_majorana(Nx,Ny,t,U,pbc)

#######
# Exact dynamics
#######
print("solving exact dynamics...")
H_sparse = H.sparse(Nmodes) # Sparse matrix of size 2^Nmodes x 2^Nmodes
rhs = lambda t,vec: -1j*H_sparse @ vec
sol = solve_ivp(rhs,[0,dt*steps],state_vec,method="RK45",t_eval=np.linspace(0,dt*steps,steps+1))
y_t = sol.y
overlap_exact = [y_t[:,step].T.conjugate() @ O_sparse @ y_t[:,step] for step in range(steps+1)]

#######
# Majorana propagation
#######
print("start majorana propagation...")
initial_overlap = state(O)
circuit = trotterize(H,dt)
O_mp = O
overlap_mp = [state(O_mp)]
# start propagation
for step in range(steps):
    # propagate for 1 time-step
    O_mp = propagate(circuit,O_mp,maxdegree=maxdegree,mincoeff=mincoeff)
    # compute overlap
    overlap = state(O_mp)
    overlap_mp.append(overlap)
    if step % 10 == 0:
        print(f"t={step*dt} (step {step}), overlap={overlap}, len(O_mp)={len(O_mp)}")

#######
# Plot results
#######

plt.rcParams.update(rcParamsDefault)
plt.rc("savefig", dpi=300)
plt.rc('text', usetex=True)
plt.rc("font", family="serif",size=18.)
plt.rc('text.latex', preamble="\\usepackage{moresize}\\usepackage{amsmath}\\renewcommand{\\vec}[1]{\\boldsymbol{#1}}")
plt.rc("lines", linewidth=1.5, markersize=10, markeredgewidth=1.5)

steps_vec = [dt*(i+1) for i in range(steps+1)]
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(steps_vec,overlap_exact,label="exact",color="black",linewidth=3)
ax.plot(steps_vec,overlap_mp, label=fr"$\ell={maxdegree}$",linestyle="-")

ax.set_xlabel("time $t$")
ax.set_ylabel(r"$\langle (1-n_{5,\uparrow})(1-n_{5,\downarrow})\rangle$")
plt.show()
