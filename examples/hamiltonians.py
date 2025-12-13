from fastfermion import MajoranaPolynomial, PauliPolynomial, FermiPolynomial, majoranas, paulis, fermis

def gridedges(Nx,Ny,pbcx,pbcy) -> list[tuple[int,int]]:
    """
    Return list of edges with a Nx * Ny square grid
    """
    edgelist = []
    def sub2ind(x,y): return x*Nx+y
    for x in range(Nx+pbcx):
        for y in range(Ny+pbcy):
            cur = sub2ind(x,y)
            top = sub2ind(x,(y+1)%Ny)
            right = sub2ind((x+1)%Nx,y)
            if Nx > 2 or x == 0: edgelist.append((cur,right))
            if Ny > 2 or y == 0: edgelist.append((cur,top))
    return edgelist

def fermihubbard2d_majorana(Nx,Ny,t,U,pbc=[False,False]) -> MajoranaPolynomial:
    """
    Returns Fermi-Hubbard model as a MajoranaPolynomial
    """
    edgelist = gridedges(Nx,Ny,pbc[0],pbc[1])
    m = majoranas(4*Nx*Ny)
    H = MajoranaPolynomial()
    for i,j in edgelist:
        H += -1j*0.5*t* ( m[4*i]*m[4*j+1] + m[4*j]*m[4*i+1] + m[4*i+2]*m[4*j+3] + m[4*j+2]*m[4*i+3] )
    if U != 0:
        for i in range(Nx*Ny):
            H += 0.25*U * (1 + 1j*m[4*i]*m[4*i+1]) * (1 + 1j*m[4*i+2]*m[4*i+3])
    return H


def fermihubbard2d_creann(Nx,Ny,t,U,pbc=[False,False]) -> FermiPolynomial:
    """
    Returns Fermi-Hubbard model as a FermiPolynomial
    """
    edgelist = gridedges(Nx,Ny,pbc[0],pbc[1])
    N = Nx*Ny
    a = fermis(2*N)
    n = [a[i].dagger()*a[i] for i in range(2*N)]
    H = FermiPolynomial()
    if t != 0:
        for i,j in edgelist:
            H += -t*(a[2*i].dagger()*a[2*j] + a[2*j].dagger()*a[2*i] + a[2*i+1].dagger()*a[2*j+1] + a[2*j+1].dagger()*a[2*i+1])
    if U != 0:
        for i in range(N):
            H += U*n[2*i]*n[2*i+1]
    return H


def heisenberg1d(N: int) -> PauliPolynomial:
    """
    Construct Heisenberg hamiltonian on a 1D chain
    """
    [sx,sy,sz] = paulis(N)
    edgelist = [(i, (i + 1)%N) for i in range(N)]
    H = PauliPolynomial()
    for i,j in edgelist:
        H += (sx[i]*sx[j] + sy[i]*sy[j] + sz[i]*sz[j])
    return H