"""Test sparse matrix representations of fastfermion polynomials"""

from common import (
    check_sparse_matrices_approx_equal,
    check_sparse_matrices_exactly_equal,
    fermistringstuple,
    paulistringstuple,
    subsets,
)
from fastfermion import (
    poly,
    FermiPolynomial,
    FermiString,
    MajoranaPolynomial,
    MajoranaString,
    PauliPolynomial,
    PauliString,
    paulis
)
import numpy as np
from scipy.special import comb
import itertools
import openfermion as of


def test_sparse_pauli():
    """Test sparse matrix representation of Pauli polynomials"""
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])
    assert np.max(np.abs(PauliString("X").sparse().toarray() - X)) == 0
    assert np.max(np.abs(PauliString("Y").sparse().toarray() - Y)) == 0
    assert np.max(np.abs(PauliString("Z").sparse().toarray() - Z)) == 0
    assert np.max(np.abs(PauliString("XY").sparse().toarray()
                          - np.kron(X,Y))) == 0
    assert np.max(np.abs(PauliString("XZ").sparse().toarray()
                          - np.kron(X,Z))) == 0
    assert np.max(np.abs(PauliString("YZ").sparse().toarray()
                          - np.kron(Y,Z))) == 0
    # PauliStrings
    for ps in paulistringstuple(5):
        check_sparse_matrices_exactly_equal(
            PauliString(ps).sparse(),
            of.get_sparse_operator(of.QubitOperator(ps))
        )
    # PauliPolynomials
    for k in range(1,5):
        # Form PauliPolynomial and QubitOperator
        p1 = PauliPolynomial()
        p2 = of.QubitOperator()
        for i,ps in enumerate(paulistringstuple(k)):
            coeff = np.cos(k*i) + np.sin(2*k*i)*1j
            p1 += coeff * PauliString(ps)
            p2 += of.QubitOperator(ps,coeff)
        check_sparse_matrices_approx_equal(p1.sparse(), 
                                           of.get_sparse_operator(p2))

def test_sparse_pauli_nup_subspace():
    """
    Test sparse matrix representation of Pauli polynomials
    when restricted to a subspace with a total spin
    """

    # Construct polynomial which preserves total spin
    sx, sy, sz = paulis(10)

    def h(i,j):
        return sx[i]*sx[j] + sy[i]*sy[j] + sz[i]*sz[j]

    # Some polynomial which preserves total spins
    p1 = sum(np.sin(1+i)*np.cos(2+j)*h(i,j) for i in range(4) for j in range(4))
    p2 = sum((1+i)*1j*h(i,j) for i in range(5) for j in range(3))

    testpolys = [p1, p2]
    
    for p in testpolys:
        n = p.extent()
        A = p.sparse(n).toarray()
        for nup in range(n+1):
            ind = [i for i in range(2**n) if i.bit_count() == nup]
            assert len(ind) == comb(n,nup)
            Asub = A[np.ix_(ind,ind)]
            A2 = p.sparse(n,nup).toarray()
            assert np.max(np.abs(A2 - Asub)) <= 1e-14


def test_sparse_fermi():
    """Test sparse matrix representation of Fermi polynomials"""
    assert np.all(FermiString().sparse().toarray() == 1)
    assert np.max(np.abs(FermiString([],[0]).sparse().toarray() - np.array([[0,1],[0,0]]))) == 0
    assert np.max(np.abs(FermiString([0],[]).sparse().toarray() - np.array([[0,0],[1,0]]))) == 0
    assert np.max(np.abs(FermiString([],[0]).sparse(2).toarray() - np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]))) == 0
    assert np.max(np.abs(FermiString([],[1]).sparse(2).toarray() - np.array([[0,1,0,0],[0,0,0,0],[0,0,0,-1],[0,0,0,0]]))) == 0
    
    # FermiStrings
    for fs in fermistringstuple(5):
        check_sparse_matrices_exactly_equal(
            FermiString(fs).sparse(),
            of.get_sparse_operator(of.FermionOperator(fs))
        )
    
    # FermiPolynomials
    for k in range(1,5):
        # Form PauliPolynomial and QubitOperator
        p1 = FermiPolynomial()
        p2 = of.FermionOperator()
        for i,fs in enumerate(fermistringstuple(k)):
            coeff = np.cos(k*i) + np.sin(2*k*i)*1j
            p1 += coeff * FermiString(fs)
            p2 += of.FermionOperator(fs,coeff)
        check_sparse_matrices_approx_equal(p1.sparse(),
                                           of.get_sparse_operator(p2))

def test_sparse_fermi_subspace():
    """
    Test sparse matrix representation of Fermi polynomials
    when restricted to a subspace with a fixed occupation number
    """

    def generate_poly(n,deg,seed):
        """
        Generate a polynomial of degree 4 which preserves
        particle number
        """
        p = FermiPolynomial()
        for k in range(2,deg+1,2):
            ksubsets = itertools.combinations(reversed(range(n)),k)
            p += sum((np.cos(1+seed+i)*np.sin(2+seed+j))*FermiString(U,S) for i,U in enumerate(ksubsets) for j,S in enumerate(ksubsets))
        return p

    test_polys = [poly("f1^ f0 + 2*f2^ f1 - 3j*f0^ f0 + 5*f1^ f0^ f2^ f0 f1 f3"),
                  generate_poly(4,2,0), generate_poly(3,4,1)]

    for p in test_polys:
        n = p.extent()
        A = p.sparse(n).toarray()
        for ntot in range(n+1):
            ind = [i for i in range(2**n) if i.bit_count() == ntot]
            assert len(ind) == comb(n,ntot)
            Asub = A[np.ix_(ind,ind)]
            A2 = p.sparse(n,ntot).toarray()
            assert np.max(np.abs(A2 - Asub)) <= 1e-14


def test_sparse_majorana():
    """Test sparse matrix representation of Majorana polynomials"""

    # Note: OpenFermion's get_sparse_operator doesn't work on
    # MajoranaOperators directly.
    
    # MajoranaStrings
    for ms in subsets(range(7)):
        check_sparse_matrices_exactly_equal(
            MajoranaString(ms).sparse(),
            of.get_sparse_operator(of.get_fermion_operator(of.MajoranaOperator(ms)))
        )
    
    # MajoranaPolynomials
    for k in range(1,5):
        # Form MajoranaPolynomial and of.MajoranaOperator
        p1 = MajoranaPolynomial()
        p2 = of.MajoranaOperator()
        for i,ms in enumerate(subsets(range(k))):
            coeff = np.cos(k*i) + np.sin(2*k*i)*1j
            p1 += coeff * MajoranaString(ms)
            p2 += of.MajoranaOperator(ms,coeff)
        check_sparse_matrices_approx_equal(p1.sparse(),
                                           of.get_sparse_operator(of.get_fermion_operator(p2)))