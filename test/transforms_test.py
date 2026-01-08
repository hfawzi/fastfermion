"""Test transform functions of fastfermion (Jordan-Wigner, etc.)"""

from common import check_equal, subsets
from fastfermion import (
    MAX_QUBITS,
    FermiString,
    MajoranaString,
    PauliString,
    tomajorana,
    jw,
    tofermi,
    rjw,
)
import openfermion as of
import itertools

def test_fermi_to_majorana():
    """Test the Fermi to Majorana transform"""
    m = 4
    m1 = m//2
    m2 = m - m1
    M = MAX_QUBITS
    # To test edge cases, we take the set of modes to be [0...m1] [M-m2...M]
    # where M = MAX_QUBITS, and m1+m2 = m
    modes = sorted(list(range(m1)) + list(range(M-m2, M)),reverse=True)
    for U in subsets(modes):
        for S in subsets(modes):
            x = of.FermionOperator([(u,1) for u in U] + [(s,0) for s in S])
            check_equal(
                tomajorana(FermiString(U,S)),
                of.get_majorana_operator(x)
            )

def test_majorana_to_fermi():
    """Test the Majorana to Fermi transform"""
    m = 6
    m1 = m//2
    m2 = m - m1
    M = MAX_QUBITS
    modes = sorted(list(range(m1)) + list(range(M-m2, M)),reverse=False)
    for U in subsets(modes):
        check_equal(
            tofermi(MajoranaString(U)),
            of.normal_ordered(of.get_fermion_operator(of.MajoranaOperator(U)))
        )

def test_jw():
    """Test the Jordan-Wigner transform, i.e., the Pauli to Fermi transform"""
    m = 4
    m1 = m//2
    m2 = m - m1
    M = MAX_QUBITS
    modes = sorted(list(range(m1)) + list(range(M-m2, M)),reverse=True)
    for U in subsets(modes):
        for S in subsets(modes):
            x = of.FermionOperator([(u,1) for u in U] + [(s,0) for s in S])
            check_equal(
                jw(FermiString(U,S)),
                of.jordan_wigner(x)
            )

def test_reverse_jw():
    """Test the reverse Jordan-Wigner transform, i.e., the Pauli to Fermi
    transform"""
    sys = [0,1,2,3]
    for U in subsets(sys):
        for actions in itertools.product(('X','Y','Z'), repeat=len(U)):
            # pstr is of the form ((0,'X'),(1,'Y'))
            pstr = tuple(zip(U,actions))
            check_equal(
                rjw(PauliString(pstr)),
                of.normal_ordered(of.reverse_jordan_wigner(of.QubitOperator(pstr)))
            )
