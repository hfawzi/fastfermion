from common import (
    check_equal,
    check_majoranapolynomial_equal,
    fermistringstuple,
    paulistringstuple,
    majoranastringstuple,
    subsets
)

import fastfermion
from fastfermion import (
    PauliString,
    PauliPolynomial,
    FermiString,
    FermiPolynomial,
    MajoranaString,
    MajoranaPolynomial,
    commutator,
    commutes,
    from_openfermion,
    to_openfermion,
    poly,
    coefficient
)

import itertools
import numpy as np
import openfermion as of

def test_pauli_multiplication():
    """Test multiplication of PauliStrings and compare against OpenFermion"""
    n = 4
    # Generates all PauliStrings in OpenFermion format
    # ((0,'X'),), ((0,'Y'),), ..., ((0,'Z'),(1,'Z'),...,(n-1,'Z'))
    all_strings = [tuple((i,a) for i,a in enumerate(s) if a != 'I')
                   for s in itertools.product(('I','X','Y','Z'),repeat=n)]
    assert len(all_strings) == 4**n
    for p in all_strings:
        for q in all_strings:
            check_equal(
                PauliString(p)*PauliString(q),
                of.QubitOperator(p)*of.QubitOperator(q)
            )

    # Polynomials
    p = poly("(X0 Z0 + .4 Y1 + .5 X2) X2")
    q = poly("1 + X0 Y1 + X1 X2 Z3 - 0.1 Z2")
    check_equal(p*q, to_openfermion(p)*to_openfermion(q))

    # In-place operations
    p = poly("X0 Z0")
    p += poly("Z1")
    assert p == poly("X0 Z0 + Z1")
    p /= 2
    assert p == poly("(X0 Z0 + Z1)/2")
    p -= PauliString("Z1")
    assert p == poly("(X0 Z0 - Z1) / 2")
    p += PauliString("Z1")
    assert p == poly("(X0 Z0 + Z1) / 2")
    p *= PauliString("X1")
    assert p == poly("(X0 Z0 + Z1) X1 / 2")
    p *= 2
    p *= poly("1+X3")
    assert p == poly("(X0 Z0 + Z1) X1 (1+X3)")


def test_fermi_multiplication():
    """Test multiplication of FermiStrings and compare with OpenFermion"""
    # Compute all possible products (a_U^{dagger} a_S)*(a_T^{dagger} a_V)
    # where U,V,S,T are subsets of {0,...,m-1}
    m = 3
    modes = sorted(range(m),reverse=True)
    for U in subsets(modes):
        for S in subsets(modes):
            for T in subsets(modes):
                for V in subsets(modes):
                    x = of.FermionOperator([(u,1) for u in U] + [(v,0) for v in V])
                    y = of.FermionOperator([(s,1) for s in S] + [(t,0) for t in T])
                    check_equal(
                        FermiString(U,V)*FermiString(S,T),
                        of.normal_ordered(x*y)
                    )
    # Polynomials
    p = poly("3 f2^ f1 - f1")
    q = 2*FermiString([3],[1]) + .1j*FermiString([2,1,0],[2])
    check_equal(p*q, to_openfermion(p)*to_openfermion(q))

def test_majorana_multiplication():
    """Test multiplication of MajoranaStrings and compare with OpenFermion"""
    m = 3
    modes = tuple(range(2*m))
    for S in subsets(modes):
        for T in subsets(modes):
            check_equal(
                MajoranaString(S)*MajoranaString(T),
                of.MajoranaOperator(S)*of.MajoranaOperator(T)
            )
    
    # Polynomials
    p = poly("m1 m2 + 2*m2 m3")
    q = poly("m3 m4 + m5 m6")
    check_majoranapolynomial_equal(p*q, poly("m1 m2 m3 m4 + 2*m2 m4 + m1 m2 m5 m6 + 2*m2 m3 m5 m6"))

def test_coefficient():
    """Test coefficient"""

    assert poly("(X0 Z0 + .4 Y1 + .5 X2) X2").coefficient("1") == 0.5
    assert coefficient(poly("(X0 Z0 + .4 Y1 + .5 X2) X2"), ((1,'Y'),(2,'X'))) == 0.4
    assert poly("f1 f1^").coefficient("1") == 1
    assert poly("f1 f1^").coefficient("f1^ f1") == -1
    assert poly("3 f2^ f1 - f1").coefficient(((2,1),(1,0))) == 3
    assert poly("m0 m2 m1 + 3").coefficient((0,1,2)) == -1
    assert poly("m0 m2 m1 + 3").coefficient((0,1)) == 0

    coeffs = [0.1,1.2j,-3]
    strings = {
        "Pauli": [PauliString("XX"), PauliString("Z"), PauliString()],
        "Fermi": [FermiString([3,2],[0]), FermiString([],[2]), FermiString()],
        "Majorana": [MajoranaString([1,2]), MajoranaString([3]), MajoranaString()]
    }
    for poly_type, poly_strings in strings.items():
        p = sum(c*s for c,s in zip(coeffs,poly_strings))
        if poly_type == "Pauli":
            ops = paulistringstuple(2)
        elif poly_type == "Fermi":
            ops = fermistringstuple(4)
        elif poly_type == "Majorana":
            ops = majoranastringstuple(4)
        p_terms = p.terms
        string_type = getattr(fastfermion, poly_type + "String")
        for op in ops:
            assert p.coefficient(op) == p_terms.get(string_type(op),0)


def test_norm():
    """Test the function norm"""
    coeffs = [0.1,1.2j,-3]
    strings = {
        "Pauli": [PauliString("XX"), PauliString("Z"), PauliString()],
        "Fermi": [FermiString([3,2],[0]), FermiString([],[2]), FermiString()],
        "Majorana": [MajoranaString([1,4]), MajoranaString([9]), MajoranaString()]
    }
    for _, poly_strings in strings.items():
        p = sum(c*s for c,s in zip(coeffs,poly_strings))
        assert p.norm(0) == len(coeffs)
        assert p.norm(1) == np.sum(np.abs(coeffs))
        assert p.norm(2) == np.sqrt(np.sum(np.abs(coeffs)**2))
        assert p.norm('inf') == np.max(np.abs(coeffs))

def test_degree():
    """Test degree"""
    assert PauliString("XYZ").degree() == 3
    assert PauliString("XYZX").degree('x') == 2
    assert PauliString("XYZ").degree('y') == 1
    assert PauliString("XYZ").degree('z') == 1
    assert poly("1 + X0 Y1 + X1 X2 Z3 - 0.1 Z2").degree() == 3
    assert poly("1 + X0 Y1 + X1 X2 Z3 - 0.1 Z2").degree('x') == 2
    assert poly("1 + X0 Y1 + X1 X2 Z3 - 0.1 Z2").degree('y') == 1
    assert poly("1 + X0 Y1 + X1 X2 Z3 - 0.1 Z2").degree('z') == 1
    assert FermiString([3,2,1],[3,2]).degree() == 5
    assert FermiString([3,2,1],[3,2]).degree(1) == 3
    assert FermiString([3,2,1],[3,2]).degree(0) == 2
    assert poly("f1 f1^").degree() == 2
    assert poly("f1 f1^").degree(1) == 1
    assert poly("f1 f1^").degree(0) == 1
    assert MajoranaString([0,4]).degree() == 2
    assert poly("m0 m2 m1 + 3").degree() == 3

def test_indices():
    """Test indices"""
    assert PauliString("XX").indices() == [(0,'X'),(1,'X')]
    assert PauliString("IZ").indices() == [(1,'Z')]
    assert PauliString("").indices() == []
    assert FermiString([3,2],[0]).indices() == [(3,1),(2,1),(0,0)]
    assert FermiString([],[2]).indices() == [(2,0)]
    assert FermiString().indices() == []
    assert MajoranaString([1,4]).indices() == [1,4]
    assert MajoranaString([9]).indices() == [9]
    assert MajoranaString().indices() == []


def test_permute():
    """Test permutations of PauliString, FermiString and MajoranaString"""
    assert PauliString("XYZ").permute([1,2,0]) == PauliString("ZXY")
    assert poly("1 + X0 Y1 Z4").permute([1,2,0,3,4]) == poly("1 + X1 Y2 Z4")
    assert FermiString([3,2],[1,0]).permute([0,1,3,2]) == -1*FermiString([3,2],[1,0])
    assert poly("f3^ f1^ f0 f1 + 2 f2").permute([3,1,0,2]) == poly("f2^ f1^ f3 f1 + 2 f0")
    assert FermiString([3,2],[1,0]).permute([1,0,3,2]) == +1*FermiString([3,2],[1,0])
    assert MajoranaString([0,1,4]).permute([1,0]) == -MajoranaString([0,1,4])
    assert poly("1 + m0 m1").permute([1,0]) == poly("1 + m1 m0")

    n = 4
    for ps in paulistringstuple(n):
        for perm in itertools.permutations(range(n),n):
            assert PauliString(ps).permute(perm) == PauliString([(perm[ind],op) for ind,op in ps])
    for fs in fermistringstuple(n):
        for perm in itertools.permutations(range(n),n):
            assert FermiString(fs).permute(perm) == FermiPolynomial([(perm[ind],op) for ind,op in fs])
    for ms in subsets(range(n)):
        for perm in itertools.permutations(range(n),n):
            assert MajoranaString(ms).permute(perm) == MajoranaPolynomial([perm[ind] for ind in ms])
    


def test_pauli_commutation():
    """Test commutation"""
    assert commutes(poly("1+X0 Z1"),poly("X3 Z4"))
    assert commutator(poly("1+2 X0 Z1"),poly("Y0+Y1")) == poly("2 (X0 Z1 (Y0+Y1) - (Y0+Y1) X0 Z1)")
    for a in paulistringstuple(4):
        for b in paulistringstuple(4):
            pa = PauliString(a)
            pb = PauliString(b)
            _comm = pa*pb-pb*pa
            _commutes = all(coeff==0. for coeff in (pa*pb-pb*pa).terms.values())
            assert commutator(pa,pb) == _comm
            assert commutator(PauliPolynomial(pa),PauliPolynomial(pb)) == _comm
            assert commutes(pa,pb) == _commutes
            assert commutes(PauliPolynomial(pa),PauliPolynomial(pb)) == _commutes
            
def test_fermi_commutation():
    """Test commutation"""
    for a in fermistringstuple(4):
        for b in fermistringstuple(4):
            pa = FermiString(a)
            pb = FermiString(b)
            _comm = pa*pb-pb*pa
            _commutes = all(coeff==0. for coeff in (pa*pb-pb*pa).terms.values())
            assert commutator(pa,pb) == _comm
            assert commutator(FermiPolynomial(pa),FermiPolynomial(pb)) == _comm
            assert commutes(pa,pb) == _commutes
            assert commutes(FermiPolynomial(pa),FermiPolynomial(pb)) == _commutes

def test_majorana_commutation():
    """Test commutation"""
    for a in majoranastringstuple(4):
        for b in majoranastringstuple(4):
            pa = MajoranaString(a)
            pb = MajoranaString(b)
            _comm = pa*pb-pb*pa
            _commutes = all(coeff==0. for coeff in (pa*pb-pb*pa).terms.values())
            assert commutator(pa,pb) == _comm
            assert commutator(MajoranaPolynomial(pa),MajoranaPolynomial(pb)) == _comm
            assert commutes(pa,pb) == _commutes
            assert commutes(MajoranaPolynomial(pa),MajoranaPolynomial(pb)) == _commutes

def test_init_fermipolynomial():
    """Test initializing a FermiPolynomial with a non-normal-ordered sequence"""
    assert FermiPolynomial(((0,0),(1,1))) == poly("-f1^ f0")
    assert FermiPolynomial(((0,0),(1,0),(2,0))) + 2 == poly("-f2 f1 f0 + 2")
    max_degree = 6
    for degree in range(0,max_degree):
        for modes in itertools.permutations(range(degree)):
            for actions in itertools.product((0,1),repeat=degree):
                fstr = tuple(zip(modes,actions))
                check_equal(
                    FermiPolynomial(fstr),
                    of.normal_ordered(of.FermionOperator(fstr))
                )

def test_from_openfermion_pauli():
    """Test the from_openfermion function for Pauli polynomials"""
    max_degree = 4
    max_terms = 3
    allpstrings = [tuple(zip(loc,actions)) \
                    for degree in range(0,max_degree)
                    for loc in itertools.permutations(range(degree)) \
                    for actions in itertools.product(('X','Y','Z'),repeat=degree)]
    for num_terms in range(max_terms):
        for terms in itertools.combinations(allpstrings, num_terms):
            coeffs = [2+1.5*i for i,_ in enumerate(terms)]
            of_poly = of.QubitOperator()
            for coeff, term in zip(coeffs,terms):
                of_poly += of.QubitOperator(term, coeff)
            check_equal(
                    from_openfermion(of_poly),
                    of_poly
                )

def test_from_openfermion_fermi():
    """Test the from_openfermion function for Fermi polynomials"""
    max_degree = 5
    max_terms = 2
    max_modes = 5
    allfstrings = [tuple(zip(modes,actions)) \
                    for degree in range(0,max_degree)
                    for modes in itertools.product(range(max_modes),repeat=degree) \
                    for actions in itertools.product((0,1),repeat=degree)]
    for num_terms in range(max_terms):
        for terms in itertools.combinations(allfstrings, num_terms):
            coeffs = [2+1.5*i for i,_ in enumerate(terms)]
            of_poly = of.FermionOperator()
            for coeff, term in zip(coeffs,terms):
                of_poly += of.FermionOperator(term, coeff)
            check_equal(
                    from_openfermion(of_poly),
                    of.normal_ordered(of_poly)
            )

def test_from_openfermion_majorana():
    """Test the from_openfermion function for Majorana polynomials"""
    max_degree = 4
    max_modes = 6
    max_terms = 3
    allmstrings = [loc \
                    for degree in range(0,max_degree)
                    for loc in itertools.product(range(max_modes),repeat=degree)]
    for num_terms in range(max_terms):
        for terms in itertools.combinations(allmstrings, num_terms):
            coeffs = [2+1.5*i for i,_ in enumerate(terms)]
            of_poly = of.MajoranaOperator()
            for coeff, term in zip(coeffs,terms):
                of_poly += of.MajoranaOperator(term, coeff)
            check_equal(
                    from_openfermion(of_poly),
                    of_poly
                )

def test_to_openfermion():
    """Test to_openfermion"""
    assert to_openfermion(poly("3.1 X1 X2 - 1.2j Z0")) == \
         of.QubitOperator(((1,'X'),(2,'X')),3.1) + of.QubitOperator(((0,'Z'),),-1.2j)
    assert to_openfermion(poly("f1^ f0 - 1j f2")) == \
         of.FermionOperator(((1,1),(0,0)),1) + of.FermionOperator(((2,0),),-1j)
    assert to_openfermion(poly("m0 m1 + 3")) == \
         of.MajoranaOperator((0,1),1) + of.MajoranaOperator((),3)