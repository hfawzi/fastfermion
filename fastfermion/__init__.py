"""
Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
All rights reserved. Use of this source code is governed
by a license that can be found in the LICENSE file.

fastfermion module
"""

from .ffcore import *
from .ffcore import __version__
from ._cirq import from_cirq, to_cirq, to_paulisum
from ._of import *
from ._parse import *


def coefficient(p,op):
    """
    Returns coefficient of a monomial `op` in a polynomial `p`

    Argument op can be a:
    * string (e.g., "X0 Z1", or "f1^ f2", or "m0 m1")
    * tuple (e.g., ((0, 'X'), (1,'Z')), or ((1,1),(2,0)), or (0,1))
    * or one of PauliString, FermiString or MajoranaString.

    In the case of Fermi polynomials, op has to be normal ordered: i.e., all
    the creation operators appear to the left of annihilation operators, and
    and creation and annihilation operators are ordered in *decreasing order*
    from left to right. For example:
    * coefficient(p,"f2^ f0 f1^ f1") is invalid
    * coefficient(p,"f2^ f1^ f0 f1") is invalid
    * coefficient(p,"f2^ f1^ f1 f0") is valid

    Similarly for Majorana polynomials, op has to be a Majorana string
    where the operators are ordered in *increasing order*.

    To get the constant coefficient of a polynomial, use op="1".

    This function can also be invoked using the dot notation,
    see examples below.

    Examples:
    ```
    >>> from fastfermion import poly, coefficient
    >>> coefficient(poly("(X0 Z0 + .4 Y1 + .5 X2) X2"), "1")
    >>> (0.5+0j)
    >>> coefficient(poly("(X0 Z0 + .4 Y1 + .5 X2) X2"), ((1,'Y'),(2,'X')))
    >>> (0.4+0j)
    >>> poly("f1 f1^").coefficient("1")
    >>> (1+0j)
    >>> poly("f1 f1^").coefficient("f1^ f1")
    >>> (-1+0j)
    >>> poly("3 f2^ f1 - f1").coefficient(((2,1),(1,0)))  # ((2,1),(1,0)) is the tuple representation of f2^ f1
    >>> (3+0j)
    >>> poly("m0 m2 m1 + 3").coefficient((0,1,2))
    >>> -1
    ```
    """
    return p.coefficient(op)

def degree(p,v=None):
    """
    Degree of a polynomial.

    If second argument is specified, it should be either "X","Y","Z" for Pauli
    polynomials, or 0 (annihilation), or 1 (creation), for Fermi polynomials.
    In this case returns the degree wrt specified variables.
    Function can also be invoked using the dot notation.

    Examples:
    ```
    >>> from fastfermion import degree, poly
    >>> A = poly("X0 Z2 + 1j X1")
    >>> degree(A)
    2
    >>> A.degree()
    2
    >>> A.degree("Y")
    0
    >>> B = poly("f5d f1 f3")
    >>> B.degree()
    3
    >>> B.degree(1)
    1
    ```
    """
    if v is None:
        return p.degree()
    return p.degree(v)

def dagger(p):
    """
    Adjoint/dagger of the polynomial.

    Examples:
    ```
    >>> from fastfermion import dagger, poly
    >>> dagger(poly("X0 Z2 + 1j X1 Y2"))
    X0 Z2 - 1j X1 Y2
    >>> dagger(poly("f5^ f3 f1"))
    - f3^ f1^ f5
    ```
    Note the negative sign since Fermi polynomials are automatically normal ordered
    ```
    >> poly("(1+2j) m0 m4 m5").dagger()
    (-1+2j) m0 m4 m5
    ```
    """
    return p.dagger()

def norm(p,k=1):
    """
    Norm of the coefficient vector.

    If polynomial has coefficients c_i, returns (sum_i |c_i|^k)^{1/k}
    By default k=1. Supported values of k are k=1, k=2, and k='inf'.
    If k='inf' returns the largest coefficient in magnitude

    Examples:
    ```
    >>> from fastfermion import norm, poly
    >>> norm(poly("3.1 X0 Z2 + 1j X1 Y2"),1)
    4.1
    >>> poly("-3 + f5^ f1 f3").norm('inf')
    3
    ```
    """
    if k == 'inf':
        return p.norm_inf()
    return p.norm(k)

def extent(p):
    """
    Extent of a polynomial.

    Returns smallest n such that the variables in p are all indexed by [0..n-1]

    Examples:
    ```
    >>> from fastfermion import extent, poly
    >>> extent(poly("3.1 X0 Z2 + 1j X1 Y2"))
    3
    >>> poly("-3 + f5^ f1 f3").extent()
    6
    ```
    """
    return p.extent()

def truncate(p,maxdegree):
    """
    Removes all terms from a polynomial which have degree larger than maxdegree

    Examples:
    ```
    >>> poly("3 X0 Y1 X2 + 0.2 X0").truncate(2)
    0.2 X0
    >>> truncate(poly("f5^ f3^ f1 + 1.2 f2"),1)
    1.2 f2
    ``` 
    """
    return p.truncate(maxdegree)

def compress(p,threshold=0):
    """
    Removes terms with small weight.

    Removes all terms from a polynomial which are zero, or smaller in magnitude
    than a threshold (if specified)

    Examples:
    ```
    >>> from fastfermion import compress, poly
    >>> compress(poly("1e-6 X0 Z2 + 1j X1 Y2"),1e-5)
    1j X1 Y2
    >> poly("(1+2j) m0 m4 m5 + 1e-8").compress(1e-7)
    (1+2j) m0 m4 m5
    ```
    """
    return p.compress(threshold)

def permute(p,perm):
    """
    Permutes the indices of the variables in polynomial

    The permutation is specified as a list of length k containing exactly
    all the integers 0 to k-1.

    Examples:
    ```
    >>> from fastfermion import permute, poly
    >>> permute(poly("X0 Z1"),[1,0])
    Z0 X1
    >>> permute(poly("X0 Y1 Z2"),[2,0,1])
    Y0 Z1 X2
    >> poly("m0 m1").permute([1,0])
    - m0 m1
    ```
    """
    return p.permute(perm)

def commutes(a,b):
    """
    Checks whether a and b commute.

    Examples:
    ```
    >>> from fastfermion import commutes, poly
    >>> commutes(poly("X0"),poly("Z1"))
    True
    >>> commutes(poly("f0d f1"),poly("f2d f3"))
    True
    ```
    """
    return a.commutes(b)

def commutator(a,b):
    """
    Returns the commutator of a and b.

    Examples:
    ```
    >>> from fastfermion import commutes, poly
    >>> commutator(poly("X0"),poly("Z0"))
    -2j Y0
    ```
    """
    return a.commutator(b)


def sparse(p,*args,**kwargs):
    """
    Returns sparse matrix representation of polynomial p
    """
    return p.sparse(*args,**kwargs)

def topauli(p):
    """
    Converts a {Fermi,Majorana} polynomial p into a PauliPolynomial
    """
    if isinstance(p, PauliPolynomial):
        return p
    return p.topauli()

def tofermi(p):
    """
    Converts a {Pauli,Majorana} polynomial p into a FermiPolynomial
    """
    if isinstance(p, FermiPolynomial):
        return p
    return p.tofermi()

def tomajorana(p):
    """
    Converts a {Pauli,Fermi} polynomial p into a MajoranaPolynomial
    """
    if isinstance(p, MajoranaPolynomial):
        return p
    return p.tomajorana()
