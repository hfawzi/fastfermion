"""
Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
All rights reserved. Use of this source code is governed
by a license that can be found in the LICENSE file.

fastfermion/OpenFermion interface
"""

import importlib
from . import ffcore as ff

try:
    of = importlib.import_module("openfermion")
    QubitOperator = of.QubitOperator
    FermionOperator = of.FermionOperator
    MajoranaOperator = of.MajoranaOperator

    def from_openfermion(op: QubitOperator | FermionOperator | MajoranaOperator):
        """Converts an OpenFermion operator to the corresponding
        fastfermion object"""
        if isinstance(op,QubitOperator):
            res = ff.PauliPolynomial()
            for pstr, v in op.terms.items():
                res += ff.PauliPolynomial(pstr,v)
            return res
        if isinstance(op,FermionOperator):
            res = ff.FermiPolynomial()
            for fstr, v in op.terms.items():
                res += ff.FermiPolynomial(fstr,v)
            return res
        if isinstance(op,MajoranaOperator):
            res = ff.MajoranaPolynomial()
            for mstr, v in op.terms.items():
                res += ff.MajoranaPolynomial(mstr,v)
            return res
        raise ValueError("Argument is not an instance of any of " \
        "QubitOperator, FermionOperator, MajoranaOperator")

    def to_openfermion(poly: ff.PauliPolynomial | ff.FermiPolynomial | ff.MajoranaPolynomial):
        """Converts a fastfermion polynomial to an OpenFermion operator"""
        if isinstance(poly,ff.PauliPolynomial):
            res = QubitOperator()
            for pstr, v in poly.terms.items():
                res += QubitOperator(tuple(pstr.indices()),v)
            return res
        if isinstance(poly,ff.FermiPolynomial):
            res = FermionOperator()
            for fstr, v in poly.terms.items():
                res += FermionOperator(tuple(fstr.indices()),v)
            return res
        if isinstance(poly,ff.MajoranaPolynomial):
            res = MajoranaOperator()
            for mstr, v in poly.terms.items():
                res += MajoranaOperator(tuple(mstr.indices()),v)
            return res
        raise ValueError("Argument is not an instance of any of " \
        "PauliPolynomial, FermiPolynomial, MajoranaPolynomial")


except ModuleNotFoundError:
    def from_openfermion(op):
        """Dummy function"""
        raise ModuleNotFoundError
    def to_openfermion(op):
        """Dummy function"""
        raise ModuleNotFoundError