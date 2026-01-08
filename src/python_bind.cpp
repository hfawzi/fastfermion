/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#include "pauli_algebra.h"
#include "fermi_algebra.h"
#include "majorana_algebra.h"
#include "pauli_sparse.h"
#include "fermi_sparse.h"
#include "majorana_sparse.h"
#include "transforms.h"
#include "fockstate.h"
#include "qubitproductstate.h"
#include "pauli_gates.h"
#include "pauli_propagation.h"
#include "majorana_gates.h"
#include "majorana_propagation.h"
#include "gen.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>

namespace py = pybind11;

// To cast ankerl::unordered_dense::map as a python dict
template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc>
struct pybind11::detail::type_caster<ankerl::unordered_dense::map<Key, Value, Hash, Equal, Alloc>>
    : pybind11::detail::map_caster<ankerl::unordered_dense::map<Key, Value, Hash, Equal, Alloc>, Key, Value> {};


namespace fastfermion {

// To cast CSCMatrix to SciPy CSCMatrix
auto CSCMatrix_to_scipy(const CSCMatrix<ff_complex>& sm) {
    py::object csc_matrix_type = py::module_::import("scipy").attr("sparse").attr("csc_matrix");
    return csc_matrix_type(std::make_tuple(std::move(sm.data), std::move(sm.indices), std::move(sm.indptr)), std::make_tuple(sm.shape[0], sm.shape[1])).release();
};

template <typename PolyT, typename StringT, typename MonT>
void add_poly_basic_ops(py::class_<PolyT>& pyPolyClass) {

    pyPolyClass.def("extent", &PolyT::extent);
    pyPolyClass.def("compress", &PolyT::compress, py::arg("threshold")=1e-8);
    pyPolyClass.def("permute", &PolyT::permute);
    pyPolyClass.def("truncate", &PolyT::truncate);
    pyPolyClass.def("commutes", py::overload_cast<const ff_complex&>(&PolyT::commutes, py::const_));
    pyPolyClass.def("commutes", py::overload_cast<const StringT&>(&PolyT::commutes, py::const_));
    pyPolyClass.def("commutes", py::overload_cast<const MonT&>(&PolyT::commutes, py::const_));
    pyPolyClass.def("commutes", py::overload_cast<const PolyT&>(&PolyT::commutes, py::const_));

    pyPolyClass.def("commutator", py::overload_cast<const ff_complex&>(&PolyT::commutator, py::const_));
    pyPolyClass.def("commutator", py::overload_cast<const StringT&>(&PolyT::commutator, py::const_));
    pyPolyClass.def("commutator", py::overload_cast<const MonT&>(&PolyT::commutator, py::const_));
    pyPolyClass.def("commutator", py::overload_cast<const PolyT&>(&PolyT::commutator, py::const_));

    pyPolyClass.def_readonly("terms", &PolyT::terms);
    pyPolyClass.def("__len__", [](const PolyT& a) { return a.terms.size(); });
    pyPolyClass.def("__str__", &PolyT::to_compact_string);
    pyPolyClass.def("__repr__", &PolyT::to_compact_string);
    pyPolyClass.def("norm", [](const PolyT& p, const std::variant<int,std::string>& v) {
        // v can take the values 0, 1, 2, or "inf"
        // That's why we use variant
        if(v.index() == 0) {
            return p.norm(std::get<int>(v));
        } else {
            if(std::get<std::string>(v) == "inf") {
                return p.norm_inf();
            } else {
                throw_error("Invalid argument");
            }
        }
    }, py::arg("v")=1);
    
    // Operator overloading
    // see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#operator-overloading
    
    // In-place operations

    pyPolyClass.def(py::self += ff_complex());
    pyPolyClass.def(py::self += StringT());
    pyPolyClass.def(py::self += py::self);

    pyPolyClass.def(py::self -= ff_complex());
    pyPolyClass.def(py::self -= StringT());
    pyPolyClass.def(py::self -= py::self);

    pyPolyClass.def(py::self *= ff_complex());
    pyPolyClass.def(py::self *= StringT());
    pyPolyClass.def(py::self *= py::self);

    pyPolyClass.def(py::self /= ff_complex());

    pyPolyClass.def(- py::self);
    
    // Binary operations

    pyPolyClass.def(py::self + ff_complex());
    pyPolyClass.def(py::self + StringT());
    pyPolyClass.def(py::self + py::self);
    pyPolyClass.def(py::self - ff_complex());
    pyPolyClass.def(py::self - StringT());
    pyPolyClass.def(py::self - py::self);
    pyPolyClass.def(py::self * ff_complex());
    pyPolyClass.def(py::self * StringT());
    pyPolyClass.def(py::self * py::self);
    pyPolyClass.def(py::self / ff_complex());
    pyPolyClass.def(ff_complex() + py::self);
    pyPolyClass.def(ff_complex() - py::self);
    pyPolyClass.def(ff_complex() * py::self);
    pyPolyClass.def(py::self == py::self);
    pyPolyClass.def(py::self != py::self);

}

template<typename PauliOpT>
void add_pauli_sparse(py::class_<PauliOpT>& pyPauliOpClass) {
    pyPauliOpClass.def("sparse",[](const PauliOpT& a, const std::optional<int>& n, const std::optional<int>& nup) {
        int _n = n.has_value() ? n.value() : a.extent();
        if(!nup.has_value()) {
            return CSCMatrix_to_scipy(a.sparse(_n));
        } else {
            return CSCMatrix_to_scipy(a.sparse(_n,nup.value()));
        }
    }, py::arg("n") = py::none(), py::arg("nup") = py::none());
}

template<typename FermiOpT>
void add_fermi_sparse(py::class_<FermiOpT>& pyFermiOpClass) {
    pyFermiOpClass.def("sparse",[](const FermiOpT& a, const std::optional<int>& n, const std::optional<int>& nocc) {
        int _n = n.has_value() ? n.value() : a.extent();
        if(!nocc.has_value()) {
            return CSCMatrix_to_scipy(a.sparse(_n));
        } else {
            return CSCMatrix_to_scipy(a.sparse(_n,nocc.value()));
        }
    }, py::arg("n") = py::none(), py::arg("nocc") = py::none());
}

template<typename MajoranaOpT>
void add_majorana_sparse(py::class_<MajoranaOpT>& pyMajoranaOpClass) {
    pyMajoranaOpClass.def("sparse",[](const MajoranaOpT& a, const std::optional<int>& n) {
        if(n.has_value()) {
            return CSCMatrix_to_scipy(a.sparse(n.value()));
        } else {
            return CSCMatrix_to_scipy(a.sparse());
        }
    }, py::arg("n") = py::none());
}

void add_transforms(py::module_& m) {

    // Majorana to Pauli and Fermi to Pauli
    m.def("jw", py::overload_cast<const MajoranaString&>(&jordan_wigner), "Convert a Majorana polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("jw", py::overload_cast<const MajoranaPolynomial&>(&jordan_wigner), "Convert a Majorana polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("jw", py::overload_cast<const FermiString&>(&jordan_wigner), "Convert a Fermi polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("jw", py::overload_cast<const FermiPolynomial&>(&jordan_wigner), "Convert a Fermi polynomial to a Pauli polynomial via the Jordan-Wigner mapping");

    // Pauli to Fermi
    m.def("rjw", py::overload_cast<const PauliString&>(&reverse_jordan_wigner), "Convert a Pauli polynomial to a Fermi polynomial via the reverse Jordan-Wigner mapping");
    m.def("rjw", py::overload_cast<const PauliPolynomial&>(&reverse_jordan_wigner), "Convert a Pauli polynomial to a Fermi polynomial via the reverse Jordan-Wigner mapping");

    // To Pauli
    m.def("topauli", py::overload_cast<const MajoranaString&>(&jordan_wigner), "Convert a Majorana polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("topauli", py::overload_cast<const MajoranaPolynomial&>(&jordan_wigner), "Convert a Majorana polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("topauli", py::overload_cast<const FermiString&>(&jordan_wigner), "Convert a Fermi polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("topauli", py::overload_cast<const FermiPolynomial&>(&jordan_wigner), "Convert a Fermi polynomial to a Pauli polynomial via the Jordan-Wigner mapping");


    // To Fermi
    m.def("tofermi", py::overload_cast<const PauliString&>(&reverse_jordan_wigner), "Convert a Pauli polynomial to a Fermi polynomial via the reverse Jordan-Wigner mapping");
    m.def("tofermi", py::overload_cast<const PauliPolynomial&>(&reverse_jordan_wigner), "Convert a Pauli polynomial to a Fermi polynomial via the reverse Jordan-Wigner mapping");
    m.def("tofermi", py::overload_cast<const MajoranaString&>(&majorana_to_fermi), "Convert a Majorana polynomial a Fermi polynomial");
    m.def("tofermi", py::overload_cast<const MajoranaPolynomial&>(&majorana_to_fermi), "Convert a Majorana polynomial a Fermi polynomial");


    // To Majorana
    m.def("tomajorana", py::overload_cast<const FermiString&>(&fermi_to_majorana), "Convert a Fermi polynomial to Majorana polynomial");
    m.def("tomajorana", py::overload_cast<const FermiPolynomial&>(&fermi_to_majorana), "Convert a Fermi polynomial to Majorana polynomial");
    m.def("tomajorana", [](const PauliString& a) {
        return MajoranaPolynomial(pauli_to_majorana(a));
    }, "Convert a Pauli polynomial to a Majorana polynomial");
    m.def("tomajorana", py::overload_cast<const PauliPolynomial&>(&pauli_to_majorana), "Convert a Pauli polynomial to a Majorana polynomial");

}

void add_states(py::module_& m) {

    py::class_<FockState>(m, "FockState")
        .def(py::init<>())
        .def(py::init<const std::uint64_t&>())
        .def(py::init<const std::vector<int>&>(), py::arg("occ")=std::vector<int>{})
        .def("__call__", py::overload_cast<const FermiString&>(&FockState::operator(), py::const_))
        .def("__call__", py::overload_cast<const FermiPolynomial&>(&FockState::operator(), py::const_))
        .def("__call__", py::overload_cast<const MajoranaString&>(&FockState::operator(), py::const_))
        .def("__call__", py::overload_cast<const MajoranaPolynomial&>(&FockState::operator(), py::const_))
        .def("occ", &FockState::occ)
        .def("vec", [](const FockState& state, const int& n, const std::optional<int>& nocc) {
            if(nocc.has_value()) {
                return state.vec(n, nocc.value());
            } else {
                return state.vec(n);
            }
        }, py::arg("n"), py::arg("nocc") = py::none());

    py::class_<QubitProductState>(m, "QubitProductState")
        .def(py::init<>())
        .def(py::init<const std::uint64_t&>())
        .def(py::init<const std::vector<int>&>(), py::arg("up")=std::vector<int>{})
        .def("__call__", py::overload_cast<const PauliString&>(&QubitProductState::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&QubitProductState::operator(), py::const_))
        .def("up", &QubitProductState::up)
        .def("vec", [](const QubitProductState& state, const int& n, const std::optional<int>& nup) {
            if(nup.has_value()) {
                return state.vec(n, nup.value());
            } else {
                return state.vec(n);
            }
        }, py::arg("n"), py::arg("nocc") = py::none());

}

void add_pauli_gates(py::module_& m) {
    py::class_<pauli_gates::H>(m, "H")
        .def(py::init<const int&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::H::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::H::operator(), py::const_))
        .def("__repr__", &pauli_gates::H::to_string)
        .def("__str__", &pauli_gates::H::to_string)
        .def_property_readonly("qubits", [](const pauli_gates::H& gate) { return std::vector<int>{gate.i}; })
        .def("aspoly", &pauli_gates::H::aspoly)
        .doc() = R"DOC(
        Hadamard unitary on qubit p:
            U = (X_p + Z_p) / sqrt(2)

        Example:
        >>> from fastfermion import H
        >>> U = H(0)
        )DOC";

    py::class_<pauli_gates::S>(m, "S")
        .def(py::init<const int&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::S::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::S::operator(), py::const_))
        .def("__repr__", &pauli_gates::S::to_string)
        .def("__str__", &pauli_gates::S::to_string)
        .def_property_readonly("qubits", [](const pauli_gates::S& gate) { return std::vector<int>{gate.i}; })
        .def("aspoly", &pauli_gates::S::aspoly)
        .doc() = R"DOC(
        S unitary on qubit p:
            U = ((1+1j) + (1-1j) Z_p) / 2

        Example:
        >>> from fastfermion import S
        >>> U = S(0)
        )DOC";

    py::class_<pauli_gates::CNOT>(m, "CNOT")
        .def(py::init<const int&, const int&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::CNOT::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::CNOT::operator(), py::const_))
        .def("__repr__", &pauli_gates::CNOT::to_string)
        .def("__str__", &pauli_gates::CNOT::to_string)
        .def_property_readonly("qubits", [](const pauli_gates::CNOT& gate) { return std::vector<int>{gate.i,gate.j}; })
        .def("aspoly", &pauli_gates::CNOT::aspoly)
        .doc() = R"DOC(
        CNOT unitary on qubits p,q:
            U = (1 + Z_p + X_q - Z_p X_q)/2

        Example:
        >>> from fastfermion import CNOT
        >>> U = CNOT(0,1)
        )DOC";

    py::class_<pauli_gates::SWAP>(m, "SWAP")
        .def(py::init<const int&, const int&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::SWAP::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::SWAP::operator(), py::const_))
        .def("__repr__", &pauli_gates::SWAP::to_string)
        .def("__str__", &pauli_gates::SWAP::to_string)
        .def_property_readonly("qubits", [](const pauli_gates::SWAP& gate) { return std::vector<int>{gate.i,gate.j}; })
        .def("aspoly", &pauli_gates::SWAP::aspoly)
        .doc() = R"DOC(
        SWAP unitary on qubits p,q:
            U = (1 + X_p X_q + Y_p Y_q + Z_p Z_q)/2

        Example:
        >>> from fastfermion import SWAP
        >>> U = SWAP(0,1)
        )DOC";

    py::class_<pauli_gates::CZ>(m, "CZ")
        .def(py::init<const int&, const int&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::CZ::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::CZ::operator(), py::const_))
        .def("__repr__", &pauli_gates::CZ::to_string)
        .def("__str__", &pauli_gates::CZ::to_string)
        .def_property_readonly("qubits", [](const pauli_gates::CZ& gate) { return std::vector<int>{gate.i,gate.j}; })
        .def("aspoly", &pauli_gates::CZ::aspoly)
        .doc() = R"DOC(
        SWAP unitary on qubits p,q:
            U = (1 + Z_p + Z_q + Z_p Z_q)/2

        Example:
        >>> from fastfermion import CZ
        >>> U = CZ(0,1)
        )DOC";

    py::class_<pauli_gates::ROT>(m, "ROT")
        .def(py::init<const std::string&, const std::vector<int>&, const ff_float&>())
        .def(py::init<const PauliString&, const ff_float&>())
        .def(py::init<const std::string&, const ff_float&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::ROT::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::ROT::operator(), py::const_))
        .def_property_readonly("qubits", [](const pauli_gates::ROT& gate) { return gate.ps.support_set(); })
        .def_readonly("axis", &pauli_gates::ROT::ps)
        .def_readonly("theta", &pauli_gates::ROT::theta)
        .def("aspoly", &pauli_gates::ROT::aspoly)
        .def("__repr__", &pauli_gates::ROT::to_string)
        .def("__str__", &pauli_gates::ROT::to_string)
        .doc() = R"DOC(
        Represents a Pauli rotation U = e^{-i P theta/2}
        where P is a Pauli string and theta is real number

        Example:
        >>> from fastfermion import ROT
        >>> U = ROT("XX",(0,1),0.75)
        )DOC";
}

void add_pauli_propagation(py::module_& m) {
    
    m.def("propagate",
        [](
            const pauli_gates::Circuit& circuit,
            const std::variant<PauliString, PauliPolynomial>& observable,
            const std::optional<int>& maxdegree,
            const std::optional<ff_float>& mincoeff
        ) {
            int _maxdegree = maxdegree.has_value() ? maxdegree.value() : ff_ulong::DIGITS;
            ff_float _mincoeff = mincoeff.has_value() ? mincoeff.value() : 0;
            if(observable.index() == 0) {
                // PauliString
                return pauli_gates::propagate(circuit, PauliPolynomial(std::get<0>(observable)), _maxdegree, _mincoeff);
            } else {
                // PauliPolynomial
                return pauli_gates::propagate(circuit, std::get<1>(observable), _maxdegree, _mincoeff);
            }
        }, py::arg("circuit"), py::arg("observable"), py::arg("maxdegree") = py::none(), py::arg("mincoeff") = py::none(),
        R"DOC(
        Backpropagates a polynomial through a circuit.
        If maxdegree is specified, truncates any term of degree larger than maxdegree.
        Note: the truncation only happens after applying non-Clifford gates (i.e., ROT gates).
        So the output of propagate may have degree larger than maxdegree.

        Examples:
        >>> from fastfermion import H, CNOT, propagate
        >>> circuit = [H(0),CNOT(0,1)]
        >>> observable = PauliString("XZ")
        >>> result = propagate(circuit,observable,maxdegree=3,mincoeff=1e-8)
        )DOC"
    );

}

void add_majorana_propagation(py::module_& m) {

    py::class_<majorana_gates::MROT>(m, "MROT")
        .def(py::init<const std::vector<int>&, const ff_complex&, const ff_float&>())
        .def(py::init<const MajoranaString&, const ff_float&>())
        .def(py::init<const MajoranaString&, const ff_complex&, const ff_float&>())
        .def("__call__", py::overload_cast<const MajoranaString&>(&majorana_gates::MROT::operator(), py::const_))
        .def("__call__", py::overload_cast<const MajoranaPolynomial&>(&majorana_gates::MROT::operator(), py::const_))
        .def_property_readonly("qubits", [](const majorana_gates::MROT& gate) { return gate.ms.support_set(); })
        .def_readonly("axis", &majorana_gates::MROT::ms)
        .def_readonly("theta", &majorana_gates::MROT::theta)
        .def("aspoly", &majorana_gates::MROT::aspoly)
        .def("__repr__", &majorana_gates::MROT::to_string)
        .def("__str__", &majorana_gates::MROT::to_string)
        .doc() = R"DOC(
        Represents a Majorana rotation U = e^{-i theta/2 M}
        where M is a Hermitian Majorana monomial and theta is a real number.
        The Majorana monomial is supplied as a pair P,c where P is
        a MajoranaString and c is a complex number, so that
            MROT(P,c,theta)
        represents the unitary e^{-i theta/2 (c*P)}.
        If c*P is not Hermitian, an error is raised

        Example:
        >>> from fastfermion import MROT, MajoranaString
        >>> U = MROT(MajoranaString([0,1]),1j,1.0)
        >>> print(U)
        MROT(m0 m1,1.000000) = e^{0.500000 m0 m1}
        >>> V = MROT(MajoranaString([0,2]),1,2.0)
        RuntimeError: Supplied Majorana monomial 1*(m0 m2) is not Hermitian
        )DOC";

    m.def("propagate",
        [](
            const majorana_gates::MajoranaCircuit& circuit,
            const std::variant<MajoranaString,MajoranaPolynomial>& observable,
            const std::optional<int>& maxdegree,
            const std::optional<ff_float>& mincoeff
        ) {
            ff_float mincoeffval = mincoeff.has_value() ? mincoeff.value() : 0;
            if(observable.index() == 0) {
                // MajoranaString
                if(maxdegree.has_value()) {
                    return majorana_gates::propagate(circuit, MajoranaPolynomial(std::get<0>(observable)), maxdegree.value(), mincoeffval);
                } else {
                    return majorana_gates::propagate(circuit, MajoranaPolynomial(std::get<0>(observable)), mincoeffval);
                }
            } else {
                // MajoranaPolynomial
                if(maxdegree.has_value()) {
                    return majorana_gates::propagate(circuit, std::get<1>(observable), maxdegree.value(), mincoeffval);
                } else {
                    return majorana_gates::propagate(circuit, std::get<1>(observable), mincoeffval);
                }
            }
        },
        py::arg("circuit"), py::arg("observable"), py::arg("maxdegree") = py::none(), py::arg("mincoeff") = 0,
        R"DOC(
        Backpropagates a Majorana polynomial through a Majorana circuit.
        If maxdegree is specified, truncates any term of degree larger than maxdegree.

        Examples:
        >>> from fastfermion import MROT, propagate
        >>> circuit = [MROT(MajoranaString([0,1]))]
        >>> observable = PauliString("XZ")
        >>> result = propagate(circuit,observable,maxdegree=3)
        )DOC"
    );

}


void add_gen(py::module_& m) {

    m.def("paulis",&paulis,R"DOC(
            Returns generators of Pauli algebra

            Example:
                >>> from fastfermion import paulis
                >>> X,Y,Z = paulis(10)
                >>> A = X[0]*Y[8] + .25 * Z[9]
        )DOC");

    m.def("fermis",&fermis,R"DOC(
            Returns annihilation operators

            Example:
                >>> from fastfermion import fermis
                >>> f = fermis(10)
                >>> B = f[0].dagger() * f[0] - f[3]
        )DOC");

    m.def("majoranas",&majoranas,R"DOC(
            Returns Majorana operators

            Example:
                >>> from fastfermion import majoranas
                >>> m = majoranas(10)
                >>> C = m[0]*m[9] - m[8]
        )DOC");

    m.def("paulistrings", py::overload_cast<int>(&paulistrings));
    m.def("paulistrings", py::overload_cast<int, int>(&paulistrings));
    m.def("paulistrings", py::overload_cast<int, int, std::function<bool(const std::vector<int>&, const std::vector<char>&)>>(&paulistrings));

    m.def("fermistrings", py::overload_cast<int>(&fermistrings));
    m.def("fermistrings", py::overload_cast<int, int>(&fermistrings));
    m.def("fermistrings", py::overload_cast<int, int, std::function<bool(const std::vector<int>&, const std::vector<int>&)>>(&fermistrings));

    m.def("majoranastrings", py::overload_cast<int>(&majoranastrings));
    m.def("majoranastrings", py::overload_cast<int, int>(&majoranastrings));
    m.def("majoranastrings", py::overload_cast<int, int, std::function<bool(const std::vector<int>&)>>(&majoranastrings));
}

PYBIND11_MODULE(ffcore, m, py::mod_gil_not_used()) {

    // The module name (ffcore) is given as the first macro argument (it should not be in quotes).
    // The second argument (m) defines a variable of type py::module_ which is the main interface for creating bindings.

    m.attr("MAX_QUBITS") = ff_ulong::DIGITS;
    m.attr("FERMI_SYMBOL") = ff_config.fermi_symbol;
    m.attr("MAJORANA_SYMBOL") = ff_config.majorana_symbol;
    m.attr("DAGGER_SYMBOL") = ff_config.dagger_symbol;

    #ifdef FF_VERSION
        m.attr("__version__") = FF_VERSION;
    #else
        m.attr("__version__") = "";
    #endif
    
    py::class_<PauliString> pyPauliString = py::class_<PauliString>(m, "PauliString")
        .def(py::init<>())
        .def(py::init<const std::string&>()) 
        .def(py::init<const std::vector<std::pair<int,char>>&>())
        .def("permute", &PauliString::permute)
        .def("to_string", &PauliString::to_string, py::arg("n") = 0)
        .def("indices", &PauliString::indices)
        .def("degree", [](const PauliString& a, const std::optional<char>& v) { return v.has_value() ? a.degree(v.value()) : a.degree(); }, py::arg("v") = py::none())
        .def("extent", &PauliString::extent)
        .def("commutes", py::overload_cast<const PauliString&>(&PauliString::commutes, py::const_))
        .def("commutes", py::overload_cast<const PauliPolynomial&>(&PauliString::commutes, py::const_))
        .def("commutator", [](const PauliString& a, const PauliString& b) { return PauliPolynomial(a.commutator(b)); })
        .def("commutator", py::overload_cast<const PauliPolynomial&>(&PauliString::commutator, py::const_))
        .def("__str__", &PauliString::to_compact_string)
        .def("__repr__", &PauliString::to_compact_string)
        .def("__hash__", &PauliString::hash)

        // .def("sparse", [](const PauliString& a) { return CSCMatrix_to_scipy(a.sparse()); })
        // .def("sparse", [](const PauliString& a, int n) { return CSCMatrix_to_scipy(a.sparse(n)); })
        
        .def("tofermi",[](const PauliString& p) { return reverse_jordan_wigner(p); })
        .def("tomajorana",[](const PauliString& p) { return MajoranaPolynomial(pauli_to_majorana(p)); })

        .def("__eq__", [](const PauliString &a, const ff_complex& b) { return a == b; })
        .def("__eq__", [](const PauliString &a, const PauliString& b) { return a == b; })

        .def("__add__", [](const PauliString &a, ff_complex b) { return b == ff_complex(0,0) ? a : (a+PauliPolynomial(b)); })
        .def("__add__", [](const PauliString &a, const PauliString& b) { return a+b; })
        .def("__add__", [](const PauliString &a, const PauliPolynomial& b) { return a+b; })

        .def("__sub__", [](const PauliString &a, ff_complex b) { return b == ff_complex(0,0) ? a : (a+PauliPolynomial(-b)); })
        .def("__sub__", [](const PauliString &a, const PauliString& b) { return a-b; })
        .def("__sub__", [](const PauliString &a, const PauliPolynomial& b) { return a-b; })

        .def("__mul__", [](const PauliString &a, ff_complex b) { return PauliPolynomial(a*b); })
        .def("__mul__", [](const PauliString &a, const PauliString& b) { return PauliPolynomial(a*b); })
        .def("__mul__", [](const PauliString &a, const PauliPolynomial& b) { return PauliPolynomial(a*b); })

        .def("__rsub__", [](const PauliString &a, ff_complex b) { return PauliPolynomial(b == ff_complex(0,0) ? -a : (PauliPolynomial(b)-a)); })
        .def("__radd__", [](const PauliString &a, ff_complex b) { return PauliPolynomial(b == ff_complex(0,0) ? a : (PauliPolynomial(b)+a)); })
        .def("__rmul__", [](const PauliString &a, ff_complex b) { return PauliPolynomial(b*a); });


    py::class_<PauliPolynomial> pyPauliPolynomial = py::class_<PauliPolynomial>(m, "PauliPolynomial")
        .def(py::init<>()) 
        .def(py::init<const ff_complex&>())
        .def(py::init<const PauliString&>())
        .def(py::init<const PauliMonomial&>())
        .def(py::init<const std::vector<std::pair<int,char>>&>())
        .def(py::init<const std::vector<std::pair<int,char>>&, const ff_complex&>())
        .def(py::init<const PauliPolynomial&>())
        .def("degree", [](const PauliPolynomial& a, const std::optional<char>& v) { return v.has_value() ? a.degree(v.value()) : a.degree(); }, py::arg("v") = py::none())
        .def("dagger", &PauliPolynomial::dagger)
        .def("coefficient", py::overload_cast<const PauliString&>(&PauliPolynomial::coefficient, py::const_))
        .def("coefficient", [](const PauliPolynomial& a, const std::vector<std::pair<int,char>>& indices) { return a.coefficient(PauliString(indices)); })
        .def("coefficient", [](const PauliPolynomial& a, const std::string& ps) { return a.coefficient(PauliString(ps)); })
        .def("support", &PauliPolynomial::support_set)
        /*.def("sparse", [](const PauliPolynomial& a) { return CSCMatrix_to_scipy(a.sparse()); })
        .def("sparse", [](const PauliPolynomial& a, int n) { return CSCMatrix_to_scipy(a.sparse(n)); })
        .def("sparse", [](const PauliPolynomial& a, int n, int nup) { return CSCMatrix_to_scipy(a.sparse(n, nup)); })*/
        .def("overlapwithzero", &PauliPolynomial::overlapwithzero)
        .def("tofermi",[](const PauliPolynomial& p) { return reverse_jordan_wigner(p); })
        .def("tomajorana",[](const PauliPolynomial& p) { return pauli_to_majorana(p); });

    add_pauli_sparse(pyPauliString);
    add_pauli_sparse(pyPauliPolynomial);

    add_poly_basic_ops<PauliPolynomial, PauliString, PauliMonomial>(pyPauliPolynomial);

    py::class_<FermiString> pyFermiString = py::class_<FermiString>(m, "FermiString")
        .def(py::init<>())
        .def(py::init<const std::vector<std::pair<int,bool>>&>())
        .def(py::init<const std::vector<int>&, const std::vector<int>&>())
        .def("extent", &FermiString::extent)
        .def("indices", &FermiString::indices)
        .def("__hash__", &FermiString::hash)
        .def("degree", [](const FermiString& a, const std::optional<int>& v) {
            if(v.has_value()) {
                return a.degree(v.value());
            } else {
                return a.degree();
            }
        }, py::arg("v") = py::none())
        .def("permute", [](const FermiString& a, const std::vector<int>& perm) { return FermiPolynomial(a.permute(perm)); })
        .def("commutes", py::overload_cast<const FermiString&>(&FermiString::commutes, py::const_))
        .def("commutes", py::overload_cast<const FermiPolynomial&>(&FermiString::commutes, py::const_))
        .def("commutator", py::overload_cast<const FermiString&>(&FermiString::commutator, py::const_))
        .def("commutator", py::overload_cast<const FermiPolynomial&>(&FermiString::commutator, py::const_))
        //.def("cre", [](const FermiString& a) { return a.cre.rsupport(); })
        //.def("ann", [](const FermiString& a) { return a.ann.rsupport(); })
        .def("dagger", [](const FermiString& a) { return FermiPolynomial(a.dagger()); })
        // .def("sparse", [](const FermiString& a) { return CSCMatrix_to_scipy(a.sparse()); })
        // .def("sparse", [](const FermiString& a, int n) { return CSCMatrix_to_scipy(a.sparse(n)); })
        .def("topauli",[](const FermiString& p) { return jordan_wigner(p); })
        .def("tomajorana",[](const FermiString& p) { return fermi_to_majorana(p); })
        .def("__eq__", [](const FermiString &a, const ff_complex& b) { return a == b; })
        .def("__eq__", [](const FermiString &a, const FermiString& b) { return a == b; })

        .def("__neg__", [](const FermiString &a) { return FermiPolynomial(-a); })
        .def("__div__", [](const FermiString &a, const ff_complex& b) { return FermiPolynomial(a*(1.0/b)); })

        .def("__add__", [](const FermiString &a, const ff_complex& b) { return a+b; })
        .def("__add__", [](const FermiString &a, const FermiString& b) { return a+b; })

        .def("__sub__", [](const FermiString &a, const ff_complex& b) { return a-b; })
        .def("__sub__", [](const FermiString &a, const FermiString& b) { return a-b; })

        .def("__mul__", [](const FermiString &a, const ff_complex& b) { return FermiPolynomial(a*b); })
        .def("__mul__", [](const FermiString &a, const FermiString& b) { return FermiPolynomial(a*b); })
        .def("__mul__", [](const FermiString &a, const FermiPolynomial& b) { return FermiPolynomial(a*b); })
        
        .def("__rsub__", [](const FermiString &a, const ff_complex& b) { return b-a; })
        .def("__radd__", [](const FermiString &a, const ff_complex& b) { return b+a; })
        .def("__rmul__", [](const FermiString &a, const ff_complex& b) { return FermiPolynomial(b*a); })

        .def("__str__", &FermiString::to_compact_string)
        .def("__repr__", &FermiString::to_compact_string);


    py::class_<FermiPolynomial> pyFermiPolynomial = py::class_<FermiPolynomial>(m, "FermiPolynomial")
        .def(py::init<>())
        .def(py::init<const ff_complex&>())
        .def(py::init<const FermiString&>())
        .def(py::init<const FermiPolynomial&>())
        .def(py::init<const std::vector<std::pair<int,bool>>&>())
        .def(py::init<const std::vector<std::pair<int,bool>>&, const ff_complex&>())
        .def(py::init<const std::vector<int>&, const std::vector<int>&, ff_complex>())
        .def("degree", [](const FermiPolynomial& a, const std::optional<int>& v) { return v.has_value() ? a.degree(v.value()) : a.degree(); }, py::arg("v") = py::none())
        .def("dagger", &FermiPolynomial::dagger)
        .def("coefficient", py::overload_cast<const FermiString&>(&FermiPolynomial::coefficient, py::const_))
        .def("coefficient", [](const FermiPolynomial& a, const std::string& fs) { return a.coefficient(FermiString(fs)); })
        .def("coefficient", [](const FermiPolynomial& a, const std::vector<std::pair<int,bool>>& fs) { return a.coefficient(FermiString(fs)); })
        .def("support", &FermiPolynomial::support_set)
        .def("overlapwithvacuum", &FermiPolynomial::overlapwithvacuum)
        .def("topauli",[](const FermiPolynomial& p) { return jordan_wigner(p); })
        .def("tomajorana",[](const FermiPolynomial& p) { return fermi_to_majorana(p); });

    
    add_fermi_sparse(pyFermiString);
    add_fermi_sparse(pyFermiPolynomial);

    add_poly_basic_ops<FermiPolynomial, FermiString, FermiMonomial>(pyFermiPolynomial);

    py::class_<MajoranaString> pyMajoranaString = py::class_<MajoranaString>(m, "MajoranaString")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&>())
        .def("extent", &MajoranaString::extent)
        .def("degree", &MajoranaString::degree)
        .def("indices", &MajoranaString::support_set)
        .def("is_hermitian", &MajoranaString::is_hermitian)
        .def("commutes", py::overload_cast<const MajoranaString&>(&MajoranaString::commutes, py::const_))
        .def("commutes", py::overload_cast<const MajoranaPolynomial&>(&MajoranaString::commutes, py::const_))
        .def("commutator", [](const MajoranaString& a, const MajoranaString& b) { return MajoranaPolynomial(a.commutator(b)); })
        .def("commutator", py::overload_cast<const MajoranaPolynomial&>(&MajoranaString::commutator, py::const_))
        .def("permute", [](const MajoranaString& a, const std::vector<int>& perm) { return MajoranaPolynomial(a.permute(perm)); })
        .def("dagger", [](const MajoranaString& a) { return MajoranaPolynomial(a.dagger()); })
        // .def("sparse", [](const MajoranaString& a) { return CSCMatrix_to_scipy(a.sparse()); })
        .def("topauli",[](const MajoranaString& p) { return jordan_wigner(p); })
        .def("tofermi",[](const MajoranaString& p) { return majorana_to_fermi(p); })

        .def("__hash__", [](const MajoranaString &a) { return a.hash(); })

        .def("__eq__", [](const MajoranaString &a, const ff_complex& b) { return a == b; })
        .def("__eq__", [](const MajoranaString &a, const MajoranaString& b) { return a == b; })

        .def("__neg__", [](const MajoranaString &a) { return MajoranaPolynomial(-a); })
        .def("__div__", [](const MajoranaString &a, const ff_complex& b) { return MajoranaPolynomial(a*(1.0/b)); })

        .def("__add__", [](const MajoranaString &a, const ff_complex& b) { return a+b; })
        .def("__add__", [](const MajoranaString &a, const MajoranaString& b) { return a+b; })
        .def("__add__", [](const MajoranaString &a, const MajoranaPolynomial& b) { return a+b; })

        .def("__sub__", [](const MajoranaString &a, const ff_complex& b) { return a-b; })
        .def("__sub__", [](const MajoranaString &a, const MajoranaString& b) { return a-b; })
        .def("__sub__", [](const MajoranaString &a, const MajoranaPolynomial& b) { return a-b; })

        .def("__mul__", [](const MajoranaString &a, const ff_complex& b) { return MajoranaPolynomial(a*b); })
        .def("__mul__", [](const MajoranaString &a, const MajoranaString& b) { return MajoranaPolynomial(a*b); })
        .def("__mul__", [](const MajoranaString &a, const MajoranaPolynomial& b) { return MajoranaPolynomial(a*b); })
        
        .def("__rsub__", [](const MajoranaString &a, const ff_complex& b) { return b-a; })
        .def("__radd__", [](const MajoranaString &a, const ff_complex& b) { return b+a; })
        .def("__rmul__", [](const MajoranaString &a, const ff_complex& b) { return MajoranaPolynomial(b*a); })

        .def("__str__", &MajoranaString::to_compact_string)
        .def("__repr__", &MajoranaString::to_compact_string);

    py::class_<MajoranaPolynomial> pyMajoranaPolynomial = py::class_<MajoranaPolynomial>(m, "MajoranaPolynomial")
        .def(py::init<>())
        .def(py::init<const ff_complex&>())
        .def(py::init<const MajoranaString&>())
        .def(py::init<const MajoranaPolynomial&>())
        .def(py::init<const std::vector<int>&>())
        .def(py::init<const std::vector<int>&, ff_complex>())
        .def("degree", &MajoranaPolynomial::degree)
        .def("dagger", &MajoranaPolynomial::dagger)
        .def("coefficient", py::overload_cast<const MajoranaString&>(&MajoranaPolynomial::coefficient, py::const_))
        .def("coefficient", [](const MajoranaPolynomial& a, const std::string& ms) { return a.coefficient(MajoranaString(ms)); })
        .def("coefficient", [](const MajoranaPolynomial& a, const std::vector<int>& ms) { return a.coefficient(MajoranaString(ms)); })
        .def("support", &MajoranaPolynomial::support_set)
        // .def("sparse", [](const MajoranaPolynomial& a) { return CSCMatrix_to_scipy(a.sparse()); })
        .def("topauli",[](const MajoranaPolynomial& a) { return jordan_wigner(a); })
        .def("tofermi",[](const MajoranaPolynomial& a) { return majorana_to_fermi(a); });

    add_poly_basic_ops<MajoranaPolynomial, MajoranaString, MajoranaMonomial>(pyMajoranaPolynomial);

    add_majorana_sparse(pyMajoranaString);
    add_majorana_sparse(pyMajoranaPolynomial);

    // UTILS

    add_gen(m);
    add_transforms(m);
    add_states(m);
    
    add_pauli_gates(m);
    add_pauli_propagation(m);

    add_majorana_propagation(m);
    
}

}