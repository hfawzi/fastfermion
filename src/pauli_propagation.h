/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#include "pauli_gates.h"

#include <variant>
#include <functional>

namespace fastfermion {

namespace pauli_gates {

// A CliffordGate is either a H, or S, CNOT, SWAP, CZ
using CliffordGate = std::variant<H,S,CNOT,SWAP,CZ>;

// A Clifford circuit is a sequence of Clifford gates
using CliffordCircuit = std::vector<CliffordGate>;

// A gate is either a CliffordGate or a Pauli Rotation
using Gate = std::variant<CliffordGate,ROT>;

// A circuit is a sequence of gates
using Circuit = std::vector<Gate>;

std::pair<PauliString, ff_complex> propagate_clifford(const CliffordCircuit& circuit, const PauliString& a) {
    ff_complex coeff = 1;
    PauliString res = a;
    for(int i=circuit.size()-1; i>=0; i--) {
        // Call Clifford gate
        // All CliffordGate structs should have a method:
        //   apply_inplace(PauliString& a, ff_complex& coeff)
        // which applies the gate to coeff*a, and modifies a and coeff in-place
        //
        // I should call circuit[i].apply_inplace(res,coeff) however this
        // raises a compilation error because there is no method called
        // apply_inplace for std::variant<H,S,CNOT>.
        // The way to do this is to use the visit function
        // https://en.cppreference.com/w/cpp/utility/variant/visit2.html
        // See e.g.,
        // https://www.cppstories.com/2020/04/variant-virtual-polymorphism.html/
        // The visit function is essentially equivalent to a switch statement
        // on number of variants (=number of possible Clifford gates)
        std::visit(
            [&res, &coeff](const auto& gate) { gate.apply_inplace(res,coeff); }
            , circuit[i]
        );
    }
    return std::make_pair(res,coeff);
}

void _apply_clifford_circuit(PauliPolynomial& poly, const Circuit& circuit, int begin, int end) {
    // begin < end
    // Explicitly form the Clifford circuit
    // This is not really needed...
    // TODO: fix this
    CliffordCircuit cc(end-begin);
    for(int j=begin; j<end; ++j) {
        try {
            cc[j-begin] = std::get<CliffordGate>(circuit[j]);
        } catch(const std::bad_variant_access& err) {
            throw_error("Internal error: circuit[begin:end] contains non-Clifford gates");
        }
    }
    // Apply the Clifford circuit to all the elements
    PauliPolynomial poly2;
    for(const auto& [x,val] : poly.terms) {
        const auto& [y,mult] = propagate_clifford(cc, x);
        poly2.terms[y] += mult*val;
    }
    // Replace poly
    poly.terms.swap(poly2.terms);
}

PauliPolynomial propagate(const Circuit& circuit, const PauliPolynomial& a, const int& maxdegree=128, const ff_float& mincoeff=0) {
    // The main Pauli propagation function
    PauliPolynomial poly(a);
    int clifford_begin;
    bool pending_clifford_operations = false;
    for(int i=circuit.size()-1; i>=0; i--) {
        // Check if gate is a Clifford or a Rotation
        if(circuit[i].index() == 0) {
            // Clifford gate
            if(!pending_clifford_operations) {
                clifford_begin = i;
                pending_clifford_operations = true;
            }
        } else if (circuit[i].index() == 1) {
            // We need to check first if there are pending Clifford
            // operations we accumulated
            if(pending_clifford_operations) {
                _apply_clifford_circuit(poly, circuit, i+1, clifford_begin+1);
                pending_clifford_operations = false;
            }

            // Apply Pauli rotation
            const ROT& gate = std::get<ROT>(circuit[i]);
            //
            // We use the fact that
            //   ROT_{ps,theta}(x) = x                                  if ps and x commute
            //                     = cos(theta)*x + i*sin(theta)*ps*x   else
            //
            // Here ps is the PauliString
            //
            
            // o_new will hold all the new terms that will be added to poly
            // which are of the form I*sin(theta)*gate.ps*x
            // where x ranges over all terms in poly that do not commute with gate.ps
            std::vector<std::pair<PauliString, ff_complex>> o_new;

            // This is an over-estimate. The true size of o_new is the number of terms
            // in poly that don't commute with gate.ps
            o_new.reserve(poly.terms.size());

            // Some precomputation
            const PauliString& ps = gate.ps;
            const ff_float& theta = gate.theta;
            const ff_float costheta = cos(theta);
            const ff_complex isintheta = ff_complex(0,sin(theta));

            // Populate o_new
            for(auto& [x,v] : poly.terms) {
                if (!x.commutes(ps)) {
                    PauliMonomial px = ps*x;
                    if(px.degree_total() <= maxdegree) {
                        o_new.emplace_back(px.pauli_string(),v*isintheta*px.coefficient());
                    }
                    // The term x will get multiplied by cos(theta)
                    v *= costheta;
                }
            }

            // Add all the new terms
            for(const auto& [x,v] : o_new) {
                poly.terms[x] += v;
            }

            // Truncate terms
            if(mincoeff > 0) {
                std::erase_if(poly.terms, [&mincoeff](const auto& term) { return std::abs(term.second) <= mincoeff; });
            }
        }
    }
    if(pending_clifford_operations) {
        _apply_clifford_circuit(poly, circuit, 0, clifford_begin+1);
        pending_clifford_operations = false;
    }
    return poly;
}


PauliPolynomial propagate(const Circuit& circuit, const PauliString& a, const int maxdegree=128) {
    return propagate(circuit, PauliPolynomial(a), maxdegree);
}

}

}