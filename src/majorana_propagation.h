/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "majorana_gates.h"

namespace fastfermion {

namespace majorana_gates {

// For now, a MajoranaCircuit is simply a sequence of Majorana Rotations
using MajoranaCircuit = std::vector<MROT>;

template<class FilterPred>
// pred is any object such that pred(a) is a boolean where a is a MajoranaString
MajoranaPolynomial _propagate(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs, FilterPred pred, const ff_float& mincoeff=0) {
    MajoranaPolynomial ret(obs);
    int circuit_size = circuit.size();
    for(int i=circuit_size-1; i>=0; i--) {
        circuit[i].apply_inplace(ret, pred);
        if(mincoeff > 0) {
            std::erase_if(ret.terms, [&mincoeff](const auto& term) { return std::abs(term.second) <= mincoeff; });
        }
    }
    return ret;
}


MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs, const ff_float& mincoeff=0) {
    return _propagate(circuit, obs, [](const MajoranaString& a) { return true; }, mincoeff);
}

MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs, const int& maxdegree, const ff_float& mincoeff=0) {
    return _propagate(circuit, obs, [&maxdegree](const MajoranaString& a) { return a.degree() <= maxdegree; }, mincoeff);
}

MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaString& obs, const ff_float& mincoeff=0) {
    return propagate(circuit, MajoranaPolynomial(obs));
}

MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaString& obs, const int& maxdegree, const ff_float& mincoeff=0) {
    return propagate(circuit, MajoranaPolynomial(obs), maxdegree);
}

}

}