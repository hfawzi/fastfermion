/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "subspaces.h"
#include "pauli_algebra.h"
#include "bits_utils.h"

namespace fastfermion {

struct QubitProductState {

    ff_ulong b;
    QubitProductState() : b(0) { } // all-zeros
    QubitProductState(const std::uint64_t& b) : b(b) { }
    QubitProductState(const ff_ulong& b) : b(b) { }
    QubitProductState(const std::vector<int>& up) : b(0) {
        for(const int& i: up) b.set(i);
    }

    std::vector<ff_float> vec(int n) const {
        // Returns a vector of length 2^n representing the state
        // Hilbert space = full Hilber space of dimension 2^n
        int last = b.rbegin();
        if(last >= n) {
            throw_error("Error: n must be at least " << last+1);
        }
        if(n > WORD_LENGTH) {
            throw_error("Error: n is too large");
        }
        FullHilbertSpace hilbertspace(n);
        std::vector<ff_float> ret(hilbertspace.dim);
        // Flip the bits to follow the convention adopted in fermi_sparse
        ret[hilbertspace.index(fliplr_bits(b, n).to_ullong())] = 1;
        return ret;
    }

    std::vector<ff_float> vec(int n, int nup) const {
        // Returns a vector of length 2^n representing the state
        if(b.popcount() != nup) {
            throw_error("Error: qubit state does not live in the supplied subspace");
        }
        int last = b.rbegin();
        if(last >= n) {
            throw_error("Error: n must be at least " << last+1);
        }
        if(n > WORD_LENGTH) {
            throw_error("Error: n is too large");
        }
        FixedCountSubspace hilbertspace(n,nup);
        std::vector<ff_float> ret(hilbertspace.dim);
        ret[hilbertspace.index(fliplr_bits(b, n).to_ullong())] = 1;
        return ret;
    }

    std::vector<int> up() {
        return b.support();
    }

    std::string to_string(int n) {
        std::string ret(n,'0');
        for(const int& i : b.support()) {
            ret[i] = '1';
        }
        return ("|" + ret + ">");
    }

    // Expectation values

    // Pauli operators
    ff_complex operator()(const PauliString& a) const {
        if(a.xory == 0) {
            return scalar_utils::m1pow((a.yorz & b).popcount());
        }
        return 0;
    }

    ff_complex operator()(const PauliMonomial& a) const {
        return a.coeff * (*this)(a.s);
    }

    ff_complex operator()(const PauliPolynomial& p) const {
        ff_complex ret = 0;
        for(const auto& [a,v] : p.terms) {
            ret += v*(*this)(a);
        }
        return ret;
    }

};

}