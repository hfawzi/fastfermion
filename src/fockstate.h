/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "subspaces.h"
#include "fermi_algebra.h"
#include "majorana_algebra.h"
#include "bits_utils.h"

namespace fastfermion {

struct FockState {

    ff_ulong b;

    FockState() : b(0) { } // vacuum
    FockState(const std::uint64_t& b) : b(b) { }
    FockState(const ff_ulong& b) : b(b) { }
    FockState(const std::vector<int>& occ) : b(0) {
        for(const int& i: occ) b.set(i);
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

    std::vector<ff_float> vec(int n, int nocc) const {
        // Returns a vector of length 2^n representing the state
        if(b.popcount() != nocc) {
            throw_error("Error: fock state does not live in the supplied subspace");
        }
        int last = b.rbegin();
        if(last >= n) {
            throw_error("Error: n must be at least " << last+1);
        }
        if(n > WORD_LENGTH) {
            throw_error("Error: n is too large");
        }
        FixedCountSubspace hilbertspace(n,nocc);
        std::vector<ff_float> ret(hilbertspace.dim);
        // Flip the bits to follow the convention adopted in fermi_sparse
        ret[hilbertspace.index(fliplr_bits(b, n).to_ullong())] = 1;
        return ret;
    }

    std::vector<int> occ() {
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

    // Fermi operators
    ff_complex operator()(const FermiString& a) const {
        if(a.cre == a.ann && (a.ann & (~b)) == 0) {
            int x = a.ann.popcount();
            return scalar_utils::m1pow(x*(x-1)/2);
        }
        return 0;
    }

    ff_complex operator()(const FermiMonomial& a) const {
        return a.coeff * (*this)(a.s);
    }

    ff_complex operator()(const FermiPolynomial& p) const {
        ff_complex ret = 0;
        for(const auto& [a,v] : p.terms) {
            ret += v*(*this)(a);
        }
        return ret;
    }

    // Majorana operators
    ff_complex operator()(const MajoranaString& a) const {
        ff_dbl_ulong a_even = a.alpha & ff_dbl_ulong::even_mask();
        ff_dbl_ulong a_odd = (a.alpha & ff_dbl_ulong::odd_mask()) >> 1;
        if(a_even == a_odd) {
            // Fully paired
            ff_ulong jset = deinterleave(a_even);
            return scalar_utils::Ipow(jset.popcount() + 2*(b & jset).popcount());
        }
        return 0;
    }

    ff_complex operator()(const MajoranaMonomial& a) const {
        return a.coeff * (*this)(a.s);
    }

    ff_complex operator()(const MajoranaPolynomial& p) const {
        ff_complex ret = 0;
        for(const auto& [a,v] : p.terms) {
            ret += v*(*this)(a);
        }
        return ret;
    }

};

}