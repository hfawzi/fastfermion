/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "hashmap/unordered_dense.h"
#include "bits_utils.h" // for next_combination
#include "error.h"

// Subspaces of (C^2)^{\otimes n}
// A subspace here refers to a sequence of bitstrings (= basis of the subspace)
// Represented as a struct with the following fields:
// - dim = dimension
// - start = first bitstring in the sequence
// - next = next bitstring in the sequence
// - index = index of the bitstring in the sequence

namespace fastfermion {

using bitstring = std::uint64_t;

// The full 2^n-dimensional space
struct FullHilbertSpace {
    std::uint64_t dim;
    bitstring start;
    FullHilbertSpace(int n) : dim(1<<n), start(0) { };
    bitstring next(const bitstring& b) const { return b+1; }
    std::uint64_t index(const bitstring& b) const { return b; }
};

// TODO: Change this
std::uint64_t _nchoosek(int n, int k) {
    if(k < 0 || k > n) {
        throw_error("invalid input (" << n << " choose " << k << ")");
    }
    std::uint64_t ret = 1;
    for(int i=0; i<k; i++) {
        ret *= n-i;
        ret /= i+1;
    }
    return ret;
}

// The subspace spanned by bitstrings b \in {0,1}^n where sum b_i = nup
struct FixedCountSubspace {
    std::uint64_t dim,start;
    ankerl::unordered_dense::map<std::uint64_t, std::uint64_t> _index;
    FixedCountSubspace(int n, int nup) : dim(_nchoosek(n,nup)), start( (1ULL<<(nup))-1 ) {
        _index.reserve(dim);
        // Construct _index map
        std::uint64_t i;
        bitstring b;
        for(i=0, b=start; i<dim; i++, b=next(b)) {
            _index[b] = i;
        }
    };
    bitstring next(const bitstring& b) const { return next_combination(b); }
    std::uint64_t index(const bitstring& b) const { return _index.at(b); }
};

}