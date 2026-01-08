/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "fermi_algebra.h"
#include "matrix.h" // for CSCMatrix
#include "subspaces.h"

#include <algorithm> // for std::sort
#include <bit> // for std::popcount

namespace fastfermion {

int merge_parity(const std::uint64_t& x, const std::uint64_t& y) {
    return merge_parity(BitSet<1>(x), BitSet<1>(y));
}

template<class Subspace>
CSCMatrix<ff_complex> _fermipoly_sparse(const FermiPolynomial& poly, int n, const Subspace& subspace) {

    // Returns a matrix of size dim x dim where dim = subspace.dim
    // If subspace = FullHilbertSpace, then dim = 2^n
    // Each row/column is indexed by a bitstring of length n. This bit string is read from *left to right*
    // i.e., the leftmost bit corresponds to whether mode *0* occupied, etc.
    //
    // Example: if we have n=2 modes then columns are arranged as follows:
    // column 1 -> |0,0> : modes 0 and 1 empty
    // column 2 -> |0,1> : mode 0 empty, mode 1 occupied
    // column 3 -> |1,0> : mode 0 occupied, mode 1 empty
    // column 4 -> |1,1> : modes 0 and 1 occupied
    //
    // This is the convention used in OpenFermion.
    // Not sure if this is best. (It would make more sense to increment mode 0 first, ...?)

    struct FermiStringExtra {
        std::uint64_t u;
        std::uint64_t s;
        std::uint64_t us;
        ff_complex coefficient;
    };

    int nops = poly.terms.size();
    int subspace_dim = subspace.dim;

    std::vector<FermiStringExtra> ops_info(nops);

    int i = 0;
    for(const auto& [fs, coefficient] : poly.terms) {
        ops_info[i].u = fliplr_bits(fs.cre, n).to_ullong();
        ops_info[i].s = fliplr_bits(fs.ann, n).to_ullong();
        ops_info[i].us = ops_info[i].u ^ ops_info[i].s;
        ops_info[i].coefficient = coefficient;
        i++;
    }

    // Sort ops_info by the value of cre+ann
    // This is because the sparse matrices of FermiStrings
    // can have overlapping supports only if they have the same
    // value of cre+ann
    // Having the FermiStrings with the same cre+ann next to
    // each other allows us to aggregate values directly
    std::sort(ops_info.begin(), ops_info.end(), [](const FermiStringExtra& a, const FermiStringExtra& b) { return a.us < b.us; });

    // Precompute number of nonzero entries
    int K = 0;
    for(i=1; i<nops; i++) {
        if(ops_info[i].us != ops_info[i-1].us) {
            K++;
        }
    }
    // Upper bound on the number of nonzeros
    // TODO: Implement a better upper bound
    int nnzBound = subspace_dim*K;

    CSCMatrix<ff_complex> A( subspace_dim, subspace_dim );
    A.indptr.resize(A.shape[1]+1);
    A.indptr[0] = 0;

    // Reserve space for indices and data
    A.indices.reserve(nnzBound);
    A.data.reserve(nnzBound);

    // Simple implementation calling merge_parity 2^{2*nnz} times
    std::uint64_t a, b, u, s, us;
    int sgn_coeff;
    int colNnz;
    int j;

    // Loop over all bit strings of length n
    for(j=0, b=subspace.start; j<subspace_dim; j++, b=subspace.next(b)) {
        colNnz = 0;
        i = 0;
        while(i < nops) {
            us = ops_info[i].us;
            // Index of row
            a = b ^ us;
            // Keep looking for other FermiStrings with
            // the same value of us = cre ^ ann, and
            // aggregate coefficient
            // Iterator i will be incremented
            ff_complex coeff = 0;
            do {
                u = ops_info[i].u;
                s = ops_info[i].s;
                if( ( ((~b) & s) == 0 ) && (b & u & (~s)) == 0 ) {
                    sgn_coeff = scalar_utils::m1pow(merge_parity(ops_info[i].s,b^ops_info[i].s) + merge_parity(ops_info[i].u,b^us));
                    // sgn_coeff = merge_parity(b^ops_info[i].s,ops_info[i].s,true)*merge_parity(b^us,ops_info[i].u,true);
                    coeff += sgn_coeff * ops_info[i].coefficient;
                }
                i++;
            } while (i < nops && ops_info[i].us == us);
            if(coeff != ff_complex(0,0)) {
                try {
                    A.indices.push_back(subspace.index(a));
                } catch(const std::out_of_range& ex) {
                    throw_error("error: subspace is not invariant by the given operator");
                }
                A.data.push_back(coeff);
                colNnz++;
            }
        }
        A.indptr[j+1] = A.indptr[j] + colNnz;
    }

    return A;

}

CSCMatrix<ff_complex> FermiPolynomial::sparse(int n) const { return _fermipoly_sparse(*this, n, FullHilbertSpace(n)); }
CSCMatrix<ff_complex> FermiPolynomial::sparse(int n, int nocc) const { return _fermipoly_sparse(*this, n, FixedCountSubspace(n, nocc)); }

CSCMatrix<ff_complex> FermiString::sparse(int n) const { return FermiPolynomial(*this).sparse(n); }
CSCMatrix<ff_complex> FermiString::sparse(int n, int nocc) const { return FermiPolynomial(*this).sparse(n,nocc); }

}