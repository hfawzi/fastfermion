/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "pauli_algebra.h"
#include "matrix.h" // for CSCMatrix
#include "bits_utils.h" // for next_combination
#include "subspaces.h"

#include <algorithm> // for std::sort
#include <bit> // for std::popcount

namespace fastfermion {

template<class Subspace>
CSCMatrix<ff_complex> _paulipoly_sparse(const PauliPolynomial& poly, int n, const Subspace& subspace) {

    // Returns sparse matrix representation of size 2^n x 2^n of PauliPolynomial
    // Subspace is any type that has the following:
    // - dim attribute = dimension of subspace
    // - start = first element of the basis
    // - next = gives next element in the basis
    // - index = index of a basis element \in {0,...,dim-1}
    // Subspaces are defined in subspaces.h

    struct PauliStringExtra {
        std::uint64_t xory_flipped;
        std::uint64_t yorz_flipped;
        ff_complex coefficient;
    };

    int nops = poly.terms.size();
    int subspace_dim = subspace.dim;

    // Precompute some information about operators
    std::vector<PauliStringExtra> ops_info(nops);
    int i = 0;
    for(const auto& [ps, coefficient] : poly.terms) {
        // Flipping bits so that resulting matrix matches the convention of "kron"
        ops_info[i].xory_flipped = fliplr_bits(ps.xory, n).to_ullong();
        ops_info[i].yorz_flipped = fliplr_bits(ps.yorz, n).to_ullong();
        ops_info[i].coefficient = scalar_utils::Ipow(ps.degree_y()) * coefficient;
        i++;
    }

    // Sort ops_info by xory and keep track of operators
    // with the same value of xory
    // This is because sparse matrices of PauliStrings
    // with the same value of xory have the same support.
    // Having the PauliStrings with the same xory next to
    // each other allows us to aggregate values.
    std::sort(ops_info.begin(), ops_info.end(), [](const PauliStringExtra& a, const PauliStringExtra& b) { return a.xory_flipped < b.xory_flipped; });

    // Count the number K of PauliStrings with distinct value
    // of xory. Upper bound on nnz of the matrix to compute is 
    // K*subspace_dim
    int K = 0;
    for(i=1; i<nops; i++) {
        if(ops_info[i].xory_flipped != ops_info[i-1].xory_flipped) {
            K++;
        }
    }
    int nnzEstimate = K*subspace_dim;

    // Construct matrix
    CSCMatrix<ff_complex> A(subspace_dim,subspace_dim);
    A.indptr.resize(A.shape[1]+1);
    A.indptr[0] = 0;

    // Reserve memory for A
    A.data.reserve(nnzEstimate);
    A.indices.reserve(nnzEstimate);

    std::uint64_t a,b,xory_flipped;
    int sgn_coeff;
    int colNnz = 0;
    int j;

    for(j=0, b=subspace.start; j<subspace_dim; j++, b=subspace.next(b)) {
        // X = [0 1]        Y = [0 -1i]           Z = [1  0]
        //     [1 0]            [1i  0]               [0 -1]
        // Compute index of row a where A_{a,b} \neq 0
        // This is precisely a = b + xory
        
        // Counter to keep track of the number of nonzero
        // entries in the this column
        // A priori, this is just equal to K (=number of
        // PauliStrings with distinct value of xory).
        // But there can be cancellations, so it could be
        // smaller than K.
        colNnz = 0;
        // Go through the operators, grouping together the ones
        // with the same value of xory
        i = 0;
        while(i < nops) {
            xory_flipped = ops_info[i].xory_flipped;
            // Index of row
            a = b ^ xory_flipped;
            // Keep looking for other PauliStrings with
            // the same value of xory and aggregate coefficient
            // Iterator i will be incremented
            ff_complex coeff = 0;
            do {
                sgn_coeff = scalar_utils::m1pow(std::popcount(b & ops_info[i].yorz_flipped));
                coeff += sgn_coeff * ops_info[i].coefficient;
                i++;
            } while (i < nops && ops_info[i].xory_flipped == xory_flipped);
            // NOTE: This if statement is time-consuming.
            // Removing it and adding spurious entries would be faster
            if(coeff != ff_complex(0,0)) {
                try {
                    A.indices.push_back(subspace.index(a));
                } catch(const std::out_of_range& ex) {
                    throw_error("error: subspace is not invariant by the given operator");
                }
                //A.indices.push_back(a);
                A.data.push_back(coeff);
                colNnz++;
            }
        }
        A.indptr[j+1] = A.indptr[j]+colNnz;
    }
    return A;
}

CSCMatrix<ff_complex> PauliPolynomial::sparse(int n) const {
    return _paulipoly_sparse(*this, n, FullHilbertSpace(n));
}

CSCMatrix<ff_complex> PauliPolynomial::sparse(int n, int nup) const {
    return _paulipoly_sparse(*this, n, FixedCountSubspace(n,nup));
}

CSCMatrix<ff_complex> PauliString::sparse(int n) const { return PauliPolynomial(*this).sparse(n); }
CSCMatrix<ff_complex> PauliString::sparse(int n, int nup) const { return PauliPolynomial(*this).sparse(n, nup); }

CSCMatrix<ff_complex> PauliMonomial::sparse(int n) const { return PauliPolynomial(*this).sparse(n); }
CSCMatrix<ff_complex> PauliMonomial::sparse(int n, int nup) const { return PauliPolynomial(*this).sparse(n, nup); }

}