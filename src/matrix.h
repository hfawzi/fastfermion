/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

namespace fastfermion {

template<class T=double>
struct Matrix {
    int shape[2];
    std::vector<T> data;
    Matrix() {
        shape[0] = 0;
        shape[1] = 0;
    }
    Matrix(int m_arg, int n_arg) {
        shape[0] = m_arg;
        shape[1] = n_arg;
        data = std::vector<T>(shape[0]*shape[1], 0);
    }
    constexpr T& operator()(int i, int j) { return data[shape[1]*i+j]; }
};

template<class T=double>
struct CSCMatrix {
    int shape[2];
    std::vector<std::size_t> indptr; // of size equal to shape[1]+1
    std::vector<std::size_t> indices; // row indices
    std::vector<T> data;
    CSCMatrix(int m, int n) {
        shape[0] = m;
        shape[1] = n;
        // Fill indptr
        indptr = std::vector<std::size_t>(n+1, 0);
    }
    CSCMatrix() : CSCMatrix(0,0) { }
    Matrix<T> todense() const {
        // assertm(indptr.size() == shape[1]+1, "CSCMatrix is not well-formed");
        Matrix<T> A(shape[0],shape[1]);
        for(int j=0; j<shape[1]; j++) {
            for(int k=indptr[j]; k<indptr[j+1]; k++) {
                A(indices[k],j) += data[k];
            }
        }
        return A;
    }
};

}