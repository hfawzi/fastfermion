/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "matrix.h"
#include "majorana_algebra.h"
#include "transforms.h"

namespace fastfermion {

CSCMatrix<ff_complex> MajoranaString::sparse() const { return majorana_to_pauli(*this).sparse(); }
CSCMatrix<ff_complex> MajoranaString::sparse(int n) const { return majorana_to_pauli(*this).sparse(n); }

CSCMatrix<ff_complex> MajoranaPolynomial::sparse() const { return majorana_to_pauli(*this).sparse(); }
CSCMatrix<ff_complex> MajoranaPolynomial::sparse(int n) const { return majorana_to_pauli(*this).sparse(n); }

}