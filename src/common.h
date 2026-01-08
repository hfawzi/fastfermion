/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include <complex>
#include <cmath>

#include "error.h"
#include "bits.h"
#include "bits_utils.h"

namespace fastfermion {

#define SYS_NUM_ULONG 2 // Number of supported qubits will be 64*SYS_NUM_ULONG

#define MIN(a,b) ((a<b) ? (a) : (b))
#define MAX(a,b) ((a>b) ? (a) : (b))

// -------------------------------------------------------------------------------

// Types used

using ff_float = double;
using ff_complex = std::complex<double>;
using ff_ulong = BitSet<SYS_NUM_ULONG>;

// -------------------------------------------------------------------------------

// Printing config

struct {
    char fermi_symbol = 'f';
    char majorana_symbol = 'm';
    char dagger_symbol = '^';
    char identity_symbol = 'I'; // What to display for empty strings
    char imaginary_symbol = 'j';
    int max_line_length = 120;
    int max_terms_to_show = 200;
} ff_config;

// -------------------------------------------------------------------------------

// Some useful arithmetic operations

ff_complex operator*(const int& a, const ff_complex& b) { return ff_complex(a*b.real(), a*b.imag()); }
ff_complex operator*(const ff_complex& b, const int& a) { return ff_complex(a*b.real(), a*b.imag()); }

namespace scalar_utils {

// (-1)^m
inline int m1pow(int m) {
    return ((m%2 == 0) ? 1 : (-1));
}

// I^m
inline ff_complex Ipow(int m) {
    if(m%2 == 0) return ff_complex(m1pow(m/2),0);
    else return ff_complex(0,m1pow((m-1)/2));
}

}

}