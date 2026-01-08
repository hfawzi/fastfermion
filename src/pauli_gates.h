/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "pauli_algebra.h"
#include "bits_utils.h"

namespace fastfermion {

namespace pauli_gates {

void H_impl(const int& i, PauliString& a, ff_complex& coeff) {
    // X -> Z
    // Z -> X
    // Y -> -Y
    // out.xory = in.yorz
    // out.yorz = in.xory
    // out.phase = (-1)^{in.xory & in.yorz}
    ff_ulong mask = ff_ulong::singleton(i);
    ff_ulong xory_i = a.xory & mask;
    ff_ulong yorz_i = a.yorz & mask;
    a.xory = (a.xory & (~mask)) | yorz_i;
    a.yorz = (a.yorz & (~mask)) | xory_i;
    if( ! ((xory_i & yorz_i) == 0) ) coeff *= -1;
}

void S_impl(const int& i, PauliString& a, ff_complex& coeff) {
    // X -> -Y
    // Z -> Z
    // Y -> X
    // out.xory = in.xory
    // out.yorz = in.xory ^ in.yorz (=in.xorz)
    // out.phase = (-1)^{in.xory & in.yorz}
    ff_ulong mask = ff_ulong::singleton(i);
    ff_ulong xory_i = a.xory & mask;
    ff_ulong yorz_i = a.yorz & mask;
    a.yorz = (a.yorz & (~mask)) | (xory_i ^ yorz_i);
    if( ! ((xory_i & (~yorz_i)) == 0) ) coeff *= -1;
}

void CNOT_impl(const int& i, const int& j, PauliString& a, ff_complex& coeff) {

    // On a generating set:
    // XI -> XX
    // ZI -> ZI
    // IX -> IX
    // IZ -> ZZ
    
    // Other cases:
    // IY -> ZY
    // XX -> XI
    // XY -> YZ
    // XZ -> -YY
    // YI -> YX
    // YX -> YI
    // YY -> -XZ
    // YZ -> XY
    // ZX -> ZX
    // ZY -> IY
    // ZZ -> ZI

    // out.xory[i] = in.xory[i]
    // out.xory[j] = in.xory[i] ^ in.xory[j]
    // out.yorz[i] = in.yorz[i] ^ in.yorz[j]
    // out.yorz[j] = in.yorz[j]
    // out.phase = -1 iff input is either XZ or YY
    ff_ulong mask_i = ff_ulong::singleton(i);
    ff_ulong mask_j = ff_ulong::singleton(j);
    
    ff_ulong xory_j = a.xory & mask_j;
    ff_ulong xory_i_at_j = ((a.xory & mask_i) >> i) << j;
    ff_ulong yorz_i = a.yorz & mask_i;
    ff_ulong yorz_j_at_i = ((a.yorz & mask_j) >> j) << i;
    a.xory = (a.xory & ~mask_j) | (xory_i_at_j ^ xory_j);
    a.yorz = (a.yorz & ~mask_i) | (yorz_i ^ yorz_j_at_i);

    if(!(xory_i_at_j == 0) && !(yorz_j_at_i == 0) && !((yorz_i == 0) ^ (xory_j == 0))) coeff *= -1;

}

void SWAP_impl(int i, int j, PauliString& a, ff_complex& coeff) {
    // XI -> IX
    // ZI -> IZ
    // IX -> XI
    // IZ -> ZI
    // Swap bits i and j of a.xory and a.yorz
    swap_bits_inplace(a.xory, i, j);
    swap_bits_inplace(a.yorz, i, j);
}

void CZ_impl(int i, int j, PauliString& a, ff_complex& coeff) {
    // XI -> XZ
    // ZI -> ZI
    // IX -> ZX
    // IZ -> IZ

    // Other cases:
    // XX -> YY
    // XZ -> XI
    // XY -> -YX

    // YI -> YZ
    // YX -> -XY
    // YY -> XX
    // YZ -> YI

    // ZX -> IX
    // ZY -> IY
    // ZZ -> ZZ
    
    // IY -> ZY

    // xory doesn't change
    // out.xory[i] = in.xory[i]
    // out.xory[j] = in.xory[j]

    // out.yorz[i] = in.xory[j] ^ in.yorz[i]
    // out.yorz[j] = in.xory[i] ^ in.yorz[j]

    // coeff = (-1)^{xory[i] & xory[j] & (yorz[i] ^ yorz[j])}

    ff_ulong mask_i = ff_ulong::singleton(i);
    ff_ulong mask_j = ff_ulong::singleton(j);

    if( a.xory.at(i) && a.xory.at(j) && (a.yorz.at(i) ^ a.yorz.at(j)) ) coeff *= -1;

    a.yorz ^= ( ((a.xory & mask_j) >> j) << i ) | ( ((a.xory & mask_i) >> i) << j );

}


struct H {
    int i;
    void apply_inplace(PauliString& a, ff_complex& coeff) const {
        H_impl(i,a,coeff);
    }
    std::pair<PauliString, ff_complex> operator()(const PauliString& a) const {
        PauliString res(a);
        ff_complex c = 1;
        apply_inplace(res,c);
        return std::make_pair(res,c);
    }
    PauliPolynomial operator()(const PauliPolynomial& p) const {
        PauliPolynomial res;
        for(const auto& [x,v] : p.terms) {
            const auto [y,w] = (*this)(x);
            res.terms[y] += w*v;
        }
        return res;
    }

    PauliPolynomial aspoly() const {
        // Return a representation of the unitary as a PauliPolynomial
        // H = (X+Z)/sqrt(2)
        PauliPolynomial poly;
        poly.terms[PauliString(std::vector<int>{i},std::vector<char>{'X'})] = 1/std::sqrt(2);
        poly.terms[PauliString(std::vector<int>{i},std::vector<char>{'Z'})] = 1/std::sqrt(2);
        return poly;
    }
    std::string to_string() const {
        return "H(" + std::to_string(i) + ")";
    }
};

struct S {
    int i;
    void apply_inplace(PauliString& a, ff_complex& coeff) const {
        S_impl(i,a,coeff);
    }
    std::pair<PauliString, ff_complex> operator()(const PauliString& a) const {
        PauliString res(a);
        ff_complex c = 1;
        apply_inplace(res,c);
        return std::make_pair(res,c);
    }
    PauliPolynomial operator()(const PauliPolynomial& p) const {
        PauliPolynomial res;
        for(const auto& [x,v] : p.terms) {
            const auto [y,w] = (*this)(x);
            res.terms[y] += w*v;
        }
        return res;
    }
    PauliPolynomial aspoly() const {
        // Return a representation of the unitary as a PauliPolynomial
        // S = (1+i)/2 + (1-i)/2 Z
        PauliPolynomial poly;
        poly.terms[PauliString()] = ff_complex(.5,.5);
        poly.terms[PauliString(std::vector<int>{i},std::vector<char>{'Z'})] = ff_complex(.5,-.5);
        return poly;
    }
    std::string to_string() const {
        return "S(" + std::to_string(i) + ")";
    }
};

struct CNOT {
    int i;
    int j;
    void apply_inplace(PauliString& a, ff_complex& coeff) const {
        CNOT_impl(i,j,a,coeff);
    }
    std::pair<PauliString, ff_complex> operator()(const PauliString& a) const {
        PauliString res(a);
        ff_complex c = 1;
        apply_inplace(res,c);
        return std::make_pair(res,c);
    }
    PauliPolynomial operator()(const PauliPolynomial& p) const {
        PauliPolynomial res;
        for(const auto& [x,v] : p.terms) {
            const auto [y,w] = (*this)(x);
            res.terms[y] += w*v;
        }
        return res;
    }
    PauliPolynomial aspoly() const {
        // Return a representation of the unitary as a PauliPolynomial
        // CNOT = I/2 + Zi/2 + Xj/2 - Zi Xj/2
        PauliPolynomial poly;
        poly.terms[PauliString()] = 0.5;
        poly.terms[PauliString(std::vector<int>{i},std::vector<char>{'Z'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{j},std::vector<char>{'X'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{i,j},std::vector<char>{'Z','X'})] = -0.5;
        return poly;
    }
    std::string to_string() const {
        return "CNOT(" + std::to_string(i) + "," + std::to_string(j) + ")";
    }
};

struct SWAP {
    int i;
    int j;
    void apply_inplace(PauliString& a, ff_complex& coeff) const {
        SWAP_impl(i,j,a,coeff);
    }
    std::pair<PauliString, ff_complex> operator()(const PauliString& a) const {
        PauliString res(a);
        ff_complex c = 1;
        apply_inplace(res,c);
        return std::make_pair(res,c);
    }
    PauliPolynomial operator()(const PauliPolynomial& p) const {
        PauliPolynomial res;
        for(const auto& [x,v] : p.terms) {
            const auto [y,w] = (*this)(x);
            res.terms[y] += w*v;
        }
        return res;
    }
    PauliPolynomial aspoly() const {
        // Return a representation of the unitary as a PauliPolynomial
        // SWAP = (1 + Xi Xj + Yi Yj + Zi Zj)/2
        PauliPolynomial poly;
        poly.terms[PauliString()] = 0.5;
        poly.terms[PauliString(std::vector<int>{i,j},std::vector<char>{'X','X'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{i,j},std::vector<char>{'Y','Y'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{i,j},std::vector<char>{'Z','Z'})] = 0.5;
        return poly;
    }
    std::string to_string() const {
        return "SWAP(" + std::to_string(i) + "," + std::to_string(j) + ")";
    }
};

struct CZ {
    int i;
    int j;
    void apply_inplace(PauliString& a, ff_complex& coeff) const {
        CZ_impl(i,j,a,coeff);
    }
    std::pair<PauliString, ff_complex> operator()(const PauliString& a) const {
        PauliString res(a);
        ff_complex c = 1;
        apply_inplace(res,c);
        return std::make_pair(res,c);
    }
    PauliPolynomial operator()(const PauliPolynomial& p) const {
        PauliPolynomial res;
        for(const auto& [x,v] : p.terms) {
            const auto [y,w] = (*this)(x);
            res.terms[y] += w*v;
        }
        return res;
    }
    PauliPolynomial aspoly() const {
        // Return a representation of the unitary as a PauliPolynomial
        // CZ = I/2 + Zi/2 + Zj/2 - Zi Zj/2
        PauliPolynomial poly;
        poly.terms[PauliString()] = 0.5;
        poly.terms[PauliString(std::vector<int>{i},std::vector<char>{'Z'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{j},std::vector<char>{'Z'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{i,j},std::vector<char>{'Z','Z'})] = -0.5;
        return poly;
    }
    std::string to_string() const {
        return "CZ(" + std::to_string(i) + "," + std::to_string(j) + ")";
    }
};

// PauliRotation
struct ROT {
    // Represents a gate U = e^{-i theta/2 P} where P is a PauliString
    PauliString ps;
    ff_float theta;
    ROT(const PauliString& ps, const ff_float& theta) :
        ps(ps), theta(theta) { }
    ROT(const std::string& str, const ff_float& theta) :
        ps(PauliString(str)), theta(theta) { }
    ROT(const std::string& str, const std::vector<int>& loc, const ff_float& theta) :
        ps(PauliString(loc,std::vector<char>(str.begin(), str.end()))),
        theta(theta) { }
    std::string to_string() const {
        return "ROT(" + ps.to_compact_string() + "," + std::to_string(theta) + ")";
    }
    PauliPolynomial aspoly() const {
        // e^{-i theta/2 P} = cos(theta/2) - 1i*sin(theta/2)*P
        PauliPolynomial poly;
        poly.terms[PauliString()] = std::cos(theta/2);
        poly.terms[ps] += ff_complex(0,-std::sin(theta/2));
        return poly;
    }
    
    PauliPolynomial operator()(const PauliPolynomial& o) const {
        // Apply U^{\dagger} o U
        PauliPolynomial ret(o);
        std::vector<std::pair<PauliString, ff_complex>> o_new;
        o_new.reserve(o.terms.size());
        const ff_float costheta = cos(theta);
        const ff_complex isintheta = ff_complex(0,sin(theta));
        for(auto& [x,v] : ret.terms) {
            if (!x.commutes(ps)) {
                PauliMonomial px = ps*x;
                o_new.emplace_back(px.pauli_string(), v*isintheta*px.coefficient());
                v *= costheta;
            }
        }
        for (const auto& [x,v] : o_new) {
            ret.terms[x] += v;
        }
        return ret;
    }

    PauliPolynomial operator()(const PauliString& o) const {
        // Apply U^{\dagger} o U
        return (*this)(PauliPolynomial(o));
    }
    
};

}

}