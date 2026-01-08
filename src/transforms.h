/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "pauli_algebra.h"
#include "fermi_algebra.h"
#include "majorana_algebra.h"

#include <bit>
#include <limits>
#include <cstdint>

namespace fastfermion {

// ----------------------------------------------------------------------------
// FERMI TO MAJORANA
// ----------------------------------------------------------------------------

void _fermi_to_majorana(const FermiString& a, MajoranaPolynomial& res, const ff_complex& multiplier=1, const bool& only_keep_hermitian_terms=false) {
    // Transform a FermiString a^dagger_U a_S into a MajoranaPolynomial
    // according to the rule
    //      a_j = (m_{2j} + i m_{2j+1})/2
    //      a^{dagger}_j = (m_{2j} - i m_{2j+1})/2
    // and add the result to res. MajoranaPolynomial res is updated in place:
    //      res += multiplier*majorana(a^{dagger}_U a_S)
    // If only_keep_hermitian_terms is true, only the hermitian Majorana terms
    // are added to res.

    ff_ulong supp = a.cre | a.ann; // U union S
    ff_ulong inter = a.cre & a.ann; // U intersect S
    ff_ulong symmdiff = a.cre ^ a.ann; // U triangle S = (U \ S) union (S \ U)

    int deg_cre = a.cre.popcount();
    int deg_ann = a.ann.popcount();
    int num_union = supp.popcount();
    int num_inter = inter.popcount();
    std::size_t num_terms = (1 << num_union);
    int g2exp = num_inter - deg_cre - deg_ann;

    ff_float global_coeff = 
        (g2exp >= 0 ? (1 << g2exp) : (1.0/(1<<(-g2exp)))) *  
        scalar_utils::m1pow(deg_cre*(deg_cre-1)/2 + deg_ann*(deg_ann-1)/2 + merge_parity(a.ann, a.cre));
        // * scalar_utils::m1pow(deg_ann*(deg_ann-1)/2) * merge_parity(a.ann, a.cre);

    std::vector<MajoranaString> terms(num_terms);
    std::vector<ff_complex> coeffs(num_terms);

    terms[0] = interleave(symmdiff);
    coeffs[0] = global_coeff*1; // |X| = 0 so c(x) = 1
    std::uint64_t count = 1;
    std::uint64_t z;
    bool jInU, jInS;
    ff_complex cx;
    ff_dbl_ulong Ax;
    ff_dbl_ulong evenj, oddj;

    // Loop over the bits of supp
    for(int j = supp.begin(); j != supp.end(); j = supp.next(j)) {
        jInU = a.cre.at(j);
        jInS = !jInU || a.ann.at(j); // if j is not in U, it must be in S (save checking bit position at a.ann)
        evenj = ff_dbl_ulong::singleton(2*j);
        oddj = ff_dbl_ulong::singleton(2*j+1);
        for(z=count; z<2*count; z++) {
            //int cardX = std::popcount(z);
            // Complex multiplications are expensive, can save time here
            cx = coeffs[z-count] * scalar_utils::Ipow(1+2*(jInU && !jInS));
            Ax = (jInU && jInS) ? terms[z-count].alpha | evenj | oddj
                    : (terms[z-count].alpha | oddj) & (~evenj);
            terms[z] = MajoranaString(Ax);
            coeffs[z] = cx;
        }
        count *= 2;
    }

    for(z=0; z<num_terms; z++) {
        bool hermitian_term = MajoranaMonomial(terms[z],coeffs[z]).dagger() == MajoranaMonomial(terms[z],coeffs[z]);
        if(!only_keep_hermitian_terms || (only_keep_hermitian_terms && hermitian_term)) {
            res.terms[terms[z]] += multiplier*coeffs[z];
        }
    }

}


MajoranaPolynomial fermi_to_majorana(const FermiString& a) {
    MajoranaPolynomial res;
    _fermi_to_majorana(a,res);
    return res.compress(0);
}

MajoranaPolynomial fermi_to_majorana(const FermiPolynomial& p) {
    MajoranaPolynomial res;
    for(const auto& [a,v] : p.terms) {
        // Optimization:
        // Check if p has the Hermitian conjugate of a with the right
        // coefficient. If so, we only need to keep the Hermitian terms
        // when expressing as Majorana
        FermiMonomial a_dagger = a.dagger();
        bool has_hermitian_term = false;
        try {
            if(p.terms.at(a_dagger.fermi_string()) == v * a_dagger.coefficient()) {
                has_hermitian_term = true;
            }
        } catch(const std::out_of_range& e) {
            has_hermitian_term = false;
        }
        _fermi_to_majorana(a,res,v,has_hermitian_term);
    }
    return res.compress(0);
}


// ----------------------------------------------------------------------------
// MAJORANA TO PAULI and FERMI TO PAULI
// ----------------------------------------------------------------------------

PauliMonomial majorana_to_pauli(const MajoranaString& a) {

    PauliString res(0,0);
    int jpow = 0;

    for(int i = a.alpha.begin(); i != a.alpha.end(); i = a.alpha.next(i)) {
        int j = i/2;
        int r = i%2;
        pauli_string_multiply(
            res, jpow,
            // Z_0 ... Z_{j-1} X_j if r = 0
            // Z_0 ... Z_{j-1} Y_j if r = 1
            PauliString(
                ff_ulong::singleton(j),
                ff_ulong::range(j,r == 1)
            )
        );
    }

    return PauliMonomial(res,scalar_utils::Ipow(jpow));

}

PauliPolynomial jordan_wigner(const MajoranaString& a) {
    return PauliPolynomial(majorana_to_pauli(a));
}

PauliPolynomial jordan_wigner(const MajoranaPolynomial& p) {
    PauliPolynomial res;
    PauliMonomial b;
    for(const auto& [a,v] : p.terms) {
        b = majorana_to_pauli(a);
        res.terms[b.pauli_string()] += v*b.coefficient();
    }
    return res.compress(0);
}

PauliPolynomial majorana_to_pauli(const MajoranaPolynomial& p) {
    return jordan_wigner(p);
}

PauliPolynomial jordan_wigner(const FermiString& a) {
    return jordan_wigner(fermi_to_majorana(a));
}

PauliPolynomial jordan_wigner(const FermiPolynomial& p) {
    return jordan_wigner(fermi_to_majorana(p));
}

// ----------------------------------------------------------------------------
// MAJORANA TO FERMI
// ----------------------------------------------------------------------------

void _majorana_to_fermi(const MajoranaString& a, FermiPolynomial& res, const ff_complex& multiplier = 1) {

    // Converts a MajoranaString into a FermiPolynomial using the identification:
    //   m_{2j} = a_j^{dagger} + a_j
    //   m_{2j+1} = i*(a_j^{dagger} - a_j)
    // Result is added to FermiPolynomial res, with given multiplier

    ff_dbl_ulong even_mask = ff_dbl_ulong::even_mask();
    ff_dbl_ulong odd_mask = ff_dbl_ulong::odd_mask();

    ff_ulong _even = deinterleave( a.alpha & even_mask );
    ff_ulong _odd = deinterleave ( (a.alpha & odd_mask) >> 1 );

    ff_ulong evenXorOdd = _even ^ _odd;
    ff_ulong evenorodd = _even | _odd;

    int evenorodd_size = evenorodd.popcount();
    int evenAndOdd_size = (_even & _odd).popcount();
    int deg = a.degree();

    int num_terms = 1 << evenorodd_size;

    ff_complex global_coeff = multiplier * scalar_utils::m1pow(deg*(deg-1)/2) * scalar_utils::Ipow(_odd.popcount());

    std::vector<FermiString> terms(num_terms);
    std::vector<ff_complex> coeffs(num_terms);
    terms[0] = FermiString(evenXorOdd,0);
    coeffs[0] = global_coeff * scalar_utils::m1pow(evenAndOdd_size);
    res.terms[terms[0]] += coeffs[0];

    std::uint64_t z = 0;
    std::uint64_t count = 1;

    ff_ulong ann, cre;
    ff_complex coeff;


    int level = 0;
    std::uint64_t mask = 0;

    // Loop through the bits of evenorodd
    for(int j = evenorodd.begin() ; j != evenorodd.end(); j = evenorodd.next(j)) {

        ff_ulong singj = ff_ulong::singleton(j);
        ff_ulong notsingj = ~singj;
        bool inEven = (_even & singj) != 0;
        bool inOdd = (_odd & singj) != 0;
        bool powerOfTwoInc = (inEven && inOdd);

        for(z=count; z != 2*count; z++) {
            int numDaggersBeforeJ = std::popcount((z-count) ^ mask);
            int powerOfNeg1Inc = inOdd + numDaggersBeforeJ;
            ann = terms[z-count].ann | singj;
            cre = (inEven && inOdd) ?
                          (terms[z-count].cre | singj)
                         : (terms[z-count].cre & notsingj);
            coeff = coeffs[z-count];
            if(powerOfTwoInc) coeff *= 2;
            if(powerOfNeg1Inc % 2) coeff = -coeff;
            terms[z] = FermiString(cre, ann);
            coeffs[z] = coeff;
            res.terms[terms[z]] += coeffs[z];
        }
        count *= 2;
        if(inEven ^ inOdd) {
            mask |= (1ULL << level);
        }
        level++;
    }

}

FermiPolynomial majorana_to_fermi(const MajoranaString& a) {
    FermiPolynomial res;
    _majorana_to_fermi(a,res);
    return res.compress(0);
}

FermiPolynomial majorana_to_fermi(const MajoranaPolynomial& p) {
    FermiPolynomial res;
    for(const auto& [a,v] : p.terms) {
        _majorana_to_fermi(a,res,v);
    }
    return res.compress(0);
}


// ----------------------------------------------------------------------------
// PAULI TO MAJORANA/FERMI
// ----------------------------------------------------------------------------

MajoranaMonomial pauli_to_majorana(const PauliString& a) {

    MajoranaString res;
    int jpow = 0;
    ff_ulong supp = a.xory | a.yorz;
    MajoranaString sqm;

    for(int j = supp.begin(); j != supp.end(); j = supp.next(j)) {
        ff_ulong singj = ff_ulong::singleton(j);
        bool xory = (a.xory & singj) != 0;
        bool yorz = (a.yorz & singj) != 0;
        // Majorana representation of single-qubit term (Xj, Yj, or Zj)
        if(!xory && !yorz) {
            // identity
            sqm = MajoranaString();
        } else if(xory && !yorz) {
            // X
            sqm = MajoranaString(ff_dbl_ulong::range(2*j,true));
            jpow -= j;
        } else if (xory && yorz) {
            // Y
            sqm = MajoranaString(ff_dbl_ulong::range(2*j,false) | ff_dbl_ulong::singleton(2*j+1));
            jpow -= j;
        } else if (!xory && yorz) {
            // Z
            // 3 is '11' in binary
            sqm = MajoranaString(3ULL << (2*j));
            jpow -= 1;
        }
        int q = 0;
        majorana_string_multiply(res, q, sqm);
        jpow += 2*q;
    }

    return MajoranaMonomial(res,scalar_utils::Ipow(jpow));

}

MajoranaPolynomial pauli_to_majorana(const PauliPolynomial& p) {
    MajoranaPolynomial res;
    for(const auto& [a,v] : p.terms) {
        MajoranaMonomial b = pauli_to_majorana(a);
        res.terms[b.majorana_string()] += b.coefficient() * v;
    }
    return res.compress(0);
}

FermiPolynomial reverse_jordan_wigner(const PauliString& a) {
    MajoranaMonomial b = pauli_to_majorana(a);
    FermiPolynomial res;
    _majorana_to_fermi(b.majorana_string(), res, b.coefficient());
    return res.compress(0);
}

FermiPolynomial reverse_jordan_wigner(const PauliPolynomial& a) {
    return majorana_to_fermi(pauli_to_majorana(a));
}

}