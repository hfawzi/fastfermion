/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "hashmap/unordered_dense.h"
#include "format_poly.h"
#include "parse.h"
#include "matrix.h"

#include <functional>

namespace fastfermion {

// Forward declaration of structs defined in this file
struct FermiString;
struct FermiMonomial; // A FermiString with a coefficient
struct FermiPolynomial; // A linear combination of FermiStrings

struct FermiStringHash;
using FermiPolynomialMap = ankerl::unordered_dense::map<FermiString, ff_complex, FermiStringHash>;

struct FermiString {
    // Represents Fermionic operator normal ordered, i.e., of the form
    //    a_{i_1}^{\dagger} ... a_{i_k}^{\dagger} a_{j_1} ... a_{j_m}
    // where i_1 > ... > i_k and j_1 > ... > j_m
    ff_ulong cre; // i'th bit is 1 if there is a_{i}^{\dagger}
    ff_ulong ann; // i'th bit is 1 if there is a_{i}
    FermiString() : cre(0), ann(0) {};
    FermiString(const ff_ulong& cre, const ff_ulong& ann) : cre(cre), ann(ann)  { };
    FermiString(const std::vector<std::pair<int, bool>>& str) {
        // Constructs a FermiString from a vector of the form
        // { (i1,1),(i2,1),(j1,0),(j2,0) }
        // (i,1) is for a creation term a_i^{\dagger} and (j,0)
        // is for an annihilation term a_j
        // Sequence must be normal ordered
        int str_size = str.size();
        for(int i=0; i<str_size; ++i) {
            // Check that it is normal ordered
            if(i > 0) {
                if((str[i].second == str[i-1].second && str[i].first >= str[i-1].first)
                    || str[i].second > str[i-1].second) {
                    throw_error("Invalid FermiString (not normal ordered)");
                }
            }
            if(str[i].first < 0 || str[i].first >= ff_ulong::DIGITS) {
                throw_error("Invalid FermiString (mode index " << str[i].first << " invalid, should be between 0 and " << ff_ulong::DIGITS-1 << ")");
            }
            if(str[i].second == 1) {
                cre.set(str[i].first);
            } else {
                ann.set(str[i].first);
            }
        }
    }
    FermiString(const std::vector<int>& cre_seq, const std::vector<int>& ann_seq) : cre(0), ann(0) {
        // ann_seq and cre_seq should both be decreasing
        // raises error if not
        std::size_t i;
        for(i=0; i<cre_seq.size(); i++) {
            //if(i > 0) assertm(cre_seq[i] < cre_seq[i-1], "Error: cre_seq should be decreasing");
            if(i > 0 && cre_seq[i] >= cre_seq[i-1]) {
                throw_error("Invalid FermiString (not normal ordered)");
            }
            if(cre_seq[i] >= ff_ulong::DIGITS) {
                throw_error("Invalid FermiString (mode index " << cre_seq[i] << " invalid, should be between 0 and " << ff_ulong::DIGITS-1 << ")");
            }
            cre.set(cre_seq[i]);
        }
        for(i=0; i<ann_seq.size(); i++) {
            //if(i > 0) assertm(ann_seq[i] < ann_seq[i-1], "Error: ann_seq should be decreasing");
            if(i > 0 && ann_seq[i] >= ann_seq[i-1]) {
                throw_error("Invalid FermiString");
            }
            if(ann_seq[i] >= ff_ulong::DIGITS) {
                throw_error("Invalid FermiString (mode index " << ann_seq[i] << " invalid, should be between 0 and " << ff_ulong::DIGITS-1 << ")");
            }
            ann.set(ann_seq[i]);
        }
    }
    FermiString(const std::string& fs) : FermiString(_parse_fermi_string(fs)) { }

    int extent() const {
        return ff_ulong::DIGITS - (cre | ann).countl_zero();
    };
    std::string to_compact_string() const {
        std::string ret = "";
        std::vector<int> cre_set = cre.rsupport();
        std::vector<int> ann_set = ann.rsupport();
        std::size_t i;
        for(i = 0; i<cre_set.size(); i++) {
            if(i != 0) ret += " ";
            ret += ff_config.fermi_symbol + std::to_string(cre_set[i]) + ff_config.dagger_symbol;
        }
        if(cre_set.size() > 0 && ann_set.size() > 0) ret += " ";
        for(i = 0; i<ann_set.size(); i++) {
            if(i != 0) ret += " ";
            ret += ff_config.fermi_symbol + std::to_string(ann_set[i]);
        }
        if(ret == "") ret = ff_config.identity_symbol; // identity
        return ret;
    }
    std::string to_string() const { return to_compact_string(); }

    std::vector<std::pair<int,int>> indices() const {
        // Returns representation of FermiString as a vector
        // {(i_1,1),...,(i_k,1),(j_1,0),...,(j_m,0)}
        // if FermiString is
        // a_{i_1}^{\dagger} ... a_{i_k}^{\dagger} a_{j_1} ... a_{j_m}
        std::vector<std::pair<int,int>> ret;
        std::vector<int> cre_set = cre.rsupport();
        std::vector<int> ann_set = ann.rsupport();
        std::size_t i;
        for(i = 0; i<cre_set.size(); i++) {
            ret.push_back({cre_set[i],1});
        }
        for(i = 0; i<ann_set.size(); i++) {
            ret.push_back({ann_set[i],0});
        }
        return ret;
    }

    std::vector<int> cre_set() const { return cre.rsupport(); } // in reverse order
    std::vector<int> ann_set() const { return ann.rsupport(); } // in reverse order
    std::vector<int> support_set(bool spinful=false) const {
        std::vector<int> modes = (cre | ann).support();
        if(spinful) {
            for(std::size_t i=0; i<modes.size(); i++) modes[i] /= 2; // Return site number
        }
        return modes;
    }

    int degree_cre() const { return cre.popcount(); }
    int degree_ann() const { return ann.popcount(); }
    int degree_total() const { return cre.popcount() + ann.popcount(); }
    int degree() const {
        return degree_total();
    }
    int degree(const int& updown) const {
        if(updown == 1) {
            return degree_cre();
        } else if (updown == 0) {
            return degree_ann();
        } else {
            throw_error("Invalid argument: should be either 0 (annihilation) or 1 (creation)");
        }
    }

    std::uint64_t hash() const {
        auto h = std::uint64_t{};
        h = ankerl::unordered_dense::tuple_hash_helper<>::mix64(h,cre.hash());
        h = ankerl::unordered_dense::tuple_hash_helper<>::mix64(h,ann.hash());
        return h;
    }

    FermiMonomial permute(const std::vector<int>& map) const;
    FermiMonomial dagger() const;

    bool commutes(const ff_complex& b) const { return true; }
    bool commutes(const FermiString& b) const;
    bool commutes(const FermiMonomial& b) const;
    bool commutes(const FermiPolynomial& b) const;

    ff_complex commutator(const ff_complex& b) const { return 0; };
    FermiPolynomial commutator(const FermiString& b) const;
    FermiPolynomial commutator(const FermiMonomial& b) const;
    FermiPolynomial commutator(const FermiPolynomial& b) const;

    CSCMatrix<ff_complex> sparse(int n) const;
    CSCMatrix<ff_complex> sparse(int n, int nocc) const;
    CSCMatrix<ff_complex> sparse() const {
        return sparse(extent());
    }

};

struct FermiStringHash {
    using is_avalanching = void;
    std::uint64_t operator()(const FermiString& a) const {
        return a.hash();
    }
};


// Define a total order on FermiString
bool operator<(const FermiString& a, const FermiString& b) {
    return std::forward_as_tuple(a.ann | a.cre, a.cre, a.ann) < std::forward_as_tuple(b.ann | b.cre, b.cre, b.ann);
}


// A FermiMonomial is a FermiString together with a coefficient
struct FermiMonomial {

    FermiString s;
    ff_complex coeff;
    FermiMonomial() : s(0,0), coeff(0) { };
    FermiMonomial(const FermiString& s) : s(s), coeff(1) { };
    FermiMonomial(const FermiString& s, ff_complex coeff) : s(s), coeff(coeff) { };
    FermiMonomial(const std::vector<int>& cre_seq, const std::vector<int>& ann_seq, ff_complex coeff) : s(cre_seq, ann_seq), coeff(coeff) { };
    FermiMonomial(const ff_ulong& cre, const ff_ulong& ann, ff_complex coeff) : s(cre, ann), coeff(coeff) { };

    FermiString fermi_string() const { return s; }
    ff_complex coefficient() const { return coeff; }
    std::string to_compact_string() const { return format_complex(coeff) + " " + s.to_compact_string(); }
    std::string to_string() const { return to_compact_string(); }
    int extent() const { return s.extent(); }
    std::vector<int> support_set() const { return s.support_set(); }
    FermiMonomial& operator*=(const ff_complex& b) { coeff *= b; return *this; }
    FermiMonomial permute(const std::vector<int>& map) const;
    FermiMonomial dagger() const;
    bool commutes(const ff_complex& b) const { return true; }
    bool commutes(const FermiString& b) const;
    bool commutes(const FermiMonomial& b) const;
    bool commutes(const FermiPolynomial& b) const;

    ff_complex commutator(const ff_complex& b) const { return 0; };
    FermiPolynomial commutator(const FermiString& b) const;
    FermiPolynomial commutator(const FermiMonomial& b) const;
    FermiPolynomial commutator(const FermiPolynomial& b) const;

};


// A FermiPolynomial is a linear combination of FermiStrings
struct FermiPolynomial {
    FermiPolynomialMap terms;
    FermiPolynomial() { };
    explicit FermiPolynomial(const ff_complex& v) { terms[FermiString()] = v; }  // constant term equal to v
    FermiPolynomial(const FermiString& a) { terms[a] = 1; }
    FermiPolynomial(const FermiMonomial& a) { terms[a.s] = a.coeff; }
    FermiPolynomial(const std::vector<int>& cre_seq, const std::vector<int>& ann_seq, ff_complex coeff) {
        terms[FermiString(cre_seq,ann_seq)] = coeff;  
    }
    FermiPolynomial(const std::vector<FermiString>& a, const std::vector<ff_complex>& coeffs) {
        assert(coeffs.size() == a.size());
        for(std::size_t i=0; i<a.size(); i++)
            if(coeffs[i] != ff_complex(0,0)) terms[a[i]] += coeffs[i];
    }
    FermiPolynomial(const std::vector<FermiString>& a, const std::vector<ff_float>& coeffs) {
        assert(coeffs.size() == a.size());
        for(std::size_t i=0; i<a.size(); i++)
            if(coeffs[i] != 0) terms[a[i]] += coeffs[i];
    }
    
    FermiPolynomial(const std::vector<std::pair<int, bool>>& str, const ff_complex& coeff) {
        // Here str is an OpenFermion's style string of the form
        // { (i1,a1), (i2,a2), ... }
        // where a1,a2, ... \in {0,1} indicates whether this is a creation (1)
        // or annihilation operator (0)
        // Note that this string may not be normal ordered
        //if(str.size() == 0) return;

        // We initially set the polynomial to coeff*I, and then
        // we will keep multiplying *this with FermiStrings
        terms[FermiString()] = coeff;
        bool total_sign = 0;
        int str_size = str.size();
        for(int i=0; i<str_size; ) {
            std::array<ff_ulong,2> creann{{0,0}};
            int j = i;
            // Read creation then annihilation
            for(int action=1; action >= 0; action--) {
                while(j < str_size && str[j].second == action) {
                    int loc = str[j].first;
                    if(loc < 0 || loc >= ff_ulong::DIGITS) {
                        throw_error("Invalid FermiString (mode index " << loc << " invalid, should be between 0 and " << ff_ulong::DIGITS-1 << ")");
                    }
                    if(creann[action].at(loc)) {
                        // The creation operator at loc appears twice
                        terms.clear();
                        return;
                    } else {
                        creann[action].set(loc);
                        // Count sign
                        total_sign ^= (creann[action] & ff_ulong::range(loc)).popcount() % 2 == 1;
                    }
                    j++;
                }
            }
            (*this) *= FermiString(creann[1],creann[0]);
            i = j;
        }
        if(total_sign) {
            (*this) *= -1;
        }
    }

    FermiPolynomial(const std::vector<std::pair<int, bool>>& str) : 
        FermiPolynomial(str,1.0) { }
    
    FermiPolynomial permute(const std::vector<int>& map) const;

    ff_complex coefficient(const FermiString& x) const {
        auto it = terms.find(x);
        if(it == terms.end()) {
            return ff_complex(0,0);
        } else {
            return it->second;
        }
    }

    int extent() const {
        int l = 0;
        for(auto& [x,v] : terms) {
            int xli = x.extent();
            if(v != ff_complex(0,0) && xli > l)
                l = xli;
        }
        return l;
    }

    int degree() const {
        int deg = 0;
        for(const auto& [x,coeff] : terms) {
            if (coeff != ff_complex(0,0)) {
                deg = MAX(deg, x.degree());
            }
        }
        return deg;
    }

    int degree(const int& updown) const {
        int deg = 0;
        for(const auto& [x,coeff] : terms) {
            if (coeff != ff_complex(0,0)) {
                deg = MAX(deg, x.degree(updown));
            }
        }
        return deg;
    }

    ff_float norm(const int& p=1) const {
        ff_float ret = 0;
        for(const auto& [x,v] : terms) {
            if(p == 0) {
                ret += 1;
            } else if (p == 1) {
                ret += std::abs(v);
            } else if (p == 2) {
                ret += std::abs(v)*std::abs(v);
            } else {
                throw_error("Invalid value of p (allowed values are p=0,p=1,p=2)");
            }
        }
        if(p == 2) {
            ret = std::sqrt(ret);
        }
        return ret;
    }

    ff_float norm_inf() const {
        ff_float ret = 0;
        for(const auto& [x,v] : terms) {
            ret = MAX(std::abs(v), ret);
        }
        return ret;
    }
    
    bool is_zero(ff_float tolerance=1e-8) {
        for(const auto& [x,v] : terms) {
            if(std::abs(v) > tolerance) return false;
        }
        return true;
    }

    FermiPolynomial dagger() const {
        FermiPolynomial b;
        FermiMonomial xd;
        for(const auto& [x,v] : terms) {
            xd = x.dagger();
            b.terms[xd.fermi_string()] += xd.coefficient()*std::conj(v);
        }
        return b;
    }

    bool is_hermitian() const {
        FermiMonomial xd;
        for(const auto& [x,v] : terms) {
            if(v != ff_complex(0,0)) {
                xd = x.dagger();
                try {
                    if(terms.at(xd.fermi_string()) != xd.coefficient()*std::conj(v)) return false;
                } catch(const std::out_of_range& ex) {
                    return false;
                }
            }
        }
        return true;
    }

    std::vector<int> support_set() const {
        ff_ulong supp = 0;
        for(auto& [x,v] : terms) {
            if(v != ff_complex(0,0)) supp |= (x.cre | x.ann);
        }
        return supp.support();
    }

    void inplace_compress(const FermiString& x) {
        if(terms[x] == ff_complex(0,0)) {
            terms.erase(x);
        }
    }

    // In-place addition
    FermiPolynomial& operator+=(const ff_complex& b) { if(b != ff_complex(0,0)) { terms[FermiString()] += b; inplace_compress(FermiString()); } return *this; };
    FermiPolynomial& operator+=(const FermiString& b) { terms[b] += 1; inplace_compress(b); return *this; };
    FermiPolynomial& operator+=(const FermiMonomial& b) { terms[b.s] += b.coeff; inplace_compress(b.s); return *this; };
    FermiPolynomial& operator+=(const FermiPolynomial& b) {
        for(auto& [x,v] : b.terms) {
            terms[x] += v;
            inplace_compress(x);
        }
        return *this;
    }

    // In-place subtraction
    FermiPolynomial& operator-=(const ff_complex& b) { if(b != ff_complex(0,0)) { terms[FermiString()] -= b; inplace_compress(FermiString()); } return *this; };
    FermiPolynomial& operator-=(const FermiString& b) { terms[b] -= 1; inplace_compress(b); return *this; };
    FermiPolynomial& operator-=(const FermiMonomial& b) { terms[b.s] -= b.coeff; inplace_compress(b.s); return *this; };
    FermiPolynomial& operator-=(const FermiPolynomial& b) {
        for(auto& [x,v] : b.terms) {
            terms[x] -= v;
            inplace_compress(x);
        }
        return *this;
    }

    // In-place multiplication
    FermiPolynomial& operator*=(const ff_complex& b) { for(auto& [x,v] : terms) { terms[x] *= b; } return *this; }
    FermiPolynomial& operator*=(const FermiString&);
    FermiPolynomial& operator*=(const FermiPolynomial&);

    // In-place division
    FermiPolynomial& operator/=(const ff_complex& b) { for(auto& [x,v] : terms) { terms[x] /= b; } return *this; }

    std::string to_compact_string() const {
        return format_poly(*this, ff_config.max_terms_to_show, ff_config.max_line_length);
    }
    std::string to_string() const { return to_compact_string(); }

    FermiPolynomial compress(ff_float threshold=1e-8) const {
        FermiPolynomial b;
        for(const auto& [x,v] : terms) {
            if(std::abs(v) > threshold) {
                b.terms[x] = v;
            }
        }
        return b;
    }

    FermiPolynomial truncate(int k) const {
        FermiPolynomial b;
        for(const auto& [x,v] : terms) {
            if(x.degree_total() <= k) {
                b.terms[x] = v;
            }
        }
        return b;
    }

    ff_complex overlapwithvacuum() const {
        // Return expectation value wrt. the vacuum state, i.e. <vac|A|vac>.
        // This is just the coefficient of the identity term
        try {
            return terms.at(FermiString());
        } catch (const std::exception& e) {
            return ff_complex(0.0,0.0);
        }
    }

    bool commutes(const ff_complex& b) const { return true; }
    bool commutes(const FermiString& b) const;
    bool commutes(const FermiMonomial& b) const;
    bool commutes(const FermiPolynomial& b) const;

    ff_complex commutator(const ff_complex& b) const { return 0; }
    FermiPolynomial commutator(const FermiString& b) const;
    FermiPolynomial commutator(const FermiMonomial& b) const;
    FermiPolynomial commutator(const FermiPolynomial& b) const;

    CSCMatrix<ff_complex> sparse(int n) const;
    CSCMatrix<ff_complex> sparse(int n, int nocc) const;
    CSCMatrix<ff_complex> sparse() const {
        return sparse(extent());
    }

};

// -------------------------------------------------------------------------------

// Dagger operation
// Swaps creation and annihilation and adjust sign

FermiMonomial FermiString::dagger() const {
    int n_ann = ann.popcount();
    int n_cre = cre.popcount();
    int ann_sign = n_ann*(n_ann-1)/2 % 2 == 0 ? 1 : -1;
    int cre_sign = n_cre*(n_cre-1)/2 % 2 == 0 ? 1 : -1;
    return FermiMonomial(ann,cre,ann_sign*cre_sign);
}

FermiMonomial FermiMonomial::dagger() const {
    FermiMonomial s_dagger = s.dagger(); // s_dagger.coefficient() = +1 or -1
    return FermiMonomial(s_dagger.fermi_string(), s_dagger.coefficient() * std::conj(coeff));
}

// -------------------------------------------------------------------------------

// Algebra operation
// The key function is the multiplication function of two FermiStrings

FermiPolynomial operator*(const FermiString& x, const FermiString& y) {

    // Takes two FermiStrings and multiplies them
    // Note that the multiplication of two FermiStrings is in general a FermiPolynomial,
    // as it can result in multiple terms, for example
    //     a_1 * a_1^{\dagger} = 1 - a_{1}^{\dagger} a_{1}

    // Assume x = a_U^{\dagger} a_S
    //        y = a_T^{\dagger} a_V
    // where U, S, T, V are ordered sets (ordered in decreasing fashion)
    // Let W0 = (U\cap T) \cup (S\cap V)
    // If W0 \not\subseteq S\cap T then x*y = 0
    // Otherwise, the total number of terms in the product x*y is exactly 2^{|(S \cap T) \ W0|}
    // See doc for proof and more details on this

    ff_ulong W0 = (x.cre & y.cre) | (x.ann & y.ann);
    ff_ulong ScapT = x.ann & y.cre;

    if ((W0 & ScapT) != W0) {
        // If W0 \not\subseteq S\cap T then x*y == 0
        return FermiPolynomial(); // Zero polynomial
    }

    // In what follows W0 is a subset of ScapT
    ff_ulong ScapTminusW0 = ScapT & (~W0);

    // Total number of terms is 2^{|(S \cap T) \ W0|}
    int ScapTminusW0_size = ScapTminusW0.popcount();
    int num_terms = 1 << ScapTminusW0_size; // 2^{|(S \cap T) \ W0|}

    // Compute the term 0
    int W0_size = W0.popcount();
    int S_size = x.ann.popcount();
    int T_size = y.cre.popcount();
    int par0 = S_size*T_size + W0_size*T_size + W0_size*(W0_size-1)/2;
    int epsW0 = scalar_utils::m1pow(merge_parity(W0, y.cre) + merge_parity(W0, x.ann) + par0);
    //int epsW0 = merge_parity(W0, y.cre)*merge_parity(W0, x.ann)*(par0%2 == 0 ? 1 : -1);
    FermiMonomial default_monomial = FermiMonomial(
        x.cre | (y.cre & ~W0),
        (x.ann & ~W0) | y.ann,
        epsW0 * scalar_utils::m1pow(merge_parity(x.cre, y.cre & ~W0) + merge_parity(x.ann & ~W0, y.ann))
        //epsW0 * merge_parity(x.cre, y.cre & ~W0) * merge_parity(x.ann & ~W0, y.ann)
    );


    std::size_t i;
    std::size_t count = 0;

    // ret will hold the result x*y
    // We also store the terms as a vector for efficiency reasons
    // because the algorithm constructs the terms of x*y in a certain order
    // NOTE: this may be a waste of memory/compute time
    FermiPolynomial ret;
    std::vector<FermiMonomial> terms(num_terms);

    // Add first term
    terms[0] = default_monomial;
    ret.terms[terms[0].fermi_string()] = terms[0].coefficient();
    count = 1;

    
    ff_ulong mask_wbar, mask_gt_wbar, mask_neq_wbar, mask_lt_wbar;
    ff_ulong SminusW;
    ff_ulong TminusW;
    int par, S_gt_wbar, T_gt_wbar, V_gt_wbar, U_lt_wbar, cW;
    int W_size_cur;

    // Iterate over the bits of (S \cap T) \ W0 from smallest to largest (right to left)
    for(int wbar = ScapTminusW0.begin(); wbar != ScapTminusW0.end(); wbar = ScapTminusW0.next(wbar)) {

        mask_lt_wbar = ff_ulong::range(wbar); // 1 for i < wbar, 0 else
        mask_gt_wbar = ~ff_ulong::range(wbar,true); // 1 for i > wbar, 0 else
        mask_neq_wbar = ~ff_ulong::singleton(wbar); // 0 at wbar, 1 everywhere else

        
        S_gt_wbar = (x.ann & mask_gt_wbar).popcount();
        T_gt_wbar = (y.cre & mask_gt_wbar).popcount();
        V_gt_wbar = (y.ann & mask_gt_wbar).popcount();
        U_lt_wbar = (x.cre & mask_lt_wbar).popcount();

        // NOTE: I only need the quantity
        // S_gt_wbar + T_gt_wbar + V_gt_wbar + U_lt_wbar
        // May be able to use less operations and calls to popcount


        // count is a power of two
        // Double the terms
        for(i=count; i<2*count; i++) {
            W_size_cur = W0_size+std::popcount(i);
            par = T_size + (W_size_cur-1) + S_gt_wbar + T_gt_wbar + U_lt_wbar + V_gt_wbar;
            // Can reduce calls to popcount to 1 instead of 2
            TminusW = terms[i-count].fermi_string().cre & mask_neq_wbar; // T \ {wbar}
            SminusW = terms[i-count].fermi_string().ann & mask_neq_wbar; // S \ {wbar}
            cW = static_cast<int>(terms[i-count].coefficient().real()) * (par%2 == 0 ? 1 : -1);
            terms[i] = FermiMonomial(
                x.cre | TminusW,
                SminusW | y.ann,
                cW
            );
            ret.terms[terms[i].fermi_string()] = terms[i].coefficient();
        }
        count *= 2;
    }
    
    return ret;

}

// Overload operators

FermiMonomial operator*(const FermiString& b, const ff_complex& a) { return FermiMonomial(b, a); }
FermiMonomial operator*(const ff_complex& a, const FermiString& b) { return FermiMonomial(b, a); }

FermiMonomial operator*(const FermiMonomial& a, const ff_complex& b) { return FermiMonomial(a.s, a.coeff * b); }
FermiMonomial operator*(const ff_complex& a, const FermiMonomial& b) { return FermiMonomial(b.s, b.coeff * a); }

FermiPolynomial operator+(const ff_complex& a, const FermiString& b) { FermiPolynomial c(a); c += b; return c; }
FermiPolynomial operator+(const ff_complex& a, const FermiMonomial& b) { FermiPolynomial c(a); c += b; return c; }
FermiPolynomial operator+(const ff_complex& a, const FermiPolynomial& b) { FermiPolynomial c(a); c += b; return c; }

FermiPolynomial operator+(const FermiString& a, const ff_complex& b) { FermiPolynomial c(a); c += b; return c; }
FermiPolynomial operator+(const FermiMonomial& a, const ff_complex& b) { FermiPolynomial c(a); c += b; return c; }
FermiPolynomial operator+(const FermiPolynomial& a, const ff_complex& b) { FermiPolynomial c(a); c += b; return c; }

FermiPolynomial operator+(const FermiString& a, const FermiString& b) { FermiPolynomial c(a); c += b; return c; }
FermiPolynomial operator+(const FermiString& a, const FermiMonomial& b) { FermiPolynomial c(a); c += b; return c; }
FermiPolynomial operator+(const FermiString& a, const FermiPolynomial& b) { FermiPolynomial c(a); c += b; return c; }
FermiPolynomial operator+(const FermiMonomial& a, const FermiString& b) { FermiPolynomial c(a); c += b; return c; }
FermiPolynomial operator+(const FermiMonomial& a, const FermiMonomial& b) { FermiPolynomial c(a); c += b; return c; }
FermiPolynomial operator+(const FermiMonomial& a, const FermiPolynomial& b) { FermiPolynomial c(a); c += b; return c; }

FermiPolynomial operator+(const FermiPolynomial& a, const FermiString& b) { FermiPolynomial c(a); c += b; return c; }
FermiPolynomial operator+(const FermiPolynomial& a, const FermiMonomial& b) { FermiPolynomial c(a); c += b; return c; }
FermiPolynomial operator+(const FermiPolynomial& a, const FermiPolynomial& b) { FermiPolynomial c(a); c += b; return c; }

FermiPolynomial operator-(const FermiString& a, const FermiString& b) { FermiPolynomial c(a); c -= b; return c; }
FermiPolynomial operator-(const FermiString& a, const FermiMonomial& b) { FermiPolynomial c(a); c -= b; return c; }
FermiPolynomial operator-(const FermiString& a, const FermiPolynomial& b) { FermiPolynomial c(a); c -= b; return c; }

FermiPolynomial operator-(const FermiMonomial& a, const FermiString& b) { FermiPolynomial c(a); c -= b; return c; }
FermiPolynomial operator-(const FermiMonomial& a, const FermiMonomial& b) { FermiPolynomial c(a); c -= b; return c; }
FermiPolynomial operator-(const FermiMonomial& a, const FermiPolynomial& b) { FermiPolynomial c(a); c -= b; return c; }

FermiPolynomial operator-(const FermiPolynomial& a, const FermiString& b) { FermiPolynomial c(a); c -= b; return c; }
FermiPolynomial operator-(const FermiPolynomial& a, const FermiMonomial& b) { FermiPolynomial c(a); c -= b; return c; }
FermiPolynomial operator-(const FermiPolynomial& a, const FermiPolynomial& b) { FermiPolynomial c(a); c -= b; return c; }

FermiPolynomial operator*(const FermiPolynomial& a, const ff_complex& b) { FermiPolynomial c(a); c *= b; return c; }
FermiPolynomial operator*(const ff_complex& b, const FermiPolynomial& a) { FermiPolynomial c(a); c *= b; return c; }

FermiPolynomial operator/(const FermiPolynomial& a, const ff_complex& b) { FermiPolynomial c(a); c /= b; return c; }

FermiMonomial operator-(const FermiString& a) { return FermiMonomial(a, -1); }
FermiMonomial operator-(const FermiMonomial& a) { return -1*a; }
FermiPolynomial operator-(const FermiPolynomial& a) { return -1*a; }

FermiPolynomial operator-(const ff_complex& a, const FermiString& b) { FermiPolynomial c(a); c -= b; return c; }
FermiPolynomial operator-(const ff_complex& a, const FermiMonomial& b) { FermiPolynomial c(a); c -= b; return c; }
FermiPolynomial operator-(const ff_complex& a, const FermiPolynomial& b) { FermiPolynomial c(a); c -= b; return c; }

FermiPolynomial operator-(const FermiString& a, const ff_complex& b) { FermiPolynomial c(a); c -= b; return c; }
FermiPolynomial operator-(const FermiMonomial& a, const ff_complex& b) { FermiPolynomial c(a); c -= b; return c; }
FermiPolynomial operator-(const FermiPolynomial& a, const ff_complex& b) { FermiPolynomial c(a); c -= b; return c; }



FermiPolynomial& FermiPolynomial::operator*=(const FermiPolynomial& b) {
    FermiPolynomialMap new_terms;
    FermiMonomial xy;
    for(auto& [x,v] : terms) {
        for(auto& [y,w] : b.terms) {
            for(auto& [z,u] : (x*y).terms) {
                new_terms[z] += v*w*u;
            }
        }
    }
    // Only include nonzero terms
    terms.clear();
    for(const auto& [x,v] : new_terms) {
        if(v != ff_complex(0,0)) {
            terms[x] = v;
        }
    }
    return *this;
}

FermiPolynomial& FermiPolynomial::operator*=(const FermiString& b) {
    (*this) *= FermiPolynomial(b);
    return *this;
}

FermiPolynomial operator*(const FermiString& a, const FermiMonomial& b) { return b.coefficient() * (a*b.fermi_string()); }
FermiPolynomial operator*(const FermiMonomial& a, const FermiString& b) { return a.coefficient() * (a.fermi_string()*b); }
FermiPolynomial operator*(const FermiMonomial& a, const FermiMonomial& b) { return a.coefficient() * b.coefficient() * (a.fermi_string()*b.fermi_string()); }
FermiPolynomial operator*(const FermiPolynomial& a, const FermiPolynomial& b) { FermiPolynomial c(a); c *= b; return c; }

// -------------------------------------------------------------------------------

// Commutation

FermiPolynomial FermiPolynomial::commutator(const FermiPolynomial& b) const { return ((*this)*b - b*(*this)); }
FermiPolynomial FermiPolynomial::commutator(const FermiString& b) const { return commutator(FermiPolynomial(b)); }
FermiPolynomial FermiPolynomial::commutator(const FermiMonomial& b) const { return commutator(FermiPolynomial(b)); }

FermiPolynomial FermiString::commutator(const FermiString& b) const { return ((*this)*b - b*(*this)); }
FermiPolynomial FermiString::commutator(const FermiMonomial& b) const { return b.coeff * commutator(b.s); }
FermiPolynomial FermiString::commutator(const FermiPolynomial& b) const { return FermiPolynomial(*this).commutator(b); }

FermiPolynomial FermiMonomial::commutator(const FermiString& b) const { return coeff * s.commutator(b); }
FermiPolynomial FermiMonomial::commutator(const FermiMonomial& b) const { return (coeff*b.coeff)*s.commutator(b.s); }
FermiPolynomial FermiMonomial::commutator(const FermiPolynomial& b) const { return FermiPolynomial(*this).commutator(b); }


bool FermiString::commutes(const FermiString& b) const { return commutator(b).is_zero(0); }
bool FermiString::commutes(const FermiMonomial& b) const { return commutes(b.s); }
bool FermiString::commutes(const FermiPolynomial& b) const { return commutator(b).is_zero(0); }

bool FermiMonomial::commutes(const FermiString& b) const { return s.commutes(b); }
bool FermiMonomial::commutes(const FermiMonomial& b) const { return s.commutes(b.s); }
bool FermiMonomial::commutes(const FermiPolynomial& b) const { return s.commutes(b); }

bool FermiPolynomial::commutes(const FermiString& b) const { return commutator(b).is_zero(0); }
bool FermiPolynomial::commutes(const FermiMonomial& b) const { return commutator(b).is_zero(0); }
bool FermiPolynomial::commutes(const FermiPolynomial& b) const { return commutator(b).is_zero(0); }

// -------------------------------------------------------------------------------

// Permutations of modes

FermiMonomial FermiString::permute(const std::vector<int>& map) const {
    auto [new_cre,sgn1] = shuffle_with_parity(cre,map);
    auto [new_ann,sgn2] = shuffle_with_parity(ann,map);
    return FermiMonomial(new_cre, new_ann, sgn1*sgn2);
}

FermiMonomial FermiMonomial::permute(const std::vector<int>& map) const { return coeff * s.permute(map); }

FermiPolynomial FermiPolynomial::permute(const std::vector<int>& map) const {
    FermiPolynomial c;
    for(auto& [x,v] : terms) {
        FermiMonomial x2 = x.permute(map);
        c.terms[x2.fermi_string()] = x2.coefficient() * v;
    }
    return c;
}

// -------------------------------------------------------------------------------

// Equality

bool operator==(const FermiPolynomial& a, const FermiPolynomial& b) {
    FermiPolynomial c = a-b;
    for(auto& [x,v] : c.terms) {
        if(v.real() != 0 || v.imag() != 0) return false;
    }
    return true;
}

bool operator==(const FermiString& a, const ff_complex& b) { return a.cre == 0 && a.ann == 0 && b == ff_complex(1,0); }
bool operator==(const FermiString& a, const FermiString& b) { return a.cre == b.cre && a.ann == b.ann; }
bool operator==(const FermiString& a, const FermiMonomial& b) { return a == b.s && b.coeff == ff_complex(1,0); }
bool operator==(const FermiString& a, const FermiPolynomial& b) { return FermiPolynomial(a) == b; }

bool operator==(const FermiMonomial& a, const ff_complex& b) { return a.s.cre == 0 && a.s.ann == 0 && a.coeff == b; }
bool operator==(const FermiMonomial& a, const FermiString& b) { return a.s == b && a.coeff == ff_complex(1,0); }
bool operator==(const FermiMonomial& a, const FermiMonomial& b) { return a.s == b.s && a.coeff == b.coeff; }
bool operator==(const FermiMonomial& a, const FermiPolynomial& b) { return FermiPolynomial(a) == b; }

bool operator==(const FermiPolynomial& a, const ff_complex& b) { return a == FermiPolynomial(b); }
bool operator==(const FermiPolynomial& a, const FermiString& b) { return a == FermiPolynomial(b); }
bool operator==(const FermiPolynomial& a, const FermiMonomial& b) { return a == FermiPolynomial(b); }

// -------------------------------------------------------------------------------

// Printing

std::ostream& operator<<(std::ostream& os, const FermiString& a) { return os << a.to_compact_string(); }
std::ostream& operator<<(std::ostream& os, const FermiMonomial& a) { return os << a.to_compact_string(); }
std::ostream& operator<<(std::ostream& os, const FermiPolynomial& p) { return os << p.to_compact_string(); }

}