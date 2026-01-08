/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "hashmap/unordered_dense.h"
#include "matrix.h"
#include "format_poly.h"
#include "parse.h"

#include <functional>

namespace fastfermion {

using ff_dbl_ulong = BitSet<2*SYS_NUM_ULONG>;

// Forward declaration of structs defined in this file
struct MajoranaString;
struct MajoranaMonomial; // A MajoranaString with a coefficient
struct MajoranaPolynomial; // A linear combination of MajoranaString

struct MajoranaStringHash;
using MajoranaPolynomialMap = ankerl::unordered_dense::map<MajoranaString, ff_complex, MajoranaStringHash>;

struct MajoranaString {
    
    ff_dbl_ulong alpha;
    // Represents Majorana string m_{i_1} ... m_{i_k}
    // where i_1 < ... < i_k
    // alpha is a BitSet representing the set {i_1,...,i_k}
    MajoranaString() : alpha(0) {};
    MajoranaString(const ff_dbl_ulong& alpha) : alpha(alpha)  { };
    MajoranaString(const std::vector<int>& supp) : alpha(0) {
        // supp should be both an increasing set
        // raises error if not
        std::size_t i;
        for(i=0; i<supp.size(); i++) {
            if(i > 0 && supp[i] <= supp[i-1]) {
                throw_error("Invalid MajoranaString (not normal ordered)");
            }
            if(supp[i] < 0 || supp[i] >= ff_dbl_ulong::DIGITS) {
                throw_error("Invalid MajoranaString (mode index " << supp[i] << " invalid, should be between 0 and " << ff_dbl_ulong::DIGITS-1 << ")");
            }
            alpha.set(supp[i]);
        }
    }
    MajoranaString(const std::string& ms) : MajoranaString(_parse_majorana_string(ms)) {
    }
    std::string to_compact_string() const {
        std::string ret = "";
        std::vector<int> supp = alpha.support();
        std::size_t i;
        for(i = 0; i<supp.size(); i++) {
            if(i != 0) ret += " ";
            ret += ff_config.majorana_symbol + std::to_string(supp[i]);
        }
        if(ret == "") ret = ff_config.identity_symbol; // identity
        return ret;
    }
    std::string to_string() const { return to_compact_string(); }
    std::vector<int> support_set() const { return alpha.support(); }
    int degree() const { return alpha.popcount(); }
    bool is_hermitian() const {
        int degmod4 = degree() % 4;
        return degmod4 == 0 || degmod4 == 1;
    }

    std::uint64_t hash() const {
        return alpha.hash();
    }


    int extent() const {
        return ff_dbl_ulong::DIGITS - alpha.countl_zero();
    };


    bool commutes(const ff_float& b) const { return true; }

    bool commutes(const MajoranaString& b) const {
        int na = alpha.popcount();
        int nb = b.alpha.popcount();
        int nab = (alpha & b.alpha).popcount();
        return ( (na*nb - nab)%2 == 0 );
    }

    bool commutes(const MajoranaMonomial& b) const;
    bool commutes(const MajoranaPolynomial& b) const;

    ff_complex commutator(const ff_complex& b) const { return 0; }
    MajoranaMonomial commutator(const MajoranaString& b) const;
    MajoranaMonomial commutator(const MajoranaMonomial& b) const;
    MajoranaPolynomial commutator(const MajoranaPolynomial& b) const;

    CSCMatrix<ff_complex> sparse() const;
    CSCMatrix<ff_complex> sparse(int n) const;
    MajoranaMonomial permute(const std::vector<int>& map) const;
    MajoranaMonomial dagger() const;

};


// Define a total order on MajoranaString
bool operator<(const MajoranaString& a, const MajoranaString& b) {
    return a.alpha < b.alpha;
}

struct MajoranaStringHash {
    using is_avalanching = void;
    std::uint64_t operator()(const MajoranaString& a) const {
        return a.hash();
    }
};

// A MajoranaMonomial is a MajoranaString together with a coefficient
struct MajoranaMonomial {

    MajoranaString s;
    ff_complex coeff;
    MajoranaMonomial() : s(), coeff(0) { };
    MajoranaMonomial(const MajoranaString& s) : s(s), coeff(1) { };
    MajoranaMonomial(const MajoranaString& s, const ff_complex& coeff) : s(s), coeff(coeff) { };
    MajoranaMonomial(const std::vector<int>& supp, const ff_complex& coeff) : s(supp), coeff(coeff) { };
    MajoranaMonomial(const ff_dbl_ulong& alpha, const ff_complex& coeff) : s(alpha), coeff(coeff) { };

    MajoranaString majorana_string() const { return s; }
    ff_complex coefficient() const { return coeff; }
    std::string to_compact_string() const { return format_complex(coeff) + " " + s.to_compact_string(); }
    std::string to_string() const { return to_compact_string(); }
    int extent() const { return s.extent(); }
    std::vector<int> support_set() const { return s.support_set(); }
    int degree() const { return s.degree(); }
    MajoranaMonomial& operator*=(const ff_complex& b) { coeff *= b; return *this; }
    bool is_hermitian() const {
        bool s_hermitian = s.is_hermitian();
        return (s_hermitian && coeff.imag() == 0) || (!s_hermitian && coeff.real() == 0);
    }
    MajoranaMonomial permute(const std::vector<int>& map) const;
    MajoranaMonomial dagger() const;

    bool commutes(const ff_complex& b) const { return true; }
    bool commutes(const MajoranaString& b) const { return s.commutes(b); }
    bool commutes(const MajoranaMonomial& b) const { return s.commutes(b.s); }
    bool commutes(const MajoranaPolynomial& b) const;

    ff_complex commutator(const ff_complex& b) const { return 0; }
    MajoranaMonomial commutator(const MajoranaString& b) const;
    MajoranaMonomial commutator(const MajoranaMonomial& b) const;
    MajoranaPolynomial commutator(const MajoranaPolynomial& b) const;

};


// A MajoranaPolynomial is a linear combination of MajoranaStrings
struct MajoranaPolynomial {
    MajoranaPolynomialMap terms;
    MajoranaPolynomial() { };
    explicit MajoranaPolynomial(const ff_complex& v) { terms[MajoranaString()] = v; }  // constant term equal to v
    MajoranaPolynomial(const MajoranaString& a) { terms[a] = 1; }
    MajoranaPolynomial(const MajoranaMonomial& a) { terms[a.s] = a.coeff; }
    MajoranaPolynomial(const std::vector<int>& supp, const ff_complex& coeff) {
        // Note: supp is not necessary sorted, so we need to determine the sign
        ff_dbl_ulong alpha;
        bool sgn = 0;
        int supp_size = supp.size();
        for(int i=0; i<supp_size; ++i) {
            if(supp[i] < 0 || supp[i] >= ff_dbl_ulong::DIGITS) {
                throw_error("Invalid MajoranaString (mode index " << supp[i] << " invalid, should be between 0 and " << ff_dbl_ulong::DIGITS-1 << ")");
            }
            alpha ^= ff_dbl_ulong::singleton(supp[i]); // XOR since m_i^2=1
            sgn ^= (alpha & (~ff_dbl_ulong::range(supp[i],true))).popcount() % 2 == 1;
        }
        terms[MajoranaString(alpha)] = (sgn ? -1 : 1) * coeff;
    }
    MajoranaPolynomial(const std::vector<int>& supp) : MajoranaPolynomial(supp,1) { }
    MajoranaPolynomial(const std::vector<MajoranaString>& a, const std::vector<ff_complex>& coeffs) {
        assert(coeffs.size() == a.size());
        for(std::size_t i=0; i<a.size(); i++)
            if(coeffs[i] != ff_complex(0,0)) terms[a[i]] += coeffs[i];
    }
    MajoranaPolynomial(const std::vector<MajoranaString>& a, const std::vector<ff_float>& coeffs) {
        assert(coeffs.size() == a.size());
        for(std::size_t i=0; i<a.size(); i++)
            if(coeffs[i] != 0) terms[a[i]] += coeffs[i];
    }

    MajoranaPolynomial permute(const std::vector<int>& map) const;
    CSCMatrix<ff_complex> sparse() const;
    CSCMatrix<ff_complex> sparse(int n) const;

    ff_complex coefficient(const MajoranaString& x) const {
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

    ff_float norm_inf() const {
        ff_float ret = 0;
        for(const auto& [x,v] : terms) {
            ret = MAX(std::abs(v), ret);
        }
        return ret;
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
    
    int degree() const {
        int deg = 0;
        for(const auto& [x,coeff] : terms) {
            if (coeff != ff_complex(0,0)) {
                deg = MAX(deg, x.degree());
            }
        }
        return deg;
    }

    bool is_zero(ff_float tolerance=1e-8) {
        for(const auto& [x,v] : terms) {
            if(std::abs(v) > tolerance) return false;
        }
        return true;
    }

    MajoranaPolynomial dagger() const {
        MajoranaPolynomial b;
        MajoranaMonomial xd;
        for(const auto& [x,v] : terms) {
            xd = x.dagger();
            b.terms[xd.majorana_string()] += xd.coefficient()*std::conj(v);
        }
        return b;
    }

    bool is_hermitian() const {
        MajoranaMonomial xd;
        for(const auto& [x,v] : terms) {
            bool x_hermitian = x.is_hermitian();
            if (x_hermitian && v.imag() != 0) return false;
            if (!x_hermitian && v.real() != 0) return false;
        }
        return true;
    }

    std::vector<int> support_set() const {
        ff_dbl_ulong supp = 0;
        for(auto& [x,v] : terms) {
            if(v != ff_complex(0,0)) supp |= x.alpha;
        }
        return supp.support();
    }

    void inplace_compress(const MajoranaString& x) {
        if(terms[x] == ff_complex(0,0)) {
            terms.erase(x);
        }
    }

    // In-place addition
    MajoranaPolynomial& operator+=(const ff_complex& b) { if(b != ff_complex(0,0)) { terms[MajoranaString()] += b; inplace_compress(MajoranaString()); } return *this; };
    MajoranaPolynomial& operator+=(const MajoranaString& b) { terms[b] += 1; inplace_compress(b); return *this; };
    MajoranaPolynomial& operator+=(const MajoranaMonomial& b) { terms[b.s] += b.coeff; inplace_compress(b.s); return *this; };
    MajoranaPolynomial& operator+=(const MajoranaPolynomial& b) {
        for(auto& [x,v] : b.terms) {
            terms[x] += v;
            inplace_compress(x);
        }
        return *this;
    }

    // In-place subtraction
    MajoranaPolynomial& operator-=(const ff_complex& b) { if(b != ff_complex(0,0)) { terms[MajoranaString()] -= b; inplace_compress(MajoranaString()); } return *this; };
    MajoranaPolynomial& operator-=(const MajoranaString& b) { terms[b] -= 1; inplace_compress(b); return *this; };
    MajoranaPolynomial& operator-=(const MajoranaMonomial& b) { terms[b.s] -= b.coeff; inplace_compress(b.s); return *this; };
    MajoranaPolynomial& operator-=(const MajoranaPolynomial& b) {
        for(auto& [x,v] : b.terms) {
            terms[x] -= v;
            inplace_compress(x);
        }
        return *this;
    }

    // In-place multiplication
    MajoranaPolynomial& operator*=(const ff_complex& b) { for(auto& [x,v] : terms) { terms[x] *= b; } return *this; }
    MajoranaPolynomial& operator*=(const MajoranaString&);
    MajoranaPolynomial& operator*=(const MajoranaPolynomial&);

    // In-place division
    MajoranaPolynomial& operator/=(const ff_complex& b) { for(auto& [x,v] : terms) { terms[x] /= b; } return *this; }

    std::string to_compact_string() const {
        return format_poly(*this, ff_config.max_terms_to_show, ff_config.max_line_length);
    }
    std::string to_string() const { return to_compact_string(); }

    MajoranaPolynomial compress(ff_float threshold=1e-8) const {
        MajoranaPolynomial b;
        for(const auto& [x,v] : terms) {
            if(std::abs(v) > threshold) {
                b.terms[x] = v;
            }
        }
        return b;
    }

    MajoranaPolynomial truncate(int k) const {
        MajoranaPolynomial b;
        for(const auto& [x,v] : terms) {
            if(x.degree() <= k) {
                b.terms[x] = v;
            }
        }
        return b;
    }

    // Computes the overlap of the MajoranaPolynomial with vacuum state in the Fock space basis
    // i.e. |0...0> where 0 is the occupation number of each site.
    ff_complex overlapwithvacuum() const {
        ff_dbl_ulong even_mask = ff_dbl_ulong::even_mask();
        ff_dbl_ulong odd_mask = ff_dbl_ulong::odd_mask();
        ff_complex exp_value(0.0,0.0);
        for(const auto& [x,v] : terms) {
            ff_dbl_ulong even_x = x.alpha & even_mask;
            ff_dbl_ulong odd_x = x.alpha & odd_mask;
            ff_complex factor(1.0,0.0);
            if((even_x << 1) == odd_x) {
                // x is paired, i.e., it is a subset of the form {2*i,2*i+1 for i \in I}
                // compute i^{number of pairs}
                factor = scalar_utils::Ipow(even_x.popcount());
                exp_value += v*factor;
            }
        }
        return exp_value;
    }

    bool commutes(const ff_complex& b) const { return true; }
    bool commutes(const MajoranaString& b) const;
    bool commutes(const MajoranaMonomial& b) const;
    bool commutes(const MajoranaPolynomial& b) const;

    ff_complex commutator(const ff_complex& b) const { return 0; }
    MajoranaPolynomial commutator(const MajoranaString& b) const;
    MajoranaPolynomial commutator(const MajoranaMonomial& b) const;
    MajoranaPolynomial commutator(const MajoranaPolynomial& b) const;

};

// -------------------------------------------------------------------------------

// Dagger operation
// Swaps creation and annihilation and adjust sign

MajoranaMonomial MajoranaString::dagger() const {
    int deg = degree();
    return MajoranaMonomial(alpha,deg*(deg-1)/2 % 2 == 0 ? 1 : -1);
}

MajoranaMonomial MajoranaMonomial::dagger() const {
    MajoranaMonomial s_dagger = s.dagger(); // s_dagger.coefficient() = +1 or -1
    return MajoranaMonomial(s_dagger.majorana_string(), s_dagger.coefficient() * std::conj(coeff));
}

// -------------------------------------------------------------------------------

// Algebra operation

void majorana_string_multiply(MajoranaString& x, int& m1p, const MajoranaString& y) {
    // x <- x*y
    m1p += merge_parity(y.alpha, x.alpha);
    // if(merge_parity(y.alpha, x.alpha) == -1) m1p += 1;
    x.alpha ^= y.alpha;
}

MajoranaMonomial operator*(const MajoranaString& x, const MajoranaString& y) {
    return MajoranaMonomial(x.alpha ^ y.alpha, scalar_utils::m1pow(merge_parity(y.alpha, x.alpha)));
    // return MajoranaMonomial(x.alpha ^ y.alpha, merge_parity(y.alpha, x.alpha));
}

// Overload operators

MajoranaMonomial operator*(const MajoranaString& b, const ff_complex& a) { return MajoranaMonomial(b, a); }
MajoranaMonomial operator*(const ff_complex& a, const MajoranaString& b) { return MajoranaMonomial(b, a); }

MajoranaMonomial operator*(const MajoranaMonomial& a, const ff_complex& b) { return MajoranaMonomial(a.s, a.coeff * b); }
MajoranaMonomial operator*(const ff_complex& a, const MajoranaMonomial& b) { return MajoranaMonomial(b.s, b.coeff * a); }

MajoranaPolynomial operator+(const ff_complex& a, const MajoranaString& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const ff_complex& a, const MajoranaMonomial& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const ff_complex& a, const MajoranaPolynomial& b) { MajoranaPolynomial c(a); c += b; return c; }

MajoranaPolynomial operator+(const MajoranaString& a, const ff_complex& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const MajoranaMonomial& a, const ff_complex& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const MajoranaPolynomial& a, const ff_complex& b) { MajoranaPolynomial c(a); c += b; return c; }

MajoranaPolynomial operator+(const MajoranaString& a, const MajoranaString& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const MajoranaString& a, const MajoranaMonomial& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const MajoranaString& a, const MajoranaPolynomial& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const MajoranaMonomial& a, const MajoranaString& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const MajoranaMonomial& a, const MajoranaMonomial& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const MajoranaMonomial& a, const MajoranaPolynomial& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const MajoranaPolynomial& a, const MajoranaString& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const MajoranaPolynomial& a, const MajoranaMonomial& b) { MajoranaPolynomial c(a); c += b; return c; }
MajoranaPolynomial operator+(const MajoranaPolynomial& a, const MajoranaPolynomial& b) { MajoranaPolynomial c(a); c += b; return c; }

MajoranaPolynomial operator-(const MajoranaString& a, const MajoranaString& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const MajoranaString& a, const MajoranaMonomial& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const MajoranaString& a, const MajoranaPolynomial& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const MajoranaMonomial& a, const MajoranaString& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const MajoranaMonomial& a, const MajoranaMonomial& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const MajoranaMonomial& a, const MajoranaPolynomial& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const MajoranaPolynomial& a, const MajoranaString& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const MajoranaPolynomial& a, const MajoranaMonomial& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const MajoranaPolynomial& a, const MajoranaPolynomial& b) { MajoranaPolynomial c(a); c -= b; return c; }

MajoranaPolynomial operator*(const MajoranaPolynomial& a, const ff_complex& b) { MajoranaPolynomial c(a); c *= b; return c; }
MajoranaPolynomial operator*(const ff_complex& b, const MajoranaPolynomial& a) { MajoranaPolynomial c(a); c *= b; return c; }
MajoranaPolynomial operator/(const MajoranaPolynomial& a, const ff_complex& b) { MajoranaPolynomial c(a); c /= b; return c; }

MajoranaMonomial operator-(const MajoranaString& a) { return MajoranaMonomial(a, -1); }
MajoranaMonomial operator-(const MajoranaMonomial& a) { return -1*a; }
MajoranaPolynomial operator-(const MajoranaPolynomial& a) { return -1*a; }

MajoranaPolynomial operator-(const ff_complex& a, const MajoranaString& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const ff_complex& a, const MajoranaMonomial& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const ff_complex& a, const MajoranaPolynomial& b) { MajoranaPolynomial c(a); c -= b; return c; }

MajoranaPolynomial operator-(const MajoranaString& a, const ff_complex& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const MajoranaMonomial& a, const ff_complex& b) { MajoranaPolynomial c(a); c -= b; return c; }
MajoranaPolynomial operator-(const MajoranaPolynomial& a, const ff_complex& b) { MajoranaPolynomial c(a); c -= b; return c; }



MajoranaPolynomial& MajoranaPolynomial::operator*=(const MajoranaPolynomial& b) {
    MajoranaPolynomialMap new_terms;
    MajoranaMonomial xy;
    for(auto& [x,v] : terms) {
        for(auto& [y,w] : b.terms) {
            xy = x*y;
            new_terms[xy.majorana_string()] += v*w*xy.coefficient();
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

MajoranaPolynomial& MajoranaPolynomial::operator*=(const MajoranaString& b) {
    (*this) *= MajoranaPolynomial(b);
    return *this;
}

MajoranaMonomial operator*(const MajoranaString& a, const MajoranaMonomial& b) { return b.coefficient() * (a*b.majorana_string()); }
MajoranaMonomial operator*(const MajoranaMonomial& a, const MajoranaString& b) { return a.coefficient() * (a.majorana_string()*b); }
MajoranaMonomial operator*(const MajoranaMonomial& a, const MajoranaMonomial& b) { return a.coefficient() * b.coefficient() * (a.majorana_string()*b.majorana_string()); }
MajoranaPolynomial operator*(const MajoranaPolynomial& a, const MajoranaPolynomial& b) { MajoranaPolynomial c(a); c *= b; return c; }

// -------------------------------------------------------------------------------

// Permutations of modes

MajoranaMonomial MajoranaString::permute(const std::vector<int>& map) const {
    auto [new_alpha,sgn1] = shuffle_with_parity(alpha,map);
    return MajoranaMonomial(new_alpha, sgn1);
}

MajoranaMonomial MajoranaMonomial::permute(const std::vector<int>& map) const { return coeff * s.permute(map); }

MajoranaPolynomial MajoranaPolynomial::permute(const std::vector<int>& map) const {
    MajoranaPolynomial c;
    for(auto& [x,v] : terms) {
        MajoranaMonomial x2 = x.permute(map);
        c.terms[x2.majorana_string()] = x2.coefficient() * v;
    }
    return c;
}

// -------------------------------------------------------------------------------

// Equality

bool operator==(const MajoranaPolynomial& a, const MajoranaPolynomial& b) {
    MajoranaPolynomial c = a-b;
    for(auto& [x,v] : c.terms) {
        if(v.real() != 0 || v.imag() != 0) return false;
    }
    return true;
}

bool operator==(const MajoranaString& a, const ff_complex& b) { return a.alpha == 0 && b == ff_complex(1,0); }
bool operator==(const MajoranaString& a, const MajoranaString& b) { return a.alpha == b.alpha; }
bool operator==(const MajoranaString& a, const MajoranaMonomial& b) { return a == b.s && b.coeff == ff_complex(1,0); }
bool operator==(const MajoranaString& a, const MajoranaPolynomial& b) { return MajoranaPolynomial(a) == b; }

bool operator==(const MajoranaMonomial& a, const ff_complex& b) { return a.s.alpha == 0 && a.coeff == b; }
bool operator==(const MajoranaMonomial& a, const MajoranaString& b) { return a.s == b && a.coeff == ff_complex(1,0); }
bool operator==(const MajoranaMonomial& a, const MajoranaMonomial& b) { return a.s == b.s && a.coeff == b.coeff; }
bool operator==(const MajoranaMonomial& a, const MajoranaPolynomial& b) { return MajoranaPolynomial(a) == b; }

bool operator==(const MajoranaPolynomial& a, const ff_complex& b) { return a == MajoranaPolynomial(b); }
bool operator==(const MajoranaPolynomial& a, const MajoranaString& b) { return a == MajoranaPolynomial(b); }
bool operator==(const MajoranaPolynomial& a, const MajoranaMonomial& b) { return a == MajoranaPolynomial(b); }

// -------------------------------------------------------------------------------

// Printing

std::ostream& operator<<(std::ostream& os, const MajoranaString& a) { return os << a.to_compact_string(); }
std::ostream& operator<<(std::ostream& os, const MajoranaMonomial& a) { return os << a.to_compact_string(); }
std::ostream& operator<<(std::ostream& os, const MajoranaPolynomial& p) { return os << p.to_compact_string(); }

// -------------------------------------------------------------------------------

// Commutation

MajoranaMonomial MajoranaString::commutator(const MajoranaString& b) const {
    if (commutes(b)) {
        return MajoranaMonomial();
    } else {
        return 2*((*this)*b);
    }
}

MajoranaMonomial MajoranaString::commutator(const MajoranaMonomial& b) const { return b.coeff * commutator(b.s); }

MajoranaMonomial MajoranaMonomial::commutator(const MajoranaString& b) const { return coeff * s.commutator(b); }
MajoranaMonomial MajoranaMonomial::commutator(const MajoranaMonomial& b) const { return (coeff*b.coeff)*s.commutator(b.s); }

MajoranaPolynomial MajoranaPolynomial::commutator(const MajoranaPolynomial& b) const {
    MajoranaPolynomial c;
    for(auto& [x,v] : terms) {
        for(auto& [y,w] : b.terms) {
            if(!x.commutes(y)) {
                MajoranaMonomial comm = 2*(x*y);
                c.terms[comm.s] += v * w * comm.coeff;
            }
        }
    }
    return c;
}


MajoranaPolynomial MajoranaPolynomial::commutator(const MajoranaString& b) const { return commutator(MajoranaPolynomial(b)); }
MajoranaPolynomial MajoranaPolynomial::commutator(const MajoranaMonomial& b) const { return commutator(MajoranaPolynomial(b)); }

MajoranaPolynomial MajoranaString::commutator(const MajoranaPolynomial& b) const { return MajoranaPolynomial(*this).commutator(b); }
MajoranaPolynomial MajoranaMonomial::commutator(const MajoranaPolynomial& b) const { return MajoranaPolynomial(*this).commutator(b); }

bool MajoranaString::commutes(const MajoranaMonomial& b) const { return commutes(b.s); }
bool MajoranaString::commutes(const MajoranaPolynomial& b) const { return b.commutator(*this).is_zero(0); }
bool MajoranaMonomial::commutes(const MajoranaPolynomial& b) const { return b.commutator(*this).is_zero(0); }

bool MajoranaPolynomial::commutes(const MajoranaString& b) const { return commutator(b).is_zero(0); }
bool MajoranaPolynomial::commutes(const MajoranaMonomial& b) const { return commutator(b).is_zero(0); }
bool MajoranaPolynomial::commutes(const MajoranaPolynomial& b) const { return commutator(b).is_zero(0); }

}