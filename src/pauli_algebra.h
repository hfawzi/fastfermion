/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "hashmap/unordered_dense.h"
#include "matrix.h"
#include "format_poly.h" // for format_poly
#include "parse.h"

#include <map>
#include <functional>

namespace fastfermion {

// Forward declaration of main structs defined in this file
struct PauliString;
struct PauliMonomial;
struct PauliPolynomial;

struct PauliStringHash;
using PauliPolynomialMap = ankerl::unordered_dense::map<PauliString, ff_complex, PauliStringHash>;

struct PauliString {
    ff_ulong xory; // i'th bit is 1 if i'th location is X or Y
    ff_ulong yorz; // i'th bit is 1 if i'th location is Y or Z

    // Constructors

    PauliString() : xory(0), yorz(0) { }
    PauliString(std::uint64_t xory, std::uint64_t yorz) : xory(xory), yorz(yorz) { }
    PauliString(ff_ulong xory, ff_ulong yorz) : xory(xory), yorz(yorz) { }
    PauliString(const PauliString& a) : xory(a.xory), yorz(a.yorz) { }
    PauliString(const std::vector<std::pair<int,char>>& str) : xory(0), yorz(0) {
        // Initialize using a sequence of location and actions (a la QubitOperator from OpenFermion)
        _init_from_indices(str);
    }

    PauliString(const std::vector<int>& loc, const std::vector<char>& actions) : xory(0), yorz(0) {
        // Initialize using a sequence of location and actions
        // This is exactly the same as the above constructor, but here locations
        // and actions are two separate vectors, rather than being the same
        // Most useful for low-degree monomials
        // Example: PauliString(vector<int>{0,4,17}, vector<char>{'X','Z','Y'})
        // TODO: remove this constructor, it doesn't add anything compared to the above
        if(loc.size() != actions.size()) {
            throw_error("Invalid arguments (both arguments should have the same size)");
        }
        for(std::size_t i=0; i<loc.size(); i++) {
            if(loc[i] >= 0 && loc[i] < ff_ulong::DIGITS) {
                if (actions[i] == 'I' || actions[i] == 'X' || actions[i] == 'Y' || actions[i] == 'Z') {
                    if (actions[i] == 'X' || actions[i] == 'Y') xory.set(loc[i]);
                    if (actions[i] == 'Y' || actions[i] == 'Z') yorz.set(loc[i]);
                } else {
                    throw_error("Invalid PauliString (symbol '" << actions[i] << "' not recognized)");
                }
            } else {
                throw_error("Invalid PauliString (qubit index " << loc[i] << " invalid, should be between 0 and " << ff_ulong::DIGITS-1 << ")");
            }
        }
    }

    PauliString(const std::string& str) : xory(0), yorz(0) {
        // Two types of strings are allowed:
        // either something of the form PauliString("XXIYZ")
        // or of the form PauliString("X0 Y1 Z4")
        // The first one only has characters in {I,X,Y,Z}, whereas the second
        // one must have a digit

        // Check which type of string it is
        bool type1 = true;
        for(const char& c : str) {
            if(!(c == 'X' || c == 'Y' || c == 'Z' || c == 'I')) {
                type1 = false;
                break;
            }
        }
        if(type1) {
            int str_size = str.size();
            for(int i=0; i<str_size; i++) {
                if (str[i] == 'X' || str[i] == 'Y') xory.set(i);
                if (str[i] == 'Y' || str[i] == 'Z') yorz.set(i);
            }
        } else {
            _init_from_indices(_parse_pauli_string(str));
        }
    }
    
    // End constructors

    void _init_from_indices(const std::vector<std::pair<int,char>>& str) {
        int str_size = str.size();
        for(int i=0; i<str_size; ++i) {
            auto [loc,action] = str[i];
            // Check that loc doesn't appear twice
            for(int j=0; j<str_size; ++j) {
                if(j != i && str[j].first == loc) {
                    throw_error("Invalid PauliString (qubit index " << loc << " appears multiple times)");
                }
            }
            if(loc >= 0 && loc < ff_ulong::DIGITS) {
                if (action == 'I' || action == 'X' || action == 'Y' || action == 'Z') {
                    if (action == 'X' || action == 'Y') xory.set(loc);
                    if (action == 'Y' || action == 'Z') yorz.set(loc);
                } else {
                    throw_error("Invalid PauliString (symbol '" << action << "' not recognized)");
                }
            } else {
                throw_error("Invalid PauliString (qubit index " << loc << " invalid, should be between 0 and " << ff_ulong::DIGITS-1 << ")");
            }
        }
    }

    std::string to_string(int n=0) const {
        // If parameter n is given, will return string of length at least n
        // Returns a string of I,X,Y,Z
        // e.g., XIIXZIIX
        int len = MAX(extent(),n);
        std::string str(len,'I');
        for(const auto& [pos,action] : indices()) {
            str[pos] = action;
        }
        return str;
    }
    std::string to_compact_string() const {
        // Returns a string representation of the form X1 Y2 Z4
        if(xory == 0 && yorz == 0) {
            // identity
            return "I";
        } else {
            // Returns a representation of the form X0 Y2 Z4
            std::vector<std::pair<int,char>> s_and_a = indices();
            std::string str = "";
            for(std::size_t i=0; i<s_and_a.size(); i++) {
                const auto [l,sigma] = s_and_a[i];
                if(i != 0) str += " ";
                str += sigma + std::to_string(l);
            }
            return str;
        }
    }
    ff_complex coefficient() const { return 1; }
    ff_ulong support() const { return (xory | yorz); }
    std::vector<int> support_set() const {
        return (xory | yorz).support();
    }
    std::vector<std::pair<int,char>> indices() const {
        // Returns a vector of the form { (0,'X'), (1,'Z'), (3,'Y') }
        ff_ulong a1 = xory | yorz;
        // Iterate over support
        std::vector<std::pair<int,char>> ret;
        for(int pos=a1.begin(); pos != a1.end(); pos = a1.next(pos)) {
            bool p = xory.at(pos);
            bool q = yorz.at(pos);
            if(p && q) ret.push_back(std::make_pair(pos,'Y'));
            else if (p && !q) ret.push_back(std::make_pair(pos,'X'));
            else if (!p && q) ret.push_back(std::make_pair(pos,'Z'));
        }
        return ret;
    }
    int degree_x() const { return (xory & ~yorz).popcount(); }
    int degree_y() const { return (xory & yorz).popcount(); }
    int degree_z() const { return (~xory & yorz).popcount(); }
    int degree_total() const { return (xory | yorz).popcount(); };
    int degree() const { return degree_total(); }
    int degree(const char& v) const {
        if (v == 'X' || v == 'x') return degree_x();
        else if (v == 'Y' || v == 'y') return degree_y();
        else if (v == 'Z' || v == 'z') return degree_z();
        else if (v == '\0') return degree_total();
        else throw_error("Invalid argument: should be either 'X', 'Y', or 'Z' or empty char");
    }
    int extent() const {
        return ff_ulong::DIGITS - (xory | yorz).countl_zero();
    };
    bool is_skew_symmetric() const {
        // Return true if PauliString has an odd number of Y's
        return degree_y()%2 == 1;
    }
    std::uint64_t hash() const {
        auto h = std::uint64_t{};
        h = ankerl::unordered_dense::tuple_hash_helper<>::mix64(h,xory.hash());
        h = ankerl::unordered_dense::tuple_hash_helper<>::mix64(h,yorz.hash());
        return h;
    }

    
    PauliString permute(const std::vector<int>& map) const { return PauliString(move_bits(xory, map), move_bits(yorz, map)); }


    bool commutes(const ff_complex& b) const { return true; }
    bool commutes(const PauliString& b) const {
        return ((xory & b.yorz).popcount() % 2 == (yorz & b.xory).popcount() % 2);
    }
    bool commutes(const PauliMonomial& b) const;
    bool commutes(const PauliPolynomial& b) const;

    ff_complex commutator(const ff_complex& b) const { return 0; };
    PauliMonomial commutator(const PauliString& b) const;
    PauliMonomial commutator(const PauliMonomial& b) const;
    PauliPolynomial commutator(const PauliPolynomial& b) const;


    CSCMatrix<ff_complex> sparse(int n, int nup) const;
    CSCMatrix<ff_complex> sparse(int n) const;
    CSCMatrix<ff_complex> sparse() const {
        return sparse(extent());
    }
};


void pauli_string_multiply(PauliString& a, int& jpow, const PauliString& b) {
    // Updates a <- a*b
    jpow += (a.xory & a.yorz).popcount() + (b.xory & b.yorz).popcount() + 2*(a.yorz & b.xory).popcount();
    a.xory ^= b.xory;
    a.yorz ^= b.yorz;
    jpow -= (a.xory & a.yorz).popcount();
}

// Define a total order on Pauli Strings
bool operator<(const PauliString& a, const PauliString& b) {
    return std::forward_as_tuple(a.xory | a.yorz, a.yorz, a.xory) < std::forward_as_tuple(b.xory | b.yorz, b.yorz, b.xory);
}

struct PauliStringHash {
    using is_avalanching = void;
    std::uint64_t operator()(const PauliString& a) const {
        return a.hash();
    }
};

// A PauliMonomial is a PauliString together with a coefficient
struct PauliMonomial {

    PauliString s;
    ff_complex coeff;

    PauliMonomial() : s(), coeff(0) { } // NOT SURE THIS SHOULD BE THE RIGHT BEHAVIOR: Because then PauliString() == I and PauliMonomial() == 0
    PauliMonomial(const PauliMonomial& a) : s(a.s), coeff(a.coeff) { }
    PauliMonomial(ff_ulong xory, ff_ulong yorz, ff_complex coeff) : s(xory, yorz), coeff(coeff) { }
    PauliMonomial(const PauliString& s) : s(s), coeff(1) { }
    PauliMonomial(const PauliString& s, ff_complex coeff) : s(s), coeff(coeff) { }
    PauliMonomial(std::string s_str) : s(s_str), coeff(1) { }
    PauliMonomial(std::string s_str, ff_complex coeff) : s(s_str), coeff(coeff) { }
    PauliMonomial(const std::vector<int>& loc, const std::vector<char>& actions) : s(loc,actions), coeff(1) { }
    PauliMonomial(const std::vector<int>& loc, const std::vector<char>& actions, ff_complex coeff) : s(loc,actions), coeff(coeff) { }

    // Redundant, consider removing one
    PauliString pauli_string() const { return s; }
    PauliString monomial_string() const { return s; }
    ff_complex coefficient() const { return coeff; }
    ff_ulong support() const { return s.support(); }
    std::vector<int> support_set() const { return s.support_set(); }
    std::vector<std::pair<int,char>> indices() const { return s.indices(); }
    int extent() const { return s.extent(); }
    PauliMonomial permute(const std::vector<int>& map) const { return PauliMonomial(s.permute(map), coeff); }
    PauliMonomial dagger() const { return PauliMonomial(s, std::conj(coeff)); }
    int degree_x() const { return s.degree_x(); }
    int degree_y() const { return s.degree_y(); }
    int degree_z() const { return s.degree_z(); }
    int degree_total() const { return s.degree_total(); }
    std::string to_string(int n=0) const { return std::to_string(coeff.real()) + "+" + std::to_string(coeff.imag()) + "i " + s.to_string(n); }
    std::string to_compact_string() const { return format_complex(coeff) + " " + s.to_compact_string(); }
    PauliMonomial& operator*=(const ff_float b) { coeff *= b; return *this; }
    bool commutes(const ff_complex& b) const { return true; }
    bool commutes(const PauliString& b) const { return s.commutes(b); }
    bool commutes(const PauliMonomial& b) const { return s.commutes(b.s); }
    bool commutes(const PauliPolynomial& b) const;
    ff_complex commutator(const ff_complex& b) const { return 0; };
    PauliMonomial commutator(const PauliString& b) const;
    PauliMonomial commutator(const PauliMonomial& b) const;
    PauliPolynomial commutator(const PauliPolynomial& b) const;
    CSCMatrix<ff_complex> sparse() const { return sparse(extent()); }
    CSCMatrix<ff_complex> sparse(int n) const;
    CSCMatrix<ff_complex> sparse(int n, int nup) const;
};



struct PauliPolynomial {
    PauliPolynomialMap terms;
    PauliPolynomial() {}
    explicit PauliPolynomial(const ff_complex& scal) { terms[PauliString()] = scal; }
    PauliPolynomial(const PauliString& a) { terms[a] = 1; }
    PauliPolynomial(const PauliMonomial& a) { terms[a.s] = a.coeff; }
    PauliPolynomial(const PauliPolynomial& a) : terms(a.terms) { };
    PauliPolynomial(const PauliPolynomialMap& terms) : terms(terms) { };
    PauliPolynomial(const std::vector<std::pair<int,char>>& str, const ff_complex& coeff) {
        // Initialize using a sequence of location and actions (a la QubitOperator from OpenFermion)
        // e.g., str = ((0,'X'),(1,'Y'))
        // Note that str may contain the same location multiple times. In this case we have to keep track
        // of the sign.
        PauliString u;
        int jpow = 0;
        int str_size = str.size();
        for(int i=0; i<str_size; ++i) {
            pauli_string_multiply(u, jpow, PauliString(std::vector<std::pair<int,char>>{{str[i].first, str[i].second}}));
        }
        terms[u] = scalar_utils::Ipow(jpow) * coeff;
    }
    PauliPolynomial(const std::vector<std::pair<int,char>>& str) : PauliPolynomial(str,1) {}
    PauliPolynomial(const std::vector<PauliString>& a, const std::vector<ff_complex>& coeffs) {
        assert(coeffs.size() == a.size());
        for(std::size_t i=0; i<a.size(); i++) if(coeffs[i] != ff_complex(0,0)) terms[a[i]] += coeffs[i];
    }
    PauliPolynomial(const std::vector<PauliString>& a, const std::vector<ff_float>& coeffs) {
        assert(coeffs.size() == a.size());
        for(std::size_t i=0; i<a.size(); i++) if(coeffs[i] != 0) terms[a[i]] += coeffs[i];
    }

    ff_complex coefficient(const PauliString& x) const {
        auto it = terms.find(x);
        if(it == terms.end()) {
            return ff_complex(0,0);
        } else {
            return it->second;
        }
    }

    void inplace_compress(const PauliString& x) {
        if(terms[x] == ff_complex(0,0)) {
            terms.erase(x);
        }
    }

    // In-place addition
    PauliPolynomial& operator+=(const ff_complex& b) {
        if(b != ff_complex(0,0)) {
            terms[PauliString()] += b;
            inplace_compress(PauliString());
        }
        return *this;
    };
    PauliPolynomial& operator+=(const PauliString& b) { terms[b] += 1; inplace_compress(b); return *this; };
    PauliPolynomial& operator+=(const PauliMonomial& b) { terms[b.s] += b.coeff; inplace_compress(b.s); return *this; };
    PauliPolynomial& operator+=(const PauliPolynomial& b) {
        for(auto& [x,v] : b.terms) {
            terms[x] += v;
            inplace_compress(x);
        }
        return *this;
    }

    // In-place subtraction
    PauliPolynomial& operator-=(const ff_complex& b) { if(b != ff_complex(0,0)) { terms[PauliString()] -= b; inplace_compress(PauliString()); } return *this; };
    PauliPolynomial& operator-=(const PauliString& b) { terms[b] -= 1; inplace_compress(b); return *this; };
    PauliPolynomial& operator-=(const PauliMonomial& b) { terms[b.s] -= b.coeff; inplace_compress(b.s); return *this; };
    PauliPolynomial& operator-=(const PauliPolynomial& b) {
        for(auto& [x,v] : b.terms) {
            terms[x] -= v;
            inplace_compress(x);
        }
        return *this;
    }

    // In-place multiplication
    PauliPolynomial& operator*=(const ff_complex& b) { for(auto& [x,v] : terms) { terms[x] *= b; } return *this; }
    PauliPolynomial& operator*=(const PauliString& y);
    PauliPolynomial& operator*=(const PauliMonomial& y);
    PauliPolynomial& operator*=(const PauliPolynomial& b);

    // In-place division
    PauliPolynomial& operator/=(const ff_complex& b) { for(auto& [x,v] : terms) { terms[x] /= b; } return *this; }

    
    // Permutations
    PauliPolynomial permute(const std::vector<int>& map) const {
        PauliPolynomial c;
        for(auto& [x,v] : terms) {
            c.terms[x.permute(map)] = v;
        }
        return c;
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
    std::string to_string() const {
        std::string ret = "";
        bool first_term = true;
        for(auto& [x,v] : terms) {
            if(!first_term) {
                ret += " + ";
            }
            ret += PauliMonomial(x,v).to_string();
            first_term = false;
        }
        return ret;
    }
    std::string to_compact_string() const {
        return format_poly(*this, ff_config.max_terms_to_show, ff_config.max_line_length);
    }
    std::vector<int> support_set() const {
        ff_ulong supp = 0;
        for(auto& [x,v] : terms) {
            if(v != ff_complex(0,0)) supp |= x.support();
        }
        return supp.support();
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
    PauliPolynomial dagger() const {
        PauliPolynomial b;
        for(const auto& [x,v] : terms) { b.terms[x] = std::conj(v); }
        return b;
    }

    bool is_hermitian() const {
        for(const auto& [x,v] : terms) {
            if(v.imag() != 0) return false;
        }
        return true;
    }

    PauliPolynomial compress(ff_float threshold=1e-8) const {
        PauliPolynomial b;
        for(const auto& [x,v] : terms) {
            if(std::abs(v) > threshold) {
                b.terms[x] = v;
            }
        }
        return b;
    }

    PauliPolynomial truncate(int k) const {
        PauliPolynomial b;
        for(const auto& [x,v] : terms) {
            if(x.degree_total() <= k) {
                b.terms[x] = v;
            }
        }
        return b;
    }

    int degree(const char& v) const {
        int deg = 0;
        for(const auto& [x,coeff] : terms) {
            if (coeff != ff_complex(0,0)) {
                deg = MAX(deg, x.degree(v));
            }
        }
        return deg;
    }

    int degree() const { return degree('\0'); }

    ff_complex overlapwithzero() const {
        ff_complex ret = 0;
        for(const auto& [pstr,psval] : terms) {
            if(pstr.degree_x() == 0 && pstr.degree_y() == 0) {
                ret += psval;
            }
        }
        return ret;
    }

    bool commutes(const ff_complex& b) const { return true; }
    bool commutes(const PauliString& b) const;
    bool commutes(const PauliMonomial& b) const;
    bool commutes(const PauliPolynomial& b) const;

    ff_complex commutator(const ff_complex& b) const { return 0; }
    PauliPolynomial commutator(const PauliString& b) const;
    PauliPolynomial commutator(const PauliMonomial& b) const;
    PauliPolynomial commutator(const PauliPolynomial& b) const;

    CSCMatrix<ff_complex> sparse(int n) const;
    CSCMatrix<ff_complex> sparse(int n, int nup) const;
    CSCMatrix<ff_complex> sparse() const { return sparse(extent()); }

};

// -------------------------------------------------------------------------------------------------------

// Product
PauliMonomial operator*(const PauliString& a, const PauliString& b) {
    PauliString u(a);
    int jpow = 0;
    pauli_string_multiply(u,jpow,b);
    return PauliMonomial(u, scalar_utils::Ipow(jpow));
}

// Addition

PauliPolynomial operator+(const PauliString& a, const ff_complex& b) { PauliPolynomial c(a); c += b; return c; }
PauliPolynomial operator+(const PauliString& a, const PauliString& b) { PauliPolynomial c(a); c += b; return c; }
PauliPolynomial operator+(const PauliString& a, const PauliMonomial& b) { PauliPolynomial c(a); c += b; return c; }
PauliPolynomial operator+(const PauliString& a, const PauliPolynomial& b) { PauliPolynomial c(a); c += b; return c; }

PauliPolynomial operator+(const PauliMonomial& a, const ff_complex& b) { PauliPolynomial c(a); c += b; return c; }
PauliPolynomial operator+(const PauliMonomial& a, const PauliString& b) { PauliPolynomial c(a); c += b; return c; }
PauliPolynomial operator+(const PauliMonomial& a, const PauliMonomial& b) { PauliPolynomial c(a); c += b; return c; }
PauliPolynomial operator+(const PauliMonomial& a, const PauliPolynomial& b) { PauliPolynomial c(a); c += b; return c; }

PauliPolynomial operator+(const ff_complex& a, const PauliPolynomial& b) { PauliPolynomial c(a); c += b; return c; }
PauliPolynomial operator+(const PauliPolynomial& a, const ff_complex& b) { PauliPolynomial c(a); c += b; return c; }
PauliPolynomial operator+(const PauliPolynomial& a, const PauliString& b) { PauliPolynomial c(a); c += b; return c; }
PauliPolynomial operator+(const PauliPolynomial& a, const PauliMonomial& b) { PauliPolynomial c(a); c += b; return c; }
PauliPolynomial operator+(const PauliPolynomial& a, const PauliPolynomial& b) { PauliPolynomial c(a); c += b; return c; }

// Subtraction


PauliPolynomial operator-(const PauliString& a, const ff_complex& b) { PauliPolynomial c(a); c -= b; return c; }
PauliPolynomial operator-(const PauliString& a, const PauliString& b) { PauliPolynomial c(a); c -= b; return c; }
PauliPolynomial operator-(const PauliString& a, const PauliMonomial& b) { PauliPolynomial c(a); c -= b; return c; }
PauliPolynomial operator-(const PauliString& a, const PauliPolynomial& b) { PauliPolynomial c(a); c -= b; return c; }

PauliPolynomial operator-(const PauliMonomial& a, const ff_complex& b) { PauliPolynomial c(a); c -= b; return c; }
PauliPolynomial operator-(const PauliMonomial& a, const PauliString& b) { PauliPolynomial c(a); c -= b; return c; }
PauliPolynomial operator-(const PauliMonomial& a, const PauliMonomial& b) { PauliPolynomial c(a); c -= b; return c; }
PauliPolynomial operator-(const PauliMonomial& a, const PauliPolynomial& b) { PauliPolynomial c(a); c -= b; return c; }

PauliPolynomial operator-(const ff_complex& a, const PauliPolynomial& b) { PauliPolynomial c(a); c -= b; return c; }
PauliPolynomial operator-(const PauliPolynomial& a, const ff_complex& b) { PauliPolynomial c(a); c -= b; return c; }
PauliPolynomial operator-(const PauliPolynomial& a, const PauliString& b) { PauliPolynomial c(a); c -= b; return c; }
PauliPolynomial operator-(const PauliPolynomial& a, const PauliMonomial& b) { PauliPolynomial c(a); c -= b; return c; }
PauliPolynomial operator-(const PauliPolynomial& a, const PauliPolynomial& b) { PauliPolynomial c(a); c -= b; return c; }


bool operator==(const PauliPolynomial& a, const PauliPolynomial& b) {
    PauliPolynomial c = a-b;
    for(auto& [x,v] : c.terms) {
        if(v.real() != 0 || v.imag() != 0) return false;
    }
    return true;
}
bool operator==(const PauliPolynomial& a, const ff_complex& b) { return a == PauliPolynomial(b); }
bool operator==(const PauliPolynomial& a, const PauliString& b) { return a == PauliPolynomial(b); }
bool operator==(const PauliPolynomial& a, const PauliMonomial& b) { return a == PauliPolynomial(b); }

// Unary negation

PauliMonomial operator-(const PauliString& a) { return PauliMonomial(a,-1); };
PauliMonomial operator-(const PauliMonomial& a) { return PauliMonomial(a.pauli_string(), -a.coefficient()); };
PauliPolynomial operator-(const PauliPolynomial& a) { PauliPolynomial c(a); c *= -1; return c; };

// Multiplication by scalar

PauliMonomial operator*(const PauliMonomial& a, const ff_complex& b) { return PauliMonomial(a.s, a.coeff * b); }
PauliMonomial operator*(const ff_complex& a, const PauliMonomial& b) { return PauliMonomial(b.s, b.coeff * a); }

PauliMonomial operator*(const ff_complex& a, const PauliString& b) { return PauliMonomial(b, a); }
PauliMonomial operator*(const PauliString& b, const ff_complex& a) { return PauliMonomial(b, a); }

PauliPolynomial operator*(const PauliPolynomial& a, const ff_complex& b) { PauliPolynomial c(a); c *= b; return c; }
PauliPolynomial operator*(const ff_complex& b, const PauliPolynomial& a) { PauliPolynomial c(a); c *= b; return c; }

// Division by scalar

PauliPolynomial operator/(const PauliPolynomial& a, const ff_complex& b) { PauliPolynomial c(a); c /= b; return c; }

// Multiplication by another operator

PauliMonomial operator*(const PauliMonomial& a, const PauliMonomial& b) {
    PauliMonomial s2 = a.s * b.s;
    return PauliMonomial(s2.s, s2.coeff * a.coeff * b.coeff);
}

PauliMonomial operator*(const PauliString& a, const PauliMonomial& b) { return PauliMonomial(a)*b; };
PauliPolynomial operator*(const PauliString& a, const PauliPolynomial& b) { PauliPolynomial c(a); c *= b; return c; };

PauliMonomial operator*(const PauliMonomial& a, const PauliString& b) { return a*PauliMonomial(b); };
PauliPolynomial operator*(const PauliMonomial& a, const PauliPolynomial& b) { PauliPolynomial c(a); c *= b; return c; };

PauliPolynomial operator*(const PauliPolynomial& a, const PauliString& b) { PauliPolynomial c(a); c *= b; return c; }
PauliPolynomial operator*(const PauliPolynomial& a, const PauliMonomial& b) { PauliPolynomial c(a); c *= b; return c; }
PauliPolynomial operator*(const PauliPolynomial& a, const PauliPolynomial& b) { PauliPolynomial c(a); c *= b; return c; }

// In-place operations
PauliPolynomial& PauliPolynomial::operator*=(const PauliPolynomial& b) {
    PauliPolynomialMap new_terms;
    PauliMonomial xy;
    for(auto& [x,v] : terms) {
        for(auto& [y,w] : b.terms) {
            xy = x*y;
            new_terms[xy.pauli_string()] += xy.coefficient()*v*w;
        }
    }
    terms.clear();
    for(const auto& [x,v] : new_terms) {
        if(v != ff_complex(0,0)) {
            terms[x] = v;
        }
    }
    return *this;
}
PauliPolynomial& PauliPolynomial::operator*=(const PauliString& y) {
    (*this) *= PauliPolynomial(y);
    return *this;
}
PauliPolynomial& PauliPolynomial::operator*=(const PauliMonomial& y) {
    (*this) *= PauliPolynomial(y);
    return *this;
}



// Equality



bool operator==(const PauliString& a, const ff_complex& b) { return a.xory == 0 && a.yorz == 0 && b == ff_complex(1,0); }
bool operator==(const PauliString& a, const PauliString& b) { return a.xory == b.xory && a.yorz == b.yorz; }
bool operator==(const PauliString& a, const PauliMonomial& b) { return a == b.s && b.coeff == ff_complex(1,0); }
bool operator==(const PauliString& a, const PauliPolynomial& b) { return PauliPolynomial(a) == b; }

bool operator==(const PauliMonomial& a, const ff_complex& b) { return a.s.xory == 0 && a.s.yorz == 0 && a.coeff == b; }
bool operator==(const PauliMonomial& a, const PauliString& b) { return a.s == b && a.coeff == ff_complex(1,0); }
bool operator==(const PauliMonomial& a, const PauliMonomial& b) { return a.s == b.s && a.coeff == b.coeff; }
bool operator==(const PauliMonomial& a, const PauliPolynomial& b) { return PauliPolynomial(a) == b; }





// -------------------------------------------------------------------------------------------------------

// Commutation

bool PauliString::commutes(const PauliMonomial& b) const { return b.commutes(*this); }

PauliMonomial PauliString::commutator(const PauliString& b) const {
    // Returns the value of the commutator given two PauliStrings a and b
    if (commutes(b)) {
        // Zero monomial
        // [a,b] = 0
        return PauliMonomial();
    } else {
        // [a,b] = 2*a*b if [a,b] \neq 0
        return 2*((*this)*b);
    }
}
PauliMonomial PauliString::commutator(const PauliMonomial& b) const { return b.coeff*commutator(b.s); }
PauliMonomial PauliMonomial::commutator(const PauliString& b) const { return coeff * s.commutator(b); }
PauliMonomial PauliMonomial::commutator(const PauliMonomial& b) const { return (coeff * b.coeff) * s.commutator(b.s); }

PauliPolynomial PauliPolynomial::commutator(const PauliPolynomial& b) const {
    // Returns the value of the commutator given two PauliPolynomials a (= *this) and b
    PauliPolynomial c;
    for(auto& [x,v] : terms) {
        for(auto& [y,w] : b.terms) {
            if(!x.commutes(y)) {
                PauliMonomial comm = 2*x*y;
                c.terms[comm.s] += v*w*comm.coeff;
            }
        }
    }
    return c;
}
PauliPolynomial PauliPolynomial::commutator(const PauliString& b) const { return commutator(PauliPolynomial(b)); }
PauliPolynomial PauliPolynomial::commutator(const PauliMonomial& b) const { return commutator(PauliPolynomial(b)); }

PauliPolynomial PauliString::commutator(const PauliPolynomial& b) const { return PauliPolynomial(*this).commutator(b); }
PauliPolynomial PauliMonomial::commutator(const PauliPolynomial& b) const { return PauliPolynomial(*this).commutator(b); }

bool PauliPolynomial::commutes(const PauliString& b) const { return commutator(b).is_zero(0); }
bool PauliPolynomial::commutes(const PauliMonomial& b) const { return commutator(b).is_zero(0); }
bool PauliPolynomial::commutes(const PauliPolynomial& b) const { return commutator(b).is_zero(0); }

bool PauliString::commutes(const PauliPolynomial& b) const { return commutator(b).is_zero(0); }
bool PauliMonomial::commutes(const PauliPolynomial& b) const { return commutator(b).is_zero(0); }


// -------------------------------------------------------------------------------

// Printing

std::ostream& operator<<(std::ostream& os, const PauliString& a) { return os << a.to_compact_string(); }
std::ostream& operator<<(std::ostream& os, const PauliMonomial& a) { return os << a.to_compact_string(); }
std::ostream& operator<<(std::ostream& os, const PauliPolynomial& p) { return os << p.to_compact_string(); }

}