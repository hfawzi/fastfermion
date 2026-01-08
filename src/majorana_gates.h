/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "format_poly.h"
#include "majorana_algebra.h"

namespace fastfermion {

namespace majorana_gates {

// MajoranaRotation
struct MROT {
    // Represents a unitary U = e^{-i theta/2 M}
    // where M = i^{r}*P where
    // where P is a MajoranaString
    // and r =  0 if P is Hermitian (deg(P)=0 or 1 mod 4)
    //          1 else
    // The coefficient i^{r} ensures that i^{r}*P is Hermitian
    // so that U is unitary
    MajoranaString ms;
    ff_float theta;
    bool _r;
    MROT() : ms(), theta(0), _r(0) { }
    MROT(const MajoranaString& ms, const ff_float& theta) :
        ms(ms), theta(theta), _r( !(ms.is_hermitian()) ) {
    }
    MROT(const MajoranaMonomial& mm, const ff_float& t) {
        // e^{-i t/2 M} where M is a Hermitian monomial
        if(mm.is_hermitian()) {
            ms = mm.s;
            _r = !mm.s.is_hermitian();
            if(_r) {
                // Majorana String is not Hermitian, so necessarily mm.coeff is purely imaginary
                theta = mm.coeff.imag()*t;
            } else {
                // Majorana String is Hermitian, so necessarily mm.coeff is purely real
                theta = mm.coeff.real()*t;
            }
        } else {
            throw_error("Supplied Majorana monomial " << format_complex(mm.coeff) << "*(" << mm.s.to_compact_string() << ") is not Hermitian");
        }
    }
    MROT(const MajoranaString& ms, const ff_complex& coeff, const ff_float& t) : MROT(MajoranaMonomial(ms,coeff),t) { }
    std::string to_string() const {
        return std::string("MROT(") + (_r ? "1j " : "") + ms.to_compact_string() + ", theta=" + std::to_string(theta) + ") = " + 
              (_r ? ("e^{" + std::to_string(theta/2) + " " + ms.to_compact_string() + "}")
                  : ("e^{-i " + std::to_string(theta/2) + " " + ms.to_compact_string() + "}"));
    }
    MajoranaPolynomial aspoly() const {
        // e^{-i theta/2 M} = cos(theta/2) - 1i*sin(theta/2)*M
        // since M^2 = 1
        MajoranaPolynomial poly;
        poly.terms[MajoranaString()] = std::cos(theta/2);
        poly.terms[ms] += _r ? ff_complex(std::sin(theta/2),0) : ff_complex(0,-std::sin(theta/2));
        return poly;
    }

    
    template<class FilterPred>
    void apply_inplace(MajoranaPolynomial& poly, FilterPred filter_pred) const {
        // Applies in-place operation poly <- U^{dagger} poly U
        //
        // We use the fact that for a MajoranaString x,
        //   MROT_{ms,theta}(x) = x                                 if ms and x commute
        //                      = cos(theta)*x + i*sin(theta)*M*x   else
        // where M = i^r * ms is a Hermitian Majorana monomial

        // o_new will hold all the new terms that will be added to poly
        // which are of the form i*sin(theta)*M*x
        // where x ranges over all terms in poly that do not commute with ms
        std::vector<std::pair<MajoranaString, ff_complex>> o_new;

        // This is an over-estimate. The true size of o_new is the number of terms
        // in poly that don't commute with gate.ms
        o_new.reserve(poly.terms.size());

        // Some precomputation
        const ff_float costheta = cos(theta);
        const ff_complex iirsintheta = _r ? ff_complex(-sin(theta),0) : ff_complex(0,sin(theta)); // i*i^r*sin(theta)

        // Populate o_new
        for(auto& [x,v] : poly.terms) {
            if (!x.commutes(ms)) {
                MajoranaMonomial mx = ms*x;
                if(filter_pred(mx.s)) {
                    o_new.emplace_back(mx.majorana_string(), v*iirsintheta*mx.coefficient());
                }
                v *= costheta;
            }
        }
        for (const auto& [x,v] : o_new) {
            poly.terms[x] += v;
        }
    }

    void apply_inplace(MajoranaPolynomial& poly) const {
        apply_inplace(poly, [](const MajoranaString& a) { return true; });
    }

    void apply_inplace(MajoranaPolynomial& poly, const int& maxdegree) const {
        apply_inplace(poly, [&maxdegree](const MajoranaString& a) { return a.degree() <= maxdegree; });
    }
    
    MajoranaPolynomial operator()(const MajoranaPolynomial& o) const {
        // Apply U^{\dagger} o U
        MajoranaPolynomial ret(o);
        apply_inplace(ret);
        return ret;
    }

    MajoranaPolynomial operator()(const MajoranaString& o) const {
        // Apply U^{\dagger} o U
        return (*this)(MajoranaPolynomial(o));
    }
    
};

}

}