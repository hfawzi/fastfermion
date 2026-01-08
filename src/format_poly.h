/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include <string>
#include <format> // for std::format

namespace fastfermion {

std::string format_float(const ff_float& v) {
    return (std::stringstream() << v).str();
};

std::string format_complex(const ff_complex& z) {
    if(z.imag() == 0) {
        return format_float(z.real());
    } else if (z.real() == 0) {
        return format_float(z.imag()) + ff_config.imaginary_symbol;
    } else {
        if(z.imag() < 0) {
            return "("+format_float(z.real()) + "-" + format_float(-z.imag()) + ff_config.imaginary_symbol + ")";
        } else {
            return "("+format_float(z.real()) + "+" + format_float(z.imag()) + ff_config.imaginary_symbol + ")";
        }
    }
}

template<class Poly>
std::string format_poly(const Poly& poly, int max_terms_to_show=50, int max_line_length=120) {
    if(poly.terms.empty()) {
        return "0";
    }

    std::string ret = "";
    // Number of spaces before and after '+' and '-' between
    // each term
    int sep_gap_len = 2;
    int i = 0;
    int current_line_length = 0;
    for(auto& [x,v] : poly.terms) {
        if(i >= max_terms_to_show) {
            //ret += std::format(" + ... ({} more terms)", poly.terms.size()-i);
            ret += (std::stringstream() << " + ... (" << poly.terms.size()-i << " more terms)").str();
            break;
        }
        bool x_is_identity = x == decltype(x)();

        std::string term;
        std::string sep = "";
        std::string coeffstr, opstr;

        // 1. Determine separation operator
        if(i > 0) {
            // We set it to '-' only when v is real and negative
            // or v is pure imaginary with a negative coefficient
            std::string sep_op = (v.imag() == 0 && v.real() < 0) || (v.real() == 0 && v.imag() < 0) ? "-" : "+";
            // Add space separation
            std::string left_gap = std::string(sep_gap_len, ' ');
            std::string right_gap = std::string(sep_gap_len, ' ');
            sep = left_gap + sep_op + right_gap;
        }

        // 2. Put coeff in string format
        if(v == ff_complex(1,0)) {
            // If the coefficient is 1, don't show it unless
            // the operator itself is identity
            coeffstr = x_is_identity ? "1" : "";
        } else if (v == ff_complex(-1,0)) {
            // If first term and identity, then -1
            // If first term and not identity, then -
            // If not first term, and identity, then 1 (the minus will come from sep)
            // If not first term and not identity, then none (the sep will have the minus)
            coeffstr = 
                std::basic_string(i == 0 ? "-" : "") + std::basic_string(x_is_identity ? "1" : "");
        } else if(v.imag() == 0) {
            // Purely real coefficient
            // If coefficient is real and negative, we don't show
            // the negative sign unless we are in the first term
            coeffstr = format_float(i == 0 ? v.real() : std::abs(v.real()));
        } else if(v.real() == 0) {
            // Purely imaginary coefficient
            // If coefficient is imaginary and negative, we don't show
            // the negative sign unless we are in the first term
            coeffstr = format_float(i == 0 ? v.imag() : std::abs(v.imag())) + ff_config.imaginary_symbol;
        } else {
            // Coefficient has real and imaginary components
            if(v.imag() < 0) {
                coeffstr = "("+format_float(v.real()) + "-" + format_float(-v.imag()) + ff_config.imaginary_symbol + ")";
            } else {
                coeffstr = "("+format_float(v.real()) + "+" + format_float(v.imag()) + ff_config.imaginary_symbol + ")";
            }
        }

        // 3. Put operator in string format
        opstr = x_is_identity ? "" : x.to_compact_string();

        // Put everything together
        term = sep + coeffstr;
        if(coeffstr != "" && opstr != "") {
            term += " "; // Add a space between coeff and operator
        }
        term += opstr;
        int term_size = term.size();

        // Add new line if needed
        // current_terms_per_line++;
        if(current_line_length + term_size > max_line_length) {
            // Need to put the term in a new line
            ret += "\n";
            // current_terms_per_line = 0;
            current_line_length = 0;
        }

        // Add term to ret
        ret += term;
        current_line_length += term.size();
        i++;
    }
    return ret;
}

}