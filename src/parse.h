/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

#include <vector>
#include <utility>
#include <string>
#include <charconv> // for from_chars, to parse string into a FermiString

namespace fastfermion {

std::vector<std::pair<int, char>> _parse_pauli_string(const std::string& ps) {
    // Parses a PauliString of the form X0 Z1 Y4
    // Go through the string character by character
    std::vector<std::pair<int,char>> ind;
    if(ps == "I" || ps == "1") return ind;
    int ps_size = ps.size();
    for(int i=0; i<ps_size; ) {
        if(i > 0 && (ps[i] == ' ' || ps[i] == '*')) {
            i++;
            continue;
        }
        if(ps[i] == 'X' || ps[i] == 'Y' || ps[i] == 'Z') {
            int j = i+1;
            while(j < ps_size && std::isdigit(ps[j])) {
                j++;
            }
            if(j == i+1) {
                throw_error("Invalid Pauli string");
            }
            // Extract substring
            int loc = std::stoi(ps.substr(i+1,j-(i+1)));
            ind.emplace_back(loc,ps[i]);
            i = j;
        } else {
            throw_error("Invalid Pauli string");
        }
    }
    return ind;
}

std::vector<std::pair<int, bool>> _parse_fermi_string(const std::string& fs) {
    // Parses a FermiString of the form f1^ f0^ f0
    // If not in normal form, raised an error
    // Go through the string character by character
    std::vector<std::pair<int,bool>> ind;
    if(fs == "I" || fs == "1") return ind;
    int fs_size = fs.size();
    for(int i=0; i<fs_size; ) {
        if(i > 0 && (fs[i] == ' ' || fs[i] == '*')) {
            i++;
            continue;
        }
        if(fs[i] == ff_config.fermi_symbol) {
            int j = i+1;
            while(j < fs_size && std::isdigit(fs[j])) {
                j++;
            }
            if(j == i+1) {
                throw_error("Invalid Fermi string");
            }
            // Extract substring
            int loc = std::stoi(fs.substr(i+1,j-(i+1)));
            bool iscre = j < fs_size && fs[j] == ff_config.dagger_symbol;
            ind.emplace_back(loc,iscre);
            if(iscre) i = j+1;
            else i = j;
        } else {
            throw_error("Invalid Fermi string");
        }
    }
    return ind;
}

std::vector<int> _parse_majorana_string(const std::string& ms) {
    // Parses a MajoranaString of the form m0 m1 m3
    // If not in normal form, raised an error
    // Go through the string character by character
    std::vector<int> ind;
    if(ms == "I" || ms == "1") return ind;
    int ms_size = ms.size();
    for(int i=0; i<ms_size; ) {
        if(i > 0 && (ms[i] == ' ' || ms[i] == '*')) {
            i++;
            continue;
        }
        if(ms[i] == ff_config.majorana_symbol) {
            int j = i+1;
            while(j < ms_size && std::isdigit(ms[j])) {
                j++;
            }
            if(j == i+1) {
                throw_error("Invalid Majorana string");
            }
            // Extract substring
            int loc = std::stoi(ms.substr(i+1,j-(i+1)));
            ind.emplace_back(loc);
            i = j;
        } else {
            throw_error("Invalid Majorana string");
        }
    }
    return ind;
}

}