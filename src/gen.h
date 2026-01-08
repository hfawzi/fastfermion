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

namespace fastfermion {

std::tuple<std::vector<PauliString>,std::vector<PauliString>,std::vector<PauliString>> paulis(int n) {
    std::vector<PauliString> sx(n);
    std::vector<PauliString> sy(n);
    std::vector<PauliString> sz(n);
    for(int i=0; i<n; i++) {
        sx[i] = PauliString(std::vector<std::pair<int,char>>{{i,'X'}});
        sy[i] = PauliString(std::vector<std::pair<int,char>>{{i,'Y'}});
        sz[i] = PauliString(std::vector<std::pair<int,char>>{{i,'Z'}});
    }
    return std::make_tuple(sx,sy,sz);
}


std::vector<FermiString> fermis(int n) {
    std::vector<FermiString> a(n);
    for(int i=0; i<n; i++) {
        a[i] = FermiString(std::vector<int>{},std::vector<int>{i});
    }
    return a;
}

// Utility function to get all the majorana operators
std::vector<MajoranaString> majoranas(const int& n) {
    std::vector<MajoranaString> maj(n);
    for(int i=0; i<n; i++) {
        maj[i] = MajoranaString(std::vector<int>{i});
    }
    return maj;
}


// Utility functions to enumerate all Pauli, Fermi, Majorana strings of certain degree

void next_combination(std::vector<int>& x) {
    // Generates next subset of \N = {0,1,2,} of same size as x.size()
    // e.g., if x={0,2,3} returns {1,2,3}
    //          x={1,2,3} returns {0,1,4}
    // Assumes that x is sorted and increasing. Otherwise behaviour is undefined
    std::size_t j=0;
    while(j<x.size()-1 && x[j+1]==x[j]+1) { x[j] = j; j++; }
    x[j]++;
}

std::vector<PauliString> paulistrings(int n, int degree, std::function<bool(const std::vector<int>&, const std::vector<char>&)> filter_fun) {
    // Return all (n choose d)*3^d Pauli strings of degree `degree` supported in `{0,...,n-1}`
    if(degree > n) return std::vector<PauliString>{};
    if(degree == 0) return std::vector<PauliString>{PauliString()};
    int pow3d = std::pow(3,degree);
    std::vector<PauliString> ret;
    PauliString a;

    // Initialization
    int i, l;
    // Iterate over all (n choose degree) subsets
    std::vector<int> supp(degree);
    for(int i=0; i<degree; i++) supp[i] = i; // initialize with supp = {0,...,degree-1}
    while(supp[degree-1] < n) {

        // Loop over all 3^d actions
        std::vector<char> actions(degree, 'X'); // action[i] = 'X', 'Y', or 'Z'
        for(i=0; i<pow3d; i++) {
            a = PauliString(supp, actions);
            if(filter_fun(supp,actions)) ret.push_back(a);
            l = degree-1;
            while(l >= 0 && actions[l] == 'Z') { actions[l] = 'X'; l--; }
            if(l == -1) break;
            else if(l >= 0) actions[l]++;
        }
        assert(l == -1);

        // Update support
        next_combination(supp);
    }
    return ret;
}

std::vector<PauliString> paulistrings(int n, int degree) {
    return paulistrings(n, degree, [](const std::vector<int>& s, const std::vector<char>& a) { return true; });
}

std::vector<PauliString> paulistrings(int n) {
    std::vector<PauliString> ret;
    for(int d=0; d<=n; d++) {
        std::vector<PauliString> pd = paulistrings(n,d);
        ret.insert(ret.end(), pd.begin(), pd.end());
    }
    return ret;
}

std::vector<FermiString> fermistrings(int n, int degree, std::function<bool(const std::vector<int>&, const std::vector<int>&)> filter_fun) {
    // Returns all (2n choose d) Fermi Strings of degree d in n modes
    if(degree > 2*n) return std::vector<FermiString>{};
    if(degree == 0) return std::vector<FermiString>{FermiString()};
    std::vector<FermiString> ret;
    std::vector<int> supp(degree,0);
    for(int i=0; i<degree; i++) supp[i] = i;
    while(supp[degree-1] < 2*n) {
        ff_ulong cre, ann;
        for(int i=0; i<degree; i++) {
            if(supp[i] < n) ann.set(supp[i]);
            else cre.set(supp[i]-n);
        }
        if(filter_fun(cre.rsupport(), ann.rsupport())) {
            ret.push_back(FermiString(cre,ann));
        }
        next_combination(supp);
    }
    return ret;
}

std::vector<FermiString> fermistrings(int n, int degree) {
    return fermistrings(n, degree, [](const std::vector<int>& cre_s, const std::vector<int>& ann_s) { return true; });
}

std::vector<FermiString> fermistrings(int n) {
    std::vector<FermiString> ret;
    for(int d=0; d<=2*n; d++) {
        std::vector<FermiString> pd = fermistrings(n,d);
        ret.insert(ret.end(), pd.begin(), pd.end());
    }
    return ret;
}


std::vector<MajoranaString> majoranastrings(int n, int degree, std::function<bool(const std::vector<int>&)> filter_fun) {
    // Returns all (n choose d) Majorana Strings of degree d in n modes
    if(degree > n) return std::vector<MajoranaString>{};
    if(degree == 0) return std::vector<MajoranaString>{MajoranaString()};
    std::vector<MajoranaString> ret;
    std::vector<int> supp(degree,0);
    for(int i=0; i<degree; i++) supp[i] = i;
    while(supp[degree-1] < n) {
        if(filter_fun(supp)) {
            ret.push_back(MajoranaString(supp));
        }
        next_combination(supp);
    }
    return ret;
}

std::vector<MajoranaString> majoranastrings(int n, int degree) {
    return majoranastrings(n, degree, [](const std::vector<int>& supp) { return true; });
}

std::vector<MajoranaString> majoranastrings(int n) {
    std::vector<MajoranaString> ret;
    for(int d=0; d<=n; d++) {
        std::vector<MajoranaString> pd = majoranastrings(n,d);
        ret.insert(ret.end(), pd.begin(), pd.end());
    }
    return ret;
}

}