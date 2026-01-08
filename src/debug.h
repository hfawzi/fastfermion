/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

#include <string>
#include <array>
#include <chrono>
#include <vector>
#include <format>

namespace fastfermion {

// Helper functions useful for debugging
// Explanation for the do/while block
// https://stackoverflow.com/questions/1067226/c-multi-line-macro-do-while0-vs-scope-block
#define assertm(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                        << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)

std::ostream& operator<<(std::ostream& os, const std::chrono::duration<double>& d) {
    double duration_sec = (double)std::chrono::duration_cast<std::chrono::microseconds>(d).count() / 1e6;
    return os << duration_sec;
}

std::ostream& operator<<(std::ostream& os, const std::array<ff_complex,4>& v) {
    std::string v_str = "[";
    for(std::size_t i=0; i<4; i++) {
        v_str += std::format("{}",v[i]);
        if(i<v.size()-1) v_str += " ";
    }
    v_str += "]";
    return os << v_str;
}

std::ostream& operator<<(std::ostream& os, const std::vector<std::size_t>& v) {
    std::string v_str = "[";
    for(std::size_t i=0; i<v.size(); i++) {
        v_str += std::to_string(v[i]);
        if(i<v.size()-1) v_str += " ";
    }
    v_str += "]";
    return os << v_str;
}

std::ostream& operator<<(std::ostream& os, const std::vector<char>& v) {
    std::string v_str = "[";
    for(std::size_t i=0; i<v.size(); i++) {
        v_str += std::to_string(v[i]);
        if(i<v.size()-1) v_str += " ";
    }
    v_str += "]";
    return os << v_str;
}

std::ostream& operator<<(std::ostream& os, const std::vector<int>& v) {
    std::string v_str = "[";
    for(std::size_t i=0; i<v.size(); i++) {
        v_str += std::to_string(v[i]);
        if(i<v.size()-1) v_str += " ";
    }
    v_str += "]";
    return os << v_str;
}

std::ostream& operator<<(std::ostream& os, const std::vector<ff_float>& v) {
    std::string v_str = "[";
    for(std::size_t i=0; i<v.size(); i++) {
        v_str += std::format("{}",v[i]);
        if(i<v.size()-1) v_str += " ";
    }
    v_str += "]";
    return os << v_str;
}

std::ostream& operator<<(std::ostream& os, const std::vector<ff_complex>& v) {
    std::string v_str = "[";
    for(std::size_t i=0; i<v.size(); i++) {
        v_str += std::format("{}",v[i]);
        if(i<v.size()-1) v_str += " ";
    }
    v_str += "]";
    return os << v_str;
}

}