/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include <cassert>
#include <vector>
#include <array>
#include <string>
#include <iostream>
#include <bit>
#include <limits>
#include <cstdint>

#include "hashmap/unordered_dense.h" // for hashing

namespace fastfermion {

constexpr int WORD_LENGTH = std::numeric_limits<std::uint64_t>::digits;

// Used to define a bitstring of arbitrary length
// BitSet<1> is a 64-bit string
// BitSet<2> is a 128-bit string, etc.
template<int N>
struct BitSet {
    std::array<std::uint64_t, N> words;
    static constexpr int DIGITS = N*WORD_LENGTH;

    BitSet() { words.fill(0); }
    BitSet(const BitSet<N>& a) : words(a.words) { }
    BitSet(const std::array<std::uint64_t, N>& words) : words(words) { }
    BitSet(std::uint64_t x) {
        words.fill(0);
        words[0] = x;
    }

    int popcount() const {
        int ret = 0;
        for(std::size_t i=0; i<N; ++i)
            ret += std::popcount(words[i]);
        return ret;
    }

    int countr_zero() const {
        // Counts the number of consecutive zeros starting from the right
        // e.g., countr_zero for 001101000 is 3
        // We iterate over the words from left to right
        // as long as they are zero
        std::size_t i = 0;
        while(i < N && words[i] == 0) i += 1;
        if(i == N) return N*WORD_LENGTH;
        return WORD_LENGTH*i + std::countr_zero(words[i]);
    }

    int countl_zero() const {
        std::size_t i = 0;
        while(i < N && words[N-1-i] == 0) i += 1;
        if(i == N) return N*WORD_LENGTH;
        // i <= N-1
        return WORD_LENGTH*i + std::countl_zero(words[N-1-i]);
    }


    // In-place operations
    BitSet<N>& operator|=(const BitSet<N>& b) { for(std::size_t i=0; i<N; ++i) { words[i] |= b.words[i]; } return *this; }
    BitSet<N>& operator&=(const BitSet<N>& b) { for(std::size_t i=0; i<N; ++i) { words[i] &= b.words[i]; } return *this; }
    BitSet<N>& operator^=(const BitSet<N>& b) { for(std::size_t i=0; i<N; ++i) { words[i] ^= b.words[i]; } return *this; }

    // In-place bitwise shifts
    BitSet<N>& operator>>=(int shift) {
        assert(shift >= 0);
        int q = shift / WORD_LENGTH;
        int r = shift % WORD_LENGTH;
        for(int i=0; i<N; i++) {
            if(i+q+1 < N) {
                // words[i] = (words[i+q] >> r) | (words[i+q+1] << (WORD_LENGTH-r));
                // When r=0, WORD_LENGTH-r = WORD_LENGTH, and shifting by WORD_LENGTH is unsupported
                // so we check first if r == 0
                words[i] = (words[i+q] >> r);
                if(r != 0) words[i] |= (words[i+q+1] << (WORD_LENGTH-r));
            } else if (i+q < N) {
                words[i] = words[i+q] >> r;
            } else {
                words[i] = 0;
            }
        }
        return *this;
    }
    BitSet<N>& operator<<=(int shift) {
        // Let D = WORD_LENGTH. The shift operation is:
        //   x[Dj+k] <- x[Dj+k-shift] = x[D(j-q)+k-r]
        // which translates to:
        //   words[j][r:D-1] <- words[j-q][0:D-1-r]
        //   words[j][0:r-1] <- words[j-q-1][D-r:D-1]
        // which can be succintly summarized as
        //   words[j] = (words[j-q] << r) | (words[j-q-1] >> D-r)
        assert(shift >= 0);
        int q = shift / WORD_LENGTH;
        int r = shift % WORD_LENGTH;
        for(int i=N-1; i>=0; i--) {
            if(i-q-1 >= 0) {
                // words[i] = (words[i-q] << r) | (words[i-q-1] >> (WORD_LENGTH-r));
                // When r=0, WORD_LENGTH-r = WORD_LENGTH, and shifting by WORD_LENGTH is unsupported
                // so we check first if r == 0
                words[i] = (words[i-q] << r);
                if(r != 0) words[i] |= (words[i-q-1] >> (WORD_LENGTH-r));
            } else if (i-q >= 0) {
                words[i] = words[i-q] << r;
            } else {
                words[i] = 0;
            }
        }
        return *this;
    }
    
    // Functions to iterate on bits
    // Usage:
    // for(int cur_bit = begin(); cur_bit != end(); cur_bit = next(cur_bit)) { ... }
    int begin() const { return countr_zero(); }
    int end() const { return N*WORD_LENGTH; }
    int next(int pos) const {
        assert(pos >= 0 && pos < N*WORD_LENGTH);
        if(pos == N*WORD_LENGTH-1) return end();
        BitSet<N> w = (*this) >> (pos+1); // pos + 1 < N*WORD_LENGTH
        if(w == 0) return end();
        return pos + 1 + w.countr_zero();
    }

    // Functions to iterate on bits in reverse order
    // Usage:
    // for(int cur_bit = rbegin(); cur_bit != rend(); cur_bit = rnext(cur_bit)) { ... }
    int rbegin() const { return N*WORD_LENGTH-countl_zero()-1; }
    int rend() const { return -1; }
    int rnext(int pos) const {
        assert(pos >= 0 && pos < N*WORD_LENGTH);
        if(pos == 0) return rend();
        BitSet<N> w = (*this) << (N*WORD_LENGTH-pos); // CHECK THIS !!!
        if(w == 0) return rend();
        return pos - 1 - w.countl_zero();
    }
    

    std::uint64_t to_ullong() const {
        return words[0];
    }

    std::vector<int> support() const {
        // Returns positions of activated bits in increasing order (from least significant bit to most significant one)
        // The length of the returned vector is equal to popcount()
        std::vector<int> supp;
        int len = popcount();
        supp.reserve(len);
        int cur_bit;
        int i;
        for(cur_bit = begin(), i = 0; cur_bit != end(); cur_bit = next(cur_bit), i++) {
            supp.push_back(cur_bit);
            assert(i < len);
            if(i >= 1) assert(supp[i] > supp[i-1]);
        }
        assert(i == len);
        return supp;
    }

    std::vector<int> rsupport() const {
        // Returns positions of activated bits in reverse order (from most significant bit to least significant one)
        // The length of the returned vector is equal to popcount()
        std::vector<int> supp;
        int len = popcount();
        supp.reserve(len);
        int cur_bit;
        int i;
        for(cur_bit = rbegin(), i = 0; cur_bit != rend(); cur_bit = rnext(cur_bit), i++) {
            supp.push_back(cur_bit);
            assert(i < len);
            if(i >= 1) assert(supp[i] < supp[i-1]);
        }
        assert(i == len);
        return supp;
    }

    std::string to_string() const {
        std::string ret(N*WORD_LENGTH,'0'); // Initialize with all zeros
        for(int cur_bit=begin(); cur_bit != end(); cur_bit = next(cur_bit)) {
            assert(cur_bit >= 0 && cur_bit < N*WORD_LENGTH);
            ret[N*WORD_LENGTH-1-cur_bit] = '1';
        }
        return ret;
    }

    void set(int pos) {
        // Set the bit at position pos to 1
        assert(pos < N*WORD_LENGTH);
        int q = pos/WORD_LENGTH;
        int r = pos%WORD_LENGTH;
        words[q] |= (1ULL << r);
    }

    bool at(int pos) const {
        assert(pos < N*WORD_LENGTH);
        int q = pos/WORD_LENGTH;
        int r = pos%WORD_LENGTH;
        return (words[q] & (1ULL << r)) != 0;
    }

    std::uint64_t hash() const {
        auto h = std::uint64_t{};
        for(const std::uint64_t& w : words) {
            h = ankerl::unordered_dense::tuple_hash_helper<>::mix64(h,w);
        }
        return h;
    }

    // ---- STATIC METHODS THAT CONSTRUCT SPECIFIC BIT STRINGS ----

    static BitSet<N> range(int end, bool inclusive_end = false) {
        // Returns BitSet where all bits 0...end-1 are set to 1
        assert(end < N*WORD_LENGTH);
        int q = end/WORD_LENGTH;
        int r = end%WORD_LENGTH;
        int i;
        BitSet<N> a(0); // initialized to 0
        for(i=0; i<q; i++) a.words[i] = ~0; // all ones
        a.words[q] = (1ULL << r)-1; // 1 from 0 to r-1
        if(inclusive_end) a.words[q] |= (1ULL << r);
        return a;
    }

    static BitSet<N> singleton(int pos) {
        // Returns BitSet where only bit at position pos is set to 1
        assert(pos < N*WORD_LENGTH);
        BitSet<N> a(0);
        a.set(pos);
        return a;
    }

    static BitSet<N> from_set(const std::vector<int>& pos) {
        // Returns BitSet a where a[i] = 1 for i \in pos
        BitSet<N> a; // initialized to 0
        int q,r;
        for(std::size_t i=0; i<pos.size(); i++) {
            assert(pos[i] < N*WORD_LENGTH);
            q = pos[i]/WORD_LENGTH;
            r = pos[i]%WORD_LENGTH;
            a.words[q] |= (1ULL << r);
        }
        return a;
    }

    static constexpr BitSet<N> even_mask() {
        // Returns Bit Set with the even bits active
        std::array<std::uint64_t, N> _words{};
        _words.fill(0x5555555555555555); // 64-bit with 1s at even location
        return BitSet<N>(_words);
    }

    static constexpr BitSet<N> odd_mask() {
        // Returns Bit Set with the odd bits active
        std::array<std::uint64_t, N> _words{};
        _words.fill(0xAAAAAAAAAAAAAAAA); // 64-bit with 1s at even location
        return BitSet<N>(_words);
    }

    // ---- END STATIC METHODS ----
    
};

template<int N>
std::ostream& operator<<(std::ostream& os, const BitSet<N>& a) {
    return os << a.to_string();
}

template<int N>
BitSet<N> operator|(const BitSet<N>& a, const BitSet<N>& b) { BitSet<N> c(a); c |= b; return c; }
template<int N>
BitSet<N> operator&(const BitSet<N>& a, const BitSet<N>& b) { BitSet<N> c(a); c &= b; return c; }
template<int N>
BitSet<N> operator^(const BitSet<N>& a, const BitSet<N>& b) { BitSet<N> c(a); c ^= b; return c; }
template<int N>
BitSet<N> operator~(const BitSet<N>& a) { BitSet<N> c; for(std::size_t i=0; i<N; i++) { c.words[i] = ~a.words[i]; }  return c; }
template<int N>
BitSet<N> operator<<(const BitSet<N>& a, int shift) { BitSet<N> c(a); c <<= shift; return c; }
template<int N>
BitSet<N> operator>>(const BitSet<N>& a, int shift) { BitSet<N> c(a); c >>= shift; return c; }
template<int N>
bool operator==(const BitSet<N>& a, const BitSet<N>& b) { return a.words == b.words; }
template<int N>
bool operator==(const BitSet<N>& a, std::uint64_t b) { return a.words == BitSet<N>(b).words; }
template<int N>
bool operator==(std::uint64_t a, const BitSet<N>& b) { return BitSet<N>(a).words == b.words; }
template<int N>
bool operator<(const BitSet<N>& a, const BitSet<N>& b) {
    int i = N-1;
    while(i >= 0 && a.words[i] == b.words[i]) i--;
    // Either i == -1 or a.words[i] != b.words[i]
    return (i >= 0 && a.words[i] < b.words[i]);
}
template<int N>
bool operator<=(const BitSet<N>& a, const BitSet<N>& b) {
    int i = N-1;
    while(i >= 0 && a.words[i] == b.words[i]) i--;
    return (i == -1) || (i >= 0 && a.words[i] < b.words[i]);
}

}