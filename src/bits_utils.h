/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "bits.h"
#include "error.h"

namespace fastfermion {

template<int N>
int merge_parity(const BitSet<N>& x, const BitSet<N>& y) {
    // Given two sets x = {x_1, ..., x_k} and y = {y_1, ..., y_l}
    // returns T mod 2 where T = the number of pairs (i,j) such that x_i < y_j.
    //
    // Both sets x and y are represented as bit strings
    // i.e.,  x has 1 at positions x_1,...,x_k
    //    and y has 1 at positions y_1,...,y_l
    //
    // We can assume wlog that x_1 > ... > x_k
    // and y_1 > ... > y_l

    // The implementation below works by iterating over the bits y_1 > ... > y_l
    // of y, and counting for each j, the number of bits of x that are in a position
    // smaller than y_j.

    // Iterate over y_1 > ... > y_l
    int tot = 0;
    for(int cur_bit=y.rbegin(); cur_bit != y.rend(); cur_bit = y.rnext(cur_bit)) {
        tot += (x & BitSet<N>::range(cur_bit)).popcount(); // Count number of bits in x having a position strictly smaller than cur_bit
    }
    return tot % 2;
}

template<int N>
std::pair<BitSet<N>, int> shuffle_with_parity(const BitSet<N>& S, const std::vector<int>& map) {
    // Given a set S = {s_1 > s_2 > ... > s_m} represented as a bitstring,
    // and an injective map pi:{0,1,...} -> {0,1,...}, returns T (a bitset)
    // and sgn \in {-1,+1}
    // where T = {t_1 > t_2 > ... > t_m} are the elements of pi(S) in decreasing order
    // and sgn = (-1)^{inv(<pi(s_1), ..., pi(s_m)>)}
    // where inv(<pi(s_1), ..., pi(s_m)>) = |{(i,j) : i<j and pi(s_i) < pi(s_j)}|
    // This is used in fermionic manipulations, e.g., when permuting a FermiString, so that
    //      a_{pi(s_1)} * ... * a_{pi(s_m)} = sgn * a_T
    //
    // Since S is sorted, note that
    // inv(<pi(s_1), ..., pi(s_m)>) = |{(i,j) : s_i > s_j and pi(s_i) < pi(s_j)}|
    // inv(S,pi) = |{(s,s') \in S^2 : s' > s and pi(s') < pi(s)}|
    //           = sum_{s} |s' > s s.t. pi(s') < pi(s)|
    //
    // Note: the above can also be applied to Majorana operators which by convention we order
    // in the opposite way m_{s_1} m_{s_2} ... m_{s_k} where s_1 < ... < s_k.
    // The reason is that going from any m_{t_1} ... m_{t_k} to m_{t_k} ... m_{t_1} we incur
    // a sign of (-1)^{k(k-1)/2} which only depends on the length of the string
    BitSet<N> new_S(0);
    int new_pos;
    int inv = 0;
    int map_size = map.size();
    for(int pos = S.rbegin() ; pos != S.rend(); pos = S.rnext(pos)) {
        //assertm(pos < map.size(), "pos = " << pos << ", map.size() = " << map.size());
        new_pos = pos < map_size ? map[pos] : pos; // If map[pos] is not defined, we assumed map[pos] = pos (identity)
        assert(new_pos >= 0 && new_pos < BitSet<N>::DIGITS);
        assert(!new_S.at(new_pos));
        // if(new_S.at(new_pos)) {
        //  // It seems this is not an injective map: two different sites map to the same location.
        //  throw_error("Given permutation is not valid; the pre-image of " << new_pos << " is not unique");
        // }
        // Count the number of bits in new_S which are at a position < new_pos
        // This is precisely |{s' > pos : pi(s') < pi(pos)}|
        inv += (new_S & BitSet<N>::range(new_pos)).popcount();
        new_S.set(new_pos);
    }
    return std::make_pair(new_S, inv%2 == 0 ? 1 : -1);
}


// ------------- BIT PERMUTATION FUNCTIONS --------------------
template<int N>
BitSet<N> move_bits(const BitSet<N>& a, const std::vector<int>& map) {
    // If we think of a as a bitstring a[0], ... a[n-1], and map : {0,...,n-1} -> {0,...,n-1}
    // Then this function returns the new bit string b such that
    //      b[map[j]] = a[j] for all j=0,...,n-1
    // NOTE: map is only accessed on the location of nonzero bits of a
    // The code below implements the following:
    //    for each j s.t. a[j] == 1, set b[map[j]] = 1
    // Naive method
    BitSet<N> b(0);
    int map_size = map.size();
    for(int pos = a.begin(); pos != a.end(); pos = a.next(pos)) {
        if(pos < map_size) {
            if(map[pos] < 0 || map[pos] >= N*WORD_LENGTH) {
                throw_error("Invalid map value " << map[pos] << " at " << pos);
            }
            b.set(map[pos]);
        } else {
            b.set(pos);
        }
    }
    return b;
}

template<int N>
BitSet<N> fliplr_bits(const BitSet<N>& a, int n) {
    // If we think of a as a bitstring a[0], ... a[n-1], then this function returns the new bit string b such that
    //      b[j] = a[n-1-j], j=0,...,n-1
    //      b[j] = 0 for j >= n
    // Note: b's bits beyond position n-1 are set to 0 no matter what a's bits were
    // Naive method
    BitSet<N> b(0);
    for(int pos=a.begin(); pos != a.end() && pos < n; pos = a.next(pos)) {
        b.set(n-1-pos);
    }
    return b;
}

template<int N>
BitSet<N> swap_even_odd(const BitSet<N>& a) {
    // One-liner:
    //   return ((a & even_mask) << 1) | ((a & odd_mask) >> 1)
    BitSet<N> x(a);
    BitSet<N> y(a);
    constexpr std::uint64_t _even_mask = 0x5555555555555555;
    constexpr std::uint64_t _odd_mask = 0xAAAAAAAAAAAAAAAA;
    for(std::size_t i=0; i<N; ++i) {
        x.words[i] &= _even_mask;
        y.words[i] &= _odd_mask;
    }
    x <<= 1;
    y >>= 1;
    return (x | y);
}

template<int N>
void swap_bits_inplace(BitSet<N>& b, int i, int j) {
    // Swap bits i and j of b
    // See https://graphics.stanford.edu/~seander/bithacks.html#SwappingBitsXOR
    // Compute x = XOR(b[i],b[j])
    // Set b[i] = XOR(b[i],x)
    // and b[j] = XOR(b[j],x)
    BitSet<N> x = ((b >> i) ^ (b >> j)) & BitSet<N>(1ULL); // Single bit
    b ^= (x << i) | (x << j);
}

// ------------- BIT INTERLEAVING --------------------

void interleave(const std::uint64_t& a, std::uint64_t* y) {
    // Interleave 64-bit integer by zeros
    // Stores the result in y[0] and y[1]

    // Follows the divide and conquer approach from
    // https://stackoverflow.com/a/39490836
    // std::array<std::uint64_t, 2> y;

    // First split a into lower 32-bits and higher 32-bits
    y[0] = a & (0x00000000FFFFFFFF);
    y[1] = (a & (0xFFFFFFFF00000000)) >> 32;

    static const std::uint64_t B[] = {
        0x5555555555555555, // 0101010101010101...0101
        0x3333333333333333, // 0011001100110011...0011
        0x0F0F0F0F0F0F0F0F, // 0000111100001111...1111
        0x00FF00FF00FF00FF, // 0000000011111111...1111
        0x0000FFFF0000FFFF,  // ...
        0x00000000FFFFFFFF,  // ... Not used for interleaving, only used for de-interleaving
    };
    static const std::uint64_t S[] = {1, 2, 4, 8, 16};

    // y[0] and y[1] are supported on the first 32-bits only
    // assert(y[0] < (1ULL<<32) && y[1] < (1ULL<<32));
    
    for(std::size_t i=0; i<2; ++i) {
        y[i] = (y[i] | (y[i] << S[4])) & B[4];
        y[i] = (y[i] | (y[i] << S[3])) & B[3];
        y[i] = (y[i] | (y[i] << S[2])) & B[2];
        y[i] = (y[i] | (y[i] << S[1])) & B[1];
        y[i] = (y[i] | (y[i] << S[0])) & B[0];
    }
}


template<int N>
BitSet<2*N> interleave(const BitSet<N>& a) {
    // Returns bitstring b on 2N bits where b[2i] = a[i]
    // and b[2i+1] = 0
    
    // NOTE: This algorithm is linear in N
    // One can do the interleaving in logarithmic complexity in N
    // However the latter works best when N is a power of two
    // For now, since N is small we prefer the simpler implementation

    BitSet<2*N> b;
    for(std::size_t i=0; i<N; ++i) {
        interleave(a.words[i], b.words.data()+(2*i));
    }

    return b;
}

std::uint64_t deinterleave64(const std::uint64_t& y) {

    // De-interleaves a 64-bit y into x, i.e., returns x s.t.
    // x[i] = y[2*i] for 0 <= i < 32
    // x[i] = 0 for i >= 32

    static const std::uint64_t B[] = {
        0x5555555555555555, // 0101010101010101...0101
        0x3333333333333333, // 0011001100110011...0011
        0x0F0F0F0F0F0F0F0F, // 0000111100001111...1111
        0x00FF00FF00FF00FF, // 0000000011111111...1111
        0x0000FFFF0000FFFF,  // ...
        0x00000000FFFFFFFF,  // ... Not used for interleaving, only used for de-interleaving
    };
    static const std::uint64_t S[] = {1, 2, 4, 8, 16};

    std::uint64_t x = y & B[0];
    x = (x | (x >> S[0])) & B[1];
    x = (x | (x >> S[1])) & B[2];
    x = (x | (x >> S[2])) & B[3];
    x = (x | (x >> S[3])) & B[4];
    x = (x | (x >> S[4])) & B[5];
    return x;
}

std::uint64_t deinterleave128(const std::uint64_t& y0, const std::uint64_t& y1) {

    // De-interleaves a 128-bit y=(y0,y1) into x, i.e., returns x s.t.
    // x[i] = y0[2*i] for 0 <= i < 32
    // x[i] = y1[2*(i-32)] for i >= 32

    return deinterleave64(y0) | (deinterleave64(y1) << 32);

}

template<int N>
BitSet<N/2> deinterleave(const BitSet<N>& a) {
    // Returns bitstring b on N bits where b[i] = a[2*i]
    
    // NOTE: This algorithm is linear in N
    // One can do the interleaving in logarithmic complexity in N
    // However the latter works best when N is a power of two
    // For now, since N is small we prefer the simpler implementation

    BitSet<N/2> b;
    for(std::size_t i=0; i<N/2; ++i) {
        b.words[i] = deinterleave128(a.words[2*i], a.words[2*i+1]);
    }

    return b;
}
// --------------------------------------------

std::uint64_t next_combination(std::uint64_t v) {
    // Returns the next subset (represented as a bitstring) with the same cardinality as v
    // If v is 00010011, the next patterns would be 00010101, 00010110, 00011001,00011010, 00011100, 00100011
    // From https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    // See also https://github.com/hcs0/Hackers-Delight/blob/master/snoob.c.txt
    std::uint64_t t = v | (v - 1); // t gets v's least significant 0 bits set to 1
    // Next set to 1 the most significant bit to change, 
    // set to 0 the least significant ones, and add the necessary 1 bits.
    return ((t + 1) | (((~t & -~t) - 1) >> (std::countr_zero(v) + 1)));
}

}