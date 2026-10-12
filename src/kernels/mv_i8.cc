// int8 weights x int16 vector -> int32 matrix-vector multiply for AIE2 (XDNA1).
//
// A tile (m x k, int8) arrives "quad-transposed" (set up by the DMA, see matvec_i8.py):
//   byte (q*m + i)*4 + j  =  A[i][4q + j]
// so one 32-byte load holds 8 rows x 4 columns. The core widens it to int16,
// multiplies it with the matching 4 values of b repeated per row, and sums the
// four lanes of each row at the end. Halves the weight bytes streamed from RAM.
// Derived in part from mlir-aie aie_kernels/linalg/mv_i16.cc
// Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define NOCPP

#include <stdint.h>
#include <aie_api/aie.hpp>

template <unsigned m, unsigned k>
void matvec_i8(const int8_t *__restrict a, const int16_t *__restrict b,
               int32_t *__restrict c) {
  static_assert(m % 8 == 0 && k % 4 == 0);
  const int32_t *__restrict b32 = (const int32_t *)b;
  event0();
  for (unsigned row = 0; row < m; row += 8) {
    aie::accum<acc32, 32> acc = aie::zeros<acc32, 32>();
    const int8_t *__restrict a_ptr = a + row * 4;
    for (unsigned q = 0; q < k / 4; q++) {
      const aie::vector<int8, 32> a8 = aie::load_v<32>(a_ptr);
      const aie::vector<int16, 32> a16 = aie::unpack(a8);
      // pattern [b0 b1 b2 b3] x 8 : two 32-bit words (b0b1, b2b3) zipped
      const auto w0 = aie::broadcast<int32, 16>(b32[2 * q]);
      const auto w1 = aie::broadcast<int32, 16>(b32[2 * q + 1]);
      const auto z = aie::interleave_zip(w0, w1, 1);
      const aie::vector<int16, 32> pat = z.first.template cast_to<int16>();
      acc = aie::mac(acc, a16, pat);
      a_ptr += m * 4;
    }
    // lane 4i+j -> row i : add the four columns' partial sums
    const aie::vector<int32, 32> v = acc.template to_vector<int32>();
    const aie::vector<int32, 16> s1 =
        aie::add(aie::filter_even(v, 1), aie::filter_odd(v, 1));
    const aie::vector<int32, 8> s2 =
        aie::add(aie::filter_even(s1, 1), aie::filter_odd(s1, 1));
    aie::store_v(c + row, aie::add(aie::load_v<8>(c + row), s2));
  }
  event1();
}

extern "C" {
#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K
#define DIM_K 32
#endif
void matvec_i8_i16_i32(int8_t *a, int16_t *b, int32_t *c) {
  matvec_i8<DIM_M, DIM_K>(a, b, c);
}
}
