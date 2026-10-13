// int8 weights x NV int16 vectors -> NV int32 vectors, for AIE2 (XDNA1).
// Same weight layout and widening as mv_i8.cc, but one weight load feeds NV accumulators, so the weights are streamed
// once for NV tokens (decode is bandwidth-bound: this is what makes batched verification of drafted tokens cheap).
//   b: NV vectors of k int16 each, [v][k]       c: NV vectors of m int32 each, [v][m]
// Derived in part from mlir-aie aie_kernels/linalg/mv_i16.cc
// Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define NOCPP

#include <stdint.h>
#include <aie_api/aie.hpp>

#ifndef NV
#define NV 4
#endif

// The pattern [b0 b1 b2 b3] x 8 of each (column-quad, vector) is the same for every group of 8 rows, so it is built once per
// tile and kept in local memory; the row loop then only loads it.
template <unsigned m, unsigned k>
void matvec_i8_nv(const int8_t *__restrict a, const int16_t *__restrict b,
                  int32_t *__restrict c) {
  static_assert(m % 8 == 0 && k % 4 == 0);
  alignas(32) static int16_t pat[NV * (k / 4) * 32];
  const int32_t *__restrict b32 = (const int32_t *)b;
  event0();
  for (unsigned v = 0; v < NV; v++)
    for (unsigned q = 0; q < k / 4; q++) {
      const auto w0 = aie::broadcast<int32, 16>(b32[v * (k / 2) + 2 * q]);
      const auto w1 = aie::broadcast<int32, 16>(b32[v * (k / 2) + 2 * q + 1]);
      const auto z = aie::interleave_zip(w0, w1, 1);
      aie::store_v(pat + (v * (k / 4) + q) * 32, z.first.template cast_to<int16>());
    }
  for (unsigned row = 0; row < m; row += 8) {
    aie::accum<acc32, 32> acc[NV];
    for (unsigned v = 0; v < NV; v++) acc[v] = aie::zeros<acc32, 32>();
    const int8_t *__restrict a_ptr = a + row * 4;
    for (unsigned q = 0; q < k / 4; q++) {
      const aie::vector<int16, 32> a16 = aie::unpack(aie::load_v<32>(a_ptr));
      for (unsigned v = 0; v < NV; v++)
        acc[v] = aie::mac(acc[v], a16, aie::load_v<32>(pat + (v * (k / 4) + q) * 32));
      a_ptr += m * 4;
    }
    for (unsigned v = 0; v < NV; v++) {
      const aie::vector<int32, 32> t = acc[v].template to_vector<int32>();
      const aie::vector<int32, 16> s1 = aie::add(aie::filter_even(t, 1), aie::filter_odd(t, 1));
      const aie::vector<int32, 8> s2 = aie::add(aie::filter_even(s1, 1), aie::filter_odd(s1, 1));
      aie::store_v(c + v * m + row, aie::add(aie::load_v<8>(c + v * m + row), s2));
    }
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
void matvec_i8_nv_i16_i32(int8_t *a, int16_t *b, int32_t *c) {
  matvec_i8_nv<DIM_M, DIM_K>(a, b, c);
}
}
