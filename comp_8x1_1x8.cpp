//------------------------------------------------------------------------
// modified from http://qiita.com/kaityo256/items/bf10b0f90809e3d2bf
#include <x86intrin.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
//------------------------------------------------------------------------
int N = 100003;
int M = 203;
const float dt = 0.01f;
const float eps2 = (1.0f / 256.f) * (1.0f / 256.f);
struct float4 { float x, y, z, w; };
float4* __restrict q = nullptr;
float4* __restrict p = nullptr;
int32_t* __restrict list = nullptr;
typedef float v8sf __attribute__((vector_size(32)));
typedef float v4sf __attribute__((vector_size(16)));
typedef int32_t v8si __attribute__((vector_size(32)));
//------------------------------------------------------------------------
void
print256(v8sf r) {
  union {
    v8sf r;
    float elem[8];
  } tmp;
  tmp.r = r;
  std::cerr << std::setprecision(5);
  std::cerr << tmp.elem[0] << " " << tmp.elem[1] << " " << tmp.elem[2] << " " << tmp.elem[3] << " ";
  std::cerr << tmp.elem[4] << " " << tmp.elem[5] << " " << tmp.elem[6] << " " << tmp.elem[7];
  std::cerr << std::endl;
}
//------------------------------------------------------------------------
static inline void
transpose_4x4x2(v8sf& va,
                v8sf& vb,
                v8sf& vc,
                v8sf& vd) {
  v8sf t_a = _mm256_blend_ps(va, _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(vb), 4)), 0xaa);
  v8sf t_b = _mm256_blend_ps(_mm256_castsi256_ps(_mm256_bsrli_epi128(_mm256_castps_si256(va), 4)), vb, 0xaa);
  v8sf t_c = _mm256_blend_ps(vc, _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(vd), 4)), 0xaa);
  v8sf t_d = _mm256_blend_ps(_mm256_castsi256_ps(_mm256_bsrli_epi128(_mm256_castps_si256(vc), 4)), vd, 0xaa);

  va = _mm256_shuffle_ps(t_a, t_c, 0x44);
  vb = _mm256_shuffle_ps(t_b, t_d, 0x44);
  vc = _mm256_shuffle_ps(t_a, t_c, 0xee);
  vd = _mm256_shuffle_ps(t_b, t_d, 0xee);
}
//------------------------------------------------------------------------
static inline void
transpose_4x4x2(const v8sf& va,
                const v8sf& vb,
                const v8sf& vc,
                const v8sf& vd,
                v8sf& vx,
                v8sf& vy,
                v8sf& vz) {
  v8sf t_a = _mm256_blend_ps(va, _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(vb), 4)), 0xaa);
  v8sf t_b = _mm256_blend_ps(_mm256_castsi256_ps(_mm256_bsrli_epi128(_mm256_castps_si256(va), 4)), vb, 0xaa);
  v8sf t_c = _mm256_blend_ps(vc, _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(vd), 4)), 0xaa);
  v8sf t_d = _mm256_blend_ps(_mm256_castsi256_ps(_mm256_bsrli_epi128(_mm256_castps_si256(vc), 4)), vd, 0xaa);

  vx = _mm256_shuffle_ps(t_a, t_c, 0x44);
  vy = _mm256_shuffle_ps(t_b, t_d, 0x44);
  vz = _mm256_shuffle_ps(t_a, t_c, 0xee);
}
//------------------------------------------------------------------------
void
calc_intrin1x8_v1() {
  const v8sf vc24 = _mm256_set1_ps(24.0f * dt);
  const v8sf vc48 = _mm256_set1_ps(48.0f * dt);
  const v8sf veps2 = _mm256_set1_ps(eps2);
  v8sf vpw = _mm256_setzero_ps();
  for (int i = 0; i < N; i++) {
    const v8sf vqxi = _mm256_broadcast_ss(&q[i].x);
    const v8sf vqyi = _mm256_broadcast_ss(&q[i].y);
    const v8sf vqzi = _mm256_broadcast_ss(&q[i].z);

    v8sf vpxi = _mm256_setzero_ps();
    v8sf vpyi = _mm256_setzero_ps();
    v8sf vpzi = _mm256_setzero_ps();

    for (int k = 0; k < (M / 8) * 8; k += 8) {
      const auto j_a = list[k    ];
      const auto j_b = list[k + 1];
      const auto j_c = list[k + 2];
      const auto j_d = list[k + 3];
      const auto j_e = list[k + 4];
      const auto j_f = list[k + 5];
      const auto j_g = list[k + 6];
      const auto j_h = list[k + 7];

      v8si vindex = _mm256_set_epi32(j_h, j_g, j_f, j_e, j_d, j_c, j_b, j_a);
      vindex = _mm256_slli_epi32(vindex, 2);

      v8sf vqxj = _mm256_i32gather_ps((const float*)(&q[0].x),
                                      vindex, 4);
      v8sf vqyj = _mm256_i32gather_ps((const float*)(&q[0].y),
                                      vindex, 4);
      v8sf vqzj = _mm256_i32gather_ps((const float*)(&q[0].z),
                                      vindex, 4);

      v8sf vpxj = _mm256_i32gather_ps((const float*)(&p[0].x),
                                      vindex, 4);
      v8sf vpyj = _mm256_i32gather_ps((const float*)(&p[0].y),
                                      vindex, 4);
      v8sf vpzj = _mm256_i32gather_ps((const float*)(&p[0].z),
                                      vindex, 4);

      v8sf vdx = vqxj - vqxi;
      v8sf vdy = vqyj - vqyi;
      v8sf vdz = vqzj - vqzi;

      v8sf vr2 = veps2 + vdx * vdx + vdy * vdy + vdz * vdz;
      v8sf vr6 = vr2 * vr2 * vr2;
      v8sf vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      vpxi += vdf * vdx;
      vpyi += vdf * vdy;
      vpzi += vdf * vdz;

      vpxj -= vdf * vdx;
      vpyj -= vdf * vdy;
      vpzj -= vdf * vdz;

      transpose_4x4x2(vpxj, vpyj, vpzj, vpw);
      /*
        vpxj = {vpja, vpje}
        vpyj = {vpjb, vpjf}
        vpzj = {vpjc, vpjg}
        vpw  = {vpjd, vpjh}
       */

      _mm256_storeu2_m128((float*)(p + j_e), (float*)(p + j_a), vpxj);
      _mm256_storeu2_m128((float*)(p + j_f), (float*)(p + j_b), vpyj);
      _mm256_storeu2_m128((float*)(p + j_g), (float*)(p + j_c), vpzj);
      _mm256_storeu2_m128((float*)(p + j_h), (float*)(p + j_d), vpw );
    }
    // horizontal sum
    transpose_4x4x2(vpxi, vpyi, vpzi, vpw);
    /*
      vpxi = {vpia, vpie}
      vpyi = {vpib, vpif}
      vpzi = {vpic, vpig}
      vpw  = {vpid, vpih}
     */
    v8sf vpi_hilo = vpxi + vpyi + vpzi + vpw;
    v8sf vpi_lohi = _mm256_permute2f128_ps(vpi_hilo, vpi_hilo, 0x01);
    v8sf vpi = vpi_hilo + vpi_lohi;

    // store
    vpi += (v8sf)(_mm256_castps128_ps256(_mm_load_ps((float*)(p + i))));
    _mm_store_ps((float*)(p + i), _mm256_castps256_ps128(vpi));

    for (int k = (M / 8) * 8; k < M; k++) {
      const auto j = list[k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = (eps2 + dx * dx + dy * dy + dz * dz);
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0f * dt * r6 - 48.0f * dt) / (r6 * r6 * r2);
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
    }
  }
}
//------------------------------------------------------------------------
void
calc_intrin1x8_v2() {
  const v8sf vc24 = _mm256_set1_ps(24.0f * dt);
  const v8sf vc48 = _mm256_set1_ps(48.0f * dt);
  const v8sf veps2 = _mm256_set1_ps(eps2);
  for (int i = 0; i < N; i++) {
    v8sf vqi = _mm256_castps128_ps256(_mm_load_ps((float*)(q + i)));
    vqi = _mm256_insertf128_ps(vqi, _mm256_castps256_ps128(vqi), 1);
    v8sf vpi = _mm256_setzero_ps();

    for (int k = 0; k < (M / 8) * 8; k += 8) {
      const auto j_a = list[k    ];
      const auto j_b = list[k + 1];
      const auto j_c = list[k + 2];
      const auto j_d = list[k + 3];

      const auto j_e = list[k + 4];
      const auto j_f = list[k + 5];
      const auto j_g = list[k + 6];
      const auto j_h = list[k + 7];

      v8sf vpj_ea = _mm256_loadu2_m128((float*)(p + j_e), (float*)(p + j_a));
      v8sf vpj_fb = _mm256_loadu2_m128((float*)(p + j_f), (float*)(p + j_b));
      v8sf vpj_gc = _mm256_loadu2_m128((float*)(p + j_g), (float*)(p + j_c));
      v8sf vpj_hd = _mm256_loadu2_m128((float*)(p + j_h), (float*)(p + j_d));

      v8sf vqj_ea = _mm256_loadu2_m128((float*)(q + j_e), (float*)(q + j_a));
      v8sf vqj_fb = _mm256_loadu2_m128((float*)(q + j_f), (float*)(q + j_b));
      v8sf vqj_gc = _mm256_loadu2_m128((float*)(q + j_g), (float*)(q + j_c));
      v8sf vqj_hd = _mm256_loadu2_m128((float*)(q + j_h), (float*)(q + j_d));

      v8sf vdq_ea = _mm256_sub_ps(vqj_ea, vqi);
      v8sf vdq_fb = _mm256_sub_ps(vqj_fb, vqi);
      v8sf vdq_gc = _mm256_sub_ps(vqj_gc, vqi);
      v8sf vdq_hd = _mm256_sub_ps(vqj_hd, vqi);

      v8sf vdx, vdy, vdz;
      transpose_4x4x2(vdq_ea, vdq_fb, vdq_gc, vdq_hd,
                      vdx, vdy, vdz);

      v8sf vr2 = _mm256_fmadd_ps(vdz, vdz,
                                 _mm256_fmadd_ps(vdy, vdy,
                                                 _mm256_fmadd_ps(vdx, vdx, veps2)));
      v8sf vr6 = _mm256_mul_ps(vr2,
                               _mm256_mul_ps(vr2, vr2));
      v8sf vdf = _mm256_div_ps(_mm256_fmsub_ps(vc24, vr6, vc48),
                               _mm256_mul_ps(_mm256_mul_ps(vr6, vr6), vr2));

      v8sf vdf_ea = _mm256_mul_ps(_mm256_permute_ps(vdf, 0x00), vdq_ea);
      v8sf vdf_fb = _mm256_mul_ps(_mm256_permute_ps(vdf, 0x55), vdq_fb);
      v8sf vdf_gc = _mm256_mul_ps(_mm256_permute_ps(vdf, 0xaa), vdq_gc);
      v8sf vdf_hd = _mm256_mul_ps(_mm256_permute_ps(vdf, 0xff), vdq_hd);

      vpi = _mm256_add_ps(vpi, vdf_ea);
      vpj_ea = _mm256_sub_ps(vpj_ea, vdf_ea);

      vpi = _mm256_add_ps(vpi, vdf_fb);
      vpj_fb = _mm256_sub_ps(vpj_fb, vdf_fb);

      vpi = _mm256_add_ps(vpi, vdf_gc);
      vpj_gc = _mm256_sub_ps(vpj_gc, vdf_gc);

      vpi = _mm256_add_ps(vpi, vdf_hd);
      vpj_hd = _mm256_sub_ps(vpj_hd, vdf_hd);

      _mm256_storeu2_m128((float*)(p + j_e), (float*)(p + j_a), vpj_ea);
      _mm256_storeu2_m128((float*)(p + j_f), (float*)(p + j_b), vpj_fb);
      _mm256_storeu2_m128((float*)(p + j_g), (float*)(p + j_c), vpj_gc);
      _mm256_storeu2_m128((float*)(p + j_h), (float*)(p + j_d), vpj_hd);
    }
    vpi = _mm256_add_ps(_mm256_permute2f128_ps(vpi, vpi, 0x01), vpi);
    vpi = _mm256_add_ps(vpi, _mm256_castps128_ps256(_mm_load_ps((float*)(p + i))));
    _mm_store_ps((float*)(p + i), _mm256_castps256_ps128(vpi));

    for (int k = (M / 8) * 8; k < M; k++) {
      const auto j = list[k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = (eps2 + dx * dx + dy * dy + dz * dz);
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0f * dt * r6 - 48.0f * dt) / (r6 * r6 * r2);
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
    }
  }
}
//------------------------------------------------------------------------
void
calc_intrin8x1_v1() {
  const v8sf vc24 = _mm256_set1_ps(24.0f * dt);
  const v8sf vc48 = _mm256_set1_ps(48.0f * dt);
  const v8sf veps2 = _mm256_set1_ps(eps2);
  v8sf vpw = _mm256_setzero_ps();
  for (int i = 0; i < (N / 8) * 8; i += 8) {
    v8si vindex = _mm256_set_epi32(i + 7, i + 6, i + 5, i + 4,
                                   i + 3, i + 2, i + 1, i);
    vindex = _mm256_slli_epi32(vindex, 2);

    v8sf vqxi = _mm256_i32gather_ps((const float*)(&q[0].x),
                                    vindex, 4);
    v8sf vqyi = _mm256_i32gather_ps((const float*)(&q[0].y),
                                    vindex, 4);
    v8sf vqzi = _mm256_i32gather_ps((const float*)(&q[0].z),
                                    vindex, 4);

    v8sf vpxi = _mm256_setzero_ps();
    v8sf vpyi = _mm256_setzero_ps();
    v8sf vpzi = _mm256_setzero_ps();

    for (int k = 0; k < M; k++) {
      const auto j = list[k];

      v8sf vqxj = _mm256_broadcast_ss(&q[j].x);
      v8sf vqyj = _mm256_broadcast_ss(&q[j].y);
      v8sf vqzj = _mm256_broadcast_ss(&q[j].z);

      v8sf vpxj = _mm256_broadcast_ss(&p[j].x);
      v8sf vpyj = _mm256_broadcast_ss(&p[j].y);
      v8sf vpzj = _mm256_broadcast_ss(&p[j].z);

      v8sf vdx = _mm256_sub_ps(vqxj, vqxi);
      v8sf vdy = _mm256_sub_ps(vqyj, vqyi);
      v8sf vdz = _mm256_sub_ps(vqzj, vqzi);

      v8sf vr2 = _mm256_fmadd_ps(vdz, vdz,
                                 _mm256_fmadd_ps(vdy, vdy,
                                                 _mm256_fmadd_ps(vdx, vdx, veps2)));
      v8sf vr6 = _mm256_mul_ps(vr2,
                               _mm256_mul_ps(vr2, vr2));
      v8sf vdf = _mm256_div_ps(_mm256_fmsub_ps(vc24, vr6, vc48),
                               _mm256_mul_ps(_mm256_mul_ps(vr6, vr6), vr2));

      vpxi = _mm256_fmadd_ps(vdf, vdx, vpxi);
      vpyi = _mm256_fmadd_ps(vdf, vdy, vpyi);
      vpzi = _mm256_fmadd_ps(vdf, vdz, vpzi);

      vpxj = _mm256_fnmadd_ps(vdf, vdx, vpxj);
      vpyj = _mm256_fnmadd_ps(vdf, vdy, vpyj);
      vpzj = _mm256_fnmadd_ps(vdf, vdz, vpzj);

      transpose_4x4x2(vpxj, vpyj, vpzj, vpw);
      v8sf vpj_hilo = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(vpxj, vpyj),
                                                  vpzj), vpw);
      v8sf vpj_lohi = _mm256_permute2f128_ps(vpj_hilo, vpj_hilo, 0x01);
      v8sf vpj = _mm256_add_ps(vpj_hilo, vpj_lohi);

      _mm_store_ps((float*)(p + j), _mm256_castps256_ps128(vpj));
    }
    vpxi += (v8sf)(_mm256_i32gather_ps((const float*)(&p[0].x),
                                       vindex, 4));
    vpyi += (v8sf)(_mm256_i32gather_ps((const float*)(&p[0].y),
                                       vindex, 4));
    vpzi += (v8sf)(_mm256_i32gather_ps((const float*)(&p[0].z),
                                                  vindex, 4));

    transpose_4x4x2(vpxi, vpyi, vpzi, vpw);

    _mm256_storeu2_m128((float*)(p + i + 4), (float*)(p + i    ), vpxi);
    _mm256_storeu2_m128((float*)(p + i + 5), (float*)(p + i + 1), vpyi);
    _mm256_storeu2_m128((float*)(p + i + 6), (float*)(p + i + 2), vpzi);
    _mm256_storeu2_m128((float*)(p + i + 7), (float*)(p + i + 3), vpw);
  }
  for (int i = (N / 8) * 8; i < N; i++) {
    const v8sf vqxi = _mm256_broadcast_ss(&q[i].x);
    const v8sf vqyi = _mm256_broadcast_ss(&q[i].y);
    const v8sf vqzi = _mm256_broadcast_ss(&q[i].z);

    v8sf vpxi = _mm256_setzero_ps();
    v8sf vpyi = _mm256_setzero_ps();
    v8sf vpzi = _mm256_setzero_ps();

    for (int k = 0; k < (M / 8) * 8; k += 8) {
      const auto j_a = list[k    ];
      const auto j_b = list[k + 1];
      const auto j_c = list[k + 2];
      const auto j_d = list[k + 3];
      const auto j_e = list[k + 4];
      const auto j_f = list[k + 5];
      const auto j_g = list[k + 6];
      const auto j_h = list[k + 7];

      v8si vindex = _mm256_set_epi32(j_h, j_g, j_f, j_e, j_d, j_c, j_b, j_a);
      vindex = _mm256_slli_epi32(vindex, 2);

      v8sf vqxj = _mm256_i32gather_ps((const float*)(&q[0].x),
                                      vindex, 4);
      v8sf vqyj = _mm256_i32gather_ps((const float*)(&q[0].y),
                                      vindex, 4);
      v8sf vqzj = _mm256_i32gather_ps((const float*)(&q[0].z),
                                      vindex, 4);

      v8sf vpxj = _mm256_i32gather_ps((const float*)(&p[0].x),
                                      vindex, 4);
      v8sf vpyj = _mm256_i32gather_ps((const float*)(&p[0].y),
                                      vindex, 4);
      v8sf vpzj = _mm256_i32gather_ps((const float*)(&p[0].z),
                                      vindex, 4);

      v8sf vdx = vqxj - vqxi;
      v8sf vdy = vqyj - vqyi;
      v8sf vdz = vqzj - vqzi;

      v8sf vr2 = veps2 + vdx * vdx + vdy * vdy + vdz * vdz;
      v8sf vr6 = vr2 * vr2 * vr2;
      v8sf vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      vpxi += vdf * vdx;
      vpyi += vdf * vdy;
      vpzi += vdf * vdz;

      vpxj -= vdf * vdx;
      vpyj -= vdf * vdy;
      vpzj -= vdf * vdz;

      transpose_4x4x2(vpxj, vpyj, vpzj, vpw);
      /*
        vpxj = {vpja, vpje}
        vpyj = {vpjb, vpjf}
        vpzj = {vpjc, vpjg}
        vpw  = {vpjd, vpjh}
       */

      _mm256_storeu2_m128((float*)(p + j_e), (float*)(p + j_a), vpxj);
      _mm256_storeu2_m128((float*)(p + j_f), (float*)(p + j_b), vpyj);
      _mm256_storeu2_m128((float*)(p + j_g), (float*)(p + j_c), vpzj);
      _mm256_storeu2_m128((float*)(p + j_h), (float*)(p + j_d), vpw );
    }
    // horizontal sum
    transpose_4x4x2(vpxi, vpyi, vpzi, vpw);
    /*
      vpxi = {vpia, vpie}
      vpyi = {vpib, vpif}
      vpzi = {vpic, vpig}
      vpw  = {vpid, vpih}
     */
    v8sf vpi_hilo = vpxi + vpyi + vpzi + vpw;
    v8sf vpi_lohi = _mm256_permute2f128_ps(vpi_hilo, vpi_hilo, 0x01);
    v8sf vpi = vpi_hilo + vpi_lohi;

    // store
    vpi += (v8sf)(_mm256_castps128_ps256(_mm_load_ps((float*)(p + i))));
    _mm_store_ps((float*)(p + i), _mm256_castps256_ps128(vpi));

    for (int k = (M / 8) * 8; k < M; k++) {
      const auto j = list[k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = (eps2 + dx * dx + dy * dy + dz * dz);
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0f * dt * r6 - 48.0f * dt) / (r6 * r6 * r2);
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
    }
  }
}
//------------------------------------------------------------------------
void
calc_intrin8x1_v2() {
  const v8sf vc24 = _mm256_set1_ps(24.0f * dt);
  const v8sf vc48 = _mm256_set1_ps(48.0f * dt);
  const v8sf veps2 = _mm256_set1_ps(eps2);
  for (int i = 0; i < (N / 8) * 8; i += 8) {
    v8sf vqi_ba = _mm256_load_ps((float*)(q + i    ));
    v8sf vqi_dc = _mm256_load_ps((float*)(q + i + 2));
    v8sf vqi_fe = _mm256_load_ps((float*)(q + i + 4));
    v8sf vqi_hg = _mm256_load_ps((float*)(q + i + 6));

    v8sf vpi_ba = _mm256_setzero_ps();
    v8sf vpi_dc = _mm256_setzero_ps();
    v8sf vpi_fe = _mm256_setzero_ps();
    v8sf vpi_hg = _mm256_setzero_ps();

    for (int k = 0; k < M; k++) {
      const auto j = list[k];

      v8sf vqj = _mm256_castps128_ps256(_mm_load_ps((float*)(q + j)));
      vqj = _mm256_insertf128_ps(vqj, _mm256_castps256_ps128(vqj), 1);
      v8sf vpj = _mm256_castps128_ps256(_mm_load_ps((float*)(p + j)));
      vpj = _mm256_insertf128_ps(vpj, _mm256_castps256_ps128(vpj), 1);

      v8sf vdq_ba = _mm256_sub_ps(vqj, vqi_ba);
      v8sf vdq_dc = _mm256_sub_ps(vqj, vqi_dc);
      v8sf vdq_fe = _mm256_sub_ps(vqj, vqi_fe);
      v8sf vdq_hg = _mm256_sub_ps(vqj, vqi_hg);

      v8sf vdx, vdy, vdz;
      transpose_4x4x2(vdq_ba, vdq_dc, vdq_fe, vdq_hg,
                      vdx, vdy, vdz);

      v8sf vr2 = _mm256_fmadd_ps(vdz, vdz,
                                 _mm256_fmadd_ps(vdy, vdy,
                                                 _mm256_fmadd_ps(vdx, vdx, veps2)));
      v8sf vr6 = _mm256_mul_ps(vr2,
                               _mm256_mul_ps(vr2, vr2));
      v8sf vdf = _mm256_div_ps(_mm256_fmsub_ps(vc24, vr6, vc48),
                               _mm256_mul_ps(_mm256_mul_ps(vr6, vr6), vr2));

      v8sf vdf_ba = _mm256_mul_ps(_mm256_permute_ps(vdf, 0x00), vdq_ba);
      v8sf vdf_dc = _mm256_mul_ps(_mm256_permute_ps(vdf, 0x55), vdq_dc);
      v8sf vdf_fe = _mm256_mul_ps(_mm256_permute_ps(vdf, 0xaa), vdq_fe);
      v8sf vdf_hg = _mm256_mul_ps(_mm256_permute_ps(vdf, 0xff), vdq_hg);

      vpi_ba = _mm256_add_ps(vpi_ba, vdf_ba);
      vpi_dc = _mm256_add_ps(vpi_dc, vdf_dc);
      vpi_fe = _mm256_add_ps(vpi_fe, vdf_fe);
      vpi_hg = _mm256_add_ps(vpi_hg, vdf_hg);

      vpj = _mm256_sub_ps(vpj, vdf_ba);
      vpj = _mm256_sub_ps(vpj, vdf_dc);
      vpj = _mm256_sub_ps(vpj, vdf_fe);
      vpj = _mm256_sub_ps(vpj, vdf_hg);

      vpj = _mm256_add_ps(_mm256_permute2f128_ps(vpj, vpj, 0x01), vpj);

      _mm_store_ps((float*)(p + j), _mm256_castps256_ps128(vpj));
    }

    vpi_ba += (v8sf)(_mm256_load_ps((float*)(p + i)));
    vpi_dc += (v8sf)(_mm256_load_ps((float*)(p + i + 2)));
    vpi_fe += (v8sf)(_mm256_load_ps((float*)(p + i + 4)));
    vpi_hg += (v8sf)(_mm256_load_ps((float*)(p + i + 6)));

    _mm256_store_ps((float*)(p + i    ), vpi_ba);
    _mm256_store_ps((float*)(p + i + 2), vpi_dc);
    _mm256_store_ps((float*)(p + i + 4), vpi_fe);
    _mm256_store_ps((float*)(p + i + 6), vpi_hg);
  }
  for (int i = (N / 8) * 8; i < N; i++) {
    v8sf vqi = _mm256_castps128_ps256(_mm_load_ps((float*)(q + i)));
    vqi = _mm256_insertf128_ps(vqi, _mm256_castps256_ps128(vqi), 1);
    v8sf vpi = _mm256_setzero_ps();

    for (int k = 0; k < (M / 8) * 8; k += 8) {
      const auto j_a = list[k    ];
      const auto j_b = list[k + 1];
      const auto j_c = list[k + 2];
      const auto j_d = list[k + 3];

      const auto j_e = list[k + 4];
      const auto j_f = list[k + 5];
      const auto j_g = list[k + 6];
      const auto j_h = list[k + 7];

      v8sf vpj_ea = _mm256_loadu2_m128((float*)(p + j_e), (float*)(p + j_a));
      v8sf vpj_fb = _mm256_loadu2_m128((float*)(p + j_f), (float*)(p + j_b));
      v8sf vpj_gc = _mm256_loadu2_m128((float*)(p + j_g), (float*)(p + j_c));
      v8sf vpj_hd = _mm256_loadu2_m128((float*)(p + j_h), (float*)(p + j_d));

      v8sf vqj_ea = _mm256_loadu2_m128((float*)(q + j_e), (float*)(q + j_a));
      v8sf vqj_fb = _mm256_loadu2_m128((float*)(q + j_f), (float*)(q + j_b));
      v8sf vqj_gc = _mm256_loadu2_m128((float*)(q + j_g), (float*)(q + j_c));
      v8sf vqj_hd = _mm256_loadu2_m128((float*)(q + j_h), (float*)(q + j_d));

      v8sf vdq_ea = _mm256_sub_ps(vqj_ea, vqi);
      v8sf vdq_fb = _mm256_sub_ps(vqj_fb, vqi);
      v8sf vdq_gc = _mm256_sub_ps(vqj_gc, vqi);
      v8sf vdq_hd = _mm256_sub_ps(vqj_hd, vqi);

      v8sf vdx, vdy, vdz;
      transpose_4x4x2(vdq_ea, vdq_fb, vdq_gc, vdq_hd,
                      vdx, vdy, vdz);

      v8sf vr2 = _mm256_fmadd_ps(vdz, vdz,
                                 _mm256_fmadd_ps(vdy, vdy,
                                                 _mm256_fmadd_ps(vdx, vdx, veps2)));
      v8sf vr6 = _mm256_mul_ps(vr2,
                               _mm256_mul_ps(vr2, vr2));
      v8sf vdf = _mm256_div_ps(_mm256_fmsub_ps(vc24, vr6, vc48),
                               _mm256_mul_ps(_mm256_mul_ps(vr6, vr6), vr2));

      v8sf vdf_ea = _mm256_mul_ps(_mm256_permute_ps(vdf, 0x00), vdq_ea);
      v8sf vdf_fb = _mm256_mul_ps(_mm256_permute_ps(vdf, 0x55), vdq_fb);
      v8sf vdf_gc = _mm256_mul_ps(_mm256_permute_ps(vdf, 0xaa), vdq_gc);
      v8sf vdf_hd = _mm256_mul_ps(_mm256_permute_ps(vdf, 0xff), vdq_hd);

      vpi = _mm256_add_ps(vpi, vdf_ea);
      vpj_ea = _mm256_sub_ps(vpj_ea, vdf_ea);

      vpi = _mm256_add_ps(vpi, vdf_fb);
      vpj_fb = _mm256_sub_ps(vpj_fb, vdf_fb);

      vpi = _mm256_add_ps(vpi, vdf_gc);
      vpj_gc = _mm256_sub_ps(vpj_gc, vdf_gc);

      vpi = _mm256_add_ps(vpi, vdf_hd);
      vpj_hd = _mm256_sub_ps(vpj_hd, vdf_hd);

      _mm256_storeu2_m128((float*)(p + j_e), (float*)(p + j_a), vpj_ea);
      _mm256_storeu2_m128((float*)(p + j_f), (float*)(p + j_b), vpj_fb);
      _mm256_storeu2_m128((float*)(p + j_g), (float*)(p + j_c), vpj_gc);
      _mm256_storeu2_m128((float*)(p + j_h), (float*)(p + j_d), vpj_hd);
    }
    vpi = _mm256_add_ps(_mm256_permute2f128_ps(vpi, vpi, 0x01), vpi);
    vpi = _mm256_add_ps(vpi, _mm256_castps128_ps256(_mm_load_ps((float*)(p + i))));
    _mm_store_ps((float*)(p + i), _mm256_castps256_ps128(vpi));

    for (int k = (M / 8) * 8; k < M; k++) {
      const auto j = list[k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = (eps2 + dx * dx + dy * dy + dz * dz);
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0f * dt * r6 - 48.0f * dt) / (r6 * r6 * r2);
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
    }
  }
}
//------------------------------------------------------------------------
void
init(void) {
  std::mt19937 mt;
  std::uniform_real_distribution<> ud(0.0, 10.0);
  for (int i = 0; i < N; i++) {
    q[i].x = ud(mt);
    q[i].y = ud(mt);
    q[i].z = ud(mt);
    p[i].x = 0.0f;
    p[i].y = 0.0f;
    p[i].z = 0.0f;
  }
}
//------------------------------------------------------------------------
void
reference(void) {
  const auto c24 = 24.0f * dt;
  const auto c48 = 48.0f * dt;
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < M; k++) {
      const auto j = list[k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = (eps2 + dx * dx + dy * dy + dz * dz);
      const auto r6 = r2 * r2 * r2;
      const auto df = (c24 * r6 - c48) / (r6 * r6 * r2);
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
    }
  }
}
//------------------------------------------------------------------------
void
gen_neighlist(const int seed) {
  std::mt19937 mt(seed);
  std::uniform_int_distribution<> dist(0, N - 1);
  std::generate(list, list + M, [&mt, &dist](){return dist(mt);});
}
//------------------------------------------------------------------------
#define BENCH(func, num_loop)                                           \
  do {                                                                  \
    const auto beg = std::chrono::system_clock::now();                  \
    for (int i = 0; i < num_loop; i++) {                                \
      func();                                                           \
    }                                                                   \
    const auto end = std::chrono::system_clock::now();                  \
    std::cerr << #func << ":\n";                                        \
    std::cout <<                                                        \
      std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() << " [microseconds]\n"; \
  } while(0)
//------------------------------------------------------------------------
int
main(int argc, char* argv[]) {
  if (argc >= 3) {
    N = std::atoi(argv[1]);
    M = std::atoi(argv[2]);
  }

  int rand_seed = 10;
  if (argc >= 4) {
    rand_seed = std::atoi(argv[3]);
  }

  bool verbose = true;
  if (argc == 5)
    if (std::atoi(argv[4]))
      verbose = true;

  q = new float4 [N];
  p = new float4 [N];
  posix_memalign((void**)(&list), 32, sizeof(int32_t) * M);

  gen_neighlist(rand_seed);

  std::cout << N << " " << M << " ";

  const int num_loop = 1;

  init();
#ifdef USE1x8_v1
  BENCH(calc_intrin1x8_v1, num_loop);
#elif USE1x8_v2
  BENCH(calc_intrin1x8_v2, num_loop);
#elif USE8x1_v1
  BENCH(calc_intrin8x1_v1, num_loop);
#elif USE8x1_v2
  BENCH(calc_intrin8x1_v2, num_loop);
#elif REFERENCE
  BENCH(reference, num_loop);
#endif

  if (verbose) {
    std::cerr << std::setprecision(5);
    for (int i = 0; i < 10; i++) {
      std::cerr << p[i].x << " " << p[i].y << " " << p[i].z << "\n";
    }
  }

  delete [] q;
  delete [] p;
  free(list);
}
//------------------------------------------------------------------------
