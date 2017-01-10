//------------------------------------------------------------------------
// code is modified from http://qiita.com/kaityo256/items/bf10b0f90809e3d2bf
#include <x86intrin.h>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <iomanip>
//------------------------------------------------------------------------
int N = 20000;
const double dt = 0.01;
struct double4 { double x, y, z, w; };
double4* __restrict q = nullptr;
double4* __restrict p = nullptr;
typedef double v4df __attribute__((vector_size(32)));
typedef int64_t v4di __attribute__((vector_size(32)));
//------------------------------------------------------------------------
void
print256(v4df r) {
  union {
    v4df r;
    double elem[4];
  } tmp;
  tmp.r = r;
  std::cerr << std::setprecision(9);
  std::cerr << tmp.elem[0] << " " << tmp.elem[1] << " " << tmp.elem[2] << " " << tmp.elem[3] << std::endl;
}
//------------------------------------------------------------------------
void
calc_intrin(void) {
  const double c24[4] = {24 * dt, 24 * dt, 24 * dt, 24 * dt};
  const double c48[4] = {48 * dt, 48 * dt, 48 * dt, 48 * dt};
  const v4df vc24 = _mm256_load_pd((double*)(c24));
  const v4df vc48 = _mm256_load_pd((double*)(c48));
  for (int i = 0; i < N - 1; i++) {
    const v4df vqi = _mm256_load_pd((double*)(q + i));
    v4df vpi = _mm256_load_pd((double*)(p + i));
    for (int j = i + 1; j < N - 3; j += 4) {
      v4df vqj_a = _mm256_load_pd((double*)(q + j));
      v4df vdq_a = (vqj_a - vqi);
      v4df vd1_a = vdq_a * vdq_a;
      v4df vd2_a = _mm256_permute4x64_pd(vd1_a, 201);
      v4df vd3_a = _mm256_permute4x64_pd(vd1_a, 210);
      v4df vr2_a = vd1_a + vd2_a + vd3_a;

      v4df vqj_b = _mm256_load_pd((double*)(q + j + 1));
      v4df vdq_b = (vqj_b - vqi);
      v4df vd1_b = vdq_b * vdq_b;
      v4df vd2_b = _mm256_permute4x64_pd(vd1_b, 201);
      v4df vd3_b = _mm256_permute4x64_pd(vd1_b, 210);
      v4df vr2_b = vd1_b + vd2_b + vd3_b;

      v4df vqj_c = _mm256_load_pd((double*)(q + j + 2));
      v4df vdq_c = (vqj_c - vqi);
      v4df vd1_c = vdq_c * vdq_c;
      v4df vd2_c = _mm256_permute4x64_pd(vd1_c, 201);
      v4df vd3_c = _mm256_permute4x64_pd(vd1_c, 210);
      v4df vr2_c = vd1_c + vd2_c + vd3_c;

      v4df vqj_d = _mm256_load_pd((double*)(q + j + 3));
      v4df vdq_d = (vqj_d - vqi);
      v4df vd1_d = vdq_d * vdq_d;
      v4df vd2_d = _mm256_permute4x64_pd(vd1_d, 201);
      v4df vd3_d = _mm256_permute4x64_pd(vd1_d, 210);
      v4df vr2_d = vd1_d + vd2_d + vd3_d;

      v4df vr2_ac = _mm256_unpacklo_pd(vr2_a, vr2_c);
      v4df vr2_bd = _mm256_unpacklo_pd(vr2_b, vr2_d);
      v4df vr2 = _mm256_shuffle_pd(vr2_ac, vr2_bd, 12);
      v4df vr6 = vr2 * vr2 * vr2;
      v4df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      v4df vdf_a = _mm256_permute4x64_pd(vdf, 0);
      v4df vdf_b = _mm256_permute4x64_pd(vdf, 85);
      v4df vdf_c = _mm256_permute4x64_pd(vdf, 170);
      v4df vdf_d = _mm256_permute4x64_pd(vdf, 255);

      v4df vpj_a = _mm256_load_pd((double*)(p + j));
      vpi += vdq_a * vdf_a;
      vpj_a -= vdq_a * vdf_a;
      _mm256_store_pd((double*)(p + j), vpj_a);

      v4df vpj_b = _mm256_load_pd((double*)(p + j + 1));
      vpi += vdq_b * vdf_b;
      vpj_b -= vdq_b * vdf_b;
      _mm256_store_pd((double*)(p + j + 1), vpj_b);

      v4df vpj_c = _mm256_load_pd((double*)(p + j + 2));
      vpi += vdq_c * vdf_c;
      vpj_c -= vdq_c * vdf_c;
      _mm256_store_pd((double*)(p + j + 2), vpj_c);

      v4df vpj_d = _mm256_load_pd((double*)(p + j + 3));
      vpi += vdq_d * vdf_d;
      vpj_d -= vdq_d * vdf_d;
      _mm256_store_pd((double*)(p + j + 3), vpj_d);
    }
    _mm256_store_pd((double*)(p + i), vpi);
    for (int j = N - (5 - i) % 4; j < N; j++) {
      const double dx = q[j].x - q[i].x;
      const double dy = q[j].y - q[i].y;
      const double dz = q[j].z - q[i].z;
      const double r2 = (dx * dx + dy * dy + dz * dz);
      double r6 = r2 * r2 * r2;
      double df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
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
calc_intrin1x4(void) {
  const v4df vc24 = _mm256_set_pd(24 * dt, 24 * dt, 24 * dt, 24 * dt);
  const v4df vc48 = _mm256_set_pd(48 * dt, 48 * dt, 48 * dt, 48 * dt);
  for (int i = 0; i < N - 1; i++) {
    const v4df vqi = _mm256_load_pd((double*)(q + i));
    v4df vpi = _mm256_load_pd((double*)(p + i));
    for (int j = i + 1; j < N - 3; j += 4) {
      v4df vqj_a = _mm256_load_pd((double*)(q + j));
      v4df vdq_a = (vqj_a - vqi);

      v4df vqj_b = _mm256_load_pd((double*)(q + j + 1));
      v4df vdq_b = (vqj_b - vqi);

      v4df vqj_c = _mm256_load_pd((double*)(q + j + 2));
      v4df vdq_c = (vqj_c - vqi);

      v4df vqj_d = _mm256_load_pd((double*)(q + j + 3));
      v4df vdq_d = (vqj_d - vqi);

      v4df tmp0 = _mm256_unpacklo_pd(vdq_a, vdq_b);
      v4df tmp1 = _mm256_unpackhi_pd(vdq_a, vdq_b);
      v4df tmp2 = _mm256_unpacklo_pd(vdq_c, vdq_d);
      v4df tmp3 = _mm256_unpackhi_pd(vdq_c, vdq_d);

      v4df vdx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
      v4df vdy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
      v4df vdz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);

      v4df vr2 = vdx * vdx + vdy * vdy + vdz * vdz;
      v4df vr6 = vr2 * vr2 * vr2;
      v4df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      v4df vdf_a = _mm256_permute4x64_pd(vdf, 0);
      v4df vdf_b = _mm256_permute4x64_pd(vdf, 85);
      v4df vdf_c = _mm256_permute4x64_pd(vdf, 170);
      v4df vdf_d = _mm256_permute4x64_pd(vdf, 255);

      v4df vpj_a = _mm256_load_pd((double*)(p + j));
      vpi += vdq_a * vdf_a;
      vpj_a -= vdq_a * vdf_a;
      _mm256_store_pd((double*)(p + j), vpj_a);

      v4df vpj_b = _mm256_load_pd((double*)(p + j + 1));
      vpi += vdq_b * vdf_b;
      vpj_b -= vdq_b * vdf_b;
      _mm256_store_pd((double*)(p + j + 1), vpj_b);

      v4df vpj_c = _mm256_load_pd((double*)(p + j + 2));
      vpi += vdq_c * vdf_c;
      vpj_c -= vdq_c * vdf_c;
      _mm256_store_pd((double*)(p + j + 2), vpj_c);

      v4df vpj_d = _mm256_load_pd((double*)(p + j + 3));
      vpi += vdq_d * vdf_d;
      vpj_d -= vdq_d * vdf_d;
      _mm256_store_pd((double*)(p + j + 3), vpj_d);
    }
    _mm256_store_pd((double*)(p + i), vpi);
    for (int j = N - (5 - i) % 4; j < N; j++) {
      const double dx = q[j].x - q[i].x;
      const double dy = q[j].y - q[i].y;
      const double dz = q[j].z - q[i].z;
      const double r2 = (dx * dx + dy * dy + dz * dz);
      double r6 = r2 * r2 * r2;
      double df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
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
calc_intrin4x1(void) {
  const v4df vc24 = _mm256_set_pd(24 * dt, 24 * dt, 24 * dt, 24 * dt);
  const v4df vc48 = _mm256_set_pd(48 * dt, 48 * dt, 48 * dt, 48 * dt);
  for (int i = 0; i < ((N - 1) / 4) * 4; i += 4) {
    const v4df vqi_a = _mm256_load_pd((double*)(q + i));
    v4df vpi_a = _mm256_load_pd((double*)(p + i));
    const v4df vqi_b = _mm256_load_pd((double*)(q + i + 1));
    v4df vpi_b = _mm256_load_pd((double*)(p + i + 1));
    const v4df vqi_c = _mm256_load_pd((double*)(q + i + 2));
    v4df vpi_c = _mm256_load_pd((double*)(p + i + 2));
    const v4df vqi_d = _mm256_load_pd((double*)(q + i + 3));
    v4df vpi_d = _mm256_load_pd((double*)(p + i + 3));

    for (int j = i + 4; j < N; j++) {
      v4df vqj = _mm256_load_pd((double*)(q + j));
      v4df vpj = _mm256_load_pd((double*)(p + j));

      v4df vdq_a = (vqj - vqi_a);
      v4df vdq_b = (vqj - vqi_b);
      v4df vdq_c = (vqj - vqi_c);
      v4df vdq_d = (vqj - vqi_d);

      v4df tmp0 = _mm256_unpacklo_pd(vdq_a, vdq_b);
      v4df tmp1 = _mm256_unpackhi_pd(vdq_a, vdq_b);
      v4df tmp2 = _mm256_unpacklo_pd(vdq_c, vdq_d);
      v4df tmp3 = _mm256_unpackhi_pd(vdq_c, vdq_d);

      v4df vdx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
      v4df vdy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
      v4df vdz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);

      v4df vr2 = vdx * vdx + vdy * vdy + vdz * vdz;
      v4df vr6 = vr2 * vr2 * vr2;
      v4df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      v4df vdf_a = _mm256_permute4x64_pd(vdf, 0);
      v4df vdf_b = _mm256_permute4x64_pd(vdf, 85);
      v4df vdf_c = _mm256_permute4x64_pd(vdf, 170);
      v4df vdf_d = _mm256_permute4x64_pd(vdf, 255);

      vpi_a += vdq_a * vdf_a;
      vpj -= vdq_a * vdf_a;

      vpi_b += vdq_b * vdf_b;
      vpj -= vdq_b * vdf_b;

      vpi_c += vdq_c * vdf_c;
      vpj -= vdq_c * vdf_c;

      vpi_d += vdq_d * vdf_d;
      vpj -= vdq_d * vdf_d;

      _mm256_store_pd((double*)(p + j), vpj);
    }
    _mm256_store_pd((double*)(p + i), vpi_a);
    _mm256_store_pd((double*)(p + i + 1), vpi_b);
    _mm256_store_pd((double*)(p + i + 2), vpi_c);
    _mm256_store_pd((double*)(p + i + 3), vpi_d);

    for (int ii = i; ii < (i + 3); ii++) {
      const auto qx = q[ii].x;
      const auto qy = q[ii].y;
      const auto qz = q[ii].z;
      double pfx = 0.0, pfy = 0.0, pfz = 0.0;
      for (int j = ii + 1; j < (i + 4); j++) {
        const double dx = q[j].x - qx;
        const double dy = q[j].y - qy;
        const double dz = q[j].z - qz;
        const double r2 = (dx * dx + dy * dy + dz * dz);
        double r6 = r2 * r2 * r2;
        double df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
        pfx += df * dx;
        pfy += df * dy;
        pfz += df * dz;
        p[j].x -= df * dx;
        p[j].y -= df * dy;
        p[j].z -= df * dz;
      }
      p[ii].x += pfx;
      p[ii].y += pfy;
      p[ii].z += pfz;
    }
  }

  for (int i = ((N - 1) / 4) * 4; i < N - 1; i++) {
    const v4df vqi = _mm256_load_pd((double*)(q + i));
    v4df vpi = _mm256_load_pd((double*)(p + i));
    for (int j = i + 1; j < N - 3; j += 4) {
      v4df vqj_a = _mm256_load_pd((double*)(q + j));
      v4df vdq_a = (vqj_a - vqi);

      v4df vqj_b = _mm256_load_pd((double*)(q + j + 1));
      v4df vdq_b = (vqj_b - vqi);

      v4df vqj_c = _mm256_load_pd((double*)(q + j + 2));
      v4df vdq_c = (vqj_c - vqi);

      v4df vqj_d = _mm256_load_pd((double*)(q + j + 3));
      v4df vdq_d = (vqj_d - vqi);

      v4df tmp0 = _mm256_unpacklo_pd(vdq_a, vdq_b);
      v4df tmp1 = _mm256_unpackhi_pd(vdq_a, vdq_b);
      v4df tmp2 = _mm256_unpacklo_pd(vdq_c, vdq_d);
      v4df tmp3 = _mm256_unpackhi_pd(vdq_c, vdq_d);

      v4df vdx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
      v4df vdy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
      v4df vdz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);

      v4df vr2 = vdx * vdx + vdy * vdy + vdz * vdz;
      v4df vr6 = vr2 * vr2 * vr2;
      v4df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      v4df vdf_a = _mm256_permute4x64_pd(vdf, 0);
      v4df vdf_b = _mm256_permute4x64_pd(vdf, 85);
      v4df vdf_c = _mm256_permute4x64_pd(vdf, 170);
      v4df vdf_d = _mm256_permute4x64_pd(vdf, 255);

      v4df vpj_a = _mm256_load_pd((double*)(p + j));
      vpi += vdq_a * vdf_a;
      vpj_a -= vdq_a * vdf_a;
      _mm256_store_pd((double*)(p + j), vpj_a);

      v4df vpj_b = _mm256_load_pd((double*)(p + j + 1));
      vpi += vdq_b * vdf_b;
      vpj_b -= vdq_b * vdf_b;
      _mm256_store_pd((double*)(p + j + 1), vpj_b);

      v4df vpj_c = _mm256_load_pd((double*)(p + j + 2));
      vpi += vdq_c * vdf_c;
      vpj_c -= vdq_c * vdf_c;
      _mm256_store_pd((double*)(p + j + 2), vpj_c);

      v4df vpj_d = _mm256_load_pd((double*)(p + j + 3));
      vpi += vdq_d * vdf_d;
      vpj_d -= vdq_d * vdf_d;
      _mm256_store_pd((double*)(p + j + 3), vpj_d);
    }
    _mm256_store_pd((double*)(p + i), vpi);
    for (int j = N - (5 - i) % 4; j < N; j++) {
      const double dx = q[j].x - q[i].x;
      const double dy = q[j].y - q[i].y;
      const double dz = q[j].z - q[i].z;
      const double r2 = (dx * dx + dy * dy + dz * dz);
      double r6 = r2 * r2 * r2;
      double df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
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
calc_intrin1x4_reactless() {
  const v4df vc24 = _mm256_set_pd(24 * dt, 24 * dt, 24 * dt, 24 * dt);
  const v4df vc48 = _mm256_set_pd(48 * dt, 48 * dt, 48 * dt, 48 * dt);
  const v4df vzero = _mm256_setzero_pd();
  for (int i = 0; i < N; i++) {
    const v4df vqi = _mm256_load_pd((double*)(q + i));
    v4df vpi = _mm256_load_pd((double*)(p + i));
    v4di vi_id = _mm256_set1_epi64x(i);
    for (int j = 0; j < (N / 4) * 4; j += 4) {
      v4di vj_id = _mm256_set_epi64x(j + 3, j + 2, j + 1, j);
      v4df mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(vi_id, vj_id));

      v4df vqj_a = _mm256_load_pd((double*)(q + j));
      v4df vdq_a = (vqj_a - vqi);

      v4df vqj_b = _mm256_load_pd((double*)(q + j + 1));
      v4df vdq_b = (vqj_b - vqi);

      v4df vqj_c = _mm256_load_pd((double*)(q + j + 2));
      v4df vdq_c = (vqj_c - vqi);

      v4df vqj_d = _mm256_load_pd((double*)(q + j + 3));
      v4df vdq_d = (vqj_d - vqi);

      v4df tmp0 = _mm256_unpacklo_pd(vdq_a, vdq_b);
      v4df tmp1 = _mm256_unpackhi_pd(vdq_a, vdq_b);
      v4df tmp2 = _mm256_unpacklo_pd(vdq_c, vdq_d);
      v4df tmp3 = _mm256_unpackhi_pd(vdq_c, vdq_d);

      v4df vdx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
      v4df vdy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
      v4df vdz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);

      v4df vr2 = vdx * vdx + vdy * vdy + vdz * vdz;
      v4df vr6 = vr2 * vr2 * vr2;
      v4df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      vdf = _mm256_blendv_pd(vdf, vzero, mask);

      v4df vdf_a = _mm256_permute4x64_pd(vdf, 0);
      v4df vdf_b = _mm256_permute4x64_pd(vdf, 85);
      v4df vdf_c = _mm256_permute4x64_pd(vdf, 170);
      v4df vdf_d = _mm256_permute4x64_pd(vdf, 255);

      vpi += vdq_a * vdf_a;
      vpi += vdq_b * vdf_b;
      vpi += vdq_c * vdf_c;
      vpi += vdq_d * vdf_d;
    }
    _mm256_store_pd((double*)(p + i), vpi);
    for (int j = (N / 4) * 4; j < N; j++) {
      const double dx = q[j].x - q[i].x;
      const double dy = q[j].y - q[i].y;
      const double dz = q[j].z - q[i].z;
      const double r2 = (dx * dx + dy * dy + dz * dz);
      double r6 = r2 * r2 * r2;
      double df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
      if (i == j) df = 0.0;
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
    }
  }
}
//------------------------------------------------------------------------
void
calc_intrin4x1_reactless() {
  const v4df vc24 = _mm256_set_pd(24 * dt, 24 * dt, 24 * dt, 24 * dt);
  const v4df vc48 = _mm256_set_pd(48 * dt, 48 * dt, 48 * dt, 48 * dt);
  const v4df vzero = _mm256_setzero_pd();
  for (int i = 0; i < (N / 4) * 4; i += 4) {
    const v4df vqi_a = _mm256_load_pd((double*)(q + i));
    v4df vpi_a = _mm256_load_pd((double*)(p + i));
    const v4df vqi_b = _mm256_load_pd((double*)(q + i + 1));
    v4df vpi_b = _mm256_load_pd((double*)(p + i + 1));
    const v4df vqi_c = _mm256_load_pd((double*)(q + i + 2));
    v4df vpi_c = _mm256_load_pd((double*)(p + i + 2));
    const v4df vqi_d = _mm256_load_pd((double*)(q + i + 3));
    v4df vpi_d = _mm256_load_pd((double*)(p + i + 3));

    v4di vi_id = _mm256_set_epi64x(i + 3, i + 2, i + 1, i);

    for (int j = 0; j < N; j++) {
      v4di vj_id = _mm256_set1_epi64x(j);

      v4df mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(vi_id, vj_id));

      v4df vqj = _mm256_load_pd((double*)(q + j));

      v4df vdq_a = (vqj - vqi_a);
      v4df vdq_b = (vqj - vqi_b);
      v4df vdq_c = (vqj - vqi_c);
      v4df vdq_d = (vqj - vqi_d);

      v4df tmp0 = _mm256_unpacklo_pd(vdq_a, vdq_b);
      v4df tmp1 = _mm256_unpackhi_pd(vdq_a, vdq_b);
      v4df tmp2 = _mm256_unpacklo_pd(vdq_c, vdq_d);
      v4df tmp3 = _mm256_unpackhi_pd(vdq_c, vdq_d);

      v4df vdx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
      v4df vdy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
      v4df vdz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);

      v4df vr2 = vdx * vdx + vdy * vdy + vdz * vdz;
      v4df vr6 = vr2 * vr2 * vr2;
      v4df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      vdf = _mm256_blendv_pd(vdf, vzero, mask);

      v4df vdf_a = _mm256_permute4x64_pd(vdf, 0);
      v4df vdf_b = _mm256_permute4x64_pd(vdf, 85);
      v4df vdf_c = _mm256_permute4x64_pd(vdf, 170);
      v4df vdf_d = _mm256_permute4x64_pd(vdf, 255);

      vpi_a += vdq_a * vdf_a;
      vpi_b += vdq_b * vdf_b;
      vpi_c += vdq_c * vdf_c;
      vpi_d += vdq_d * vdf_d;
    }
    _mm256_store_pd((double*)(p + i), vpi_a);
    _mm256_store_pd((double*)(p + i + 1), vpi_b);
    _mm256_store_pd((double*)(p + i + 2), vpi_c);
    _mm256_store_pd((double*)(p + i + 3), vpi_d);
  }
  for (int i = (N / 4) * 4; i < N; i++) {
    const v4df vqi = _mm256_load_pd((double*)(q + i));
    v4df vpi = _mm256_load_pd((double*)(p + i));
    v4di vi_id = _mm256_set1_epi64x(i);
    for (int j = 0; j < N; j += 4) {
      v4di vj_id = _mm256_set_epi64x(j + 3, j + 2, j + 1, j);
      v4df mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(vi_id, vj_id));

      v4df vqj_a = _mm256_load_pd((double*)(q + j));
      v4df vdq_a = (vqj_a - vqi);

      v4df vqj_b = _mm256_load_pd((double*)(q + j + 1));
      v4df vdq_b = (vqj_b - vqi);

      v4df vqj_c = _mm256_load_pd((double*)(q + j + 2));
      v4df vdq_c = (vqj_c - vqi);

      v4df vqj_d = _mm256_load_pd((double*)(q + j + 3));
      v4df vdq_d = (vqj_d - vqi);

      v4df tmp0 = _mm256_unpacklo_pd(vdq_a, vdq_b);
      v4df tmp1 = _mm256_unpackhi_pd(vdq_a, vdq_b);
      v4df tmp2 = _mm256_unpacklo_pd(vdq_c, vdq_d);
      v4df tmp3 = _mm256_unpackhi_pd(vdq_c, vdq_d);

      v4df vdx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
      v4df vdy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
      v4df vdz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);

      v4df vr2 = vdx * vdx + vdy * vdy + vdz * vdz;
      v4df vr6 = vr2 * vr2 * vr2;
      v4df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      vdf = _mm256_blendv_pd(vdf, vzero, mask);

      v4df vdf_a = _mm256_permute4x64_pd(vdf, 0);
      v4df vdf_b = _mm256_permute4x64_pd(vdf, 85);
      v4df vdf_c = _mm256_permute4x64_pd(vdf, 170);
      v4df vdf_d = _mm256_permute4x64_pd(vdf, 255);

      vpi += vdq_a * vdf_a;
      vpi += vdq_b * vdf_b;
      vpi += vdq_c * vdf_c;
      vpi += vdq_d * vdf_d;
    }
    _mm256_store_pd((double*)(p + i), vpi);
    for (int j = (N / 4) * 4; j < N; j++) {
      const double dx = q[j].x - q[i].x;
      const double dy = q[j].y - q[i].y;
      const double dz = q[j].z - q[i].z;
      const double r2 = (dx * dx + dy * dy + dz * dz);
      double r6 = r2 * r2 * r2;
      double df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
      if (i == j) df = 0.0;
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
    }
  }
}
//------------------------------------------------------------------------
void
init(void) {
  for (int i = 0; i < N; i++) {
    q[i].x = 1.0 + 0.4 * i;
    q[i].y = 2.0 + 0.5 * i;
    q[i].z = 3.0 + 0.6 * i;
    p[i].x = 0.0;
    p[i].y = 0.0;
    p[i].z = 0.0;
  }
}
//------------------------------------------------------------------------
void
reference(void) {
  for (int i = 0; i < N - 1; i++) {
    for (int j = i + 1; j < N; j++) {
      const double dx = q[j].x - q[i].x;
      const double dy = q[j].y - q[i].y;
      const double dz = q[j].z - q[i].z;
      const double r2 = (dx * dx + dy * dy + dz * dz);
      double r6 = r2 * r2 * r2;
      double df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
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
reference_reactless(void) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      const double dx = q[j].x - q[i].x;
      const double dy = q[j].y - q[i].y;
      const double dz = q[j].z - q[i].z;
      const double r2 = (dx * dx + dy * dy + dz * dz);
      double r6 = r2 * r2 * r2;
      double df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
      if (i == j) df = 0.0;
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
    }
  }
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
  if (argc == 2) N = std::atoi(argv[1]);

  q = new double4 [N];
  p = new double4 [N];

  std::cout << N << " ";

  const int num_loop = 1;

  init();
#ifdef USE1x4
  BENCH(calc_intrin1x4, num_loop);
#elif USE4x1
  BENCH(calc_intrin4x1, num_loop);
#elif REFERENCE
  BENCH(reference, num_loop);
#elif USE1x4_REACTLESS
  BENCH(calc_intrin1x4_reactless, num_loop);
#elif USE4x1_REACTLESS
  BENCH(calc_intrin4x1_reactless, num_loop);
#elif REFERENCE_REACTLESS
  BENCH(reference_reactless, num_loop);
#else
  BENCH(calc_intrin, num_loop);
#endif
  std::cerr << std::setprecision(9);
  for (int i = 0; i < 10; i++) {
    std::cerr << p[i].x << " " << p[i].y << " " << p[i].z << "\n";
  }

  delete [] q;
  delete [] p;
}
