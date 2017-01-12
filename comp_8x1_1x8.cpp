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
// from http://www.kaede-software.com/2014/04/post_641.html
v4sf
hadd_v8sf(v8sf sum) {
  sum = _mm256_hadd_ps(sum, sum);
  sum = _mm256_hadd_ps(sum, sum);
  v8sf rsum = _mm256_permute2f128_ps(sum, sum, 0 << 4 | 1 );
  sum = _mm256_unpacklo_ps(sum, rsum);
  sum = _mm256_hadd_ps(sum, sum);
  return _mm256_extractf128_ps(sum, 0);
}
//------------------------------------------------------------------------
void
calc_intrin1x8() {
  const v8sf vc24 = _mm256_set_ps(24.0f * dt, 24.0f * dt, 24.0f * dt, 24.0f * dt,
                                  24.0f * dt, 24.0f * dt, 24.0f * dt, 24.0f * dt);
  const v8sf vc48 = _mm256_set_ps(48.0f * dt, 48.0f * dt, 48.0f * dt, 48.0f * dt,
                                  48.0f * dt, 48.0f * dt, 48.0f * dt, 48.0f * dt);
  const v8sf veps2 = _mm256_set_ps(eps2, eps2, eps2, eps2,
                                   eps2, eps2, eps2, eps2);
  const v8sf vzero = _mm256_setzero_ps();
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

      v8sf vqxj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&q[0].x),
                                      vindex, 4);
      v8sf vqyj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&q[0].y),
                                      vindex, 4);
      v8sf vqzj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&q[0].z),
                                      vindex, 4);

      v8sf vpxj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].x),
                                      vindex, 4);
      v8sf vpyj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].y),
                                      vindex, 4);
      v8sf vpzj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].z),
                                      vindex, 4);

      v8sf vdx = vqxj - vqxi;
      v8sf vdy = vqyj - vqyi;
      v8sf vdz = vqzj - vqzi;

      v8sf vr2 = veps2 + vdx * vdx + vdy * vdy + vdz * vdz;
      v8sf vr6 = vr2 * vr2 * vr2;
      v8sf vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      v8sf vdfx = vdf * vdx;
      v8sf vdfy = vdf * vdy;
      v8sf vdfz = vdf * vdz;

      vpxi += vdfx;
      vpyi += vdfy;
      vpzi += vdfz;

      vpxj -= vdfx;
      vpyj -= vdfy;
      vpzj -= vdfz;

      v4sf vpj_a = _mm256_extractf128_ps(vpxj, 0);
      v4sf vpj_b = _mm256_extractf128_ps(vpyj, 0);
      v4sf vpj_c = _mm256_extractf128_ps(vpzj, 0);
      v4sf vpj_d = _mm_setzero_ps();

      v4sf vpj_e = _mm256_extractf128_ps(vpxj, 1);
      v4sf vpj_f = _mm256_extractf128_ps(vpyj, 1);
      v4sf vpj_g = _mm256_extractf128_ps(vpzj, 1);
      v4sf vpj_h = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(vpj_a, vpj_b, vpj_c, vpj_d);
      _MM_TRANSPOSE4_PS(vpj_e, vpj_f, vpj_g, vpj_h);

      _mm_store_ps((float*)(p + j_a), vpj_a);
      _mm_store_ps((float*)(p + j_b), vpj_b);
      _mm_store_ps((float*)(p + j_c), vpj_c);
      _mm_store_ps((float*)(p + j_d), vpj_d);
      _mm_store_ps((float*)(p + j_e), vpj_e);
      _mm_store_ps((float*)(p + j_f), vpj_f);
      _mm_store_ps((float*)(p + j_g), vpj_g);
      _mm_store_ps((float*)(p + j_h), vpj_h);
    }
    // horizontal sum
    v4sf vpi_a = _mm256_extractf128_ps(vpxi, 0);
    v4sf vpi_b = _mm256_extractf128_ps(vpyi, 0);
    v4sf vpi_c = _mm256_extractf128_ps(vpzi, 0);
    v4sf vpi_d = _mm_setzero_ps();

    v4sf vpi_e = _mm256_extractf128_ps(vpxi, 1);
    v4sf vpi_f = _mm256_extractf128_ps(vpyi, 1);
    v4sf vpi_g = _mm256_extractf128_ps(vpzi, 1);
    v4sf vpi_h = _mm_setzero_ps();

    _MM_TRANSPOSE4_PS(vpi_a, vpi_b, vpi_c, vpi_d);
    _MM_TRANSPOSE4_PS(vpi_e, vpi_f, vpi_g, vpi_h);

    v4sf vpi = vpi_a + vpi_b + vpi_c + vpi_d
      + vpi_e + vpi_f + vpi_g + vpi_h;

    // store
    vpi += static_cast<v4sf>(_mm_load_ps((float*)(p + i)));
    _mm_store_ps((float*)(p + i), vpi);

    for (int k = (M / 8) * 8; k < M; k++) {
      const auto j = list[k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = (dx * dx + dy * dy + dz * dz);
      const auto r6 = r2 * r2 * r2;
      auto df = (24.0f * r6 - 48.0f) / (r6 * r6 * r2) * dt;
      if (i == j) df = 0.0f;
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
calc_intrin8x1() {
  const v8sf vc24 = _mm256_set_ps(24.0f * dt, 24.0f * dt, 24.0f * dt, 24.0f * dt,
                                  24.0f * dt, 24.0f * dt, 24.0f * dt, 24.0f * dt);
  const v8sf vc48 = _mm256_set_ps(48.0f * dt, 48.0f * dt, 48.0f * dt, 48.0f * dt,
                                  48.0f * dt, 48.0f * dt, 48.0f * dt, 48.0f * dt);
  const v8sf veps2 = _mm256_set_ps(eps2, eps2, eps2, eps2,
                                   eps2, eps2, eps2, eps2);
  const v8sf vzero = _mm256_setzero_ps();
  for (int i = 0; i < (N / 8) * 8; i += 8) {
    v8si vindex = _mm256_set_epi32(i + 7, i + 6, i + 5, i + 4,
                                   i + 3, i + 2, i + 1, i);
    vindex = _mm256_slli_epi32(vindex, 2);

    v8sf vqxi = _mm256_i32gather_ps(reinterpret_cast<const float*>(&q[0].x),
                                    vindex, 4);
    v8sf vqyi = _mm256_i32gather_ps(reinterpret_cast<const float*>(&q[0].y),
                                    vindex, 4);
    v8sf vqzi = _mm256_i32gather_ps(reinterpret_cast<const float*>(&q[0].z),
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

      v8sf vdx = vqxj - vqxi;
      v8sf vdy = vqyj - vqyi;
      v8sf vdz = vqzj - vqzi;

      v8sf vr2 = veps2 + vdx * vdx + vdy * vdy + vdz * vdz;
      v8sf vr6 = vr2 * vr2 * vr2;
      v8sf vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      v8sf vdfx = vdf * vdx;
      v8sf vdfy = vdf * vdy;
      v8sf vdfz = vdf * vdz;

      vpxi += vdfx;
      vpyi += vdfy;
      vpzi += vdfz;

      vpxj -= vdfx;
      vpyj -= vdfy;
      vpzj -= vdfz;

      v4sf vpj_a = _mm256_extractf128_ps(vpxj, 0);
      v4sf vpj_b = _mm256_extractf128_ps(vpyj, 0);
      v4sf vpj_c = _mm256_extractf128_ps(vpzj, 0);
      v4sf vpj_d = _mm_setzero_ps();

      v4sf vpj_e = _mm256_extractf128_ps(vpxj, 1);
      v4sf vpj_f = _mm256_extractf128_ps(vpyj, 1);
      v4sf vpj_g = _mm256_extractf128_ps(vpzj, 1);
      v4sf vpj_h = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(vpj_a, vpj_b, vpj_c, vpj_d);
      _MM_TRANSPOSE4_PS(vpj_e, vpj_f, vpj_g, vpj_h);

      v4sf vpj = vpj_a + vpj_b + vpj_c + vpj_d
        + vpj_e + vpj_f + vpj_g + vpj_h;

      _mm_store_ps((float*)(p + j), vpj);
    }

    vpxi += static_cast<v8sf>(_mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].x),
                                                  vindex, 4));
    vpyi += static_cast<v8sf>(_mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].y),
                                                  vindex, 4));
    vpzi += static_cast<v8sf>(_mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].z),
                                                  vindex, 4));

    v4sf vpi_a = _mm256_extractf128_ps(vpxi, 0);
    v4sf vpi_b = _mm256_extractf128_ps(vpyi, 0);
    v4sf vpi_c = _mm256_extractf128_ps(vpzi, 0);
    v4sf vpi_d = _mm_setzero_ps();

    v4sf vpi_e = _mm256_extractf128_ps(vpxi, 1);
    v4sf vpi_f = _mm256_extractf128_ps(vpyi, 1);
    v4sf vpi_g = _mm256_extractf128_ps(vpzi, 1);
    v4sf vpi_h = _mm_setzero_ps();

    _MM_TRANSPOSE4_PS(vpi_a, vpi_b, vpi_c, vpi_d);
    _MM_TRANSPOSE4_PS(vpi_e, vpi_f, vpi_g, vpi_h);

    _mm256_store_ps((float*)(p + i), _mm256_set_m128(vpi_b, vpi_a));
    _mm256_store_ps((float*)(p + i + 2), _mm256_set_m128(vpi_d, vpi_c));
    _mm256_store_ps((float*)(p + i + 4), _mm256_set_m128(vpi_f, vpi_e));
    _mm256_store_ps((float*)(p + i + 6), _mm256_set_m128(vpi_h, vpi_g));
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

      v8sf vqxj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&q[0].x),
                                      vindex, 4);
      v8sf vqyj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&q[0].y),
                                      vindex, 4);
      v8sf vqzj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&q[0].z),
                                      vindex, 4);

      v8sf vpxj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].x),
                                      vindex, 4);
      v8sf vpyj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].y),
                                      vindex, 4);
      v8sf vpzj = _mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].z),
                                      vindex, 4);

      v8sf vdx = vqxj - vqxi;
      v8sf vdy = vqyj - vqyi;
      v8sf vdz = vqzj - vqzi;

      v8sf vr2 = veps2 + vdx * vdx + vdy * vdy + vdz * vdz;
      v8sf vr6 = vr2 * vr2 * vr2;
      v8sf vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      v8sf vdfx = vdf * vdx;
      v8sf vdfy = vdf * vdy;
      v8sf vdfz = vdf * vdz;

      vpxi += vdfx;
      vpyi += vdfy;
      vpzi += vdfz;

      vpxj -= vdfx;
      vpyj -= vdfy;
      vpzj -= vdfz;

      v4sf vpj_a = _mm256_extractf128_ps(vpxj, 0);
      v4sf vpj_b = _mm256_extractf128_ps(vpyj, 0);
      v4sf vpj_c = _mm256_extractf128_ps(vpzj, 0);
      v4sf vpj_d = _mm_setzero_ps();

      v4sf vpj_e = _mm256_extractf128_ps(vpxj, 1);
      v4sf vpj_f = _mm256_extractf128_ps(vpyj, 1);
      v4sf vpj_g = _mm256_extractf128_ps(vpzj, 1);
      v4sf vpj_h = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(vpj_a, vpj_b, vpj_c, vpj_d);
      _MM_TRANSPOSE4_PS(vpj_e, vpj_f, vpj_g, vpj_h);

      _mm_store_ps((float*)(p + j_a), vpj_a);
      _mm_store_ps((float*)(p + j_b), vpj_b);
      _mm_store_ps((float*)(p + j_c), vpj_c);
      _mm_store_ps((float*)(p + j_d), vpj_d);
      _mm_store_ps((float*)(p + j_e), vpj_e);
      _mm_store_ps((float*)(p + j_f), vpj_f);
      _mm_store_ps((float*)(p + j_g), vpj_g);
      _mm_store_ps((float*)(p + j_h), vpj_h);
    }
    // horizontal sum
    v4sf vpi_a = _mm256_extractf128_ps(vpxi, 0);
    v4sf vpi_b = _mm256_extractf128_ps(vpyi, 0);
    v4sf vpi_c = _mm256_extractf128_ps(vpzi, 0);
    v4sf vpi_d = _mm_setzero_ps();

    v4sf vpi_e = _mm256_extractf128_ps(vpxi, 1);
    v4sf vpi_f = _mm256_extractf128_ps(vpyi, 1);
    v4sf vpi_g = _mm256_extractf128_ps(vpzi, 1);
    v4sf vpi_h = _mm_setzero_ps();

    _MM_TRANSPOSE4_PS(vpi_a, vpi_b, vpi_c, vpi_d);
    _MM_TRANSPOSE4_PS(vpi_e, vpi_f, vpi_g, vpi_h);

    v4sf vpi = vpi_a + vpi_b + vpi_c + vpi_d
      + vpi_e + vpi_f + vpi_g + vpi_h;

    // store
    vpi += static_cast<v4sf>(_mm_load_ps((float*)(p + i)));
    _mm_store_ps((float*)(p + i), vpi);

    for (int k = (M / 8) * 8; k < M; k++) {
      const auto j = list[k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = (dx * dx + dy * dy + dz * dz);
      const auto r6 = r2 * r2 * r2;
      auto df = (24.0f * r6 - 48.0f) / (r6 * r6 * r2) * dt;
      if (i == j) df = 0.0f;
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
      auto df = (c24 * r6 - c48) / (r6 * r6 * r2);
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

  const int num_loop = 30;

  init();
#ifdef USE1x8
  BENCH(calc_intrin1x8, num_loop);
#elif USE8x1
  BENCH(calc_intrin8x1, num_loop);
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
