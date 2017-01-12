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
void
calc_intrin1x8() {
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
    vpi += static_cast<v8sf>(_mm256_loadu_ps((float*)(p + i)));
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
calc_intrin8x1() {
  const v8sf vc24 = _mm256_set1_ps(24.0f * dt);
  const v8sf vc48 = _mm256_set1_ps(48.0f * dt);
  const v8sf veps2 = _mm256_set1_ps(eps2);
  v8sf vpw = _mm256_setzero_ps();
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

      vpxi += vdf * vdx;
      vpyi += vdf * vdy;
      vpzi += vdf * vdz;

      vpxj -= vdf * vdx;
      vpyj -= vdf * vdy;
      vpzj -= vdf * vdz;

      transpose_4x4x2(vpxj, vpyj, vpzj, vpw);
      v8sf vpj_hilo = vpxj + vpyj + vpzj + vpw;
      v8sf vpj_lohi = _mm256_permute2f128_ps(vpj_hilo, vpj_hilo, 0x01);
      v8sf vpj = vpj_hilo + vpj_lohi;

      _mm_store_ps((float*)(p + j), _mm256_castps256_ps128(vpj));
    }

    vpxi += static_cast<v8sf>(_mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].x),
                                                  vindex, 4));
    vpyi += static_cast<v8sf>(_mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].y),
                                                  vindex, 4));
    vpzi += static_cast<v8sf>(_mm256_i32gather_ps(reinterpret_cast<const float*>(&p[0].z),
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

      vpxi += vdf * vdx;
      vpyi += vdf * vdy;
      vpzi += vdf * vdz;

      vpxj -= vdf * vdx;
      vpyj -= vdf * vdy;
      vpzj -= vdf * vdz;

      transpose_4x4x2(vpxj, vpyj, vpzj, vpw);

      _mm256_storeu2_m128((float*)(p + j_e), (float*)(p + j_a), vpxj);
      _mm256_storeu2_m128((float*)(p + j_f), (float*)(p + j_b), vpyj);
      _mm256_storeu2_m128((float*)(p + j_g), (float*)(p + j_c), vpzj);
      _mm256_storeu2_m128((float*)(p + j_h), (float*)(p + j_d), vpw );
    }
    // horizontal sum
    transpose_4x4x2(vpxi, vpyi, vpzi, vpw);
    v8sf vpi_hilo = vpxi + vpyi + vpzi + vpw;
    v8sf vpi_lohi = _mm256_permute2f128_ps(vpi_hilo, vpi_hilo, 0x01);
    v8sf vpi = vpi_hilo + vpi_lohi;

    // store
    vpi += static_cast<v8sf>(_mm256_loadu_ps((float*)(p + i)));
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

  q = new float4 [N + 1];
  p = new float4 [N + 1];
  posix_memalign((void**)(&list), 32, sizeof(int32_t) * M);

  gen_neighlist(rand_seed);

  std::cout << N << " " << M << " ";

  const int num_loop = 1;

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
