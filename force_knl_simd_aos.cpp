#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <chrono>
#include <x86intrin.h>

int N = 100003;
int M = 203;
const double dt = 0.01;
const double eps2 = (1.0 / 256.0) * (1.0 / 256.0);
typedef double v8df __attribute__((vector_size(64)));
typedef int32_t v8si __attribute__((vector_size(32)));

struct double4 {double x, y, z, w;};

double4* __restrict q = nullptr;
double4* __restrict p = nullptr;
int32_t* __restrict list = nullptr;

void print512(v8df r) {
  union {
    v8df r;
    double elem[8];
  } tmp;
  tmp.r = r;
  std::cerr << std::setprecision(10);
  std::cerr << tmp.elem[0] << " " << tmp.elem[1] << " " << tmp.elem[2] << " " << tmp.elem[3] << " ";
  std::cerr << tmp.elem[4] << " " << tmp.elem[5] << " " << tmp.elem[6] << " " << tmp.elem[7];
  std::cerr << std::endl;
}

static inline v8df _mm512_load2_m256d(const double* hiaddr,
                                      const double* loaddr) {
  v8df ret;
  ret = _mm512_insertf64x4(ret, _mm256_load_pd(loaddr), 0x0);
  ret = _mm512_insertf64x4(ret, _mm256_load_pd(hiaddr), 0x1);
  return ret;
}

static inline void _mm512_store2_m256d(double* hiaddr,
                                       double* loaddr,
                                       const v8df& dat) {
  _mm256_store_pd(loaddr, _mm512_castpd512_pd256(dat));
  _mm256_store_pd(hiaddr, _mm512_extractf64x4_pd(dat, 0x1));
}

static inline void transpose_4x4x2(v8df& va,
                                   v8df& vb,
                                   v8df& vc,
                                   v8df& vd) {
#if 0
  v8df t_a = _mm512_shuffle_pd(va, vb, 0x00);
  v8df t_b = _mm512_shuffle_pd(va, vb, 0xff);
  v8df t_c = _mm512_shuffle_pd(vc, vd, 0x00);
  v8df t_d = _mm512_shuffle_pd(vc, vd, 0xff);
#else
  v8df t_a = _mm512_unpacklo_pd(va, vb);
  v8df t_b = _mm512_unpackhi_pd(va, vb);
  v8df t_c = _mm512_unpacklo_pd(vc, vd);
  v8df t_d = _mm512_unpackhi_pd(vc, vd);
#endif

  va = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_c);
  vb = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_d);
  vc = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_c);
  vd = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_d);
}

static inline void transpose_4x4x2(const v8df& va,
                                   const v8df& vb,
                                   const v8df& vc,
                                   const v8df& vd,
                                   v8df& vx,
                                   v8df& vy,
                                   v8df& vz) {
  v8df t_a = _mm512_unpacklo_pd(va, vb);
  v8df t_b = _mm512_unpackhi_pd(va, vb);
  v8df t_c = _mm512_unpacklo_pd(vc, vd);
  v8df t_d = _mm512_unpackhi_pd(vc, vd);

  vx = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_c);
  vy = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_d);
  vz = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_c);
}

// with gather and scatter
void calc_intrin1x8_v1() {
  const v8df vc24  = _mm512_set1_pd(24.0 * dt);
  const v8df vc48  = _mm512_set1_pd(48.0 * dt);
  const v8df veps2 = _mm512_set1_pd(eps2);

  for (int i = 0; i < N; i++) {
    v8df vqxi = _mm512_set1_pd(q[i].x);
    v8df vqyi = _mm512_set1_pd(q[i].y);
    v8df vqzi = _mm512_set1_pd(q[i].z);

    v8df vpxi = _mm512_setzero_pd();
    v8df vpyi = _mm512_setzero_pd();
    v8df vpzi = _mm512_setzero_pd();

    for (int k = 0; k < (M / 8) * 8; k += 8) {
      v8si vindex = _mm256_load_si256(reinterpret_cast<__m256i*>(&list[k]));
      vindex = _mm256_slli_epi32(vindex, 2);

      v8df vqxj = _mm512_i32gather_pd(vindex, &q[0].x, 8);
      v8df vqyj = _mm512_i32gather_pd(vindex, &q[0].y, 8);
      v8df vqzj = _mm512_i32gather_pd(vindex, &q[0].z, 8);

      v8df vpxj = _mm512_i32gather_pd(vindex, &p[0].x, 8);
      v8df vpyj = _mm512_i32gather_pd(vindex, &p[0].y, 8);
      v8df vpzj = _mm512_i32gather_pd(vindex, &p[0].z, 8);
      v8df vdx = vqxj - vqxi;
      v8df vdy = vqyj - vqyi;
      v8df vdz = vqzj - vqzi;

      v8df vr2 = veps2 + vdx * vdx + vdy * vdy + vdz * vdz;
      v8df vr6 = vr2 * vr2 * vr2;
      v8df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      vpxi += vdf * vdx;
      vpyi += vdf * vdy;
      vpzi += vdf * vdz;

      vpxj -= vdf * vdx;
      vpyj -= vdf * vdy;
      vpzj -= vdf * vdz;

      _mm512_i32scatter_pd(&p[0].x, vindex, vpxj, 8);
      _mm512_i32scatter_pd(&p[0].y, vindex, vpyj, 8);
      _mm512_i32scatter_pd(&p[0].z, vindex, vpzj, 8);
    }
    // horizontal sum
    v8df vpwi = _mm512_setzero_pd();
    transpose_4x4x2(vpxi, vpyi, vpzi, vpwi);
    // vpxi = {vpia, vpie}
    // vpyi = {vpib, vpif}
    // vpzi = {vpic, vpig}
    // vpwi = {vpid, vpih}

    v8df vpi_hilo = vpxi + vpyi + vpzi + vpwi;
    v8df vpi_lohi = _mm512_permutexvar_pd(_mm512_set_epi64(0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4),
                                          vpi_hilo);
    v8df vpi = vpi_hilo + vpi_lohi;

    // store
    vpi += static_cast<v8df>(_mm512_castpd256_pd512(_mm256_load_pd(&p[i].x)));
    _mm256_store_pd(&p[i].x, _mm512_castpd512_pd256(vpi));

    for (int k = (M / 8) * 8; k < M; k++) {
      const auto j = list[k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = (eps2 + dx * dx + dy * dy + dz * dz);
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0 * dt * r6 - 48.0 * dt) / (r6 * r6 * r2);
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
    }
  }
}

// without gather and scatter
void calc_intrin1x8_v2() {
  const v8df vc24  = _mm512_set1_pd(24.0 * dt);
  const v8df vc48  = _mm512_set1_pd(48.0 * dt);
  const v8df veps2 = _mm512_set1_pd(eps2);
  for (int i = 0; i < N; i++) {
    v8df vqi = _mm512_castpd256_pd512(_mm256_load_pd(&q[i].x));
    vqi = _mm512_insertf64x4(vqi, _mm512_castpd512_pd256(vqi), 0x1);
    v8df vpi = _mm512_setzero_pd();

    for (int k = 0; k < (M / 8) * 8; k += 8) {
      const auto j_a = list[k    ];
      const auto j_b = list[k + 1];
      const auto j_c = list[k + 2];
      const auto j_d = list[k + 3];

      const auto j_e = list[k + 4];
      const auto j_f = list[k + 5];
      const auto j_g = list[k + 6];
      const auto j_h = list[k + 7];

      v8df vpj_ea = _mm512_load2_m256d(&p[j_e].x, &p[j_a].x);
      v8df vpj_fb = _mm512_load2_m256d(&p[j_f].x, &p[j_b].x);
      v8df vpj_gc = _mm512_load2_m256d(&p[j_g].x, &p[j_c].x);
      v8df vpj_hd = _mm512_load2_m256d(&p[j_h].x, &p[j_d].x);

      v8df vqj_ea = _mm512_load2_m256d(&q[j_e].x, &q[j_a].x);
      v8df vqj_fb = _mm512_load2_m256d(&q[j_f].x, &q[j_b].x);
      v8df vqj_gc = _mm512_load2_m256d(&q[j_g].x, &q[j_c].x);
      v8df vqj_hd = _mm512_load2_m256d(&q[j_h].x, &q[j_d].x);

      v8df vdq_ea = vqj_ea - vqi;
      v8df vdq_fb = vqj_fb - vqi;
      v8df vdq_gc = vqj_gc - vqi;
      v8df vdq_hd = vqj_hd - vqi;

      v8df vdx, vdy, vdz;
      transpose_4x4x2(vdq_ea, vdq_fb, vdq_gc, vdq_hd,
                      vdx, vdy, vdz);

      v8df vr2 = veps2 + vdx * vdx + vdy * vdy + vdz * vdz;
      v8df vr6 = vr2 * vr2 * vr2;
      v8df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      v8df vdf_ea = _mm512_permutex_pd(vdf, 0x00);
      v8df vdf_fb = _mm512_permutex_pd(vdf, 0x55);
      v8df vdf_gc = _mm512_permutex_pd(vdf, 0xaa);
      v8df vdf_hd = _mm512_permutex_pd(vdf, 0xff);

      vpi    += vdf_ea * vdq_ea;
      vpj_ea -= vdf_ea * vdq_ea;

      vpi    += vdf_fb * vdq_fb;
      vpj_fb -= vdf_fb * vdq_fb;

      vpi    += vdf_gc * vdq_gc;
      vpj_gc -= vdf_gc * vdq_gc;

      vpi    += vdf_hd * vdq_hd;
      vpj_hd -= vdf_hd * vdq_hd;

      _mm512_store2_m256d(&p[j_e].x, &p[j_a].x, vpj_ea);
      _mm512_store2_m256d(&p[j_f].x, &p[j_b].x, vpj_fb);
      _mm512_store2_m256d(&p[j_g].x, &p[j_c].x, vpj_gc);
      _mm512_store2_m256d(&p[j_h].x, &p[j_d].x, vpj_hd);
    }
    vpi = _mm512_add_pd(vpi,
                        _mm512_permutexvar_pd(_mm512_set_epi64(0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4),
                                              vpi));
    vpi = _mm512_add_pd(vpi, _mm512_castpd256_pd512(_mm256_load_pd(&p[i].x)));
    _mm256_store_pd(&p[i].x, _mm512_castpd512_pd256(vpi));

    for (int k = (M / 8) * 8; k < M; k++) {
      const auto j = list[k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = (eps2 + dx * dx + dy * dy + dz * dz);
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0 * dt * r6 - 48.0 * dt) / (r6 * r6 * r2);
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
    }
  }
}

void init() {
  std::mt19937 mt;
  std::uniform_real_distribution<> ud(0.0, 10.0);
  for (int i = 0; i < N; i++) {
    q[i].x = ud(mt);
    q[i].y = ud(mt);
    q[i].z = ud(mt);
    p[i].x = 0.0;
    p[i].y = 0.0;
    p[i].z = 0.0;
  }
}

void gen_neighlist(const int seed) {
  std::mt19937 mt(seed);
  std::uniform_int_distribution<> dist(0, N - 1);
  std::generate(list, list + M, [&mt, &dist](){return dist(mt);});
}

void reference() {
  const auto c24 = 24.0 * dt;
  const auto c48 = 48.0 * dt;
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

int main(int argc, char* argv[]) {
  if (argc >= 3) {
    N = std::atoi(argv[1]);
    M = std::atoi(argv[2]);
  }

  int rand_seed = 10;
  if (argc >= 4) {
    rand_seed = std::atoi(argv[3]);
  }

  bool verbose = false;
  if (argc == 5)
    if (std::atoi(argv[4]))
      verbose = true;

  posix_memalign((void**)(&q), 64, sizeof(double4) * N);
  posix_memalign((void**)(&p), 64, sizeof(double4) * N);
  posix_memalign((void**)(&list), 32, sizeof(int32_t) * M);

  gen_neighlist(rand_seed);

  std::cout << N << " " << M << " ";

  const int num_loop = 10;

  init();
#ifdef USE1x8_v1
  BENCH(calc_intrin1x8_v1, num_loop);
#elif USE1x8_v2
  BENCH(calc_intrin1x8_v2, num_loop);
#elif REFERENCE
  BENCH(reference, num_loop);
#endif

  if (verbose) {
    std::cerr << std::setprecision(9);
    for (int i = 0; i < 10; i++) {
      std::cerr << p[i].x << " " << p[i].y << " " << p[i].z << "\n";
    }
  }

  free(q);
  free(p);
  free(list);
}
