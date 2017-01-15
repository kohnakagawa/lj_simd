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
typedef double v4df __attribute__((vector_size(32)));
typedef int32_t v8si __attribute__((vector_size(32)));

double* __restrict qx = nullptr;
double* __restrict qy = nullptr;
double* __restrict qz = nullptr;

double* __restrict px = nullptr;
double* __restrict py = nullptr;
double* __restrict pz = nullptr;

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
    v8df vqxi = _mm512_set1_pd(qx[i]);
    v8df vqyi = _mm512_set1_pd(qy[i]);
    v8df vqzi = _mm512_set1_pd(qz[i]);

    v8df vpxi = _mm512_setzero_pd();
    v8df vpyi = _mm512_setzero_pd();
    v8df vpzi = _mm512_setzero_pd();

    for (int k = 0; k < (M / 8) * 8; k += 8) {
      v8si vindex = _mm256_load_si256(reinterpret_cast<__m256i*>(&list[k]));

      v8df vqxj = _mm512_i32gather_pd(vindex, &qx[0], 8);
      v8df vqyj = _mm512_i32gather_pd(vindex, &qy[0], 8);
      v8df vqzj = _mm512_i32gather_pd(vindex, &qz[0], 8);

      v8df vpxj = _mm512_i32gather_pd(vindex, &px[0], 8);
      v8df vpyj = _mm512_i32gather_pd(vindex, &py[0], 8);
      v8df vpzj = _mm512_i32gather_pd(vindex, &pz[0], 8);

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

      _mm512_i32scatter_pd(&px[0], vindex, vpxj, 8);
      _mm512_i32scatter_pd(&py[0], vindex, vpyj, 8);
      _mm512_i32scatter_pd(&pz[0], vindex, vpzj, 8);
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
    v4df vpi = static_cast<v4df>(_mm512_castpd512_pd256(vpi_hilo + vpi_lohi));
    vpi += static_cast<v4df>(_mm256_set_pd(0.0, pz[i], py[i], px[i]));
    double* pi = reinterpret_cast<double*>(&vpi);

    px[i] = pi[0];
    py[i] = pi[1];
    pz[i] = pi[2];

    for (int k = (M / 8) * 8; k < M; k++) {
      const auto j = list[k];
      const auto dx = qx[j] - qx[i];
      const auto dy = qy[j] - qy[i];
      const auto dz = qz[j] - qz[i];
      const auto r2 = (eps2 + dx * dx + dy * dy + dz * dz);
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0 * dt * r6 - 48.0 * dt) / (r6 * r6 * r2);
      px[i] += df * dx;
      py[i] += df * dy;
      pz[i] += df * dz;
      px[j] -= df * dx;
      py[j] -= df * dy;
      pz[j] -= df * dz;
    }
  }
}

void init() {
  std::mt19937 mt;
  std::uniform_real_distribution<> ud(0.0, 10.0);
  for (int i = 0; i < N; i++) {
    qx[i] = ud(mt);
    qy[i] = ud(mt);
    qz[i] = ud(mt);
    px[i] = 0.0;
    py[i] = 0.0;
    pz[i] = 0.0;
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
      const auto dx = qx[j] - qx[i];
      const auto dy = qy[j] - qy[i];
      const auto dz = qz[j] - qz[i];
      const auto r2 = (eps2 + dx * dx + dy * dy + dz * dz);
      const auto r6 = r2 * r2 * r2;
      const auto df = (c24 * r6 - c48) / (r6 * r6 * r2);
      px[i] += df * dx;
      py[i] += df * dy;
      pz[i] += df * dz;
      px[j] -= df * dx;
      py[j] -= df * dy;
      pz[j] -= df * dz;
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

  posix_memalign((void**)(&qx), 64, sizeof(double) * N);
  posix_memalign((void**)(&qy), 64, sizeof(double) * N);
  posix_memalign((void**)(&qz), 64, sizeof(double) * N);

  posix_memalign((void**)(&px), 64, sizeof(double) * N);
  posix_memalign((void**)(&py), 64, sizeof(double) * N);
  posix_memalign((void**)(&pz), 64, sizeof(double) * N);

  posix_memalign((void**)(&list), 32, sizeof(int32_t) * M);

  gen_neighlist(rand_seed);

  std::cout << N << " " << M << " ";

  const int num_loop = 10;

  init();
#ifdef USE1x8_v1
  BENCH(calc_intrin1x8_v1, num_loop);
#elif REFERENCE
  BENCH(reference, num_loop);
#endif

  if (verbose) {
    std::cerr << std::setprecision(9);
    for (int i = 0; i < 10; i++) {
      std::cerr << px[i] << " " << py[i] << " " << pz[i] << "\n";
    }
  }

  free(qx);
  free(qy);
  free(qz);

  free(px);
  free(py);
  free(pz);

  free(list);
}
