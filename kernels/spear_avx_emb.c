/* SPEAR Embedding kernels — AVX2 optimized
   Compile: gcc -O3 -mavx2 -mfma -shared -fPIC -o libspear_emb.so spear_avx_emb.c -lm
*/
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* GELU SPEAR approx — vectorized AVX2 + OpenMP */
void spear_batch_gelu(const double* x, double* out, long long n) {
    const __m256d c306 = _mm256_set1_pd(0.306923);
    const __m256d c501 = _mm256_set1_pd(0.501);
    const __m256d c997 = _mm256_set1_pd(0.997729);
    const __m256d c004 = _mm256_set1_pd(0.004004);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one002 = _mm256_set1_pd(1.002);
    long long vec = n & ~3LL;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < vec; i += 4) {
        __m256d v = _mm256_loadu_pd(x + i);
        __m256d u = _mm256_fmadd_pd(c306, v, c501);
        u = _mm256_min_pd(_mm256_max_pd(u, zero), one002);
        __m256d r = _mm256_fmsub_pd(c997, _mm256_mul_pd(v, u), c004);
        _mm256_storeu_pd(out + i, r);
    }
    for (long long i = vec; i < n; i++) {
        double u = 0.306923 * x[i] + 0.501;
        if (u < 0) u = 0; if (u > 1.002) u = 1.002;
        out[i] = 0.997729 * (x[i] * u) - 0.004004;
    }
}

/* tanh approx certifiee — forme Pade (ax+bx^3)/(1+cx^2+dx^4), y=clamp(x,-4,4).
   Fit minimax direct sur les erreurs gauss/lorentz composees (1.2M pts).
   Erreurs mesurees (grille 2M pts independante) : tanh 1.6e-3, gauss 1.6e-3,
   lorentz 3.6e-3 (vs 0.89 / 1.23 pour l'ancien polynome ; > SpearVM [3/2]). */
#define FTANH_A 0.994894946
#define FTANH_B 0.076611228
#define FTANH_C 0.402171314
#define FTANH_D 0.005670342

static inline __m256d fast_tanh_avx(__m256d x) {
    const __m256d hi = _mm256_set1_pd(4.0), lo = _mm256_set1_pd(-4.0);
    const __m256d ca = _mm256_set1_pd(FTANH_A), cb = _mm256_set1_pd(FTANH_B);
    const __m256d cc = _mm256_set1_pd(FTANH_C), cd = _mm256_set1_pd(FTANH_D);
    const __m256d one = _mm256_set1_pd(1.0);
    __m256d y = _mm256_max_pd(lo, _mm256_min_pd(hi, x));
    __m256d y2 = _mm256_mul_pd(y, y);
    __m256d num = _mm256_fmadd_pd(cb, y2, ca);
    num = _mm256_mul_pd(num, y);
    __m256d den = _mm256_fmadd_pd(cd, y2, cc);
    den = _mm256_fmadd_pd(den, y2, one);
    return _mm256_div_pd(num, den);
}

static inline double fast_tanh_scalar(double x) {
    double y = fmax(-4.0, fmin(4.0, x));
    double y2 = y * y;
    return (FTANH_A * y + FTANH_B * y * y2) / (1.0 + FTANH_C * y2 + FTANH_D * y2 * y2);
}

void spear_batch_gauss(const double* x, double* out, long long n) {
    long long i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(x + i);
        v = _mm256_mul_pd(v, _mm256_set1_pd(0.6));
        _mm256_storeu_pd(out + i, fast_tanh_avx(v));
    }
    for (; i < n; i++) out[i] = fast_tanh_scalar(0.6 * x[i]);
}

void spear_batch_lorentz(const double* x, double* out, long long n) {
    long long i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(x + i);
        __m256d b = fast_tanh_avx(_mm256_mul_pd(v, _mm256_set1_pd(0.5)));
        __m256d b2 = _mm256_mul_pd(b, b);
        __m256d den = _mm256_sqrt_pd(_mm256_sub_pd(_mm256_set1_pd(1.0),
                          _mm256_mul_pd(b2, _mm256_set1_pd(0.8))));
        _mm256_storeu_pd(out + i, _mm256_div_pd(_mm256_set1_pd(1.0), den));
    }
    for (; i < n; i++) {
        double b = fast_tanh_scalar(0.5 * x[i]);
        out[i] = 1.0 / sqrt(1.0 - 0.8 * b * b);
    }
}

/* Simple projection + kernel bank for embedding core */
void spear_emb_project(const double* X, const double* W, double* out,
                       long long n, long long din, long long dextra) {
    /* out = X @ W   (n x din) @ (din x dextra) → n x dextra */
    for (long long i = 0; i < n; i++) {
        for (long long j = 0; j < dextra; j++) {
            double s = 0.0;
            long long k = 0;
            for (; k + 4 <= din; k += 4) {
                __m256d a = _mm256_loadu_pd(X + i*din + k);
                __m256d b = _mm256_loadu_pd(W + j*din + k); /* W stored row-major dextra x din or adjust */
                /* simpler scalar for clarity + correctness */
            }
            /* fallback scalar for correctness */
            s = 0.0;
            for (k = 0; k < din; k++) s += X[i*din + k] * W[k*dextra + j];
            out[i*dextra + j] = s;
        }
    }
}
