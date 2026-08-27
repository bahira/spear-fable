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

/* tanh approx (fast) for gauss / lorentz helpers */
static inline __m256d fast_tanh_avx(__m256d x) {
    /* clamp */
    x = _mm256_min_pd(_mm256_max_pd(x, _mm256_set1_pd(-5.0)), _mm256_set1_pd(5.0));
    /* rational-ish approx */
    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d a = _mm256_fmadd_pd(x2, _mm256_set1_pd(0.037326), _mm256_set1_pd(0.856115));
    a = _mm256_fmadd_pd(x2, a, _mm256_set1_pd(1.0));
    return _mm256_div_pd(x, a);  /* rough but fast */
}

void spear_batch_gauss(const double* x, double* out, long long n) {
    long long i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(x + i);
        v = _mm256_mul_pd(v, _mm256_set1_pd(0.6));
        _mm256_storeu_pd(out + i, fast_tanh_avx(v));
    }
    for (; i < n; i++) out[i] = tanh(0.6 * x[i]);
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
        double b = tanh(0.5 * x[i]);
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
