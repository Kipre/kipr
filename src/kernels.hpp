
typedef __m256(*intrinsic_op)(__m256, __m256);




void inline
add_kernel(float * dest, float * lhs, float * rhs, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, v_b;
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_b = _mm256_load_ps(&rhs[k]);
        v_a = _mm256_add_ps(v_a, v_b);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
    while (k < length) {
        dest[k] = lhs[k] + rhs[k];
        ++k;
    }
}


void inline
sub_kernel(float * dest, float * lhs, float * rhs, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, v_b;
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_b = _mm256_load_ps(&rhs[k]);
        v_a = _mm256_sub_ps(v_a, v_b);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
    while (k < length) {
        dest[k] = lhs[k] - rhs[k];
        ++k;
    }
}


void inline
mul_kernel(float * dest, float * lhs, float * rhs, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, v_b;
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_b = _mm256_load_ps(&rhs[k]);
        v_a = _mm256_mul_ps(v_a, v_b);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
    while (k < length) {
        dest[k] = lhs[k] * rhs[k];
        ++k;
    }
}


void inline
div_kernel(float * dest, float * lhs, float * rhs, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, v_b;
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_b = _mm256_load_ps(&rhs[k]);
        v_a = _mm256_div_ps(v_a, v_b);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
    while (k < length) {
        dest[k] = lhs[k] / rhs[k];
        ++k;
    }
}

void inline
add_dkernel(float * self, float * lhs, float * rhs, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a;
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&self[k]);
        _mm256_store_ps(&rhs[k], v_a);
        _mm256_store_ps(&lhs[k], v_a);
    }
#endif
    while (k < length) {
        lhs[k] = self[k];
        rhs[k] = self[k];
        ++k;
    }
}

void inline
sub_dkernel(float * self, float * lhs, float * rhs, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, minus = _mm256_set1_ps(-1);
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&self[k]);
        _mm256_store_ps(&lhs[k], v_a);
        v_a = _mm256_mul_ps(v_a, minus);
        _mm256_store_ps(&rhs[k], v_a);
    }
#endif
    while (k < length) {
        lhs[k] = self[k];
        rhs[k] = -self[k];
        ++k;
    }
}

void inline
mul_dkernel(float * self, float * lhs, float * rhs, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, v_b, v_c, tmp;
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_b = _mm256_load_ps(&rhs[k]);
        v_c = _mm256_load_ps(&self[k]);
        tmp = _mm256_mul_ps(v_a, v_c);
        _mm256_store_ps(&rhs[k], tmp);
        tmp = _mm256_mul_ps(v_b, v_c);
        _mm256_store_ps(&lhs[k], tmp);
    }
#endif
    float fmp;
    while (k < length) {
        fmp = rhs[k] * self[k];
        rhs[k] = lhs[k] * self[k];
        lhs[k] = fmp;
        ++k;
    }
}

void inline
div_dkernel(float * self, float * lhs, float * rhs, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, v_b, v_c, tmp;
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_b = _mm256_load_ps(&rhs[k]);
        v_c = _mm256_load_ps(&self[k]);
        tmp = _mm256_mul_ps(v_a, v_c);
        _mm256_store_ps(&rhs[k], tmp);
        tmp = _mm256_mul_ps(v_b, v_c);
        _mm256_store_ps(&lhs[k], tmp);
    }
#endif
    float fmp;
    while (k < length) {
        fmp = rhs[k] * self[k];
        rhs[k] = lhs[k] * self[k];
        lhs[k] = fmp;
        ++k;
    }
}

void print_m256(__m256 a, const char * msg = "") {
    float tmp[8];
    _mm256_store_ps(tmp, a);
    printf("__m256 %s %f, %f, %f, %f, %f, %f, %f, %f\n", msg,
           tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]);
}

inline void general_matmul(float * c, float * a, float * b, size_t I, size_t J, size_t K) {

    int i, j;
    // #pragma omp parallel for num_threads(4)
    for (i = 0; i < (int) I - 3; i += 4) {
        for (j = 0; j < (int) J - 23; j += 24) {
            __m256 h00, h01, h02, h03,
                   h10, h11, h12, h13,
                   h20, h21, h22, h23,
                   b0, b1, b2, a0;

            b0 = _mm256_load_ps(&b[j]);
            b1 = _mm256_load_ps(&b[j + 8]);
            b2 = _mm256_load_ps(&b[j + 16]);

            a0 = _mm256_set1_ps(a[(0 + i) * K]);

            h00 = _mm256_mul_ps(a0, b0);
            h10 = _mm256_mul_ps(a0, b1);
            h20 = _mm256_mul_ps(a0, b2);

            a0 = _mm256_set1_ps(a[(1 + i) * K]);

            h01 = _mm256_mul_ps(a0, b0);
            h11 = _mm256_mul_ps(a0, b1);
            h21 = _mm256_mul_ps(a0, b2);

            a0 = _mm256_set1_ps(a[(2 + i) * K]);

            h02 = _mm256_mul_ps(a0, b0);
            h12 = _mm256_mul_ps(a0, b1);
            h22 = _mm256_mul_ps(a0, b2);

            a0 = _mm256_set1_ps(a[(3 + i) * K]);

            h03 = _mm256_mul_ps(a0, b0);
            h13 = _mm256_mul_ps(a0, b1);
            h23 = _mm256_mul_ps(a0, b2);

            for (int k = 1; k < K; ++k) {

                b0 = _mm256_load_ps(&b[k * J + j]);
                b1 = _mm256_load_ps(&b[k * J + j + 8]);
                b2 = _mm256_load_ps(&b[k * J + j + 16]);

                a0 = _mm256_set1_ps(a[(0 + i) * K + k]);

                h00 = _mm256_fmadd_ps(a0, b0, h00);
                h10 = _mm256_fmadd_ps(a0, b1, h10);
                h20 = _mm256_fmadd_ps(a0, b2, h20);

                a0 = _mm256_set1_ps(a[(1 + i) * K + k]);

                h01 = _mm256_fmadd_ps(a0, b0, h01);
                h11 = _mm256_fmadd_ps(a0, b1, h11);
                h21 = _mm256_fmadd_ps(a0, b2, h21);

                a0 = _mm256_set1_ps(a[(2 + i) * K + k]);

                h02 = _mm256_fmadd_ps(a0, b0, h02);
                h12 = _mm256_fmadd_ps(a0, b1, h12);
                h22 = _mm256_fmadd_ps(a0, b2, h22);

                a0 = _mm256_set1_ps(a[(3 + i) * K + k]);

                h03 = _mm256_fmadd_ps(a0, b0, h03);
                h13 = _mm256_fmadd_ps(a0, b1, h13);
                h23 = _mm256_fmadd_ps(a0, b2, h23);
            }
            float * w = c + i * J + j;
            _mm256_store_ps(w, h00);
            w += 8;
            _mm256_store_ps(w, h10);
            w += 8;
            _mm256_store_ps(w, h20);
            w = w - 2 * 8 + J;
            _mm256_store_ps(w, h01);
            w += 8;
            _mm256_store_ps(w, h11);
            w += 8;
            _mm256_store_ps(w, h21);
            w = w - 2 * 8 + J;
            _mm256_store_ps(w, h02);
            w += 8;
            _mm256_store_ps(w, h12);
            w += 8;
            _mm256_store_ps(w, h22);
            w = w - 2 * 8 + J;
            _mm256_store_ps(w, h03);
            w += 8;
            _mm256_store_ps(w, h13);
            w += 8;
            _mm256_store_ps(w, h23);
        }
    }


    for (; j < (int)J - 7; j += 8) {
        for (int ii = 0; ii < i; ii += 4) {
            __m256 h0, h1, h2, h3,
                   b0, a0;


            b0 = _mm256_load_ps(&b[j]);

            a0 = _mm256_set1_ps(a[(0 + ii) * K]);
            h0 = _mm256_mul_ps(a0, b0);

            a0 = _mm256_set1_ps(a[(1 + ii) * K]);
            h1 = _mm256_mul_ps(a0, b0);

            a0 = _mm256_set1_ps(a[(2 + ii) * K]);
            h2 = _mm256_mul_ps(a0, b0);

            a0 = _mm256_set1_ps(a[(3 + ii) * K]);
            h3 = _mm256_mul_ps(a0, b0);


            for (int k = 0; k < K; ++k) {
                b0 = _mm256_load_ps(&b[k * J + j]);

                a0 = _mm256_set1_ps(a[(0 + ii) * K + k]);
                h0 = _mm256_fmadd_ps(a0, b0, h0);

                a0 = _mm256_set1_ps(a[(1 + ii) * K + k]);
                h1 = _mm256_fmadd_ps(a0, b0, h1);

                a0 = _mm256_set1_ps(a[(2 + ii) * K + k]);
                h2 = _mm256_fmadd_ps(a0, b0, h2);

                a0 = _mm256_set1_ps(a[(3 + ii) * K + k]);
                h3 = _mm256_fmadd_ps(a0, b0, h3);
            }

            float * w = c + ii * J + j;
            _mm256_store_ps(w, h0);
            w += J;
            _mm256_store_ps(w, h1);
            w += J;
            _mm256_store_ps(w, h2);
            w += J;
            _mm256_store_ps(w, h3);
        }
    }

    // #pragma omp parallel for num_threads(4)
    for (; j < J; ++j) {
        for (int ii = 0; ii < i; ++ii) {
            float acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += a[ii * K + k] * b[k * J + j];
            }
            c[ii * J + j] = acc;
        }
    }

    // #pragma omp parallel for num_threads(4)
    for (; i < I; ++i) {
        for (j = 0; j < J; ++j) {
            float acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += a[i * K + k] * b[k * J + j];
            }
            c[i * J + j] = acc;
        }
    }

}




inline float _add(float a, float b) {
    return a + b;
}

inline float _mul(float a, float b) {
    return a * b;
}

inline float _sub(float a, float b) {
    return a - b;
}

inline float _div(float a, float b) {
    return a / b;
}


void
exp_kernel(float * dest, float * src, ssize_t length) {
    int k = 0;
#if __AVX__
#ifdef __SVML__
    __m256 v_a;
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&src[k]);
        v_a = _mm256_exp_ps(v_a);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
#endif
    while (k < length) {
        dest[k] = exp(src[k]);
        ++k;
    }
}


void
log_kernel(float * dest, float * src, ssize_t length) {
    int k = 0;
#if __AVX__
#ifdef __SVML__
    __m256 v_a;
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&src[k]);
        v_a = _mm256_log_ps(v_a);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
#endif
    while (k < length) {
        dest[k] = log(src[k]);
        ++k;
    }
}


void inline
val_add_kernel(float * dest, float * lhs, float value, ssize_t length) {
    int k = 0;
#if __AVX__
#ifdef __SVML__
    __m256 v_a, constant = _mm256_set1_ps(value);
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_a = _mm256_add_ps(v_a, constant);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
#endif
    while (k < length) {
        dest[k] = lhs[k] + value;
        ++k;
    }
}


void inline
val_mul_kernel(float * dest, float * lhs, float value, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, constant = _mm256_set1_ps(value);
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_a = _mm256_mul_ps(v_a, constant);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
    while (k < length) {
        dest[k] = lhs[k] * value;
        ++k;
    }
}


void inline
val_div_kernel(float * dest, float * lhs, float value, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, constant = _mm256_set1_ps(value);
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_a = _mm256_div_ps(v_a, constant);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
    while (k < length) {
        dest[k] = lhs[k] * value;
        ++k;
    }
}


void inline
val_max_kernel(float * dest, float * lhs, float value, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, constant = _mm256_set1_ps(value);
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_a = _mm256_max_ps(v_a, constant);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
    while (k < length) {
        dest[k] = std::max(lhs[k], value);
        ++k;
    }
}


void inline
relu_kernel(float * dest, float * lhs, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, constant = _mm256_set1_ps(0);
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_a = _mm256_max_ps(v_a, constant);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
    while (k < length) {
        dest[k] = std::max(lhs[k], (float) 0);
        ++k;
    }
}

void inline
drelu_kernel(float * arg, float * self, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, v_b, constant = _mm256_set1_ps(0);
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&arg[k]);
        v_a = _mm256_cmp_ps(v_a, constant, 14); // -> greater than
        v_b = _mm256_load_ps(&self[k]);
        v_a = _mm256_mul_ps(v_a, v_b);
        _mm256_store_ps(&arg[k], v_a);
    }
#endif
    while (k < length) {
        arg[k] = self[k] * (arg[k] > 0);
        ++k;
    }
}


