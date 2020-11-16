

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


inline void rec_binary_op(float * dest, float * lhs, float * rhs, Shape &shape,
                          NDVector &l_strides, NDVector &r_strides, size_t * positions,
                          binary_op op, int depth) {
    if (depth < shape.nd - 1) {
        for (int k = 0; k < shape[depth]; ++k) {
            rec_binary_op(dest, lhs, rhs, shape, l_strides, r_strides, positions, op, depth + 1);
            positions[1] += l_strides[depth];
            positions[2] += r_strides[depth];
        }
        positions[1] -= l_strides[depth] * shape[depth];
        positions[2] -= r_strides[depth] * shape[depth];
    } else {
        for (int k = 0; k < shape[depth]; ++k) {
            dest[positions[0]] = op(lhs[positions[1] + l_strides[depth] * k],
                                    rhs[positions[2] + r_strides[depth] * k]);
            ++positions[0];
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
    __m256 v_a;
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&src[k]);
        v_a = _mm256_exp_ps(v_a);
        _mm256_store_ps(&dest[k], v_a);
    }
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
    __m256 v_a;
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&src[k]);
        v_a = _mm256_log_ps(v_a);
        _mm256_store_ps(&dest[k], v_a);
    }
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
    __m256 v_a, constant = _mm256_set_ps(value, value, value, value, value, value, value, value);
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_a = _mm256_add_ps(v_a, constant);
        _mm256_store_ps(&dest[k], v_a);
    }
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
    __m256 v_a, constant = _mm256_set_ps(value, value, value, value, value, value, value, value);
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
val_max_kernel(float * dest, float * lhs, float value, ssize_t length) {
    int k = 0;
#if __AVX__
    __m256 v_a, constant = _mm256_set_ps(value, value, value, value, value, value, value, value);
    for (k = 0; k < length - 8; k += 8) {
        v_a = _mm256_load_ps(&lhs[k]);
        v_a = _mm256_max_ps(v_a, constant);
        _mm256_store_ps(&dest[k], v_a);
    }
#endif
    while (k < length) {
        dest[k] = max(lhs[k], value);
        ++k;
    }
}

