
void
add_kernel(float * destination, float * other, ssize_t length) {
#if __AVX__
    int k;
    for (k = 0; k < length - 8; k += 8) {
        __m256 v_a = _mm256_load_ps(&destination[k]);
        __m256 v_b = _mm256_load_ps(&other[k]);
        v_a = _mm256_add_ps(v_a, v_b);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] += other[k];
        k++;
    }
#else
    for (int k = 0; k < length; k++) {
        destination[k] += other[k];
    }
#endif
}

inline void binary_op(float * dest, float * lhs, float * rhs, Shape &shape, 
                      NDVector &l_strides, NDVector &r_strides, size_t * positions, 
                      float (*op)(float, float), int depth) {
    if (depth < shape.nd) {
        for (int k = 0; k < shape[depth]; ++k) {
            binary_op(dest, lhs, rhs, shape, l_strides, r_strides, positions, op, depth + 1);
            positions[1] += l_strides[depth];
            positions[2] += r_strides[depth];
        }
        positions[1] -= l_strides[depth] * shape[depth];
        positions[2] -= r_strides[depth] * shape[depth];
    } else {
        printf("positions %i %i %i\n", positions[0], positions[1], positions[2]);
        dest[positions[0]] = op(lhs[positions[1]], rhs[positions[2]]);
        ++positions[0];
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
sub_kernel(float * destination, float * other, ssize_t length) {
#if __AVX__
    int k;
    for (k = 0; k < length - 8; k += 8) {
        __m256 v_a = _mm256_load_ps(&destination[k]);
        __m256 v_b = _mm256_load_ps(&other[k]);
        v_a = _mm256_sub_ps(v_a, v_b);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] -= other[k];
        k++;
    }
#else
    for (int k = 0; k < length; k++) {
        destination[k] -= other[k];
    }
#endif
}


void
mul_kernel(float * destination, float * other, ssize_t length) {
#if __AVX__
    int k;
    for (k = 0; k < length - 8; k += 8) {
        __m256 v_a = _mm256_load_ps(&destination[k]);
        __m256 v_b = _mm256_load_ps(&other[k]);
        v_a = _mm256_mul_ps(v_a, v_b);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] *= other[k];
        k++;
    }
#else
    for (int k = 0; k < length; k++) {
        destination[k] *= other[k];
    }
#endif
}


void
div_kernel(float * destination, float * other, ssize_t length) {
#if __AVX__
    int k;
    for (k = 0; k < length - 8; k += 8) {
        __m256 v_a = _mm256_load_ps(&destination[k]);
        __m256 v_b = _mm256_load_ps(&other[k]);
        v_a = _mm256_div_ps(v_a, v_b);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] /= other[k];
        k++;
    }
#else
    for (int k = 0; k < length; k++) {
        if (other[k] == 0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "");
            PyErr_Print();
            PyErr_Clear();
        }
        destination[k] /= other[k];
    }
#endif
}


void
exp_kernel(float * destination, float * other, ssize_t length) {
#if __AVX__
    int k;
    for (k = 0; k < length - 8; k += 8) {
        __m256 v_a = _mm256_load_ps(&other[k]);
        v_a = _mm256_exp_ps(v_a);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] = exp(other[k]);
        k++;
    }
#else
    for (int k = 0; k < length; k++) {
        destination[k] = exp(other[k]);
    }
#endif
}


void
log_kernel(float * destination, float * other, ssize_t length) {
#if __AVX__
    int k;
    for (k = 0; k < length - 8; k += 8) {
        __m256 v_a = _mm256_load_ps(&other[k]);
        v_a = _mm256_log_ps(v_a);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] = log(other[k]);
        k++;
    }
#else
    for (int k = 0; k < length; k++) {
        destination[k] = log(other[k]);
    }
#endif
}


void
val_mul_kernel(float * destination, float value, ssize_t length) {
#if __AVX__
    int k;
    __m256 values, constant = _mm256_set_ps(value, value, value, value, value, value, value, value);
    for (k = 0; k < length - 8; k += 8) {
        values = _mm256_load_ps(&destination[k]);
        values = _mm256_mul_ps(values, constant);
        _mm256_store_ps(&destination[k], values);
    }
    while (k < length) {
        destination[k] *= value;
        k++;
    }
#else
    for (int k = 0; k < length; k++) {
        destination[k] *= value;
    }
#endif
}

void
max_val_kernel(float * destination, float value, ssize_t length) {
#if __AVX__
    int k;
    __m256 values, val = _mm256_set_ps (value, value, value, value, value, value, value, value);
    for (k = 0; k < length - 8; k += 8) {
        values = _mm256_load_ps(&destination[k]);
        values = _mm256_max_ps(values, val);
        _mm256_store_ps(&destination[k], values);
    }
    while (k < length) {
        destination[k] = Py_MAX(value, destination[k]);
        k++;
    }
#else
    for (int k = 0; k < length; k++) {
        destination[k] = Py_MAX(value, destination[k]);
    }
#endif
}
