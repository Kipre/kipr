
void
add_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
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
    for (int k=0; k < length; k++) {
        destination[k] += other[k];
    }
#endif
}

void
sub_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
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
    for (int k=0; k < length; k++) {
        destination[k] -= other[k];
    }
#endif
}


void
mul_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
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
    for (int k=0; k < length; k++) {
        destination[k] *= other[k];
    }
#endif
}


void
div_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
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
    for (int k=0; k < length; k++) {
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
exp_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
        __m256 v_a = _mm256_load_ps(&other[k]);
        v_a = _mm256_exp_ps(v_a);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] = exp(other[k]);
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        destination[k] = exp(other[k]);
    }
#endif
}
