

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


Karray loop_matmul(const Karray & rhs, const Karray & lhs) {
    Shape result_shape = (rhs.shape.nd < lhs.shape.nd) ? lhs.shape : rhs.shape;

    size_t I, J, K, rmat, lmat, wmat = 0, rnb = 1, lnb = 1;
    int r = rhs.shape.nd - 1, l = lhs.shape.nd - 1;

    K = rhs.shape[r];
    J = lhs.shape[l];
    --r; --l;

    // printf("K = %i, J = %i, r = %i, l = %i\n", K, J, r, l);
    if (r < 0 || l < 0 || K != lhs.shape[l]) {
        PyErr_Format(Karray_error,
                     "Matmul not possible with shapes %s ans %s.",
                     rhs.shape.str(), lhs.shape.str());
        return Karray();
    }
    I = rhs.shape[r];
    --r; --l;

    result_shape.set(result_shape.nd - 1, J);
    result_shape.set(result_shape.nd - 2, I);


    // printf("K = %i, I = %i, J = %i, r = %i, l = %i\n", K, I, J, r, l);
    while (r >= 0 && l >= 0 ) {
        if (rhs.shape[r] != lhs.shape[l]) {
            PyErr_Format(Karray_error,
                         "Matmul not possible with shapes %s ans %s.",
                         rhs.shape.str(), lhs.shape.str());
            return Karray();
        }
        rnb *= rhs.shape[r];
        lnb *= lhs.shape[l];
        --r; --l;
    }
    while (r >= 0) {
        rnb *= rhs.shape[r];
        --r;
    }
    while (l >= 0) {
        lnb *= lhs.shape[l];
        --l;
    }
    // result_shape.print("result_shape");
    Karray result(result_shape);

    size_t nb_loops = max(rnb, lnb);
    size_t write = 0;
    // printf("nb_loops %i\n", nb_loops);
    for (int m = 0; m < nb_loops; ++m) {
        rmat = (m % rnb) * I * K;
        lmat = (m % lnb) * K * J;
        // printf("rmat %i, lmat %i\n", lmat, rmat);
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                result.data[write] = 0;
                for (int k = 0; k < K; ++k) {
                    // printf("%i <- %i * %i\n", write, rmat + k, lmat + j + k*J);
                    // printf("%f += %f * %f\n", result.data[write], rhs.data[rmat + k], lhs.data[lmat + j + k*J]);
                    // printf("i = %i, j = %i, k = %i\n", i, j, k);
                    result.data[write] += rhs.data[rmat + k] * lhs.data[lmat + j + k * J];
                }
                ++write;
            }
            rmat += K;
        }
    }

    return result;
}


void transpose(float * from, float * to, Positions * pos,
               Shape & shape, const NDVector& strides, int depth) {
    if (depth < shape.nd) {
        for (int k = 0; k < shape[depth]; ++k) {
            transpose(from, to, pos, shape, strides, depth + 1);
            pos->left += strides[depth];
        }
        pos->left -= shape[depth] * strides[depth];
    } else {
        // printf("writing from %i to %i\n", pos->left, pos->write);
        to[pos->write] = from[pos->left];
        ++pos->write;
    }
}



Karray loop_transpose_matmul(const Karray & rhs, const Karray & lhs) {
    Shape result_shape = (rhs.shape.nd < lhs.shape.nd) ? lhs.shape : rhs.shape;

    size_t I, J, K, rmat, lmat, wmat = 0, rnb = 1, lnb = 1;
    int r = rhs.shape.nd - 1, l = lhs.shape.nd - 1;

    K = rhs.shape[r];
    J = lhs.shape[l];
    --r; --l;

    // printf("K = %i, J = %i, r = %i, l = %i\n", K, J, r, l);
    if (r < 0 || l < 0 || K != lhs.shape[l]) {
        PyErr_Format(Karray_error,
                     "Matmul not possible with shapes %s ans %s.",
                     rhs.shape.str(), lhs.shape.str());
        return Karray();
    }
    I = rhs.shape[r];
    --r; --l;

    result_shape.set(result_shape.nd - 1, J);
    result_shape.set(result_shape.nd - 2, I);


    // printf("K = %i, I = %i, J = %i, r = %i, l = %i\n", K, I, J, r, l);
    while (r >= 0 && l >= 0 ) {
        if (rhs.shape[r] != lhs.shape[l]) {
            PyErr_Format(Karray_error,
                         "Matmul not possible with shapes %s ans %s.",
                         rhs.shape.str(), lhs.shape.str());
            return Karray();
        }
        rnb *= rhs.shape[r];
        lnb *= lhs.shape[l];
        --r; --l;
    }
    while (r >= 0) {
        rnb *= rhs.shape[r];
        --r;
    }
    while (l >= 0) {
        lnb *= lhs.shape[l];
        --l;
    }
    Karray result(result_shape);

    float * trans = new float[rhs.shape.length];

    auto [shape_t, strides_t] = lhs.shape.transpose();

    Positions pos {0, 0, 0};
    transpose(lhs.data, trans, &pos, shape_t, strides_t, 0);

    // printf("pos %i %i %i\n", pos.write, pos.left, pos.right);


    size_t nb_loops = max(rnb, lnb);
    size_t write = 0;
    // printf("nb_loops %i\n", nb_loops);
    for (int m = 0; m < nb_loops; ++m) {
        rmat = (m % rnb) * I * K;
        lmat = (m % lnb) * J * K;
        // printf("rmat %i, lmat %i\n", lmat, rmat);
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                result.data[write] = 0;
                for (int k = 0; k < K; ++k) {
                    // printf("%i <- %i * %i\n", write, rmat + k, lmat + k);
                    // printf("%f += %f * %f\n", result.data[write], rhs.data[rmat + k], lhs.data[lmat + j + k*J]);
                    // printf("i = %i, j = %i, k = %i\n", i, j, k);
                    result.data[write] += rhs.data[rmat + k] * trans[lmat + k];
                }
                ++write;
                lmat += K;
            }
            rmat += K;
            lmat -= K * J;
        }
    }

    return result;

}


void print_m256(__m256 a, const char * msg = "") {
    float tmp[8];
    _mm256_store_ps(tmp, a);
    printf("__m256 %s %f, %f, %f, %f, %f, %f, %f, %f\n", msg,
           tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]);
}

// template <unsigned u, unsigned v>
// void matmul(float * c, float * a, float * b, size_t I, size_t J, size_t K) {

//     for (int ii=0; ii < I; ii += u) {
//         for (int jj=0; jj < J / 8; jj += v) {
//             __m256 csum[u * v] = {0};
//             for (int k=0; k < K; ++k) {
//                 for (int j=0; j < v; ++j) {
//                     __m256 bb = _mm256_load_ps(&b[k * J + (j + jj) * 8]);
//                     for (int i=0; i < u; ++i) {
//                         __m256 aa = _mm256_set1_ps(a[(i + ii) * K + k]);
//                         csum[i * v + j] = _mm256_fmadd_ps(aa, bb, csum[i * v + j]);
//                     }
//                 }
//             }

//             for (int j=0; j < v; ++j)
//                 for (int i=0; i < u; ++i)
//                     _mm256_store_ps(&c[(i + ii) * J + (j + jj) * 8], csum[i * v + j]);
//         }
//     }

// }


void matmul(float * c, float * a, float * b, size_t I, size_t J, size_t K) {


    #pragma omp parallel for num_threads(8)
    for (int ii = 0; ii < I; ii += 4) {
        for (int jj = 0; jj < J / 8; jj += 3) {
            __m256 h00, h01, h02, h03,
                   h10, h11, h12, h13,
                   h20, h21, h22, h23,
                   b0, b1, b2, a0;

            b0 = _mm256_load_ps(&b[(0 + jj) * 8]);
            b1 = _mm256_load_ps(&b[(1 + jj) * 8]);
            b2 = _mm256_load_ps(&b[(2 + jj) * 8]);

            a0 = _mm256_set1_ps(a[0 + ii]);

            h00 = _mm256_mul_ps(a0, b0);
            h10 = _mm256_mul_ps(a0, b1);
            h20 = _mm256_mul_ps(a0, b2);

            a0 = _mm256_set1_ps(a[1 + ii]);

            h01 = _mm256_mul_ps(a0, b0);
            h11 = _mm256_mul_ps(a0, b1);
            h21 = _mm256_mul_ps(a0, b2);

            a0 = _mm256_set1_ps(a[2 + ii]);

            h02 = _mm256_mul_ps(a0, b0);
            h12 = _mm256_mul_ps(a0, b1);
            h22 = _mm256_mul_ps(a0, b2);

            a0 = _mm256_set1_ps(a[3 + ii]);

            h03 = _mm256_mul_ps(a0, b0);
            h13 = _mm256_mul_ps(a0, b1);
            h23 = _mm256_mul_ps(a0, b2);


            for (int k = 1; k < K; ++k) {

                b0 = _mm256_load_ps(&b[k * J + (0 + jj) * 8]);
                b1 = _mm256_load_ps(&b[k * J + (1 + jj) * 8]);
                b2 = _mm256_load_ps(&b[k * J + (2 + jj) * 8]);

                a0 = _mm256_set1_ps(a[(0 + ii) * K + k]);

                h00 = _mm256_fmadd_ps(a0, b0, h00);
                h10 = _mm256_fmadd_ps(a0, b1, h10);
                h20 = _mm256_fmadd_ps(a0, b2, h20);

                a0 = _mm256_set1_ps(a[(1 + ii) * K + k]);

                h01 = _mm256_fmadd_ps(a0, b0, h01);
                h11 = _mm256_fmadd_ps(a0, b1, h11);
                h21 = _mm256_fmadd_ps(a0, b2, h21);

                a0 = _mm256_set1_ps(a[(2 + ii) * K + k]);

                h02 = _mm256_fmadd_ps(a0, b0, h02);
                h12 = _mm256_fmadd_ps(a0, b1, h12);
                h22 = _mm256_fmadd_ps(a0, b2, h22);

                a0 = _mm256_set1_ps(a[(3 + ii) * K + k]);

                h03 = _mm256_fmadd_ps(a0, b0, h03);
                h13 = _mm256_fmadd_ps(a0, b1, h13);
                h23 = _mm256_fmadd_ps(a0, b2, h23);
            }
            float * w = c + ii * J + jj * 8;
            _mm256_store_ps(w, h00);
            printf("%lli \n", w - c);
            w += 8;
            _mm256_store_ps(w, h10);
            printf("%lli \n", w - c);
            w += 8;
            _mm256_store_ps(w, h20);
            printf("%lli \n", w - c);
            w = w - 2 * 8 + J;
            _mm256_store_ps(w, h01);
            printf("%lli \n", w - c);
            w += 8;
            _mm256_store_ps(w, h11);
            printf("%lli \n", w - c);
            w += 8;
            _mm256_store_ps(w, h21);
            printf("%lli \n", w - c);
            w = w - 2 * 8 + J;
            _mm256_store_ps(w, h02);
            printf("%lli \n", w - c);
            w += 8;
            _mm256_store_ps(w, h12);
            printf("%lli \n", w - c);
            w += 8;
            _mm256_store_ps(w, h22);
            printf("%lli \n", w - c);
            w = w - 2 * 8 + J;
            _mm256_store_ps(w, h03);
            printf("%lli \n", w - c);
            w += 8;
            _mm256_store_ps(w, h13);
            printf("%lli \n", w - c);
            w += 8;
            _mm256_store_ps(w, h23);
            printf("%lli \n", w - c);
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
    __m256 v_a, constant = _mm256_set1_ps(value);
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
        dest[k] = max(lhs[k], value);
        ++k;
    }
}

