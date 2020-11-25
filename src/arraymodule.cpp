#include "arraymodule.hpp" 
static PyMemberDef Karray_members[] = {
    // {"attr", T_INT, offsetof(PyKarray, attr), 0,
    //  "Arbitrary attribute."},
    {NULL}  /* Sentinel */
};

static PyGetSetDef Karray_getsetters[] = {
    {"refcnt", (getter) Karray_getrefcnt, NULL,
     "Python refcount of the object.", NULL},
    {"shape", (getter) Karray_getshape, NULL,
     "Shape of the array.", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef Karray_methods[] = {
    {"reshape", (PyCFunction) Karray_reshape, METH_O,
     "Return the kipr.arr with the new shape."},
    {"broadcast", (PyCFunction) Karray_broadcast, METH_O,
     "Return the kipr.arr with the breadcasted shape."},
    {"mean", (PyCFunction) Karray_mean, METH_VARARGS | METH_KEYWORDS,
     "Return the averaged array."},
    {"sum", (PyCFunction) Karray_sum, METH_VARARGS | METH_KEYWORDS,
     "Return the sum of the array along all or a particular dim."},
    {"numpy", (PyCFunction) Karray_numpy, METH_NOARGS,
     "Return a numpy representtion of the PyKarray."},
    {"execute", (PyCFunction)  execute_func, METH_O,
     "Testing function to execute C code."},
    {"transpose", (PyCFunction)  Karray_transpose, METH_NOARGS,
     "Get the transpose of <kipr.arr>."},
    {NULL}  /* Sentinel */
};



static PyMethodDef Graph_methods[] = {
    {NULL}  /* Sentinel */
};


static PyGetSetDef Graph_getsetters[] = {
    {NULL}  /* Sentinel */
};

static PyMemberDef Graph_members[] = {
    // {"attr", T_INT, offsetof(PyKarray, attr), 0,
    //  "Arbitrary attribute."},
    {NULL}  /* Sentinel */
};

static PyMethodDef arraymodule_methods[] = {
    // {"max_nd", max_nd, METH_NOARGS,
    //  "Get maximum number of dimensions for a kipr.arr() array."},
    {"execute", execute_func, METH_O,
     "Testing function to execute C code."},
    {"function", function_decorator, METH_O,
     "Function decorator."},
    {"internal_test", internal_test, METH_NOARGS,
     "Execute C/C++ side tests."},
    {"relu", Karray_relu, METH_O,
     "ReLU function for <kipr.arr> arrays."},
    {"exp", Karray_exp, METH_O,
     "Exponential function for <kipr.arr> arrays."},
    {"softmax", (PyCFunction) Karray_softmax, METH_FASTCALL,
     "Softmax function for <kipr.arr> arrays, computes along the last axis."},
    {"log", Karray_log, METH_O,
     "Log function for <kipr.arr> arrays."},
    {"cache_info", cache_info, METH_NOARGS,
     "Function to query CPU info about cache configuration."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyModuleDef arraymodule = {
    PyModuleDef_HEAD_INIT,
    "kipr_array",
    "Array backend.",
    -1,
    arraymodule_methods
};



static PyNumberMethods Karray_as_number = {
    .nb_add = Karray_add,
    .nb_subtract = Karray_sub,
    .nb_multiply = Karray_mul,

    .nb_negative = Karray_negative,

    .nb_inplace_add = Karray_inplace_add,
    .nb_inplace_subtract = Karray_inplace_sub,
    .nb_inplace_multiply = Karray_inplace_mul,

    .nb_true_divide = Karray_div,
    .nb_inplace_true_divide = Karray_inplace_div,

    .nb_matrix_multiply = Karray_matmul
};

static PyMappingMethods Karray_as_mapping = {
    .mp_subscript = Karray_subscript
};

static PyTypeObject KarrayType = {
    Karray_HEAD_INIT
    .tp_name = KARRAY_NAME,
    .tp_basicsize = sizeof(PyKarray) - sizeof(float),
    .tp_itemsize = sizeof(float),
    .tp_dealloc = (destructor) Karray_dealloc,
    .tp_repr = (reprfunc) Karray_str, 
    .tp_as_number = &Karray_as_number,
    .tp_as_mapping = &Karray_as_mapping,
    .tp_str = (reprfunc) Karray_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Array object from kipr",
    .tp_methods = Karray_methods,
    .tp_members = Karray_members,
    .tp_getset = Karray_getsetters,
    .tp_init = (initproc) Karray_init,
    .tp_new = Karray_new,
};



static PyTypeObject GraphType = {
    Karray_HEAD_INIT
    .tp_name = "kipr.graph",
    .tp_basicsize = sizeof(PyGraph),
    .tp_dealloc = (destructor) Graph_dealloc,
    .tp_repr = (reprfunc) Graph_str, 
    .tp_str = (reprfunc) Graph_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Graph object from kipr",
    .tp_methods = Graph_methods,
    .tp_members = Graph_members,
    .tp_getset = Graph_getsetters,
    .tp_init = (initproc) Graph_init,
    .tp_new = Graph_new,
};

PyMODINIT_FUNC
PyInit_kipr_array(void)
{
    Karray_error = PyErr_NewException("kipr.KarrayError", NULL, NULL);
    import_array();
    PyObject *m;
    if (PyType_Ready(&KarrayType) < 0)
        return NULL;
    if (PyType_Ready(&GraphType) < 0)
        return NULL;

    m = PyModule_Create(&arraymodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&KarrayType);
    if (PyModule_AddObject(m, "arr", (PyObject *) &KarrayType) < 0) {
        Py_DECREF(&KarrayType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&GraphType);
    if (PyModule_AddObject(m, "graph", (PyObject *) &GraphType) < 0) {
        Py_DECREF(&GraphType);
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "KarrayError", Karray_error) < 0) {
        Py_DECREF(Karray_error);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
const int OP_POP_TOP = 1;
const int OP_ROT_TWO = 2;
const int OP_ROT_THREE = 3;
const int OP_DUP_TOP = 4;
const int OP_DUP_TOP_TWO = 5;
const int OP_ROT_FOUR = 6;
const int OP_NOP = 9;
const int OP_UNARY_POSITIVE = 10;
const int OP_UNARY_NEGATIVE = 11;
const int OP_UNARY_NOT = 12;
const int OP_UNARY_INVERT = 15;
const int OP_BINARY_MATRIX_MULTIPLY = 16;
const int OP_INPLACE_MATRIX_MULTIPLY = 17;
const int OP_BINARY_POWER = 19;
const int OP_BINARY_MULTIPLY = 20;
const int OP_BINARY_MODULO = 22;
const int OP_BINARY_ADD = 23;
const int OP_BINARY_SUBTRACT = 24;
const int OP_BINARY_SUBSCR = 25;
const int OP_BINARY_FLOOR_DIVIDE = 26;
const int OP_BINARY_TRUE_DIVIDE = 27;
const int OP_INPLACE_FLOOR_DIVIDE = 28;
const int OP_INPLACE_TRUE_DIVIDE = 29;
const int OP_RERAISE = 48;
const int OP_WITH_EXCEPT_START = 49;
const int OP_GET_AITER = 50;
const int OP_GET_ANEXT = 51;
const int OP_BEFORE_ASYNC_WITH = 52;
const int OP_END_ASYNC_FOR = 54;
const int OP_INPLACE_ADD = 55;
const int OP_INPLACE_SUBTRACT = 56;
const int OP_INPLACE_MULTIPLY = 57;
const int OP_INPLACE_MODULO = 59;
const int OP_STORE_SUBSCR = 60;
const int OP_DELETE_SUBSCR = 61;
const int OP_BINARY_LSHIFT = 62;
const int OP_BINARY_RSHIFT = 63;
const int OP_BINARY_AND = 64;
const int OP_BINARY_XOR = 65;
const int OP_BINARY_OR = 66;
const int OP_INPLACE_POWER = 67;
const int OP_GET_ITER = 68;
const int OP_GET_YIELD_FROM_ITER = 69;
const int OP_PRINT_EXPR = 70;
const int OP_LOAD_BUILD_CLASS = 71;
const int OP_YIELD_FROM = 72;
const int OP_GET_AWAITABLE = 73;
const int OP_LOAD_ASSERTION_ERROR = 74;
const int OP_INPLACE_LSHIFT = 75;
const int OP_INPLACE_RSHIFT = 76;
const int OP_INPLACE_AND = 77;
const int OP_INPLACE_XOR = 78;
const int OP_INPLACE_OR = 79;
const int OP_LIST_TO_TUPLE = 82;
const int OP_RETURN_VALUE = 83;
const int OP_IMPORT_STAR = 84;
const int OP_SETUP_ANNOTATIONS = 85;
const int OP_YIELD_VALUE = 86;
const int OP_POP_BLOCK = 87;
const int OP_POP_EXCEPT = 89;
const int OP_STORE_NAME = 90;
const int OP_DELETE_NAME = 91;
const int OP_UNPACK_SEQUENCE = 92;
const int OP_FOR_ITER = 93;
const int OP_UNPACK_EX = 94;
const int OP_STORE_ATTR = 95;
const int OP_DELETE_ATTR = 96;
const int OP_STORE_GLOBAL = 97;
const int OP_DELETE_GLOBAL = 98;
const int OP_LOAD_CONST = 100;
const int OP_LOAD_NAME = 101;
const int OP_BUILD_TUPLE = 102;
const int OP_BUILD_LIST = 103;
const int OP_BUILD_SET = 104;
const int OP_BUILD_MAP = 105;
const int OP_LOAD_ATTR = 106;
const int OP_COMPARE_OP = 107;
const int OP_IMPORT_NAME = 108;
const int OP_IMPORT_FROM = 109;
const int OP_JUMP_FORWARD = 110;
const int OP_JUMP_IF_FALSE_OR_POP = 111;
const int OP_JUMP_IF_TRUE_OR_POP = 112;
const int OP_JUMP_ABSOLUTE = 113;
const int OP_POP_JUMP_IF_FALSE = 114;
const int OP_POP_JUMP_IF_TRUE = 115;
const int OP_LOAD_GLOBAL = 116;
const int OP_IS_OP = 117;
const int OP_CONTAINS_OP = 118;
const int OP_JUMP_IF_NOT_EXC_MATCH = 121;
const int OP_SETUP_FINALLY = 122;
const int OP_LOAD_FAST = 124;
const int OP_STORE_FAST = 125;
const int OP_DELETE_FAST = 126;
const int OP_RAISE_VARARGS = 130;
const int OP_CALL_FUNCTION = 131;
const int OP_MAKE_FUNCTION = 132;
const int OP_BUILD_SLICE = 133;
const int OP_LOAD_CLOSURE = 135;
const int OP_LOAD_DEREF = 136;
const int OP_STORE_DEREF = 137;
const int OP_DELETE_DEREF = 138;
const int OP_CALL_FUNCTION_KW = 141;
const int OP_CALL_FUNCTION_EX = 142;
const int OP_SETUP_WITH = 143;
const int OP_LIST_APPEND = 145;
const int OP_SET_ADD = 146;
const int OP_MAP_ADD = 147;
const int OP_LOAD_CLASSDEREF = 148;
const int OP_EXTENDED_ARG = 144;
const int OP_SETUP_ASYNC_WITH = 154;
const int OP_FORMAT_VALUE = 155;
const int OP_BUILD_CONST_KEY_MAP = 156;
const int OP_BUILD_STRING = 157;
const int OP_LOAD_METHOD = 160;
const int OP_CALL_METHOD = 161;
const int OP_LIST_EXTEND = 162;
const int OP_SET_UPDATE = 163;
const int OP_DICT_MERGE = 164;
const int OP_DICT_UPDATE = 165;


void transfer(float * from, float * to, Positions * pos, size_t * strides,
              const Filter & filter, int nd, int depth) {
	if (depth < nd) {
		size_t current_value, last_value = 0;
		for (int k = filter.offset[depth]; k < filter.offset[depth + 1]; ++k) {
			current_value = filter.vec[k];
			pos->right += (current_value - last_value) * strides[depth];
			last_value = current_value;
			transfer(from, to, pos, strides, filter, nd, depth + 1);
		}
		pos->right -= last_value * strides[depth];
	} else {
		// printf("writing from %i to %i\n", pos->right, pos->write);
		to[pos->write++] = from[pos->right];
	}
}


void simple_transfer(float * from, float * to, Positions * pos, 
	Shape & shape, NDVector & strides, int depth) {
	if (depth < shape.nd) {
		for (int k=0; k < shape[depth]; ++k) {
			simple_transfer(from, to, pos, shape, strides, depth + 1);
			pos->left += strides[depth];
		}
        pos->left -= shape[depth] * strides[depth];
	} else {
		to[pos->write++] = from[pos->left];
	}

}


size_t read_mode(PyObject * o) {
	if (!PyUnicode_Check(o))
		return ERROR_MODE;
	if (PyUnicode_Compare(o, PyUnicode_FromString("rand")) == 0 ||
	        PyUnicode_Compare(o, PyUnicode_FromString("random")) == 0) {
		return RANDOM_UNIFORM;
	}
	if (PyUnicode_Compare(o, PyUnicode_FromString("randn")) == 0) {
		return RANDOM_NORMAL;
	}
	if (PyUnicode_Compare(o, PyUnicode_FromString("range")) == 0) {
		return RANGE;
	}
	PyErr_Format(PyExc_ValueError,
	             "String magic %s not understood.", PyUnicode_AsUTF8(o));
	return ERROR_MODE;
}

size_t py_type(PyObject * o) {
	if ((PyTypeObject *) PyObject_Type(o) == &KarrayType)
		return KARRAY;
	if (PyArray_Check(o))
		return NUMPY_ARRAY;
	if (PyUnicode_Check(o))
		return STRING;
	if (PyNumber_Check(o))
		return NUMBER;
	if (PySequence_Check(o))
		return SEQUENCE;
	if (PySlice_Check(o))
		return SLICE;
	return 0;
}

size_t subscript_type(PyObject * o) {
	if (PyNumber_Check(o))
		return NUMBER;
	if (PySlice_Check(o))
		return SLICE;
	if (PySequence_Check(o))
		return SEQUENCE;
	return 0;
}

size_t align_index(Py_ssize_t i, size_t dim_length) {
	if (abs(i) >= dim_length) {
		PyErr_Format(PyExc_ValueError,
		             "Index %i out of range on axis with length %i.",
		             i, dim_length);
		return 0;
	} else {
		int result = (i + dim_length) % dim_length;
		return (size_t) result;
	}
}



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



void
inline _sum(float * self_data, float * result_data, float * weights_data,
            Shape &self_shape, NDVector &strides, bool multiple_weights,
            bool mean, int axis, int depth) {
    if (axis != depth) {
        for (int k = 0; k < self_shape[depth]; ++k) {
            _sum(self_data + strides[depth]*k, result_data + strides[depth]*k / self_shape[axis],
                 weights_data, self_shape, strides, multiple_weights, mean, axis, depth + 1);
        }
    } else {
        for (int i = 0; i < self_shape[axis]; ++i) {
            for (int k = 0; k < strides[axis]; ++k) {
                // printf("val and result: %f %f %i %i\n", self_data[strides[axis] * i + k], result_data[k], strides[axis] * i + k, i);
                result_data[k] += self_data[strides[axis] * i + k] * weights_data[multiple_weights * i];
            }
        }
        if (mean)
            for (int k = 0; k < strides[axis]; ++k)
                result_data[k] /= (float) self_shape[axis];
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
        dest[k] = max(lhs[k], 0);
        ++k;
    }
}

Karray Karray::elementwise_binary_op(const Karray &other,
                                     binary_kernel kernel,
                                     binary_op op) {
	size_t length = shape.length;
	if (length == other.shape.length) {
		Karray result(shape);
		kernel(result.data, data, other.data, length);
		return result;
	} else {
		auto [common, a_strides, b_strides] =
		    paired_strides(shape, other.shape);
		PYERR_RETURN_VAL(Karray());
		Karray result(common);
		size_t positions[3] = {0};
		rec_binary_op(result.data, data, other.data,
		              common, a_strides, b_strides, positions, op, 0);
		return result;
	}
}

void Karray::inplace_binary_op(const Karray  &rhs,
                               binary_kernel kernel,
                               binary_op op) {
	if (shape.length == rhs.shape.length) {
		kernel(data, data, rhs.data, shape.length);
	} else {
		auto [a_strides, b_strides] =
		    shape.paired_strides(rhs.shape);
		PYERR_RETURN;
		size_t positions[3] = {0};
		rec_binary_op(data, data, rhs.data, shape,
		              a_strides, b_strides, positions, op, 0);
	}
}

Karray::Karray(Shape new_shape) {
	seed = rand();
	shape = new_shape;
	data = new float[shape.length];
}


Karray::Karray(Shape new_shape, float value) {
	seed = rand();
	shape = new_shape;
	data = new float[shape.length];
	std::fill(data, data + shape.length, value);
}

Karray::Karray(float val) {
	// printf("creating generic new karr\n");
	seed = rand();
	shape = Shape();
	data = new float[1];
	data[0] = val;
}


Karray::Karray() {
	// printf("creating generic new karr\n");
	seed = rand();
	shape = Shape();
	data = new float[1];
	data[0] = 0;
}

Karray::Karray(const Karray& other)
	: seed{other.seed + 1},
	  shape{other.shape} {
	data = new float[shape.length];
	std::copy(other.data, other.data + shape.length, data);
}

Karray& Karray::operator=(const Karray& other) {
	// printf("copying array %i into %i\n", other.seed, seed);
	shape = other.shape;
	delete[] data;
	data = new float[shape.length];
	std::copy(other.data, other.data + shape.length, data);
	return *this;
}

Karray::Karray(Karray&& other)
	: seed{other.seed + 1},
	  shape{other.shape} {
	seed = other.seed + 1;
	// printf("moving array %i into %i\n", other.seed, seed);
	data = other.data;
	other.shape = Shape();
	other.data = new float[1];
	other.data[0] = 0;
}

Karray& Karray::operator=(Karray&& other) {
	// printf("moving array %i into %i\n", other.seed, seed);
	shape = other.shape;
	delete[] data;
	data = other.data;
	other.shape = Shape();
	other.data = new float[1];
	other.data[0] = 0;
	return *this;
}

Karray& Karray::operator+=(const Karray& other) {
	inplace_binary_op(other, add_kernel, _add);
	return *this;
}

Karray& Karray::operator/=(const Karray& other) {
	inplace_binary_op(other, div_kernel, _div);
	return *this;
}

Karray& Karray::operator-=(const Karray& other) {
	inplace_binary_op(other, sub_kernel, _sub);
	return *this;
}

Karray& Karray::operator*=(const Karray& other) {
	inplace_binary_op(other, mul_kernel, _mul);
	return *this;
}

Karray Karray::operator-(const Karray& rhs) {
	return elementwise_binary_op(rhs, sub_kernel, _sub);
}

Karray Karray::operator*(const Karray& rhs) {
	return elementwise_binary_op(rhs, mul_kernel, _mul);
}

Karray Karray::operator+(const Karray& rhs) {
	return elementwise_binary_op(rhs, add_kernel, _add);
}

Karray Karray::operator/(const Karray& rhs) {
	return elementwise_binary_op(rhs, div_kernel, _div);
}

// Karray Karray::matmul(const Karray& rhs) {
// 	if (!shape.compatible_for_matmul(rhs.shape))
// 		throw std::exception("shapes incompatible for matmul");

// }

void Karray::swap(Karray& other) {
	// printf("swapping %i and %i\n", seed, other.seed);
	std::swap(shape, other.shape);
	std::swap(data, other.data);
}

Karray::Karray(Shape new_shape, std::vector<float> vec) {
	seed = rand();
	shape = new_shape;
	// printf("shape.length, vec.size(): %i %i\n", shape.length, vec.size());
	data = new float[shape.length];
	std::copy(vec.begin(), vec.end(), data);
}

Karray::Karray(Shape new_shape, float * new_data) {
	seed = rand();
	shape = new_shape;
	data = new_data;
}

void Karray::from_mode(Shape new_shape, size_t mode) noexcept {
	delete[] data;
	shape = new_shape;
	data = new float[shape.length];

	if (mode == RANDOM_NORMAL || mode == RANDOM_UNIFORM) {
		std::random_device rd{};
		std::mt19937 gen{rd()};

		if (mode == RANDOM_NORMAL) {
			std::normal_distribution<float> d{0, 1};
			for (int n = 0; n < shape.length; ++n) {
				data[n] = d(gen);
			}
		} else if (mode == RANDOM_UNIFORM) {
			for (int n = 0; n < shape.length; ++n) {
				data[n] = gen() / (float) 4294967295;
			}
		}
	} else if (mode == RANGE) {
		for (int n = 0; n < shape.length; ++n) {
			data[n] = (float) n;
		}
	} else {
		throw std::exception("unknown mode");
	}
}

Karray Karray::subscript(PyObject * key) {

	Karray result;
	Filter filter;
	Positions pos {0, 0, 0};

	NDVector strides(shape.strides());
	Shape new_shape(filter.from_subscript(key, shape));
	PYERR_PRINT_GOTO_FAIL;

	result.shape = new_shape;
	delete[] result.data;
	result.data = new float[new_shape.length];

	// filter.print();
	// strides.print();
	transfer(data, result.data, &pos,
	         strides.buf, filter, shape.nd, 0);
	// printf("positions[0], positions[1]: %i %i\n", positions[0], positions[1]);
	if (pos.write != new_shape.length)
		goto fail;

	return result;
fail:
	PyErr_SetString(PyExc_ValueError, "Failed to subscript array.");
	return result;
}

Karray::~Karray() {
	// printf("deallocating karr with shape %s and seed %i\n", shape.str().c_str(), seed);
	delete[] data;
	shape.~Shape();
}

void Karray::print(const char * message) {
	std::cout << "Printing Karray " << seed << " " << message << "\n\t";
	shape.print();
	std::cout << "\t";
	auto limit = min(shape.length, MAX_PRINT_SIZE);
	for (int i = 0; i < limit; ++i) {
		std::cout << data[i] << ", ";
	}
	std::cout << "\n";
}

std::string Karray::str() {
	NDVector strides = shape.strides();
	auto matches = [&](int i) {
		int result = 0;
		for (int k = 0; k < shape.nd - 1; ++k)
			if (i % strides[k] == 0)
				++result;
		return result;
	};
	std::ostringstream ss;
	ss << "kipr.arr(";
	auto limit = min(shape.length, MAX_PRINT_SIZE);
	int match = matches(0);
	for (int i = 0; i < limit; ++i) {
		if (i == 0)
			ss << std::string(match + 1, '[');
		if (match > 0 && i > 0) {
			ss << std::string(match, ']') << ",\n";
			ss << std::string(shape.nd - match + 9, ' ');
			ss << std::string(match, '[');
		}
		ss.width(5);
		ss << data[i];
		match = matches(i + 1);
		if (!match)
			ss << ", ";
		if (i == shape.length - 1) {
			ss << std::string(match + 1, ']') << ", ";
		} else if (i == limit - 1) {
			ss << " ... , ";
		}
	}
	ss << "shape=" << shape.str()
	   << ")";
	return ss.str();
}


Karray Karray::matmul(Karray & other) {

	if (shape.nd < 2 && other.shape.nd < 2) {
		KERR_RETURN_VAL("Both arrays must be at least two-dimensional for matmul.", NULL);
	}

	size_t M, N, I, J, K;
	I = shape[-2];
	K = shape[-1];
	J = other.shape[-1];

	M = shape.nbmats();
	N = other.shape.nbmats();

	if (K != other.shape[-2] ||
		(M % N != 0 && N % M != 0)) {
		PyErr_Format(Karray_error,
		             "Matmul not possible with shapes %s and %s.",
		             shape.str(), other.shape.str());
		return NULL;
	}

	Shape new_shape((M > N) ? shape : other.shape);
	new_shape.set(new_shape.nd - 2, I);
	new_shape.set(new_shape.nd - 1, J);

	auto result = Karray(new_shape);

	for (int m = 0; m < max(M, N); ++m) {
		int ia = m % M;
		int ib = m % N;

		general_matmul(result.data + m * I * J,
		       data + ia * I * K,
		       other.data + ib * K * J,
		       I, J, K);
	}

	return result;
}


Karray Karray::broadcast(Shape new_shape) {
	// shortcuts
	if (shape.length == 1) {
		Karray result(new_shape);
		std::fill(result.data, result.data + new_shape.length, data[0]);
		return result;
	} else if (shape.length == new_shape.length) {
		Karray result(*this);
		result.shape = new_shape;
		return result;
	} else {
		Positions pos {0, 0, 0};
		Karray result(new_shape);

		NDVector strides = shape.broadcast_to(new_shape);
		PYERR_RETURN_VAL(result);
		simple_transfer(data, result.data, &pos, new_shape, strides, 0);

		return result;
	}

}

void Karray::from_numpy(PyObject * obj) noexcept {
	auto arr = (PyArrayObject *) obj;
	npy_intp nd;
	float * arr_data;
	if ((nd = PyArray_NDIM(arr)) < MAX_ND &&
	        PyArray_TYPE(arr) == NPY_FLOAT) {
		shape = Shape(PyArray_SHAPE(arr), (int) nd);
		auto length = (size_t) PyArray_SIZE(arr);
		if (shape.length != length) goto fail;
		// printf("length, shape_length %i %i\n", length, shape.length);
		arr_data = (float *) PyArray_DATA(arr);
		delete[] data;
		data = new float[length];
		for (int i = 0; i < length; ++i) {
			data[i] = arr_data[i];
		}
	} else {
fail:
		PyErr_Clear();
		PyErr_SetString(PyExc_ValueError,
		                "Failed to copy numpy array.");
	}
}

Karray Karray::flat_sum(bool mean) {
	Karray result;
	for (int k = 0; k < shape.length; ++k) {
		result.data[0] += data[k];
	}
	if (mean)
		result.data[0] /= (float) shape.length;
	return result;
}


Karray Karray::sum(size_t axis, const Karray &weights, bool mean) {
	Shape new_shape = shape;
	size_t reduction = new_shape.pop((int) axis);
	if (weights.shape.length != 1 &&
	        weights.shape.length != reduction)
		KERR_RETURN_VAL("Weights do not correspond to sum reduction.", Karray{});
	NDVector strides = shape.strides();
	Karray result(new_shape, 0.);
	_sum(data, result.data, weights.data, shape, strides, (weights.shape.length != 1), mean, axis, 0);
	return result;
}
Shape::Shape(int ndims...) {
	va_list args;
	va_start(args, ndims);
	nd = ndims;
	length = 1;
	for (int i = 0; i < MAX_ND; ++i) {
		if (i < nd) {
			length *= (buf[i] = (size_t) va_arg(args, int));
		} else {
			buf[i] = 0;
		}
	}
	va_end(args);
}

Shape::Shape(Py_ssize_t * input, int size) {
	nd = 0;
	length = 1;
	size = min(MAX_ND, size);
	while (nd < size) {
		length *= (size_t) input[nd];
		buf[nd] = (size_t) input[nd];
		++nd;
	}
	int i = nd;
	while (i < MAX_ND) {
		buf[i++] = 0;
	}
}


Shape::Shape(size_t * input, int size) {
	nd = 0;
	length = 1;
	size = min(MAX_ND, size);
	while (nd < size) {
		length *= input[nd];
		buf[nd] = input[nd];
		++nd;
	}
	int i = nd;
	while (i < MAX_ND) {
		buf[i++] = 0;
	}
}

Shape::Shape(PyObject * o, size_t target_length) {
	nd = 0;
	length = 1;
	int wildcard = -1;
	Py_ssize_t value;
	if (!PyList_Check(o) && !PyTuple_Check(o))
		KERR_RETURN("Shape must be a list or a tuple.");
	Py_ssize_t seq_length = PySequence_Length(o);
	if (seq_length == 0)
		KERR_RETURN("Shape must be a list or a tuple with at least one element.");
	PyObject ** items = PySequence_Fast_ITEMS(o);
	for (int i = 0; i < seq_length; ++i) {
		value = PyLong_AsSsize_t(items[i]);

		PYERR_SET_RETURN("Shape must be a sequence of integers.");
		if (value == 0 || value < -1) {
			PyErr_Format(Karray_error,
			             "Proposed shape is invalid with value %I64i at %i", value, nd);
			return;
		}
		if (value == -1) {
			if (wildcard != -1)
				KERR_RETURN("Can't have more than one wildcard (-1) in the shape.");
			wildcard = nd;
		} else {
			buf[nd] = value;
			length *= value;
		}
		++nd;
	}
	int i = nd;
	while (i != MAX_ND) {
		buf[i] = 0;
		++i;
	}
	if (wildcard >= 0) {
		if (!target_length)
			KERR_RETURN("Wildcard (-1) not allowed here.");
		if (target_length % length != 0)
			KERR_RETURN("Proposed shape cannot be adapted to data.");
		buf[wildcard] = target_length / length;
		length = target_length;
	} else if (target_length) {
		if (target_length != length)
			KERR_RETURN("Proposed shape doesn't align with data.");

	}
	return;

}

Shape::Shape() {
	nd = 1;
	length = 1;
	buf[0] = 1;
	std::fill(buf + 1, buf + MAX_ND, 0);
}

Shape::Shape(Shape a, Shape b) noexcept { // [3, 4, 5] & [3, 1]
	if (a.nd < b.nd)
		std::swap(a, b);

	nd = a.nd;
	length = 1;
	int i = MAX_ND - 1;
	int dim_diff = a.nd - b.nd;
	while (i >= 0) {
		if (i >= dim_diff &&
		        a[i] != b[i - dim_diff]) {
			if (a[i] == 1) {
				buf[i] = b[i - dim_diff];
			} else if (b[i - dim_diff] == 1) {
				buf[i] = a[i];
			} else {
				PyErr_Format(Karray_error,
				             "Shapes %s and %s are not compatible.", a.str(), b.str());
				return;
			}
		} else {
			buf[i] = a[i];
		}
		length *= max(1, buf[i]);
		--i;
	}
}

static std::tuple<Shape, NDVector, NDVector>
paired_strides(Shape a, Shape b) noexcept {
	Shape common = Shape(a, b);
	NDVector astr, bstr;
	size_t acc = 1, bcc = 1;
	while (a.nd > b.nd)
		b.insert_one(0);
	while (b.nd > a.nd)
		a.insert_one(0);

	for (int k = a.nd - 1; k >= 0; --k) {
		if (a[k] == common[k]) {
			astr.buf[k] = acc;
			acc *= a[k];
		} else {
			astr.buf[k] = 0;
		}
		if (b[k] == common[k]) {
			bstr.buf[k] = bcc;
			bcc *= b[k];
		} else {
			bstr.buf[k] = 0;
		}
	}
	return {common, astr, bstr};
}

std::tuple<NDVector, NDVector>
Shape::paired_strides(Shape b) noexcept {
	NDVector astr, bstr;
	size_t acc = 1, bcc = 1;
	while ((nd - b.nd) > 0)
		b.insert_one(0);

	for (int k = nd - 1; k >= 0; --k) {
		if (b[k] == buf[k]) {
			bstr.buf[k] = bcc;
			bcc *= b[k];
			astr.buf[k] = acc;
			acc *= buf[k];
		} else if (b[k] == 1) {
			astr.buf[k] = acc;
			acc *= buf[k];
			bstr.buf[k] = 0;
		} else {
			PyErr_Format(Karray_error,
			             "Shapes %s and %s not compatible for inplace binary op.",
			             str(), b.str());
			return {astr, bstr};
		}
	}
	return {astr, bstr};
}


void Shape::swap(Shape & other) {
	std::swap(nd, other.nd);
	std::swap(length, other.length);
	size_t * tmp = buf;
	*buf = *other.buf;
	*other.buf = *tmp;
}

bool Shape::operator==(Shape &other) {
	if (nd != other.nd)
		return false;
	if (length != other.length)
		return false;
	for (int k = 0; k < MAX_ND; ++k)
		if (buf[k] != other[k])
			return false;
	return true;
}

void Shape::print(const char * message) const {
	std::cout << "Shape " << message <<
	          " nd=" << nd << ", length=" << length << "\n\t";
	for (int k = 0; k < MAX_ND; ++k) {
		std::cout << buf[k] << ", ";
	}
	std::cout << '\n';
}

bool Shape::assert_or_set(size_t value, int dim) {
	if ((dim == 0 && buf[0] == 1) ||
	        buf[dim] == 0) {
		buf[dim] = value;
		return true;
	} else if (buf[dim] == value) {
		return true;
	} else {
		return false;
	}
}

// void Shape::set(int i, size_t val) {
// 	buf[i] = val;
// }

void Shape::set(int i, size_t val) {
	if (i < nd) {
		length /= buf[i];
		buf[i] = val;
		length *= val;
	} else {
		for (int k=nd; k < i; ++k) {
			buf[k] = 1;
		}
		buf[i] = val;
		length *= val;
		nd = i + 1;
	}
}

size_t Shape::nbmats() {
	if (nd == 1)
		return 0;
	if (nd == 2)
		return 1;
	return length / (buf[nd-1] * buf[nd-2]);
}

size_t Shape::validate() {
	int i, new_nd = 0;
	length = 1;
	if (buf[0] == 0) goto fail;
	while (buf[new_nd] != 0 && new_nd < MAX_ND) {
		length *= buf[new_nd];
		++new_nd;
	}
	i = new_nd;
	while (i < MAX_ND) {
		if (buf[i++] != 0) goto fail;
	}
	nd = new_nd;
	return length;

fail:
	PyErr_Format(PyExc_ValueError,
	             "Shape %s is corrupted.", str());
	return 0;

}

void Shape::write(size_t * destination) {
	for (int i = 0; i < MAX_ND; ++i) {
		destination[i] = buf[i];
	}
}



std::string Shape::str() const {
	std::string result("[");
	result += std::to_string(buf[0]);
	for (int i = 1; i < nd; ++i)
		result += ", " + std::to_string(buf[i]);
	return result + "]";
}

NDVector Shape::broadcast_to(Shape & other) {
	NDVector result;
	int dim_diff = other.nd - nd;
	if (dim_diff < 0) {
		PyErr_Format(Karray_error,
		             "Cannot broadcast shape %s to %s.", str(), other.str());
		return result;
	}
	int acc = 1;
	for (int i = other.nd - 1; i > dim_diff; --i) {
		if (buf[i - dim_diff] == 1 && other[i] != 1) {
			result.buf[i] = 0;
		} else if (buf[i - dim_diff] == other[i]) {
			result.buf[i] = acc;
			acc *= other[i];
		} else {
		PyErr_Format(Karray_error,
		             "Cannot broadcast shape %s to %s.", str(), other.str());
		return result;
		}
	}
	for (int i = 0; i < dim_diff; ++i) {
		result.buf[i] = 0;
	}
	return result;

}

size_t Shape::sum() {
	size_t result = 0;
	for (int i = 0; i < nd; ++i) {
		result += buf[i];
	}
	return result;
}

size_t Shape::operator[](int i) const {
	if (i >= 0 && i < MAX_ND) {
		return buf[i];
	} else if (i > - nd - 1 && i < 0) {
		return buf[nd + i];
	} else {
		throw std::exception("index out of range");
	}
}

size_t Shape::pop(int i) noexcept {
	if (abs(i) >= nd)
		throw std::exception("Shape::pop out of range");
	if (i == -1)
		i = nd - 1;
	size_t tmp = buf[i];
	if (i == 0 && nd == 1) {
		buf[0] = 1;
		length = 1;
		return tmp;
	}
	while (i != MAX_ND - 1) {
		buf[i] = buf[i + 1];
		++i;
	}
	buf[MAX_ND - 1] = 0;
	--nd;
	length /= tmp;
	return tmp;
}

NDVector Shape::strides(int depth_diff) const {
	NDVector result;
	size_t acc = 1;
	// printf("depth diff %i, nd %i\n", depth_diff, nd);
	for (int i = nd - 1; i >= 0; --i) {
		result.buf[i + depth_diff] = acc;
		acc *= buf[i];
	}
	for (int i = 0; i < depth_diff; ++i) {
		result.buf[i] = 0;
	}
	return result;
}

void Shape::insert_one(int i) {
	if (i < 0 || i > nd)
		KERR_RETURN("Cannot insert 1 into shape because index is out of bounds.");
	++nd;
	int k = MAX_ND - 1;
	while (k > i) {
		buf[k] = buf[k - 1];
		--k;
	}
	buf[i] = 1;
}

void Shape::push_back(size_t dim) {
	if (def) {
		nd = 0;
		def = false;
	}
	buf[nd] = dim;
	++nd;
	length *= dim;
}

size_t Shape::axis(PyObject * o) {
	if (!PyIndex_Check(o))
		KERR_RETURN_VAL("Axis is invalid.", 9);
	Py_ssize_t value = PyLong_AsSsize_t(o);
	if (abs(value) > nd - 1)
		KERR_RETURN_VAL("Axis is out of range.", 9);
	return (size_t) (value % nd + nd) % nd;

}

size_t Shape::axis(int ax) {
	if (abs(ax) > nd - 1)
		KERR_RETURN_VAL("Axis is out of range.", 9);
	return (size_t) (ax % nd + nd) % nd;

}

// static compatible_for_matmul(Shape & a, Shape & b) {
// 	if (a.nd < 2 || b.nd < 2)
// 		return {};
// 	if (a[a.nd - 1] != b[b.nd - 2])
// 		return {};
// 	Shape result;
// 	if (a.nd > b.nd)
// 		result = b
// 	else 
// 		result = a;


// 	return true;
// }

std::tuple<Shape, NDVector> Shape::transpose() const {
	Shape result;
	result.nd = nd;
	result.length = length;
	std::copy(buf, buf + MAX_ND, result.buf);
	NDVector strides_t = strides();
	result.buf[nd-1] = buf[nd-2];
	result.buf[nd-2] = buf[nd-1];
	size_t tmp = strides_t.buf[nd-1];
	strides_t.buf[nd-1] = strides_t.buf[nd-2];
	strides_t.buf[nd-2] = tmp;
	return {result, strides_t};
}

int Shape::last_axis() {
	int i = nd-1;
	while (buf[i] == 1 && i > 0)
		--i;
	return i;
}Filter::Filter(Shape& shape) {
    size_t total = 0;
    int i;
    offset[0] = 0;
    for (i = 0; i < shape.nd;) {
        total += shape[i];
        offset[++i] = total;
    }
    while (i != MAX_ND) {
        offset[i++] = total;
    }
    vec.reserve(total);
    // std::fill(vec, vec + total, -1);
}

Filter::Filter(Filter&& other) noexcept : vec{}, offset{0} {
    vec = std::move(other.vec);
    for (int i = 0; i < MAX_ND; ++i) {
        offset[i] = other.offset[i];
        other.offset[i] = 0;
    }
}

void Filter::set_val_along_axis(int axis , size_t value) {
    // printf("writing val from %i to %i on axis %i\n", offset[axis], offset[axis + 1], axis);
    std::fill(vec.begin() + offset[axis], vec.begin() + offset[axis + 1], value);
}

void Filter::set_range_along_axis(int axis) {
    // printf("writing range from %i to %i on axis %i\n", offset[axis], offset[axis + 1], axis);
    std::iota (vec.begin() + offset[axis], vec.begin() + offset[axis + 1], 0);
}

void Filter::print(const char * message) {
    std::cout << "Filter " << message << "\n\t";
    int o = 1;
    for (int k = 0; k < vec.size(); ++k) {
        if (k == offset[o]) {
            std::cout << "| ";
            ++o;
        }
        std::cout << vec[k] << ", ";
    }
    std::cout << "\n\toffsets:";
    for (int k = 0; k < MAX_ND + 1; ++k) {
        std::cout << offset[k] << ", ";
    }
    std::cout << '\n';
}

Filter& Filter::operator=(Filter&& other) noexcept {
    if (this != &other) {
        vec = std::move(other.vec);
        for (int i = 0; i < MAX_ND; ++i) {
            offset[i] = other.offset[i];
            other.offset[i] = 0;
        }
    }
    return *this;
}

void Filter::push_back(size_t number, int index) {
    vec.push_back(number);
    offset[index + 1] = vec.size();
}

Shape Filter::from_subscript(PyObject * key, Shape &current_shape) {

    Shape new_shape;
    size_t ind;
    int rest;

    std::vector<PyObject *> subs = full_subscript(key, current_shape);
    PYERR_PRINT_GOTO_FAIL;
    for (int i = 0; i < subs.size(); ++i) {
        switch (subscript_type(subs[i])) {
        case (NUMBER):
            ind = align_index(PyLong_AsSsize_t(subs[i]), current_shape[i]);
            push_back(ind, i);
            break;
        case (SLICE): {
            Py_ssize_t start, stop, step, slicelength;
            PySlice_GetIndicesEx(subs[i], current_shape[i],
                                 &start, &stop, &step, &slicelength);
            if (start == stop) {
                push_back((size_t) start, i);
            } else {
                for (int k = 0; k < slicelength; ++k) {
                    push_back(k * step + start, i);
                }
                new_shape.push_back(slicelength);
            }
        }
        break;
        case (SEQUENCE): {
            Py_ssize_t length = PySequence_Length(subs[i]);
            PyObject ** items = PySequence_Fast_ITEMS(subs[i]);
            printf("seq length: %i\n", length);
            for (int k = 0; k < length; ++k) {
                ind = align_index(PyLong_AsSsize_t(items[k]), current_shape[i]);
                PYERR_PRINT_GOTO_FAIL;
                push_back(ind, i);
            }
            new_shape.push_back(length);
        }
        }
    }
    rest = subs.size();
    offset[rest] = vec.size();
    while (rest < MAX_ND) {
        ++rest;
        offset[rest] = offset[rest - 1];
    }
    return new_shape;


fail:
    PyErr_SetString(PyExc_ValueError, "Failed to understand subscript.");
    return new_shape;
}

std::vector<PyObject *> full_subscript(PyObject * tuple, Shape& current_shape) {
    std::vector<PyObject *> elements;
    elements.reserve(current_shape.nd);
    Py_ssize_t tup_length = PySequence_Length(tuple);
    bool found_ellipsis = false;
    int missing_dims;

    if (tup_length > current_shape.nd) {
        VALERR_PRINT_GOTO_FAIL("Subscript has too much elements.");
    } else {

        PyObject * full_slice = PySlice_New(NULL, NULL, NULL);
        // Py_INCREF(full_slice);
        PyObject ** items = PySequence_Fast_ITEMS(tuple);

        for (int i = 0; i < tup_length; ++i) {
            if (items[i] == Py_Ellipsis && !found_ellipsis) {
                for (int k = 0; k < current_shape.nd - (tup_length - 1); ++k)
                    elements.push_back(full_slice);
                found_ellipsis = true;
            } else if (items[i] == Py_Ellipsis && found_ellipsis) {
                VALERR_PRINT_GOTO_FAIL("Ellipsis cannot appear twice in subscript.");
            } else {
                // Py_INCREF(items[i]);
                elements.push_back(items[i]);
            }
        }
        missing_dims = current_shape.nd - elements.size();
        for (int i = 0; i < missing_dims; ++i)
            elements.push_back(full_slice);

        return elements;
    }

fail:
    PyErr_SetString(PyExc_ValueError, "Failed to understand subscript.");
    return elements;
}

void Karray_dealloc(PyKarray *self) {
    // printf("from python with refcount=%i\n", self->ob_base.ob_refcnt);
    self->arr.~Karray();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *
Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    return type->tp_alloc(type, 0);
}

PyKarray *
new_PyKarray() {
    return reinterpret_cast<PyKarray *>(KarrayType.tp_alloc(&KarrayType, 0));
}

PyKarray *
new_PyKarray(Shape &shape) {
    PyKarray * result = reinterpret_cast<PyKarray *>(KarrayType.tp_alloc(&KarrayType, 0));
    result->arr = Karray(shape);
    return result;
}

PyKarray *
new_PyKarray(Shape &shape, float val) {
    PyKarray * result = reinterpret_cast<PyKarray *>(KarrayType.tp_alloc(&KarrayType, 0));
    result->arr = Karray(shape, val);
    return result;
}

PyKarray *
new_PyKarray(const Karray &arr) {
    PyKarray * result = reinterpret_cast<PyKarray *>(KarrayType.tp_alloc(&KarrayType, 0));
    result->arr = Karray(arr);
    return result;
}


int
Karray_init(PyKarray *self, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"data", "shape", NULL};
    PyObject *input = NULL, *shape = NULL;
    Karray candidate;
    Shape proposed_shape;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", kwlist,
                                     &input, &shape))
        return -1;

    if (shape) {
        Py_INCREF(shape);
        proposed_shape = Shape(shape);
        PYERR_PRINT_GOTO_FAIL;
    }

    Py_INCREF(input);
    switch (py_type(input)) {
    case (KARRAY): {
        // printf("initializin from karray\n");
        Py_INCREF(input);
        PyKarray * karr = reinterpret_cast<PyKarray *>(input);
        candidate = karr->arr;
        break;
    }
    case (STRING): {
        auto mode = read_mode(input);
        PYERR_PRINT_GOTO_FAIL;
        candidate.from_mode(proposed_shape, mode);
        break;
    }
    case (NUMPY_ARRAY):
        Py_INCREF(input);
        candidate.from_numpy(input);
        if (PyErr_Occurred()) {
            PyErr_Clear();
        } else {
            break;
        }
    case (NUMBER):
    case (SEQUENCE): {
        NestedSequence<float> nest(input);
        PYERR_PRINT_GOTO_FAIL;
        candidate = nest.to_Karray();
        if (shape) {
            candidate = candidate.broadcast(proposed_shape);
        }
    }
    break;
    default:
        PyErr_SetString(PyExc_TypeError,
                        "Input object not understood.");
    }
    PYERR_PRINT_GOTO_FAIL;

    self->arr.swap(candidate);

    Py_DECREF(input);
    Py_XDECREF(shape);
    return 0;

fail:
    Py_DECREF(input);
    Py_XDECREF(shape);
    PyErr_SetString(PyExc_TypeError,
                    "Failed to initialize kipr.arr.");
    return -1;
}

PyObject *
Karray_str(PyKarray * self) {
    return PyUnicode_FromString(self->arr.str().c_str());
}

PyObject *
Karray_subscript(PyObject *here, PyObject * key) {
    auto self = reinterpret_cast<PyKarray *>(here);

    Py_INCREF(key);
    if (!PyTuple_Check(key))
        key = Py_BuildValue("(O)", key);

    auto result = new_PyKarray();
    result->arr = self->arr.subscript(key);
    PYERR_PRINT_GOTO_FAIL;
    Py_DECREF(key);
    return reinterpret_cast<PyObject *>(result);

fail:
    Py_DECREF(key);
    PyErr_SetString(PyExc_ValueError, "Failed to apply subscript.");
    return NULL;
}

PyObject *
Karray_reshape(PyKarray * self, PyObject * shape) {
    Shape new_shape(shape, self->arr.shape.length);
    PYERR_RETURN_VAL(NULL);
    self->arr.shape = new_shape;

    auto result = reinterpret_cast<PyObject *>(self);
    Py_INCREF(result);
    return result;
}

PyObject *
Karray_getshape(PyKarray *self, void *closure) {
    int nd = self->arr.shape.nd;
    PyObject * result = PyTuple_New(nd);
    for (int k = 0; k < nd; k++) {
        PyTuple_SET_ITEM(result, k, PyLong_FromSize_t(self->arr.shape[k]));
    }
    return result;
}

PyObject *
Karray_getrefcnt(PyKarray *self, void *closure) {
    Py_ssize_t refcnt = self->ob_base.ob_refcnt;
    return PyLong_FromSsize_t(refcnt);
}

PyObject *
Karray_numpy(PyKarray *self, PyObject *Py_UNUSED(ignored)) {
    int nd = self->arr.shape.nd;
    npy_intp * dims = new npy_intp[nd];
    for (int k = 0; k < nd; k++) {
        dims[k] = (npy_intp) self->arr.shape[k];
    }
    float * buffer = new float[self->arr.shape.length];
    std::copy(self->arr.data, self->arr.data + self->arr.shape.length, buffer);
    PyObject * result = PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT, buffer);
    PyArray_UpdateFlags(reinterpret_cast<PyArrayObject *>(result), NPY_ARRAY_OWNDATA);
    return result;
}

PyObject *
Karray_broadcast(PyKarray * self, PyObject * shape) {
    Py_INCREF(reinterpret_cast<PyObject *>(self));
    Shape new_shape(shape);
    PYERR_RETURN_VAL(NULL);
    auto result = new_PyKarray(self->arr.broadcast(new_shape));
    PYERR_RETURN_VAL(NULL);
    return reinterpret_cast<PyObject *>(result);
}


PyObject *
Karray_sum(PyKarray *here, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"axis", "weights", NULL};

    int axis = NO_AXIS;
    PyKarray * weights_obj = NULL;
    PyKarray * result = new_PyKarray();


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i$O!", kwlist,
                                     &axis, &KarrayType, &weights_obj))
        return NULL;

    PyKarray * self = reinterpret_cast<PyKarray *>(here);

    if (axis == NO_AXIS) {
        result->arr = self->arr.flat_sum();
    } else {
        size_t ax = self->arr.shape.axis(axis);
        PYERR_RETURN_VAL(NULL);
        if (weights_obj == NULL) {
            result->arr = self->arr.sum(ax, Karray(Shape(), 1.0));
        } else {
            result->arr = self->arr.sum(ax, weights_obj->arr);
            PYERR_RETURN_VAL(NULL);
        }
    }

    return reinterpret_cast<PyObject *>(result);
}


PyObject *
Karray_mean(PyKarray *here, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"axis", "weights", NULL};

    int axis = NO_AXIS;
    PyKarray * weights_obj = NULL;
    PyKarray * result = new_PyKarray();


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i$O!", kwlist,
                                     &axis, &KarrayType, &weights_obj))
        return NULL;

    PyKarray * self = reinterpret_cast<PyKarray *>(here);

    if (axis == NO_AXIS) {
        result->arr = self->arr.flat_sum(true);
    } else {
        size_t ax = self->arr.shape.axis(axis);
        PYERR_RETURN_VAL(NULL);
        if (weights_obj == NULL) {
            result->arr = self->arr.sum(ax, Karray(Shape(), 1.0), true);
        } else {
            result->arr = self->arr.sum(ax, weights_obj->arr, true);
            PYERR_RETURN_VAL(NULL);
        }
    }

    return reinterpret_cast<PyObject *>(result);
}

PyObject *Karray_transpose(PyObject *here, PyObject *Py_UNUSED(ignored)) {
    PyKarray * self = reinterpret_cast<PyKarray *>(here);
    auto [shape_t, strides_t] = self->arr.shape.transpose();
    PyKarray * result = new_PyKarray(shape_t);
    Positions pos {0, 0, 0};
    simple_transfer(self->arr.data, result->arr.data, &pos, shape_t, strides_t, 0);
    return reinterpret_cast<PyObject *>(result);
}


PyObject *
cache_info(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    int i;
    for (i = 0; i < 32; i++) {

        // Variables to hold the contents of the 4 i386 legacy registers
        int cpu_info[4] = {0};

        __cpuidex(cpu_info, 4, i);


                // See the page 3-191 of the manual.
        int cache_type = cpu_info[0] & 0x1F; 

        if (cache_type == 0) // end of valid cache identifiers
            break;

        char * cache_type_string;
        switch (cache_type) {
            case 1: cache_type_string = "Data Cache"; break;
            case 2: cache_type_string = "Instruction Cache"; break;
            case 3: cache_type_string = "Unified Cache"; break;
            default: cache_type_string = "Unknown Type Cache"; break;
        }

        int cache_level = (cpu_info[0] >>= 5) & 0x7;

        int cache_is_self_initializing = (cpu_info[0] >>= 3) & 0x1; // does not need SW initialization
        int cache_is_fully_associative = (cpu_info[0] >>= 1) & 0x1;

        // See the page 3-192 of the manual.
        // cpu_info[1] contains 3 integers of 10, 10 and 12 bits respectively
        unsigned int cache_sets = cpu_info[2] + 1;
        unsigned int cache_coherency_line_size = (cpu_info[1] & 0xFFF) + 1;
        unsigned int cache_physical_line_partitions = ((cpu_info[1] >>= 12) & 0x3FF) + 1;
        unsigned int cache_ways_of_associativity = ((cpu_info[1] >>= 10) & 0x3FF) + 1;

        // Total cache size is the product
        size_t cache_total_size = cache_ways_of_associativity * cache_physical_line_partitions * cache_coherency_line_size * cache_sets;

        printf(
            "Cache ID %d:\n"
            "- Level: %d\n"
            "- Type: %s\n"
            "- Sets: %d\n"
            "- System Coherency Line Size: %d bytes\n"
            "- Physical Line partitions: %d\n"
            "- Ways of associativity: %d\n"
            "- Total Size: %zu bytes (%zu kb)\n"
            "- Is fully associative: %s\n"
            "- Is Self Initializing: %s\n"
            "\n"
            , i
            , cache_level
            , cache_type_string
            , cache_sets
            , cache_coherency_line_size
            , cache_physical_line_partitions
            , cache_ways_of_associativity
            , cache_total_size, cache_total_size >> 10
            , cache_is_fully_associative ? "true" : "false"
            , cache_is_self_initializing ? "true" : "false"
        );
    }
    Py_RETURN_NONE;
}

void Graph::print(const char * msg) {
	std::cout << "graph: " << msg << "\n";
	for (auto op : ops)
		std::cout << "\t" << op.str() << "\n";
}

std::string Graph::str() const {
	std::ostringstream ss;
	ss << "graph\n";
	for (auto op : ops)
		ss << "\t" << op.str() << "\n";
	return ss.str();
}

void Graph_dealloc(PyGraph *self) {
	// printf("from python with refcount=%i\n", self->ob_base.ob_refcnt);
	self->g.~Graph();
	Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *
Graph_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
	return type->tp_alloc(type, 0);
}



template<class T>
std::vector<T> fast_to_vector(PyObject * o) {
	printf("fast_to_vector is not implemented for this type\n");
}

template<>
std::vector<std::string> fast_to_vector<std::string>(PyObject * o) {
	Py_ssize_t len = PySequence_Fast_GET_SIZE(o);
	std::vector<std::string> result(len);
	PyObject ** items = PySequence_Fast_ITEMS(o);
	for (int i = 0; i < len; ++i)
		result[i] = std::string(PyUnicode_AsUTF8(items[i]));
	return result;
}

const int KIPR_MODULE = -2;
const int RELU_FUNCTION = -12;
const int SOFTMAX_FUNCTION = -13;

int karray_function_code(std::string & name) {
	if (name == "relu")
		return RELU_FUNCTION;
	if (name == "softmax")
		return SOFTMAX_FUNCTION;
	return 0;
}

void print(std::stack<int> s) {
	std::cout << "stack [";
	while (!s.empty()) {
		std::cout << s.top() << ", ";
		s.pop();
	}
	std::cout << "]\n";
}

Op func_to_op(int function) {
	if (function == RELU_FUNCTION)
		return ElementwiseUnaryOp(relu_kernel, "relu");
	if (function == SOFTMAX_FUNCTION)
		return ElementwiseUnaryOp(exp_kernel, "softmax");
	if (function == OP_BINARY_ADD)
		return ElementwiseBinaryOp(add_kernel, "add");
	if (function == OP_BINARY_MATRIX_MULTIPLY)
		return MatMulOp();
	if (function == OP_UNARY_NEGATIVE)
		return ElementwiseUnaryOp(exp_kernel, "negative");

	return Op();
}

int Graph_init(PyGraph *self, PyObject *args, PyObject *kwds) {
	char *kwlist[] = {"function", "trainable", NULL};
	PyObject *function = NULL, *trainable = NULL;

	PyObject * code;
	PyCodeObject * code_ob;
	Graph * g = &self->g;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", kwlist,
	                                 &function, &trainable))
		return -1;

	if (trainable) {
		Py_INCREF(trainable);
		DEBUG_Obj(trainable, "trainable");
	}

	Py_INCREF(function);
	if (!PyFunction_Check(function))
		return -1;


	code = PyFunction_GetCode(function);
	code_ob = reinterpret_cast<PyCodeObject *>(code);
	Py_ssize_t len = PyBytes_Size(code_ob->co_code);
	char* items =  PyBytes_AsString(code_ob->co_code);

	PyObject * globals = PyEval_GetGlobals();
	auto names = fast_to_vector<std::string>(code_ob->co_names);
	auto varnames = fast_to_vector<std::string>(code_ob->co_varnames);
	std::stack<int> stack;
	std::map<int, int> local;
	std::map<int, int> global;

	int op, arg;
	for (int i = 0; i < len; i += 2) {
		op = (int) (unsigned char) items[i];
		arg = (int) (unsigned char) items[i + 1];
		// printf("opcode %i, arg %i\n", op, arg);
		switch (op) {

		case (OP_LOAD_FAST):
			if (!local.contains(arg)) {
				local[arg] = g->ops.size();
				g->ops.push_back(LoadInput(varnames[arg]));
			}
			stack.push(local[arg]);
			break;

		case (OP_LOAD_GLOBAL): {
			PyObject * var = PyDict_GetItemString(globals, names[arg].c_str());
			if (var == 0) {
				PyErr_Format(Karray_error,
				             "Global variable %s is not defined.",
				             names[arg].c_str());
				return -1;
			}
			if (PyModule_Check(var)) {
				std::string module_name(PyModule_GetName(var));
				if (module_name == std::string("kipr")) {
					stack.push(KIPR_MODULE);
				} else {
					PyErr_Format(Karray_error,
					             "Only 'kipr' modle can be used within kipr.graph functions but %s was used.",
					             module_name);
					return -1;
				}
			} else if (py_type(var) == KARRAY) {
				if (!global.contains(arg)) {
					global[arg] = g->ops.size();
					g->ops.push_back(LoadGlobal(names[arg]));
				}
				stack.push(global[arg]);
			} else {
				PyErr_Format(Karray_error, "Unknown global variable %s is used in function.", names[arg]);
				return -1;
			}
		}
		break;

		case (OP_LOAD_METHOD):
			if (stack.top() != KIPR_MODULE) {
				PyErr_SetString(Karray_error, "Expected stack to have KIPR_MODULE on top.");
				return -1;
			}
			stack.pop();
			stack.push(karray_function_code(names[arg]));
			break;

		case (OP_CALL_METHOD): {
			if (arg != 1) {
				PyErr_SetString(Karray_error, "At the moment, only one argument per function call is supported.");
				return -1;
			}
			int argument = stack.top(); stack.pop();
			int function = stack.top(); stack.pop();
			Op oper = func_to_op(function);
			oper.operands.push_back(argument);
			g->ops[argument].add_child(g->ops.size());
			stack.push(g->ops.size());
			g->ops.push_back(oper);
		}
		break;

		case (OP_BINARY_MATRIX_MULTIPLY):
		case (OP_BINARY_ADD): {
			int a = stack.top(); stack.pop();
			int b = stack.top(); stack.pop();
			Op oper = func_to_op(op);
			oper.operands.push_back(a);
			oper.operands.push_back(b);
			g->ops[a].add_child(g->ops.size());
			g->ops[b].add_child(g->ops.size());
			stack.push(g->ops.size());
			g->ops.push_back(oper);
		}
		break;

		case (OP_UNARY_NEGATIVE): {
			int a = stack.top(); stack.pop();
			Op oper = func_to_op(op);
			oper.operands.push_back(a);
			g->ops[a].add_child(g->ops.size());
			stack.push(g->ops.size());
			g->ops.push_back(oper);
		}
		break;

		case (OP_STORE_FAST):
			local[arg] = stack.top(); stack.pop();
			break;

		case (OP_RETURN_VALUE):
			g->ret = stack.top(); stack.pop();
			break;

		default:
			PyErr_Format(Karray_error, "Unknown opcode %i.", op);
			return -1;
		}
		// print(stack);

	}

	Py_DECREF(function);
	Py_XDECREF(trainable);
	return 0;

fail:
	Py_DECREF(function);
	Py_XDECREF(trainable);
	PyErr_SetString(PyExc_TypeError,
	                "Failed to initialize <kipr.graph>.");
	return -1;
}



PyObject *
Graph_str(PyGraph * self) {
    return PyUnicode_FromString(self->g.str().c_str());
}
inline PyObject * py_binary_op(PyObject *here,
                               PyObject *other,
                               binary_kernel kernel,
                               binary_op op) {
	if (py_type(here) != KARRAY || py_type(other) != KARRAY) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	auto self = reinterpret_cast<PyKarray *>(here);
	auto rhs = reinterpret_cast<PyKarray *>(other);
	auto result = new_PyKarray();
	result->arr = self->arr.elementwise_binary_op(rhs->arr, kernel, op);
	PYERR_RETURN_VAL(NULL);
	return reinterpret_cast<PyObject *>(result);
}

inline PyObject * py_inplace_binary_op(PyObject *here,
                                       PyObject *other,
                                       binary_kernel kernel,
                                       binary_op op) {
	if (py_type(here) != KARRAY || py_type(other) != KARRAY) {
		Py_RETURN_NOTIMPLEMENTED;
	}
	auto self = reinterpret_cast<PyKarray *>(here);
	auto rhs = reinterpret_cast<PyKarray *>(other);
	self->arr.inplace_binary_op(rhs->arr, kernel, op);
	Py_INCREF(here);
	return here;
}

PyObject *
Karray_add(PyObject * self, PyObject * other) {
	return py_binary_op(self, other, add_kernel, _add);
}

PyObject *
Karray_sub(PyObject * self, PyObject * other) {
	return py_binary_op(self, other, sub_kernel, _sub);
}

PyObject *
Karray_mul(PyObject * self, PyObject * other) {
	return py_binary_op(self, other, mul_kernel, _mul);
}

PyObject *
Karray_div(PyObject * self, PyObject * other) {
	return py_binary_op(self, other, div_kernel, _div);
}


PyObject *
Karray_inplace_add(PyObject * self, PyObject * other) {
	return py_inplace_binary_op(self, other, add_kernel, _add);
}

PyObject *
Karray_inplace_sub(PyObject * self, PyObject * other) {
	return py_inplace_binary_op(self, other, sub_kernel, _sub);
}

PyObject *
Karray_inplace_mul(PyObject * self, PyObject * other) {
	return py_inplace_binary_op(self, other, mul_kernel, _mul);
}

PyObject *
Karray_inplace_div(PyObject * self, PyObject * other) {
	return py_inplace_binary_op(self, other, div_kernel, _div);
}



PyObject *
Karray_matmul(PyObject * here, PyObject * other) {

	if (py_type(here) != KARRAY || py_type(other) != KARRAY) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	auto self = reinterpret_cast<PyKarray *>(here);
	auto rhs = reinterpret_cast<PyKarray *>(other);

	if (self->arr.shape.nd < 2 && rhs->arr.shape.nd < 2) {
		KERR_RETURN_VAL("Both arrays must be at least two-dimensional for matmul.", NULL);
	}

	size_t M, N, I, J, K;
	I = self->arr.shape[-2];
	K = self->arr.shape[-1];
	J = rhs->arr.shape[-1];

	M = self->arr.shape.nbmats();
	N = rhs->arr.shape.nbmats();

	if (K != rhs->arr.shape[-2] ||
		(M % N != 0 && N % M != 0)) {
		PyErr_Format(Karray_error,
		             "Matmul not possible with shapes %s and %s.",
		             self->arr.shape.str(), rhs->arr.shape.str());
		return NULL;
	}

	Shape new_shape((M > N) ? self->arr.shape : rhs->arr.shape);
	new_shape.set(new_shape.nd - 2, I);
	new_shape.set(new_shape.nd - 1, J);

	auto result = new_PyKarray(new_shape);

	for (int m = 0; m < max(M, N); ++m) {
		int ia = m % M;
		int ib = m % N;

		general_matmul(result->arr.data + m * I * J,
		       self->arr.data + ia * I * K,
		       rhs->arr.data + ib * K * J,
		       I, J, K);
	}

	return reinterpret_cast<PyObject *>(result);
}


inline PyObject *
inplace_val_binary_op(PyObject * o,  float val,
                      binary_val_kernel kernel) {
	if (!(py_type(o) == KARRAY)) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	PyKarray * self = reinterpret_cast<PyKarray *>(o);
	PyKarray * result = new_PyKarray(self->arr.shape);

	kernel(result->arr.data, self->arr.data, val, self->arr.shape.length);

	return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_negative(PyObject * here) {
	return inplace_val_binary_op(here, -1.0, val_mul_kernel);
}

PyObject *
execute_func(PyObject *self, PyObject * input) {
    
    Py_RETURN_NONE;
}

inline PyObject *
overwrite_unary_op(PyObject * o,  void (*op_kernel)(float *, float*, ssize_t)) {
	if (!(py_type(o) == KARRAY)) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	PyKarray * self = reinterpret_cast<PyKarray *>(o);
	PyKarray * result = new_PyKarray(self->arr.shape);

	size_t length = self->arr.shape.length;

	op_kernel(result->arr.data, self->arr.data, length);

	return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_relu(PyObject *module, PyObject * o) {
	return inplace_val_binary_op(o, 0, val_max_kernel);
}

PyObject *Karray_softmax(PyObject *module,
                           PyObject *const *args,
                           Py_ssize_t nargs) {
	if (nargs == 0 || nargs > 2)
		KERR_RETURN_VAL("Wrong number of arguments", NULL);
	if (py_type(args[0]) != KARRAY) {
		Py_RETURN_NOTIMPLEMENTED;
	}
	PyKarray * self = reinterpret_cast<PyKarray *>(args[0]);
	int ax = self->arr.shape.nd - 1;
	if (nargs == 2)
		ax = self->arr.shape.axis(args[1]);

	PyKarray * result = new_PyKarray(self->arr.shape);
	exp_kernel(result->arr.data, self->arr.data, self->arr.shape.length);
	Karray summed_exp = result->arr.sum(ax, Karray(1.), false);
	summed_exp.shape.insert_one(ax);
	summed_exp.broadcast(result->arr.shape);
	result->arr /= summed_exp;

	return reinterpret_cast<PyObject *>(result);
}


PyObject *
Karray_log(PyObject *module, PyObject * o) {
	return overwrite_unary_op(o, log_kernel);
}

PyObject *
Karray_exp(PyObject *module, PyObject * o) {
	return overwrite_unary_op(o, exp_kernel);
}



PyObject * function_decorator(PyObject *self, PyObject *func) {
	PyObject * code;
	PyCodeObject * code_ob;
	PyBytesObject * bytes;
	if (PyFunction_Check(func)) {
		code = PyFunction_GetCode(func);
		code_ob = reinterpret_cast<PyCodeObject *>(code);
		bytes = reinterpret_cast<PyBytesObject *>(code_ob->co_code);


		Py_ssize_t len = PyBytes_Size(code_ob->co_code);
		char* items =  PyBytes_AsString(code_ob->co_code);
		PYERR_RETURN_VAL(NULL);
		// auto maps = op_name();
		// int op;
		// for (int i = 0; i < len; ++i) {
		// 	op = (int) (unsigned char) items[i];
		// 	if (op >= 90) {
		// 		std::cout.width(4);
		// 		std::cout << op << " ";
		// 		std::cout.width(15);
		// 		std::cout << maps[op] << " ";
		// 		std::cout.width(4);
		// 		std::cout << (int) (unsigned char) items[++i] << std::endl;
		// 	} else {
		// 		std::cout.width(4);
		// 		std::cout << op << " ";
		// 		std::cout.width(15);
		// 		std::cout << maps[op] << "\n";
		// 	}
		// }

	}


	Py_RETURN_NONE;
}
#include "test.hpp" 
