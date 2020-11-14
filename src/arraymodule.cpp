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
    // {"val", (PyCFunction) Karray_val, METH_NOARGS,
    //  "Return the float value of a scalar <kipr.arr>."},    
    {"execute", (PyCFunction)  execute_func, METH_O,
     "Testing function to execute C code."},
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

    // .nb_matrix_multiply = Karray_matmul
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

PyMODINIT_FUNC
PyInit_kipr_array(void)
{
    Karray_error = PyErr_NewException("kipr.KarrayError", NULL, NULL);
    import_array();
    PyObject *m;
    if (PyType_Ready(&KarrayType) < 0)
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

    return m;
}
void transfer(float * from, float * to, size_t * positions, size_t * strides,
              const Filter & filter, int nd, int depth) {
	if (depth < nd) {
		size_t current_value, last_value = 0;
		for (int k = filter.offset[depth]; k < filter.offset[depth + 1]; ++k) {
			current_value = filter.vec[k];
			positions[1] += (current_value - last_value) * strides[depth];
			last_value = current_value;
			transfer(from, to, positions, strides, filter, nd, depth + 1);
		}
		positions[1] -= last_value * strides[depth];
	} else {
		// printf("writing from %i to %i\n", positions[1], positions[0]);
		to[positions[0]++] = from[positions[1]];
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
		return (size_t) (i % dim_length + dim_length) % dim_length;
	}
}


std::map<int, std::string> op_name() {
	std::map<int, std::string> correspondance;
	correspondance[  0 ] = std::string("STOP_CODE");
	correspondance[  1 ] = std::string("POP_TOP");
	correspondance[  2 ] = std::string("ROT_TWO");
	correspondance[  3 ] = std::string("ROT_THREE");
	correspondance[  4 ] = std::string("DUP_TOP");
	correspondance[  5 ] = std::string("ROT_FOUR");
	correspondance[  9 ] = std::string("NOP");
	correspondance[ 10 ] = std::string("UNARY_POSITIVE");
	correspondance[ 11 ] = std::string("UNARY_NEGATIVE");
	correspondance[ 12 ] = std::string("UNARY_NOT");
	correspondance[ 13 ] = std::string("UNARY_CONVERT");
	correspondance[ 15 ] = std::string("UNARY_INVERT");
	correspondance[ 19 ] = std::string("BINARY_POWER");
	correspondance[ 20 ] = std::string("BINARY_MULTIPLY");
	correspondance[ 21 ] = std::string("BINARY_DIVIDE");
	correspondance[ 22 ] = std::string("BINARY_MODULO");
	correspondance[ 23 ] = std::string("BINARY_ADD");
	correspondance[ 24 ] = std::string("BINARY_SUBTRACT");
	correspondance[ 25 ] = std::string("BINARY_SUBSCR");
	correspondance[ 26 ] = std::string("BINARY_FLOOR_DIVIDE");
	correspondance[ 27 ] = std::string("BINARY_TRUE_DIVIDE");
	correspondance[ 28 ] = std::string("INPLACE_FLOOR_DIVIDE");
	correspondance[ 29 ] = std::string("INPLACE_TRUE_DIVIDE");
	correspondance[ 30 ] = std::string("SLICE+0");
	correspondance[ 31 ] = std::string("SLICE+1");
	correspondance[ 32 ] = std::string("SLICE+2");
	correspondance[ 33 ] = std::string("SLICE+3");
	correspondance[ 40 ] = std::string("STORE_SLICE+0");
	correspondance[ 41 ] = std::string("STORE_SLICE+1");
	correspondance[ 42 ] = std::string("STORE_SLICE+2");
	correspondance[ 43 ] = std::string("STORE_SLICE+3");
	correspondance[ 50 ] = std::string("DELETE_SLICE+0");
	correspondance[ 51 ] = std::string("DELETE_SLICE+1");
	correspondance[ 52 ] = std::string("DELETE_SLICE+2");
	correspondance[ 53 ] = std::string("DELETE_SLICE+3");
	correspondance[ 54 ] = std::string("STORE_MAP");
	correspondance[ 55 ] = std::string("INPLACE_ADD");
	correspondance[ 56 ] = std::string("INPLACE_SUBTRACT");
	correspondance[ 57 ] = std::string("INPLACE_MULTIPLY");
	correspondance[ 58 ] = std::string("INPLACE_DIVIDE");
	correspondance[ 59 ] = std::string("INPLACE_MODULO");
	correspondance[ 60 ] = std::string("STORE_SUBSCR");
	correspondance[ 61 ] = std::string("DELETE_SUBSCR");
	correspondance[ 62 ] = std::string("BINARY_LSHIFT");
	correspondance[ 63 ] = std::string("BINARY_RSHIFT");
	correspondance[ 64 ] = std::string("BINARY_AND");
	correspondance[ 65 ] = std::string("BINARY_XOR");
	correspondance[ 66 ] = std::string("BINARY_OR");
	correspondance[ 67 ] = std::string("INPLACE_POWER");
	correspondance[ 68 ] = std::string("GET_ITER");
	correspondance[ 70 ] = std::string("PRINT_EXPR");
	correspondance[ 71 ] = std::string("PRINT_ITEM");
	correspondance[ 72 ] = std::string("PRINT_NEWLINE");
	correspondance[ 73 ] = std::string("PRINT_ITEM_TO");
	correspondance[ 74 ] = std::string("PRINT_NEWLINE_TO");
	correspondance[ 75 ] = std::string("INPLACE_LSHIFT");
	correspondance[ 76 ] = std::string("INPLACE_RSHIFT");
	correspondance[ 77 ] = std::string("INPLACE_AND");
	correspondance[ 78 ] = std::string("INPLACE_XOR");
	correspondance[ 79 ] = std::string("INPLACE_OR");
	correspondance[ 80 ] = std::string("BREAK_LOOP");
	correspondance[ 81 ] = std::string("WITH_CLEANUP");
	correspondance[ 82 ] = std::string("LOAD_LOCALS");
	correspondance[ 83 ] = std::string("RETURN_VALUE");
	correspondance[ 84 ] = std::string("IMPORT_STAR");
	correspondance[ 85 ] = std::string("EXEC_STMT");
	correspondance[ 86 ] = std::string("YIELD_VALUE");
	correspondance[ 87 ] = std::string("POP_BLOCK");
	correspondance[ 88 ] = std::string("END_FINALLY");
	correspondance[ 89 ] = std::string("BUILD_CLASS");
	correspondance[ 90 ] = std::string("STORE_NAME");       // Index in name list
	correspondance[ 91 ] = std::string("DELETE_NAME");      // ""
	correspondance[ 92 ] = std::string("UNPACK_SEQUENCE");   // Number of tuple items
	correspondance[ 93 ] = std::string("FOR_ITER");
	correspondance[ 94 ] = std::string("LIST_APEND");
	correspondance[ 95 ] = std::string("STORE_ATTR");       // Index in name list
	correspondance[ 96 ] = std::string("DELETE_ATTR");      // ""
	correspondance[ 97 ] = std::string("STORE_GLOBAL");     // ""
	correspondance[ 98 ] = std::string("DELETE_GLOBAL");    // ""
	correspondance[ 99 ] = std::string("DUP_TOPX");          // number of items to duplicate
	correspondance[100 ] = std::string("LOAD_CONST");       // Index in const list
	correspondance[101 ] = std::string("LOAD_NAME");       // Index in name list
	correspondance[102 ] = std::string("BUILD_TUPLE");      // Number of tuple items
	correspondance[103 ] = std::string("BUILD_LIST");       // Number of list items
	correspondance[104 ] = std::string("BUILD_SET");        // Number of set items
	correspondance[105 ] = std::string("BUILD_MAP");        // Number of dict entries (upto 255);
	correspondance[106 ] = std::string("LOAD_ATTR");       // Index in name list
	correspondance[107 ] = std::string("COMPARE_OP");       // Comparison operator
	correspondance[108 ] = std::string("IMPORT_NAME");     // Index in name list
	correspondance[109 ] = std::string("IMPORT_FROM");     // Index in name list
	correspondance[110 ] = std::string("JUMP_FORWARD");    // Number of bytes to skip
	correspondance[111 ] = std::string("JUMP_IF_FALSE_OR_POP"); // Target byte offset from beginning of code
	correspondance[112 ] = std::string("JUMP_IF_TRUE_OR_POP");  // ""
	correspondance[113 ] = std::string("JUMP_ABSOLUTE");        // ""
	correspondance[114 ] = std::string("POP_JUMP_IF_FALSE");    // ""
	correspondance[115 ] = std::string("POP_JUMP_IF_TRUE");     // ""
	correspondance[116 ] = std::string("LOAD_GLOBAL");     // Index in name list
	correspondance[119 ] = std::string("CONTINUE_LOOP");   // Target address
	correspondance[120 ] = std::string("SETUP_LOOP");      // Distance to target address
	correspondance[121 ] = std::string("SETUP_EXCEPT");    // ""
	correspondance[122 ] = std::string("SETUP_FINALLY");   // ""
	correspondance[124 ] = std::string("LOAD_FAST");        // Local variable number
	correspondance[125 ] = std::string("STORE_FAST");       // Local variable number
	correspondance[126 ] = std::string("DELETE_FAST");      // Local variable number
	correspondance[130 ] = std::string("RAISE_VARARGS");    // Number of raise arguments (1, or 3);
	correspondance[131 ] = std::string("CALL_FUNCTION");    // //args + (//kwargs << 8);
	correspondance[132 ] = std::string("MAKE_FUNCTION");    // Number of args with default values
	correspondance[133 ] = std::string("BUILD_SLICE");      // Number of items
	correspondance[134 ] = std::string("MAKE_CLOSURE");
	correspondance[135 ] = std::string("LOAD_CLOSURE");
	correspondance[136 ] = std::string("LOAD_DEREF");
	correspondance[137 ] = std::string("STORE_DEREF");
	correspondance[140 ] = std::string("CALL_FUNCTION_VAR");     // //args + (//kwargs << 8);
	correspondance[141 ] = std::string("CALL_FUNCTION_KW");      // //args + (//kwargs << 8);
	correspondance[142 ] = std::string("CALL_FUNCTION_VAR_KW");  // //args + (//kwargs << 8);
	correspondance[143 ] = std::string("SETUP_WITH");
	correspondance[145 ] = std::string("EXTENDED_ARG");
	correspondance[146 ] = std::string("SET_ADD");
	correspondance[147 ] = std::string("MAP_ADD");

	return correspondance;
}

void
add_kernel(float * destination, float * other, ssize_t length) {
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
sub_kernel(float * destination, float * other, ssize_t length) {
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
mul_kernel(float * destination, float * other, ssize_t length) {
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
div_kernel(float * destination, float * other, ssize_t length) {
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
exp_kernel(float * destination, float * other, ssize_t length) {
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


void
log_kernel(float * destination, float * other, ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
        __m256 v_a = _mm256_load_ps(&other[k]);
        v_a = _mm256_log_ps(v_a);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] = log(other[k]);
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        destination[k] = log(other[k]);
    }
#endif
}


void
val_mul_kernel(float * destination, float value, ssize_t length) {
#if __AVX__
    int k;
    __m256 values, constant = _mm256_set_ps(value, value, value, value, value, value, value, value);
    for (k=0; k < length-8; k += 8) {
        values = _mm256_load_ps(&destination[k]);
        values = _mm256_mul_ps(values, constant);
        _mm256_store_ps(&destination[k], values);
    }
    while (k < length) {
        destination[k] *= value;
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        destination[k] *= value;
    }
#endif
}

void
max_val_kernel(float * destination, float value, ssize_t length) {
    #if __AVX__
    int k;
    __m256 values, val = _mm256_set_ps (value, value, value, value, value, value, value, value);
    for (k=0; k < length-8; k += 8) {
        values = _mm256_load_ps(&destination[k]);
        values = _mm256_max_ps(values, val);
        _mm256_store_ps(&destination[k], values);
    }
    while (k < length) {
        destination[k] = Py_MAX(value, destination[k]);
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        destination[k] = Py_MAX(value, destination[k]);
    }
#endif
}

inline Karray elementwise_binary_op(Karray &here, const Karray  &other, void (*op_kernel)(float *, float*, Py_ssize_t)) {
	if (here.shape.length != other.shape.length) {
		Karray self(here);
		Shape common(here.shape, other.shape);
		PYERR_RETURN_VAL(self);
		self.broadcast(common);
		Karray other(other);
		other.broadcast(common);
		op_kernel(self.data, other.data, common.length);
		return self;
	} else {
		Karray self(here);
		op_kernel(self.data, other.data, here.shape.length);
		return self;
	}
}

inline Karray& elementwise_inplace_binary_op(Karray &here, const Karray  &other, void (*op_kernel)(float *, float*, Py_ssize_t)) {
	if (here.shape.length == other.shape.length) {
		op_kernel(here.data, other.data, here.shape.length);
	} else {
		Karray tmp(other);
		tmp.broadcast(here.shape);
		PYERR_RETURN_VAL(here);
		op_kernel(here.data, tmp.data, here.shape.length);
	}
	return here; // return the result by reference
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
	printf("creating generic new karr\n");
	seed = rand();
	shape = Shape();
	data = new float[1];
	data[0] = val;
}


Karray::Karray() {
	printf("creating generic new karr\n");
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
	printf("copying array %i into %i\n", other.seed, seed);
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
	printf("moving array %i into %i\n", other.seed, seed);
	data = other.data;
	other.shape = Shape();
	other.data = new float[1];
	other.data[0] = 0;
}

Karray& Karray::operator=(Karray&& other) {
	printf("moving array %i into %i\n", other.seed, seed);
	shape = other.shape;
	std::cout << "null pointer " << data << std::endl;
	delete[] data;
	data = other.data;
	other.shape = Shape();
	other.data = new float[1];
	other.data[0] = 0;
	return *this;
}

Karray& Karray::operator+=(const Karray& other) {
	return elementwise_inplace_binary_op(*this, other, add_kernel);
}

Karray& Karray::operator/=(const Karray& other) {
	return elementwise_inplace_binary_op(*this, other, div_kernel);
}

Karray& Karray::operator-=(const Karray& other) {
	return elementwise_inplace_binary_op(*this, other, sub_kernel);
}

Karray& Karray::operator*=(const Karray& other) {
	return elementwise_inplace_binary_op(*this, other, mul_kernel);
}


Karray Karray::operator-(const Karray& rhs) {
	return elementwise_binary_op(*this, rhs, sub_kernel);
}

Karray Karray::operator*(const Karray& rhs) {
	return elementwise_binary_op(*this, rhs, mul_kernel);
}

Karray Karray::operator+(const Karray& rhs) {
	return elementwise_binary_op(*this, rhs, add_kernel);
}

Karray Karray::operator/(const Karray& rhs) {
	return elementwise_binary_op(*this, rhs, div_kernel);
}

void Karray::swap(Karray& other) {
	printf("swapping %i and %i\n", seed, other.seed);
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
	size_t positions[2] = {0, 0};

	NDVector strides(shape.strides());
	Shape new_shape(filter.from_subscript(key, shape));
	PYERR_PRINT_GOTO_FAIL;

	strides.print();
	new_shape.print();
	filter.print();

	result.shape = new_shape;
	delete[] result.data;
	result.data = new float[new_shape.length];

	// strides.print();
	transfer(data, result.data, positions,
	         strides.buf, filter, shape.nd, 0);
	// printf("positions[0], positions[1]: %i %i\n", positions[0], positions[1]);
	if (positions[0] != new_shape.length)
		goto fail;

	return result;
fail:
	PyErr_SetString(PyExc_ValueError, "Failed to subscript array.");
	return result;
}

Karray::~Karray() {
	printf("deallocating karr with shape %s and seed %i\n", shape.str().c_str(), seed);
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
	   << ")\n";
	return ss.str();
}


void Karray::broadcast(Shape new_shape) {
	// shortcuts
	if (shape.length == 1) {
		float tmp = data[0];
		delete[] data;
		data = new float[new_shape.length];
		std::fill(data, data + new_shape.length, tmp);
		shape = new_shape;
	} else if (shape.length == new_shape.length) {
		// do nothing
	} else {
		size_t positions[2] = {0, 0};
		auto buffer = new float[new_shape.length];

		NDVector strides(shape.strides(new_shape.nd - shape.nd));
		Filter filter(shape.broadcast_to(new_shape));
		PYERR_PRINT_GOTO_FAIL;
		// filter.print();

		// strides.print();
		transfer(data, buffer, positions,
		         strides.buf,
		         filter, new_shape.nd, 0);


		shape = new_shape;
		delete[] data;
		data = buffer;
		return;

fail:
		delete[] buffer;
		PyErr_SetString(PyExc_ValueError, "Failed to broadcast karray.");
		return;
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

// PyObject *
// Karray_binary_op(PyObject * self, PyObject * other,
//                 ) {
//     Karray *a, *b, *c;
//     Py_ssize_t data_length, *cmn_shape;
//     bool a_owned = false, b_owned = false;


//     if (!is_Karray(self) || !is_Karray(other)) {
//         Py_RETURN_NOTIMPLEMENTED;
//     }

//     a = reinterpret_cast<Karray *>(self);
//     b = reinterpret_cast<Karray *>(other);

//     data_length = Karray_length(a);
//     if (Karray_length(b) != data_length) {
//         cmn_shape = common_shape(a, b);
//         Karray_IF_ERR_GOTO_FAIL;
//         a = broadcast(a, cmn_shape);
//         a_owned = true;
//         Karray_IF_ERR_GOTO_FAIL;
//         b = broadcast(b, cmn_shape);
//         b_owned = true;
//         Karray_IF_ERR_GOTO_FAIL;
//         data_length = Karray_length(a);
//     } else {
//         c = new_Karray_as(a);
//         Karray_copy(a, c);
//         a = c;

//     }

//     op_kernel(a->data, b->data, data_length);

//     // Py_INCREF(a);


//     if (b_owned)
//         Py_DECREF(b);

//     return reinterpret_cast<PyObject *>(a);

//     fail:
//         Py_XDECREF(a);
//         Py_XDECREF(b);
//         PyErr_SetString(PyExc_TypeError,
//             "Failed to apply binary operation.");
//         return NULL;
// }
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


template<typename T>
Shape::Shape(T * input, int size) {
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

Shape::Shape(PyObject * o, size_t target_length) {
	nd = 0;
	length = 1;
	int wildcard = -1;
	Py_ssize_t value;
	if (!PyList_Check(o) && !PyTuple_Check(o))
		KERR_RETURN("Shape must be a list or a tuple.");
	Py_ssize_t seq_length = PySequence_Length(o);
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
	int curr_dim = MAX_ND - 1;
	int dim_diff = a.nd - b.nd;
	while (curr_dim >= 0) {
		printf("curr_dim: %i\n", curr_dim);
		if (curr_dim >= dim_diff &&
		        a[curr_dim] != b[curr_dim - dim_diff]) {
			if (a[curr_dim] == 1) {
				buf[curr_dim] = b[curr_dim - dim_diff];
			} else if (b[curr_dim - dim_diff] == 1) {
				buf[curr_dim] = a[curr_dim];
			} else goto fail;
		} else {
			buf[curr_dim] = a[curr_dim];
		}
		length *= max(1, buf[curr_dim]);
		--curr_dim;
	}
	return;

fail:
	PyErr_Format(PyExc_ValueError,
	             "Shapes %s and %s are not compatible.", a.str(), b.str());
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



std::string Shape::str() {
	std::string result("[");
	result += std::to_string(buf[0]);
	for (int i = 1; i < nd; ++i)
		result += ", " + std::to_string(buf[i]);
	return result + "]";
}

Filter Shape::broadcast_to(Shape & other) {
	Filter filter(other);
	int dim_diff = other.nd - nd;
	// printf("dim_diff: %i\n", dim_diff);
	if (dim_diff < 0) goto fail;
	for (int i = other.nd - 1; i > dim_diff - 1; --i) {
		// printf("braodcast dims: %i %i\n", buf[i - dim_diff], other[i]);
		if (buf[i - dim_diff] == 1 && other[i] != 1) {
			filter.set_val_along_axis(i, 0);
		} else if (buf[i - dim_diff] == other[i]) {
			filter.set_range_along_axis(i);
		} else {
			goto fail;
		}
	}
	for (int i = 0; i < dim_diff; ++i) {
		filter.set_val_along_axis(i, 0);
	}
	return filter;

fail:
	PyErr_Format(PyExc_ValueError,
	             "Cannot broadcast shape %s to %s.", str(), other.str());
	return filter;

}

size_t Shape::sum() {
	size_t result = 0;
	for (int i = 0; i < nd; ++i) {
		result += buf[i];
	}
	return result;
}

size_t Shape::operator[](size_t i) {
	return buf[i];
}

size_t Shape::pop(int i) {
	if (abs(i) >= nd)
		KERR_RETURN_VAL("Shape::pop out of range.", 0);
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

NDVector Shape::strides(int depth_diff) {
	NDVector result;
	size_t acc = 1;
	printf("depth diff %i, nd %i\n", depth_diff, nd);
	result.print();
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
		KERR_RETURN("Cannot insert 1 into shape becaise index is out of bounds.");
	if (i == 0 && nd == 1 && buf[0] == 1)
		return;
	++nd;
	int k = MAX_ND - 1;
	while (k > i) {
		buf[k] = buf[k-1];
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
	if (abs(value) > nd-1)
		KERR_RETURN_VAL("Axis is out of range.", 9);
	return (size_t) (value % nd + nd) % nd;

}

size_t Shape::axis(int ax) {
	if (abs(ax) > nd-1)
		KERR_RETURN_VAL("Axis is out of range.", 9);
	return (size_t) (ax % nd + nd) % nd;

}
Filter::Filter(Shape& shape) {
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

void
Karray_dealloc(PyKarray *self) {
    printf("from python with refcount=%i\n", self->ob_base.ob_refcnt);
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
        proposed_shape = Shape(shape);
        PYERR_PRINT_GOTO_FAIL;
    }

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
            candidate.broadcast(proposed_shape);
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
execute_func(PyObject *self, PyObject * input) {
    DEBUG_Obj(input, "");


    Shape shape(input, (size_t) 120);
    shape.print();

    Py_RETURN_NONE;
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
    return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_reshape(PyKarray * self, PyObject * shape) {
    Py_INCREF(reinterpret_cast<PyObject *>(self));
    Shape new_shape(shape, self->arr.shape.length);
    PYERR_RETURN_VAL(NULL);
    new_shape.print();
    self->arr.shape = new_shape;
    return reinterpret_cast<PyObject *>(self);
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
    new_shape.print();
    self->arr.broadcast(new_shape);
    return reinterpret_cast<PyObject *>(self);
}



PyObject *
Karray_sum(PyKarray * self, PyObject * shape) {
    Py_INCREF(reinterpret_cast<PyObject *>(self));
    Shape new_shape(shape);
    PYERR_RETURN_VAL(NULL);
    new_shape.print();
    self->arr.broadcast(new_shape);
    return reinterpret_cast<PyObject *>(self);
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


inline PyObject * binary_op(PyObject *self, PyObject *other, void (*op_kernel)(float *, float*, ssize_t)) {
	if (py_type(self) != KARRAY || py_type(other) != KARRAY) {
		Py_RETURN_NOTIMPLEMENTED;
	}
	PyObject * result;
	auto karr = reinterpret_cast<PyKarray *>(self);
	auto other_karr = reinterpret_cast<PyKarray *>(other);
	auto third_karr = new_PyKarray();
	third_karr->arr = elementwise_binary_op(karr->arr, other_karr->arr, op_kernel);
	PYERR_RETURN_VAL(NULL);
	result = reinterpret_cast<PyObject *>(third_karr);
	return result;
}

inline PyObject * inplace_binary_op(PyObject *self, PyObject *other, void (*op_kernel)(float *, float*, ssize_t)) {
	if (py_type(self) != KARRAY || py_type(other) != KARRAY) {
		Py_RETURN_NOTIMPLEMENTED;
	}
	auto karr = reinterpret_cast<PyKarray *>(self);
	auto other_karr = reinterpret_cast<PyKarray *>(other);
	elementwise_inplace_binary_op(karr->arr, other_karr->arr, op_kernel);
	Py_INCREF(self);
	return self;
}


PyObject *
Karray_add(PyObject * self, PyObject * other) {
	return binary_op(self, other, add_kernel);
}

PyObject *
Karray_sub(PyObject * self, PyObject * other) {
	return binary_op(self, other, sub_kernel);
}

PyObject *
Karray_mul(PyObject * self, PyObject * other) {
	return binary_op(self, other, mul_kernel);
}

PyObject *
Karray_div(PyObject * self, PyObject * other) {
	return binary_op(self, other, div_kernel);
}


PyObject *
Karray_inplace_add(PyObject * self, PyObject * other) {
	return inplace_binary_op(self, other, add_kernel);
}

PyObject *
Karray_inplace_sub(PyObject * self, PyObject * other) {
	return inplace_binary_op(self, other, sub_kernel);
}

PyObject *
Karray_inplace_mul(PyObject * self, PyObject * other) {
	return inplace_binary_op(self, other, mul_kernel);
}

PyObject *
Karray_inplace_div(PyObject * self, PyObject * other) {
	return inplace_binary_op(self, other, div_kernel);
}

// PyObject *
// Karray_matmul(PyObject * self, PyObject * other) {
//     Karray *a, *b, *c;
//     Py_ssize_t left_dim, mid_dim, right_dim,
//                nb_mat_a, nb_mat_b,
//                pos_a = 0, pos_b = 0, pos_c = 0;
//     Py_ssize_t result_shape[MAX_NDIMS] = {};

//     if (!is_Karray(self) || !is_Karray(other)) {
//         Py_RETURN_NOTIMPLEMENTED;
//     }

//     a = reinterpret_cast<Karray *>(self);
//     b = reinterpret_cast<Karray *>(other);

//     if (a->nd < 2 || b->nd < 2) {
//         PyErr_SetString(PyExc_TypeError,
//             "MatMul works on at least 2-dimensional arrays.");
//         PyErr_Print();
//         goto fail;
//     }

//     if (a->shape[a->nd - 1] != b->shape[b->nd - 2]) {
//         PyErr_SetString(PyExc_TypeError,
//             "Arrays not compatible for MatMul.");
//         PyErr_Print();
//         goto fail;
//     }

//     left_dim = a->shape[a->nd - 2];
//     mid_dim = a->shape[a->nd - 1];
//     right_dim = b->shape[b->nd - 1];

//     nb_mat_a = product(a->shape, a->nd-2);
//     nb_mat_b = product(b->shape, b->nd-2);

//     // printf("nb_mat_a, nb_mat_b: %i %i\n", nb_mat_a, nb_mat_b);

//     if (nb_mat_a == nb_mat_b ||
//         nb_mat_a == 1 ||
//         nb_mat_b == 1) {
//         result_shape[0] = Py_MAX(nb_mat_a, nb_mat_b);
//         result_shape[1] = left_dim;
//         result_shape[2] = right_dim;
//     } else {
//         PyErr_SetString(PyExc_TypeError,
//             "Arrays not compatible for MatMul.");
//         PyErr_Print();
//         goto fail;
//     }
//     c = new_Karray_from_shape(result_shape);
//     for (int m=0; m < result_shape[0]; ++m) {
//         pos_a = (m % nb_mat_a) * left_dim*mid_dim;
//         pos_b = (m % nb_mat_b) * mid_dim*right_dim;
//         for (int i=0; i < left_dim; ++i) {
//             for (int j=0; j < right_dim; ++j) {
//                 c->data[pos_c] = 0;
//                 for (int k=0; k < mid_dim; ++k) {
//                     // printf("indexes: %i %i\n", pos_a + k + mid_dim*i, pos_b + j + k*right_dim);
//                     c->data[pos_c] += a->data[pos_a + k + mid_dim*i] * b->data[pos_b + j + k*right_dim];
//                 }
//                 ++pos_c;
//             }
//         }
//     }
//     // risky
//     if (nb_mat_a >= nb_mat_b) {
//         c->nd = a->nd;
//         set_shape(c, a->shape);
//         c->shape[c->nd-1] = right_dim;
//         c->shape[c->nd-2] = left_dim;
//     } else {
//         c->nd = b->nd;
//         set_shape(c, b->shape);
//         c->shape[c->nd-1] = right_dim;
//         c->shape[c->nd-2] = left_dim;
//     }
//     // Py_INCREF(c);
//     return reinterpret_cast<PyObject *>(c);
//     fail:
//         PyErr_SetString(PyExc_TypeError,
//             "Failed to mat-mutiply arrays.");
//         return NULL;
// }


inline PyObject *
inplace_val_unary_op(PyObject * o,  float val, void (*op_kernel)(float *, float, ssize_t)) {
	if (!(py_type(o) == KARRAY)) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	PyKarray * self = reinterpret_cast<PyKarray *>(o);
	PyKarray * result = new_PyKarray(self->arr);

	size_t length = self->arr.shape.length;

	op_kernel(result->arr.data, val, length);

	return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_negative(PyObject * here) {
	return inplace_val_unary_op(here, -1.0, val_mul_kernel);
}

// PyObject *
// execute_func(PyObject *self, PyObject * input) {
//     DEBUG_Obj(input);

//     auto values = FastSequence<Int>(input, true);
//     if (PyErr_Occurred()) {
//         PyErr_Print();
//         Py_RETURN_NONE;
//     }


//     for(std::vector<Int>::iterator it = values.elements.begin(); it != values.elements.end(); ++it) {
//      	std::cout << (*it).value << " ";
// 	}
// 	std::cout << std::endl;
//     Py_RETURN_NONE;
// }



// PyObject *
// max_nd(PyObject *self, PyObject *Py_UNUSED(ignored)) {
//     return PyLong_FromLong(static_cast<long>(MAX_NDIMS));
// }

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
	return inplace_val_unary_op(o, 0, max_val_kernel);
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

// PyObject *
// Karray_softmax(PyObject *module, PyObject * o) {
// 	if (!(py_type(o) == KARRAY)) {
// 		Py_RETURN_NOTIMPLEMENTED;
// 	}

// 	PyKarray * self = reinterpret_cast<PyKarray *>(o);

// 	Shape new_shape = self->arr.shape
// 	reduction = shape_pop(sum_shape);
// 	nb_sums = product(sum_shape, arr->nd-1);

// 	PyKarray * result = new_PyKarray(self->arr.shape, 0.0);

// 	float * tmp_sums = new float[nb_sums];
// 	std::fill(tmp_sums, tmp_sums+nb_sums, 0);

// 	Py_ssize_t length = Karray_length(arr);

// 	exp_kernel(result->data, arr->data, Karray_length(arr));

// 	for (int i=0; i < nb_sums; ++i) {
// 		for (int k=0; k < reduction; ++k) {
// 			tmp_sums[i] += result->data[k + i*reduction];
// 		}

// 		for (int k=0; k < reduction; ++k) {
// 			result->data[k + i*reduction] /= tmp_sums[i];
// 		}
// 	}

// 	delete[] tmp_sums;

//     return reinterpret_cast<PyObject *>(result);
// }

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
		auto maps = op_name();
		int op;
		for (int i = 0; i < len; ++i) {
			op = (int) (unsigned char) items[i];
			if (op >= 90) {
				std::cout.width(4);
				std::cout << op << " ";
				std::cout.width(15);
				std::cout << maps[op] << " ";
				std::cout.width(4);
				std::cout << (int) (unsigned char) items[++i] << std::endl;
			} else {
				std::cout.width(4);
				std::cout << op << " ";
				std::cout.width(15);
				std::cout << maps[op] << "\n";
			}
		}

	}


	Py_RETURN_NONE;
}
#include "test.hpp" 
