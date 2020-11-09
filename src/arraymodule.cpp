#include "arraymodule.hpp" 
static PyMemberDef Karray_members[] = {
    // {"attr", T_INT, offsetof(PyKarray, attr), 0,
    //  "Arbitrary attribute."},
    {NULL}  /* Sentinel */
};

static PyGetSetDef Karray_getsetters[] = {
    // {"shape", (getter) Karray_getshape, NULL,
    //  "Shape of the array.", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef Karray_methods[] = {
    // {"reshape", (PyCFunction) Karray_reshape, METH_O,
    //  "Return the kipr.arr with the new shape."},
    // {"broadcast", (PyCFunction) Karray_broadcast, METH_O,
    //  "Return the kipr.arr with the breadcasted shape."},
    // {"mean", (PyCFunction) Karray_mean, METH_VARARGS | METH_KEYWORDS,
    //  "Return the averaged array."},
    // {"sum", (PyCFunction) Karray_sum, METH_VARARGS | METH_KEYWORDS,
    //  "Return the sum of the array along all or a particular dim."},
    // {"numpy", (PyCFunction) Karray_numpy, METH_NOARGS,
    //  "Return a numpy representtion of the PyKarray."},
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
    // {"internal", internal_test, METH_NOARGS,
    //  "Execute C code tests."},
    // {"relu", Karray_relu, METH_O,
    //  "ReLU function for <kipr.arr> arrays."},
    // {"exp", Karray_exp, METH_O,
    //  "Exponential function for <kipr.arr> arrays."},
    // {"softmax", Karray_softmax, METH_O,
    //  "Softmax function for <kipr.arr> arrays, computes along the last axis."},
    // {"ln", Karray_log, METH_O,
    //  "Log function for <kipr.arr> arrays."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyModuleDef arraymodule = {
    PyModuleDef_HEAD_INIT,
    "kipr_array",
    "Array backend.",
    -1,
    arraymodule_methods
};



// static PyNumberMethods Karray_as_number = {
//     .nb_add = Karray_add,
//     .nb_subtract = Karray_sub,
//     .nb_multiply = Karray_mul,

//     .nb_negative = Karray_negative,

//     .nb_inplace_add = Karray_inplace_add,
//     .nb_inplace_subtract = Karray_inplace_sub,
//     .nb_inplace_multiply = Karray_inplace_mul,

//     .nb_true_divide = Karray_div,
//     .nb_inplace_true_divide = Karray_inplace_div,

//     .nb_matrix_multiply = Karray_matmul
// };

// static PyMappingMethods Karray_as_mapping = {
//     .mp_subscript = Karray_subscript
// };

static PyTypeObject KarrayType = {
    Karray_HEAD_INIT
    .tp_name = KARRAY_NAME,
    .tp_basicsize = sizeof(PyKarray) - sizeof(float),
    .tp_itemsize = sizeof(float),
    .tp_dealloc = (destructor) Karray_dealloc,
    .tp_repr = (reprfunc) Karray_str, 
    // .tp_as_number = &Karray_as_number,
    // .tp_as_mapping = &Karray_as_mapping,
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
Karray::Karray() {
	owned = false;
	seed = rand();
	shape = Shape();
	data = new float[1];
	data[0] = 0;
}

Karray::Karray(Shape new_shape, std::vector<float> vec) {
	owned = false;
	seed = rand();
	shape = new_shape;
	// printf("shape.length, vec.size(): %i %i\n", shape.length, vec.size());
	data = new float[shape.length];
	std::copy(vec.begin(), vec.end(), data);
}

Karray::Karray(Shape new_shape, float * new_data) {
	owned = false;
	seed = rand();
	shape = new_shape;
	data = new_data;
}

void Karray::steal(Karray& other) {
	other.owned = true;
	seed = other.seed;
	shape = other.shape;
	data = other.data;
}

Karray::~Karray() {
	if (!owned) {
		delete[] data;
	}
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
		if (match > 0) {
			if (i == 0) {
				ss << std::string(match + 1, '[');
			} else {
				ss << std::string(match, ']') << ",\n";
				ss << std::string(shape.nd - match + 9, ' ');
				ss << std::string(match, '[');
			}
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

void transfer(float * from, float * to, size_t * positions, size_t * strides,
              const Filter & filter, Shape & to_shape, int depth) {
	if (depth < to_shape.nd) {
		size_t current_value, last_value = 0;
		for (int k = 0; k < to_shape[depth]; ++k) {
			current_value = filter.buf[filter.offset[depth] + k];
			positions[1] += (current_value - last_value) * strides[depth];
			last_value = current_value;
			transfer(from, to, positions, strides, filter, to_shape, depth + 1);
		}
		positions[1] -= last_value * strides[depth];
	} else {
		// printf("writing from %i to %i\n", positions[1], positions[0]);
		to[positions[0]++] = from[positions[1]];
	}
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
		NDVector strides(shape.strides(new_shape.nd - shape.nd));
		Filter filter(shape.broadcast_to(new_shape));
		PYERR_PRINT_GOTO_FAIL;
		// filter.print();

		auto buffer = new float[new_shape.length];
		size_t positions[2] = {0, 0};

		// strides.print();



		transfer(data, buffer, positions,
		         strides.buf,
		         filter, new_shape, 0);


		shape = new_shape;
		delete[] data;
		data = buffer;
		return;

fail:
		PyErr_SetString(PyExc_ValueError, "Failed to broadcast karray.");
		return;
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

Shape::Shape(PyObject * o, bool accept_singleton) {
	nd = 0;
	Py_ssize_t value;
	if (accept_singleton) {
		value = PyLong_AsSsize_t(o);
		if (!PyErr_Occurred() && value > 0) {
			buf[nd] = (size_t) value;
			length = nd;
			return;
		} else {
			PyErr_Clear();
		}
	}
	if (!PyList_Check(o) && !PyTuple_Check(o))
		goto fail;

	Py_ssize_t length = PySequence_Length(o);
	PyObject ** items = PySequence_Fast_ITEMS(o);
	for (int i = 0; i < length; ++i) {
		value = PyLong_AsSsize_t(items[i]);
		PYERR_CLEAR_GOTO_FAIL;
		buf[nd] = value;
		++nd;
	}
	while(nd != MAX_ND) {
		buf[nd] = 0;
		++nd;
	}
	cohere();
	PYERR_PRINT_GOTO_FAIL;
	return;

fail:
	PyErr_Format(PyExc_TypeError,
	             "Failed to parse the shape.");
	return;
}

Shape::Shape() {
	nd = 1;
	length = 1;
	buf[0] = 1;
	std::fill(buf + 1, buf + MAX_ND, 0);
}



void Shape::print(const char * message) {
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

size_t Shape::cohere() {
	int new_nd = 0;
	length = 1;
	if (buf[0] == 0) goto fail;
	while (buf[new_nd] != 0 && new_nd < MAX_ND) {
		length *= buf[new_nd];
		++new_nd;
	}
	int i = new_nd;
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

Filter Shape::broadcast_to(Shape other) {
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

NDVector Shape::strides(int depth_diff) {
	NDVector result;
	size_t acc = 1;
	for (int i = nd - 1; i >= 0; --i) {
		result.buf[i + depth_diff] = acc;
		acc *= buf[i];
	}
	for (int i = 0; i < depth_diff; ++i) {
		result.buf[i] = 0;
	}
	return result;
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
	buf = new size_t[total];
	// std::fill(buf, buf + total, -1);
}

Filter::Filter(Filter&& other) noexcept : buf(nullptr), offset{0} {
	buf = other.buf;
	other.buf = nullptr;
	for (int i = 0; i < MAX_ND; ++i) {
		offset[i] = other.offset[i];
		other.offset[i] = 0;
	}
}

void Filter::set_val_along_axis(int axis , size_t value) {
	// printf("writing val from %i to %i on axis %i\n", offset[axis], offset[axis + 1], axis);
	std::fill(buf + offset[axis], buf + offset[axis + 1], value);
}

void Filter::set_range_along_axis(int axis) {
	// printf("writing range from %i to %i on axis %i\n", offset[axis], offset[axis + 1], axis);
	for (int k = offset[axis]; k < offset[axis + 1]; ++k) {
		buf[k] = k - offset[axis];
	}
}

void Filter::print(const char * message) {
	std::cout << "Filter " << message << "\n\t";
	int o = 1;
	for (int k = 0; k < offset[MAX_ND - 1]; ++k) {
		if (k == offset[o]) {
			std::cout << "| ";
			++o;
		}
		std::cout << buf[k] << ", ";
	}
	std::cout << "\n\toffsets:";
	for (int k = 0; k < MAX_ND; ++k) {
		std::cout << offset[k] << ", ";
	}
	std::cout << '\n';
}

Filter::~Filter() {
	// printf("freeing filter\n");
	delete[] buf;
}

Filter& Filter::operator=(Filter&& other) noexcept {
	if (this != &other) {
		delete[] buf;

		buf = other.buf;
		other.buf = nullptr;
		for (int i = 0; i < MAX_ND; ++i) {
			offset[i] = other.offset[i];
			other.offset[i] = 0;
		}
	}
	return *this;
}

// Filter& Filter::operator=(Filter && other) {
// 	for (int i = 0; i < MAX_ND; ++i) {
// 		offset[i] = other.offset[i];
// 		other.offset[i] = 0;
// 	}
// 	buf = other.buf;
// 	other.buf = nullptr;
// }

// bool Filter::test() {
// 	TEST(FilterInitialization) {
// 		size_t a[MAX_ND] = {3, 4, 5, 0, 0, 0, 0, 0};
// 		Shape shape(a);
// 		Filter f(shape);
// 	}
// }
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


void
log_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
        __m256 v_a = _mm256_load_ps(&other[k]);
        v_a = _mm256_log_ps(v_a);
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
val_mul_kernel(float * destination, float value, Py_ssize_t length) {
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
max_val_kernel(float * destination, float value, Py_ssize_t length) {
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


void
Karray_dealloc(PyKarray *self) {
    DEBUG_Obj((PyObject *) self, "deallocating from python")
    // self->arr.print("deallocating from python");
    self->arr.~Karray();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *
Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyKarray *self;
    self = reinterpret_cast<PyKarray *>(type->tp_alloc(type, 0));
    // if (self != NULL) {
    //     self->arr = Karray();
    // }
    return reinterpret_cast<PyObject *>(self);
}

int
Karray_init(PyKarray *self, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"data", "shape", NULL};
    PyObject *input = NULL, *shape = NULL;
    Karray candidate;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", kwlist,
                                     &input, &shape))
        return -1;

    auto nest = NestedSequence<float>(input);
    PYERR_PRINT_GOTO_FAIL;

    candidate.steal(nest.to_Karray());

    if (shape) {
        Shape proposed_shape(shape);
        PYERR_PRINT_GOTO_FAIL;
        proposed_shape.print();
        candidate.broadcast(proposed_shape);
        PYERR_PRINT_GOTO_FAIL;
    }

    self->arr.steal(candidate);

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


    PyObject * out = Karray_new(&KarrayType, NULL, NULL);

    return out;
}

PyObject *
Karray_str(PyKarray * self) {
    return PyUnicode_FromString(self->arr.str().c_str());
}

// PyObject *
// Karray_binary_op(PyObject * self, PyObject * other, 
//                 void (*op_kernel)(float *, float*, Py_ssize_t)) {
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


// PyObject *
// Karray_inplace_binary_op(PyObject * self, PyObject * other, 
//                          void (*op_kernel)(float *, float*, Py_ssize_t)) {
//     Karray *a, *b;
//     Py_ssize_t data_length;

//     if (!is_Karray(self) || !is_Karray(other)) {
//         Py_RETURN_NOTIMPLEMENTED;
//     }

//     a = reinterpret_cast<Karray *>(self);
//     b = reinterpret_cast<Karray *>(other);
    
//     data_length = Karray_length(a);
//     if (Karray_length(b) != data_length) {
//         b = broadcast(b, a->shape);
//         Karray_IF_ERR_GOTO_FAIL;
//     }

//     op_kernel(a->data, b->data, data_length);

//     Py_INCREF(self);
//     return self;

//     fail:
//         Py_XDECREF(a);
//         Py_XDECREF(b);
//         PyErr_SetString(PyExc_TypeError, 
//             "Failed to apply binary operation.");
//         return NULL;
// }



// PyObject *
// Karray_add(PyObject * self, PyObject * other) {
//     return Karray_binary_op(self, other, add_kernel);
// }

// PyObject *
// Karray_inplace_add(PyObject * self, PyObject * other) {
//     return Karray_inplace_binary_op(self, other, add_kernel);
// }

// PyObject *
// Karray_sub(PyObject * self, PyObject * other) {
//     return Karray_binary_op(self, other, sub_kernel);
// }

// PyObject *
// Karray_inplace_sub(PyObject * self, PyObject * other) {
//     return Karray_inplace_binary_op(self, other, sub_kernel);
// }

// PyObject *
// Karray_mul(PyObject * self, PyObject * other) {
//     return Karray_binary_op(self, other, mul_kernel);
// }

// PyObject *
// Karray_inplace_mul(PyObject * self, PyObject * other) {
//     return Karray_inplace_binary_op(self, other, mul_kernel);
// }

// PyObject *
// Karray_div(PyObject * self, PyObject * other) {
//     return Karray_binary_op(self, other, div_kernel);
// }

// PyObject *
// Karray_inplace_div(PyObject * self, PyObject * other) {
//     return Karray_inplace_binary_op(self, other, div_kernel);
// }


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

// PyObject *
// Karray_negative(PyObject * self) {
//     Karray * result = new_Karray();
//     Karray_copy(reinterpret_cast<Karray *>(self), result);

//     val_mul_kernel(result->data, -1, Karray_length(result));

//     return reinterpret_cast<PyObject *>(result);
// }

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



// PyObject *
// Karray_relu(PyObject *self, PyObject * o) {

// 	if (!is_Karray(o)) {
// 		Py_RETURN_NOTIMPLEMENTED;
// 	}
	
// 	Karray * result = new_Karray();
// 	Karray * arr = reinterpret_cast<Karray *>(o);
// 	Karray_copy(arr, result);

// 	Py_ssize_t length = Karray_length(arr);
// 	max_val_kernel(result->data, 0, Karray_length(result));

//     return reinterpret_cast<PyObject *>(result);
// }

// PyObject *
// Karray_exp(PyObject *self, PyObject * o) {

// 	if (!is_Karray(o)) {
// 		Py_RETURN_NOTIMPLEMENTED;
// 	}

// 	Karray * arr = reinterpret_cast<Karray *>(o);
// 	Karray * result = new_Karray_from_shape(arr->shape);

// 	Py_ssize_t length = Karray_length(arr);

// 	exp_kernel(result->data, arr->data, Karray_length(arr));

//     return reinterpret_cast<PyObject *>(result);
// }

// PyObject *
// Karray_softmax(PyObject *self, PyObject * o) {

// 	if (!is_Karray(o)) {
// 		Py_RETURN_NOTIMPLEMENTED;
// 	}

// 	Py_ssize_t reduction, nb_sums, sum_shape[MAX_NDIMS] = {};
// 	Karray * arr = reinterpret_cast<Karray *>(o);
// 	Karray * result = new_Karray_from_shape(arr->shape);

// 	copy_shape(arr->shape, sum_shape);
// 	reduction = shape_pop(sum_shape);
// 	nb_sums = product(sum_shape, arr->nd-1);

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

// PyObject *
// Karray_log(PyObject *self, PyObject * o) {

// 	if (!is_Karray(o)) {
// 		Py_RETURN_NOTIMPLEMENTED;
// 	}

// 	Karray * arr = reinterpret_cast<Karray *>(o);
// 	Karray * result = new_Karray_from_shape(arr->shape);

// 	Py_ssize_t length = Karray_length(arr);

// 	log_kernel(result->data, arr->data, Karray_length(arr));

//     return reinterpret_cast<PyObject *>(result);
// }

#include "test.hpp" 
