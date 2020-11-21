

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
