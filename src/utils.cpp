
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
}

size_t py_type(PyObject * o) {
	if (PyArray_Check(o))
		return NUMPY_ARRAY;
	if (PyUnicode_Check(o))
		return STRING;
	if (PyNumber_Check(o))
		return NUMBER;
	if (PySequence_Check(o))
		return SEQUENCE;
	return 0;
}