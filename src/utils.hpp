


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
		return ERROR_CODE;
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
	return ERROR_CODE;
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
	return ERROR_CODE;
}

size_t subscript_type(PyObject * o) {
	if (PyNumber_Check(o))
		return NUMBER;
	if (PySlice_Check(o))
		return SLICE;
	if (PySequence_Check(o))
		return SEQUENCE;
	return ERROR_CODE;
}

size_t align_index(Py_ssize_t i, size_t dim_length) {
	if (abs(i) >= (int) dim_length) {
		PyErr_Format(PyExc_ValueError,
		             "Index %i out of range on axis with length %i.",
		             i, dim_length);
		return 0;
	} else {
		size_t result = (i + dim_length) % dim_length;
		return result;
	}
}

