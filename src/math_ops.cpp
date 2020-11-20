
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
	I = self->arr.shape[0];
	K = self->arr.shape[1];
	J = rhs->arr.shape[1];

	if (K != rhs->arr.shape[0]) {
		PyErr_Format(Karray_error,
		             "Matmul not possible with shapes %s and %s.",
		             self->arr.shape.str(), rhs->arr.shape.str());
		return NULL;
	}

	// M = self->arr.shape.nbmats();
	// N = rhs->arr.shape.nbmats();

	// if (I < 4 || J < 8) {
	// 	if (I < 4 && J < 8) {
	// 		for
	// 	} else if (I == 1) {

	// 	} else if (J == 1) {

	// 	} else {}
	// }

	// if ()

	Shape new_shape;
	new_shape.set(0, I);
	new_shape.set(1, J);

	auto result = new_PyKarray(new_shape);

	matmul(result->arr.data, self->arr.data, rhs->arr.data, I, J, K);

	return reinterpret_cast<PyObject *>(result);
}


PyObject *
Karray_matmul_loop(PyObject * here, PyObject * other) {
	if (py_type(here) != KARRAY || py_type(other) != KARRAY) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	auto self = reinterpret_cast<PyKarray *>(here);
	auto rhs = reinterpret_cast<PyKarray *>(other);
	Karray mult = loop_matmul(self->arr, rhs->arr);
	PYERR_RETURN_VAL(NULL);
	auto result = new_PyKarray();
	result->arr.swap(mult);
	return reinterpret_cast<PyObject *>(result);
}


PyObject *
Karray_matmul_loop_transpose(PyObject * here, PyObject * other) {
	if (py_type(here) != KARRAY || py_type(other) != KARRAY) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	auto self = reinterpret_cast<PyKarray *>(here);
	auto rhs = reinterpret_cast<PyKarray *>(other);
	Karray mult = loop_transpose_matmul(self->arr, rhs->arr);
	PYERR_RETURN_VAL(NULL);
	auto result = new_PyKarray();
	result->arr.swap(mult);
	PyObject * final = reinterpret_cast<PyObject *>(result);
	return final;
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
