
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
