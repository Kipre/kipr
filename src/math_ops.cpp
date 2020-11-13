
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

inline PyObject * binary_op(PyObject *self, PyObject *other, void (*op_kernel)(float *, float*, Py_ssize_t)) {
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


PyObject *
Karray_add(PyObject * self, PyObject * other) {
	return binary_op(self, other, add_kernel);
}

// PyObject *
// Karray_add(PyObject * self, PyObject * other) {
// 	if (py_type(self) != KARRAY || py_type(other) != KARRAY) {
// 		Py_RETURN_NOTIMPLEMENTED;
// 	}
// 	PyObject * result;
// 	auto karr = reinterpret_cast<PyKarray *>(self);
// 	auto other_karr = reinterpret_cast<PyKarray *>(other);
// 	auto third_karr = new_PyKarray();
// 	third_karr->arr = karr->arr + other_karr->arr;
// 	PYERR_RETURN_VAL(NULL);
// 	result = reinterpret_cast<PyObject *>(third_karr);
// 	return result;
// }

PyObject *
Karray_sub(PyObject * self, PyObject * other) {
	if (py_type(self) != KARRAY || py_type(other) != KARRAY) {
		Py_RETURN_NOTIMPLEMENTED;
	}
	PyObject * result;
	auto karr = reinterpret_cast<PyKarray *>(self);
	auto other_karr = reinterpret_cast<PyKarray *>(other);
	auto third_karr = new_PyKarray();
	third_karr->arr = karr->arr - other_karr->arr;
	PYERR_RETURN_VAL(NULL);
	result = reinterpret_cast<PyObject *>(third_karr);
	return result;
}

PyObject *
Karray_mul(PyObject * self, PyObject * other) {
	if (py_type(self) != KARRAY || py_type(other) != KARRAY) {
		Py_RETURN_NOTIMPLEMENTED;
	}
	PyObject * result;
	auto karr = reinterpret_cast<PyKarray *>(self);
	auto other_karr = reinterpret_cast<PyKarray *>(other);
	auto third_karr = new_PyKarray();
	third_karr->arr = karr->arr * other_karr->arr;
	PYERR_RETURN_VAL(NULL);
	result = reinterpret_cast<PyObject *>(third_karr);
	return result;
}

PyObject *
Karray_div(PyObject * self, PyObject * other) {
	if (py_type(self) != KARRAY || py_type(other) != KARRAY) {
		Py_RETURN_NOTIMPLEMENTED;
	}
	PyObject * result;
	auto karr = reinterpret_cast<PyKarray *>(self);
	auto other_karr = reinterpret_cast<PyKarray *>(other);
	auto third_karr = new_PyKarray();
	third_karr->arr = karr->arr / other_karr->arr;
	PYERR_RETURN_VAL(NULL);
	result = reinterpret_cast<PyObject *>(third_karr);
	return result;
}


PyObject *
Karray_inplace_add(PyObject * self, PyObject * other) {
	auto karr = reinterpret_cast<PyKarray *>(self);
	auto other_karr = reinterpret_cast<PyKarray *>(other);
	karr->arr += other_karr->arr;
	Py_INCREF(self);
	return self;
}

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
