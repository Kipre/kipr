
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
