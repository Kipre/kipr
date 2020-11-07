
PyObject *
execute_func(PyObject *self, PyObject * input) {
    DEBUG_Obj(input);

    auto values = FastSequence<Int>(input, true);
    if (PyErr_Occurred()) { 
        PyErr_Print(); 
        Py_RETURN_NONE; 
    }


    for(std::vector<Int>::iterator it = values.elements.begin(); it != values.elements.end(); ++it) {
     	std::cout << (*it).value << " ";
	}
	std::cout << std::endl;
    Py_RETURN_NONE;
}



PyObject *
max_nd(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromLong(static_cast<long>(MAX_NDIMS));
}



PyObject *
Karray_relu(PyObject *self, PyObject * o) {

	if (!is_Karray(o)) {
		Py_RETURN_NOTIMPLEMENTED;
	}
	
	Karray * result = new_Karray();
	Karray * arr = reinterpret_cast<Karray *>(o);
	Karray_copy(arr, result);

	Py_ssize_t length = Karray_length(arr);
	max_val_kernel(result->data, 0, Karray_length(result));

    return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_exp(PyObject *self, PyObject * o) {

	if (!is_Karray(o)) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	Karray * arr = reinterpret_cast<Karray *>(o);
	Karray * result = new_Karray_from_shape(arr->shape);

	Py_ssize_t length = Karray_length(arr);

	exp_kernel(result->data, arr->data, Karray_length(arr));

    return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_softmax(PyObject *self, PyObject * o) {

	if (!is_Karray(o)) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	Py_ssize_t reduction, nb_sums, sum_shape[MAX_NDIMS] = {};
	Karray * arr = reinterpret_cast<Karray *>(o);
	Karray * result = new_Karray_from_shape(arr->shape);

	copy_shape(arr->shape, sum_shape);
	reduction = shape_pop(sum_shape);
	nb_sums = product(sum_shape, arr->nd-1);

	float * tmp_sums = new float[nb_sums];
	std::fill(tmp_sums, tmp_sums+nb_sums, 0);

	Py_ssize_t length = Karray_length(arr);

	exp_kernel(result->data, arr->data, Karray_length(arr));

	for (int i=0; i < nb_sums; ++i) {
		for (int k=0; k < reduction; ++k) {
			tmp_sums[i] += result->data[k + i*reduction];
		}

		for (int k=0; k < reduction; ++k) {
			result->data[k + i*reduction] /= tmp_sums[i];
		}
	}

	delete[] tmp_sums;

    return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_log(PyObject *self, PyObject * o) {

	if (!is_Karray(o)) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	Karray * arr = reinterpret_cast<Karray *>(o);
	Karray * result = new_Karray_from_shape(arr->shape);

	Py_ssize_t length = Karray_length(arr);

	log_kernel(result->data, arr->data, Karray_length(arr));

    return reinterpret_cast<PyObject *>(result);
}

