
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

	Karray * arr = reinterpret_cast<Karray *>(o);
	Karray * result = new_Karray_from_shape(arr->shape);

	Py_ssize_t length = Karray_length(arr);
#if __AVX__
    int k;
    __m256 values, zero = _mm256_set_ps (0., 0., 0., 0., 0., 0., 0., 0.);
    for (k=0; k < length-8; k += 8) {
        values = _mm256_load_ps(&arr->data[k]);
        values = _mm256_max_ps(values, zero);
        _mm256_store_ps(&result->data[k], values);
    }
    while (k < length) {
        result->data[k] = Py_MAX(0, arr->data[k]);
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        result->data[k] = Py_MAX(0, arr->data[k]);
    }
#endif

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

