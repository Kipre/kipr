
PyObject *
execute_func(PyObject *self, PyObject * input) {

	Py_RETURN_NONE;
}

// PyObject *
// langid_tokenize(PyObject *self, PyObject * input) {
// 	int SEQ_LENGTH = 64;
// 	Py_ssize_t size = 0;
// 	PyObject ** strings = PySequence_Fast_ITEMS(input);
// 	Py_ssize_t len  = PySequence_Fast_GET_SIZE(input);

// 	uint16_t * buffer = new uint16_t[SEQ_LENGTH * len];

// 	for (int i = 0; i < len; ++i) {
// 		const char * utf8 = PyUnicode_AsUTF8AndSize(strings[i], &size);
// 		char_vectorizer(utf8, size, SEQ_LENGTH, buffer + i*SEQ_LENGTH, );
// 	}

//     npy_intp * dims = new npy_intp[2];
//     dims[0] = len;
//     dims[1] = SEQ_LENGTH;
//     PyObject * result = PyArray_SimpleNewFromData(2, dims, NPY_UINT16, buffer);
//     PyArray_UpdateFlags(reinterpret_cast<PyArrayObject *>(result), NPY_ARRAY_OWNDATA);
//     return result;
// }

PyObject *
char_tokenize(PyObject *self, PyObject *args, PyObject *keywds) {
	PyObject * sentences;
    int seq_length = 64;
    PyObject * mapping = NULL;
    std::unordered_map<uint32_t, uint16_t> charmap;
    Py_ssize_t len;

    static char *kwlist[] = {"sentences", "mapping", "seq_length", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|i", kwlist,
                                     &sentences, &mapping, &seq_length))
        return NULL;

    if (!PySequence_Check(sentences) || PyUnicode_Check(sentences)) {
    	PyErr_SetString(PyExc_ValueError, "'sentences' should be a sequence of strings");
    	return NULL;
    }

	if (!PyDict_Check(mapping)) {
		PyErr_SetString(PyExc_ValueError, "'mapping' should be a dict with single-char strings mapping to integers");
		return NULL;
	}

	PyObject * map = PyDict_Items(mapping);
	len  = PySequence_Fast_GET_SIZE(map);
	PyObject ** items = PySequence_Fast_ITEMS(map);
	uint32_t key;
	uint16_t value;
	for (int i = 0; i < len; ++i) {
		PyArg_ParseTuple(items[i], "IH", &key, &value);
		charmap[key] = value;
	}
	Py_DECREF(map);
    
	Py_ssize_t size = 0;
	PyObject ** strings = PySequence_Fast_ITEMS(sentences);
	len  = PySequence_Fast_GET_SIZE(sentences);

	uint16_t * buffer = new uint16_t[seq_length * len];

	for (int i = 0; i < len; ++i) {
		const char * utf8 = PyUnicode_AsUTF8AndSize(strings[i], &size);
		char_vectorizer(utf8, size, seq_length, buffer + i*seq_length, charmap);
	}

    npy_intp * dims = new npy_intp[2];
    dims[0] = len;
    dims[1] = seq_length;
    PyObject * result = PyArray_SimpleNewFromData(2, dims, NPY_UINT16, buffer);
    PyArray_UpdateFlags(reinterpret_cast<PyArrayObject *>(result), NPY_ARRAY_OWNDATA);

    Py_DECREF(sentences);
    Py_XDECREF(mapping);
    return result;
}

PyObject *
count_characters(PyObject *self, PyObject * input) {
	Py_ssize_t size = 0;
	const char * utf8 = PyUnicode_AsUTF8AndSize(input, &size);
	std::unordered_map<uint32_t, size_t> countmap = count_relevant_chars(utf8, size);
	PyObject* dict = PyDict_New();

	for (auto& [k, v]: countmap)
		PyDict_SetItem(
			dict,
			PyLong_FromUnsignedLong((unsigned long) k), 
			PyLong_FromSize_t(v)
		);

    Py_INCREF(dict);
    return dict;
}

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
	return inplace_val_binary_op(o, 0, val_max_kernel);
}

PyObject *Karray_softmax(PyObject *module,
                           PyObject *const *args,
                           Py_ssize_t nargs) {
	if (nargs == 0 || nargs > 2) {
		PyErr_Format(Karray_error, "Wrong number of arguments, got %I64i but expecting exactly 1 or 2.", nargs);
		return NULL;
	}
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
	div_kernel(result->arr.data,  result->arr.data, summed_exp.data, result->arr.shape.length);

	return reinterpret_cast<PyObject *>(result);
}


PyObject *
Karray_log(PyObject *module, PyObject * o) {
	return overwrite_unary_op(o, log_kernel);
}

PyObject *
Karray_exp(PyObject *module, PyObject * o) {
	return overwrite_unary_op(o, exp_kernel);
}



PyObject * function_decorator(PyObject *self, PyObject *func) {


	Py_RETURN_NONE;
}
