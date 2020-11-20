
PyObject *
execute_func(PyObject *self, PyObject * input) {
    int i;
    for (i = 0; i < 32; i++) {

        // Variables to hold the contents of the 4 i386 legacy registers
        int cpu_info[4] = {0};

        __cpuidex(cpu_info, 4, i);


                // See the page 3-191 of the manual.
        int cache_type = cpu_info[0] & 0x1F; 

        if (cache_type == 0) // end of valid cache identifiers
            break;

        char * cache_type_string;
        switch (cache_type) {
            case 1: cache_type_string = "Data Cache"; break;
            case 2: cache_type_string = "Instruction Cache"; break;
            case 3: cache_type_string = "Unified Cache"; break;
            default: cache_type_string = "Unknown Type Cache"; break;
        }

        int cache_level = (cpu_info[0] >>= 5) & 0x7;

        int cache_is_self_initializing = (cpu_info[0] >>= 3) & 0x1; // does not need SW initialization
        int cache_is_fully_associative = (cpu_info[0] >>= 1) & 0x1;

        // See the page 3-192 of the manual.
        // cpu_info[1] contains 3 integers of 10, 10 and 12 bits respectively
        unsigned int cache_sets = cpu_info[2] + 1;
        unsigned int cache_coherency_line_size = (cpu_info[1] & 0xFFF) + 1;
        unsigned int cache_physical_line_partitions = ((cpu_info[1] >>= 12) & 0x3FF) + 1;
        unsigned int cache_ways_of_associativity = ((cpu_info[1] >>= 10) & 0x3FF) + 1;

        // Total cache size is the product
        size_t cache_total_size = cache_ways_of_associativity * cache_physical_line_partitions * cache_coherency_line_size * cache_sets;

        printf(
            "Cache ID %d:\n"
            "- Level: %d\n"
            "- Type: %s\n"
            "- Sets: %d\n"
            "- System Coherency Line Size: %d bytes\n"
            "- Physical Line partitions: %d\n"
            "- Ways of associativity: %d\n"
            "- Total Size: %zu bytes (%zu kb)\n"
            "- Is fully associative: %s\n"
            "- Is Self Initializing: %s\n"
            "\n"
            , i
            , cache_level
            , cache_type_string
            , cache_sets
            , cache_coherency_line_size
            , cache_physical_line_partitions
            , cache_ways_of_associativity
            , cache_total_size, cache_total_size >> 10
            , cache_is_fully_associative ? "true" : "false"
            , cache_is_self_initializing ? "true" : "false"
        );
    }
    Py_RETURN_NONE;
}



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
	return inplace_val_binary_op(o, 0, val_max_kernel);
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
