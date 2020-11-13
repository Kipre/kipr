
Karray inline elementwise_binary_op(Karray &here, const Karray  &other, void (*op_kernel)(float *, float*, Py_ssize_t)) {
	if (here.shape.length != other.shape.length) {
		Karray self(here);
		Shape common(here.shape, other.shape);
		PYERR_RETURN_VAL(self);
		self.broadcast(common);
		Karray other(other);
		other.broadcast(common);
		op_kernel(self.data, other.data, common.length);
		return self;
	} else {
		Karray self(here);
		op_kernel(self.data, other.data, here.shape.length);
		return self;
	}
}

Karray::Karray(Shape new_shape) {
	seed = rand();
	shape = new_shape;
	data = new float[shape.length];
}


Karray::Karray(Shape new_shape, float value) {
	seed = rand();
	shape = new_shape;
	data = new float[shape.length];
	std::fill(data, data + shape.length, value);
}


Karray::Karray() {
	printf("creating generic new karr\n");
	seed = rand();
	shape = Shape();
	data = new float[1];
	data[0] = 0;
}

Karray::Karray(const Karray& other)
	: seed{other.seed + 1},
	  shape{other.shape} {
	data = new float[shape.length];
	std::copy(other.data, other.data + shape.length, data);
}

Karray& Karray::operator=(const Karray& other) {
	printf("copying array %i into %i\n", other.seed, seed);
	shape = other.shape;
	delete[] data;
	data = new float[shape.length];
	std::copy(other.data, other.data + shape.length, data);
	return *this;
}

Karray::Karray(Karray&& other)
	: seed{other.seed + 1},
	  shape{other.shape} {
	seed = other.seed + 1;
	printf("moving array %i into %i\n", other.seed, seed);
	data = other.data;
	other.shape = Shape();
	other.data = new float[1];
	other.data[0] = 0;
}

Karray& Karray::operator=(Karray&& other) {
	printf("moving array %i into %i\n", other.seed, seed);
	shape = other.shape;
	std::cout << "null pointer " << data << std::endl;
	delete[] data;
	data = other.data;
	other.shape = Shape();
	other.data = new float[1];
	other.data[0] = 0;
	return *this;
}

Karray& Karray::operator+=(const Karray& other) {
	if (shape.length == other.shape.length) {
		shape.print();
		add_kernel(data, other.data, shape.length);
	} else {
		Karray tmp(other);
		tmp.broadcast(shape);
		PYERR_PRINT_GOTO_FAIL;
		add_kernel(data, tmp.data, shape.length);
	}
	return *this; // return the result by reference
fail:
	PyErr_SetString(PyExc_ValueError,
	                "Failed to execute addition.");
	return *this;
}

Karray& Karray::operator/=(const Karray& other) {
	if (shape.length == other.shape.length) {
		shape.print();
		add_kernel(data, other.data, shape.length);
	} else {
		Karray tmp(other);
		tmp.broadcast(shape);
		PYERR_PRINT_GOTO_FAIL;
		div_kernel(data, tmp.data, shape.length);
	}
	return *this; // return the result by reference
fail:
	PyErr_SetString(PyExc_ValueError,
	                "Failed to execute addition.");
	return *this;
}

Karray Karray::operator-(const Karray& rhs) {
	return elementwise_binary_op(*this, rhs, sub_kernel);
}

Karray Karray::operator*(const Karray& rhs) {
	return elementwise_binary_op(*this, rhs, mul_kernel);
}

Karray Karray::operator+(const Karray& rhs) {
	return elementwise_binary_op(*this, rhs, add_kernel);
}

Karray Karray::operator/(const Karray& rhs) {
	return elementwise_binary_op(*this, rhs, div_kernel);
}

void Karray::swap(Karray& other) {
	printf("swapping %i and %i\n", seed, other.seed);
	std::swap(shape, other.shape);
	std::swap(data, other.data);
}

Karray::Karray(Shape new_shape, std::vector<float> vec) {
	seed = rand();
	shape = new_shape;
	// printf("shape.length, vec.size(): %i %i\n", shape.length, vec.size());
	data = new float[shape.length];
	std::copy(vec.begin(), vec.end(), data);
}

Karray::Karray(Shape new_shape, float * new_data) {
	seed = rand();
	shape = new_shape;
	data = new_data;
}

void Karray::from_mode(Shape new_shape, size_t mode) noexcept {
	delete[] data;
	shape = new_shape;
	data = new float[shape.length];

	if (mode == RANDOM_NORMAL || mode == RANDOM_UNIFORM) {
		std::random_device rd{};
		std::mt19937 gen{rd()};

		if (mode == RANDOM_NORMAL) {
			std::normal_distribution<float> d{0, 1};
			for (int n = 0; n < shape.length; ++n) {
				data[n] = d(gen);
			}
		} else if (mode == RANDOM_UNIFORM) {
			for (int n = 0; n < shape.length; ++n) {
				data[n] = gen() / (float) 4294967295;
			}
		}
	} else if (mode == RANGE) {
		for (int n = 0; n < shape.length; ++n) {
			data[n] = (float) n;
		}
	} else {
		throw std::exception("unknown mode");
	}
}

Karray Karray::subscript(PyObject * key) {

	Karray result;
	Filter filter;
	size_t positions[2] = {0, 0};

	NDVector strides(shape.strides());
	Shape new_shape(filter.from_subscript(key, shape));
	PYERR_PRINT_GOTO_FAIL;

	strides.print();
	new_shape.print();
	filter.print();

	result.shape = new_shape;
	delete[] result.data;
	result.data = new float[new_shape.length];

	// strides.print();
	transfer(data, result.data, positions,
	         strides.buf, filter, shape.nd, 0);
	// printf("positions[0], positions[1]: %i %i\n", positions[0], positions[1]);
	if (positions[0] != new_shape.length)
		goto fail;

	return result;
fail:
	PyErr_SetString(PyExc_ValueError, "Failed to subscript array.");
	return result;
}

Karray::~Karray() {
	printf("deallocating karr with shape %s and seed %i\n", shape.str().c_str(), seed);
	delete[] data;
	shape.~Shape();
}

void Karray::print(const char * message) {
	std::cout << "Printing Karray " << seed << " " << message << "\n\t";
	shape.print();
	std::cout << "\t";
	auto limit = min(shape.length, MAX_PRINT_SIZE);
	for (int i = 0; i < limit; ++i) {
		std::cout << data[i] << ", ";
	}
	std::cout << "\n";
}

std::string Karray::str() {
	NDVector strides = shape.strides();
	auto matches = [&](int i) {
		int result = 0;
		for (int k = 0; k < shape.nd - 1; ++k)
			if (i % strides[k] == 0)
				++result;
		return result;
	};
	std::ostringstream ss;
	ss << "kipr.arr(";
	auto limit = min(shape.length, MAX_PRINT_SIZE);
	int match = matches(0);
	for (int i = 0; i < limit; ++i) {
		if (i == 0)
			ss << std::string(match + 1, '[');
		if (match > 0 && i > 0) {
			ss << std::string(match, ']') << ",\n";
			ss << std::string(shape.nd - match + 9, ' ');
			ss << std::string(match, '[');
		}
		ss.width(5);
		ss << data[i];
		match = matches(i + 1);
		if (!match)
			ss << ", ";
		if (i == shape.length - 1) {
			ss << std::string(match + 1, ']') << ", ";
		} else if (i == limit - 1) {
			ss << " ... , ";
		}
	}
	ss << "shape=" << shape.str()
	   << ")\n";
	return ss.str();
}


void Karray::broadcast(Shape new_shape) {
	// shortcuts
	if (shape.length == 1) {
		float tmp = data[0];
		delete[] data;
		data = new float[new_shape.length];
		std::fill(data, data + new_shape.length, tmp);
		shape = new_shape;
	} else if (shape.length == new_shape.length) {
		// do nothing
	} else {
		size_t positions[2] = {0, 0};
		auto buffer = new float[new_shape.length];

		NDVector strides(shape.strides(new_shape.nd - shape.nd));
		Filter filter(shape.broadcast_to(new_shape));
		PYERR_PRINT_GOTO_FAIL;
		// filter.print();

		// strides.print();
		transfer(data, buffer, positions,
		         strides.buf,
		         filter, new_shape.nd, 0);


		shape = new_shape;
		delete[] data;
		data = buffer;
		return;

fail:
		delete[] buffer;
		PyErr_SetString(PyExc_ValueError, "Failed to broadcast karray.");
		return;
	}

}

void Karray::from_numpy(PyObject * obj) noexcept {
	auto arr = (PyArrayObject *) obj;
	npy_intp nd;
	float * arr_data;
	if ((nd = PyArray_NDIM(arr)) < MAX_ND &&
	        PyArray_TYPE(arr) == NPY_FLOAT) {
		shape = Shape(PyArray_SHAPE(arr), (int) nd);
		auto length = (size_t) PyArray_SIZE(arr);
		if (shape.length != length) goto fail;
		// printf("length, shape_length %i %i\n", length, shape.length);
		arr_data = (float *) PyArray_DATA(arr);
		delete[] data;
		data = new float[length];
		for (int i = 0; i < length; ++i) {
			data[i] = arr_data[i];
		}
	} else {
fail:
		PyErr_Clear();
		PyErr_SetString(PyExc_ValueError,
		                "Failed to copy numpy array.");
	}
}

Karray Karray::flat_sum(bool mean) {
	Karray result;
	for (int k = 0; k < shape.length; ++k) {
		result.data[0] += data[k];
	}
	if (mean)
		result.data[0] /= (float) shape.length;
	return result;
}


Karray Karray::sum(size_t axis, const Karray &weights, bool mean) {
	Shape new_shape = shape;
	size_t reduction = new_shape.pop((int) axis);
	if (weights.shape.length != 1 &&
	        weights.shape.length != reduction)
		KERR_RETURN_VAL("Weights do not correspond to sum reduction.", Karray{});
	NDVector strides = shape.strides();
	Karray result(new_shape, 0.);
	_sum(data, result.data, weights.data, shape, strides, (weights.shape.length != 1), mean, axis, 0);
	return result;
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

// PyObject *
// Karray_binary_op(PyObject * self, PyObject * other,
//                 ) {
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