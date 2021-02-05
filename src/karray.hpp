Karray Karray::elementwise_binary_op(const Karray &other,
                                     binary_kernel kernel,
                                     binary_op op) {
	size_t length = shape.length;
	if (length == other.shape.length) {
		Karray result(shape);
		kernel(result.data, data, other.data, length);
		return result;
	} else {
		auto [common, a_strides, b_strides] =
		    paired_strides(shape, other.shape);
		IF_ERROR_RETURN(Karray {});
		Karray result(common);
		Positions pos {0, 0, 0};
		rec_binary_op(result.data, data, other.data,
		              common, a_strides, b_strides, &pos, op, 0);
		return result;
	}
}

void Karray::inplace_binary_op(const Karray  &rhs,
                               binary_kernel kernel,
                               binary_op op) {
	if (shape.length == rhs.shape.length) {
		kernel(data, data, rhs.data, shape.length);
	} else {
		auto [a_strides, b_strides] =
		    shape.paired_strides(rhs.shape);
		IF_ERROR_RETURN();
		Positions pos {0, 0, 0};
		rec_binary_op(data, data, rhs.data, shape,
		              a_strides, b_strides, &pos, op, 0);
	}
}

Karray::Karray(Shape new_shape) {
	shape = new_shape;
	data = new float[shape.length];
}


Karray::Karray(Shape new_shape, float value) {
	shape = new_shape;
	data = new float[shape.length];
	std::fill(data, data + shape.length, value);
}

Karray::Karray(float val) {
	// printf("creating generic new karr\n");
	shape = Shape();
	data = new float[1];
	data[0] = val;
}


Karray::Karray() {
	// printf("creating generic new karr\n");
	shape = Shape();
	data = new float[1];
	data[0] = 0;
}

Karray::Karray(const Karray& other)
	: shape{other.shape} {
	data = new float[shape.length];
	std::copy(other.data, other.data + shape.length, data);
}

Karray& Karray::operator=(const Karray& other) {
	// printf("copying array %s into %s\n", other.shape.str().c_str(), shape.str().c_str());
	shape = other.shape;
	delete[] data;
	data = new float[shape.length];
	std::copy(other.data, other.data + shape.length, data);
	return *this;
}

Karray::Karray(Karray&& other)
	: shape{other.shape} {
	// printf("moving array %i into %i\n", other.seed, seed);
	data = other.data;
	other.shape = Shape();
	other.data = new float[1];
	other.data[0] = 0;
}

Karray& Karray::operator=(Karray&& other) {
	// printf("moving array %i into %i\n", other.seed, seed);
	shape = other.shape;
	delete[] data;
	data = other.data;
	other.shape = Shape();
	other.data = new float[1];
	other.data[0] = 0;
	return *this;
}

void Karray::reset(Shape & new_shape) {
	delete [] data;
	shape = new_shape;
	data = new float[shape.length];
}

// Karray& Karray::operator+=(const Karray& other) {
// 	inplace_binary_op(other, add_kernel, _add);
// 	return *this;
// }

// Karray& Karray::operator/=(const Karray& other) {
// 	inplace_binary_op(other, div_kernel, _div);
// 	return *this;
// }

// Karray& Karray::operator-=(const Karray& other) {
// 	inplace_binary_op(other, sub_kernel, _sub);
// 	return *this;
// }

// Karray& Karray::operator*=(const Karray& other) {
// 	inplace_binary_op(other, mul_kernel, _mul);
// 	return *this;
// }

// Karray Karray::operator-(const Karray& rhs) {
// 	return elementwise_binary_op(rhs, sub_kernel, _sub);
// }

// Karray Karray::operator*(const Karray& rhs) {
// 	return elementwise_binary_op(rhs, mul_kernel, _mul);
// }

// Karray Karray::operator+(const Karray& rhs) {
// 	return elementwise_binary_op(rhs, add_kernel, _add);
// }

// Karray Karray::operator/(const Karray& rhs) {
// 	return elementwise_binary_op(rhs, div_kernel, _div);
// }

// Karray Karray::matmul(const Karray& rhs) {
// 	if (!shape.compatible_for_matmul(rhs.shape))
// 		throw std::exception("shapes incompatible for matmul");

// }

void Karray::swap(Karray& other) {
	// printf("swapping %i and %i\n", seed, other.seed);
	std::swap(shape, other.shape);
	std::swap(data, other.data);
}

Karray::Karray(Shape new_shape, std::vector<float> vec) {
	shape = new_shape;
	// printf("shape.length, vec.size(): %i %i\n", shape.length, vec.size());
	data = new float[shape.length];
	std::copy(vec.begin(), vec.end(), data);
}

Karray::Karray(Shape new_shape, float * new_data) {
	shape = new_shape;
	data = new_data;
}

Karray Karray::subscript(PyObject * key) {
	Karray result;
	Filter filter;
	Positions pos {0, 0, 0};

	NDVector strides(shape.strides());
	Shape new_shape(filter.from_subscript(key, shape));
	IF_ERROR_RETURN({});

	result.shape = new_shape;
	delete[] result.data;
	result.data = new float[new_shape.length];

	// filter.print();
	// strides.print();
	transfer(data, result.data, &pos,
	         strides.buf, filter, shape.nd, 0);
	// printf("positions[0], positions[1]: %i %i\n", positions[0], positions[1]);
	if (pos.write != new_shape.length)
		PyErr_SetString(PyExc_ValueError, "Failed to subscript array.");

	return result;
}

Karray::~Karray() {
	// printf("deallocating karr with shape %s and seed %i\n", shape.str().c_str(), seed);
	delete[] data;
	shape.~Shape();
}

void Karray::print(const char * message) {
	std::cout << "Printing Karray " << message << "\n\t";
	shape.print();
	std::cout << "\t";
	auto limit = std::min(shape.length, MAX_PRINT_SIZE);
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
	auto limit = std::min(shape.length, MAX_PRINT_SIZE);
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
	   << ")";
	return ss.str();
}


// Karray Karray::matmul(Karray & other) {

// 	if (shape.nd < 2 && other.shape.nd < 2) {
// 		KERR_RETURN_VAL("Both arrays must be at least two-dimensional for matmul.", NULL);
// 	}

// 	size_t M, N, I, J, K;
// 	I = shape[-2];
// 	K = shape[-1];
// 	J = other.shape[-1];

// 	M = shape.nbmats();
// 	N = other.shape.nbmats();

// 	if (K != other.shape[-2] ||
// 		(M % N != 0 && N % M != 0)) {
// 		PyErr_Format(Karray_error,
// 		             "Matmul not possible with shapes %s and %s.",
// 		             shape.str(), other.shape.str());
// 		return NULL;
// 	}

// 	Shape new_shape((M > N) ? shape : other.shape);
// 	new_shape.set(new_shape.nd - 2, I);
// 	new_shape.set(new_shape.nd - 1, J);

// 	auto result = Karray(new_shape);

// 	for (int m = 0; m < max(M, N); ++m) {
// 		int ia = m % M;
// 		int ib = m % N;

// 		general_matmul(result.data + m * I * J,
// 		       data + ia * I * K,
// 		       other.data + ib * K * J,
// 		       I, J, K);
// 	}

// 	return result;
// }


Karray Karray::broadcast(Shape new_shape) {
	// shortcuts
	if (shape.length == 1) {
		Karray result(new_shape);
		std::fill(result.data, result.data + new_shape.length, data[0]);
		return result;
	} else if (shape.length == new_shape.length) {
		Karray result(*this);
		result.shape = new_shape;
		return result;
	} else {
		Positions pos {0, 0, 0};
		Karray result(new_shape);

		auto strides = shape.broadcast_to(new_shape);
		IF_ERROR_RETURN(result);
		simple_transfer(data, result.data, &pos, new_shape, strides, 0);

		return result;
	}

}

void Karray::from_numpy(PyObject * obj) noexcept {
	auto arr = (PyArrayObject *) obj;
	npy_intp nd;
	float * arr_data;
	if ((nd = PyArray_NDIM(arr)) > MAX_ND || PyArray_TYPE(arr) != NPY_FLOAT) {
		PyErr_SetString(Karray_error, "Failed to copy numpy array.");
		return;
	}

	shape = Shape(PyArray_SHAPE(arr), (int) nd);
	auto length = (size_t) PyArray_SIZE(arr);
	if (shape.length != length) {
		PyErr_SetString(Karray_error, "Failed to copy numpy array.");
		return;
	}
	arr_data = (float *) PyArray_DATA(arr);
	delete[] data;
	data = new float[length];
	for (int i = 0; i < length; ++i) {
		data[i] = arr_data[i];
	}
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
		std::abort(); // unknown mode
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
	        weights.shape.length != reduction) {
		PyErr_SetString(Karray_error, "Weights do not correspond to sum reduction.");
		return Karray {};
	}
	NDVector strides = shape.strides();
	Karray result(new_shape, 0.);
	_sum(data, result.data, weights.data, shape, strides, (weights.shape.length != 1), mean, axis, 0);
	return result;
}