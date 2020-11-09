
Karray::Karray() {
	owned = false;
	seed = rand();
	shape = Shape();
	data = new float[1];
	data[0] = 0;
}

Karray::Karray(Shape new_shape, std::vector<float> vec) {
	owned = false;
	seed = rand();
	shape = new_shape;
	// printf("shape.length, vec.size(): %i %i\n", shape.length, vec.size());
	data = new float[shape.length];
	std::copy(vec.begin(), vec.end(), data);
}

Karray::Karray(Shape new_shape, float * new_data) {
	owned = false;
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

void Karray::steal(Karray& other) {
	other.owned = true;
	seed = other.seed;
	shape = other.shape;
	data = other.data;
}

Karray::~Karray() {
	if (!owned) {
		delete[] data;
	}
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
		NDVector strides(shape.strides(new_shape.nd - shape.nd));
		Filter filter(shape.broadcast_to(new_shape));
		PYERR_PRINT_GOTO_FAIL;
		// filter.print();

		auto buffer = new float[new_shape.length];
		size_t positions[2] = {0, 0};

		// strides.print();
		transfer(data, buffer, positions,
		         strides.buf,
		         filter, new_shape, 0);


		shape = new_shape;
		delete[] data;
		data = buffer;
		return;

fail:
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