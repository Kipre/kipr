
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
		if (match > 0) {
			if (i == 0) {
				ss << std::string(match + 1, '[');
			} else {
				ss << std::string(match, ']') << ",\n";
				ss << std::string(shape.nd - match + 9, ' ');
				ss << std::string(match, '[');
			}
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

void transfer(float * from, float * to, size_t * positions, size_t * strides,
              const Filter & filter, Shape & to_shape, int depth) {
	if (depth < to_shape.nd) {
		size_t current_value, last_value = 0;
		for (int k = 0; k < to_shape[depth]; ++k) {
			current_value = filter.buf[filter.offset[depth] + k];
			positions[1] += (current_value - last_value) * strides[depth];
			last_value = current_value;
			transfer(from, to, positions, strides, filter, to_shape, depth + 1);
		}
		positions[1] -= last_value * strides[depth];
	} else {
		// printf("writing from %i to %i\n", positions[1], positions[0]);
		to[positions[0]++] = from[positions[1]];
	}
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
