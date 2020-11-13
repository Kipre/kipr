
Shape::Shape(int ndims...) {
	va_list args;
	va_start(args, ndims);
	nd = ndims;
	length = 1;
	for (int i = 0; i < MAX_ND; ++i) {
		if (i < nd) {
			length *= (buf[i] = (size_t) va_arg(args, int));
		} else {
			buf[i] = 0;
		}
	}
	va_end(args);
}


template<typename T>
Shape::Shape(T * input, int size) {
	nd = 0;
	length = 1;
	size = min(MAX_ND, size);
	while (nd < size) {
		length *= (size_t) input[nd];
		buf[nd] = (size_t) input[nd];
		++nd;
	}
	int i = nd;
	while (i < MAX_ND) {
		buf[i++] = 0;
	}
}

Shape::Shape(PyObject * o, size_t target_length) {
	nd = 0;
	length = 1;
	int wildcard = -1;
	Py_ssize_t value;
	if (!PyList_Check(o) && !PyTuple_Check(o))
		KERR_RETURN("Shape must be a list or a tuple.");
	Py_ssize_t seq_length = PySequence_Length(o);
	PyObject ** items = PySequence_Fast_ITEMS(o);
	for (int i = 0; i < seq_length; ++i) {
		value = PyLong_AsSsize_t(items[i]);

		PYERR_SET_RETURN("Shape must be a sequence of integers.");
		if (value == 0 || value < -1) {
			PyErr_Format(Karray_error,
			             "Proposed shape is invalid with value %I64i at %i", value, nd);
			return;
		}
		if (value == -1) {
			if (wildcard != -1)
				KERR_RETURN("Can't have more than one wildcard (-1) in the shape.");
			wildcard = nd;
		} else {
			buf[nd] = value;
			length *= value;
		}
		++nd;
	}
	int i = nd;
	while (i != MAX_ND) {
		buf[i] = 0;
		++i;
	}
	if (wildcard >= 0) {
		if (!target_length)
			KERR_RETURN("Wildcard (-1) not allowed here.");
		if (target_length % length != 0)
			KERR_RETURN("Proposed shape cannot be adapted to data.");
		buf[wildcard] = target_length / length;
		length = target_length;
	} else if (target_length) {
		if (target_length != length)
			KERR_RETURN("Proposed shape doesn't align with data.");

	}
	return;

}

Shape::Shape() {
	nd = 1;
	length = 1;
	buf[0] = 1;
	std::fill(buf + 1, buf + MAX_ND, 0);
}

Shape::Shape(Shape a, Shape b) noexcept { // [3, 4, 5] & [3, 1]
	if (a.nd < b.nd)
		std::swap(a, b);

	nd = a.nd;
	length = 1;
	int curr_dim = MAX_ND - 1;
	int dim_diff = a.nd - b.nd;
	while (curr_dim >= 0) {
		printf("curr_dim: %i\n", curr_dim);
		if (curr_dim >= dim_diff &&
		        a[curr_dim] != b[curr_dim - dim_diff]) {
			if (a[curr_dim] == 1) {
				buf[curr_dim] = b[curr_dim - dim_diff];
			} else if (b[curr_dim - dim_diff] == 1) {
				buf[curr_dim] = a[curr_dim];
			} else goto fail;
		} else {
			buf[curr_dim] = a[curr_dim];
		}
		length *= max(1, buf[curr_dim]);
		--curr_dim;
	}
	return;

fail:
	PyErr_Format(PyExc_ValueError,
	             "Shapes %s and %s are not compatible.", a.str(), b.str());
}


void Shape::swap(Shape & other) {
	std::swap(nd, other.nd);
	std::swap(length, other.length);
	size_t * tmp = buf;
	*buf = *other.buf;
	*other.buf = *tmp;
}

bool Shape::operator==(Shape &other) {
	if (nd != other.nd)
		return false;
	if (length != other.length)
		return false;
	for (int k = 0; k < MAX_ND; ++k)
		if (buf[k] != other[k])
			return false;
	return true;
}

void Shape::print(const char * message) const {
	std::cout << "Shape " << message <<
	          " nd=" << nd << ", length=" << length << "\n\t";
	for (int k = 0; k < MAX_ND; ++k) {
		std::cout << buf[k] << ", ";
	}
	std::cout << '\n';
}

bool Shape::assert_or_set(size_t value, int dim) {
	if ((dim == 0 && buf[0] == 1) ||
	        buf[dim] == 0) {
		buf[dim] = value;
		return true;
	} else if (buf[dim] == value) {
		return true;
	} else {
		return false;
	}
}

size_t Shape::validate() {
	int i, new_nd = 0;
	length = 1;
	if (buf[0] == 0) goto fail;
	while (buf[new_nd] != 0 && new_nd < MAX_ND) {
		length *= buf[new_nd];
		++new_nd;
	}
	i = new_nd;
	while (i < MAX_ND) {
		if (buf[i++] != 0) goto fail;
	}
	nd = new_nd;
	return length;

fail:
	PyErr_Format(PyExc_ValueError,
	             "Shape %s is corrupted.", str());
	return 0;

}

void Shape::write(size_t * destination) {
	for (int i = 0; i < MAX_ND; ++i) {
		destination[i] = buf[i];
	}
}



std::string Shape::str() {
	std::string result("[");
	result += std::to_string(buf[0]);
	for (int i = 1; i < nd; ++i)
		result += ", " + std::to_string(buf[i]);
	return result + "]";
}

Filter Shape::broadcast_to(Shape & other) {
	Filter filter(other);
	int dim_diff = other.nd - nd;
	// printf("dim_diff: %i\n", dim_diff);
	if (dim_diff < 0) goto fail;
	for (int i = other.nd - 1; i > dim_diff - 1; --i) {
		// printf("braodcast dims: %i %i\n", buf[i - dim_diff], other[i]);
		if (buf[i - dim_diff] == 1 && other[i] != 1) {
			filter.set_val_along_axis(i, 0);
		} else if (buf[i - dim_diff] == other[i]) {
			filter.set_range_along_axis(i);
		} else {
			goto fail;
		}
	}
	for (int i = 0; i < dim_diff; ++i) {
		filter.set_val_along_axis(i, 0);
	}
	return filter;

fail:
	PyErr_Format(PyExc_ValueError,
	             "Cannot broadcast shape %s to %s.", str(), other.str());
	return filter;

}

size_t Shape::sum() {
	size_t result = 0;
	for (int i = 0; i < nd; ++i) {
		result += buf[i];
	}
	return result;
}

size_t Shape::operator[](size_t i) {
	return buf[i];
}

size_t Shape::pop(int i) {
	if (abs(i) >= nd)
		KERR_RETURN_VAL("Shape::pop out of range.", 0);
	if (i == -1)
		i = nd - 1;
	size_t tmp = buf[i];
	while (i != MAX_ND - 1) {
		buf[i] = buf[i + 1];
		++i;
	}
	buf[MAX_ND - 1] = 0;
	--nd;
	length /= tmp;
	return tmp;
}

NDVector Shape::strides(int depth_diff) {
	NDVector result;
	size_t acc = 1;
	printf("depth diff %i, nd %i\n", depth_diff, nd);
	result.print();
	for (int i = nd - 1; i >= 0; --i) {
		result.buf[i + depth_diff] = acc;
		acc *= buf[i];
	}
	for (int i = 0; i < depth_diff; ++i) {
		result.buf[i] = 0;
	}
	return result;
}

void Shape::push_back(size_t dim) {
	if (def) {
		nd = 0;
		def = false;
	}
	buf[nd] = dim;
	++nd;
	length *= dim;
}

size_t Shape::axis(PyObject * o) {
	if (!PyIndex_Check(o))
		KERR_RETURN_VAL("Axis is invalid.", 9);
	Py_ssize_t value = PyLong_AsSsize_t(o);
	if (abs(value) > nd-1)
		KERR_RETURN_VAL("Axis is out of range.", 9);
	return (size_t) (value % nd + nd) % nd;

}

size_t Shape::axis(int ax) {
	if (abs(ax) > nd-1)
		KERR_RETURN_VAL("Axis is out of range.", 9);
	return (size_t) (ax % nd + nd) % nd;

}
