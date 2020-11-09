

Shape::Shape(size_t * input, int size) {
	nd = 0;
	length = 1;
	size = min(MAX_ND, size);
	while (nd < size) {
		length *= input[nd];
		buf[nd] = input[nd];
		++nd;
	}
	int i = nd;
	while (i < MAX_ND) {
		buf[i++] = 0;
	}
}

Shape::Shape(PyObject * o, bool accept_singleton) {
	nd = 0;
	Py_ssize_t value;
	if (accept_singleton) {
		value = PyLong_AsSsize_t(o);
		if (!PyErr_Occurred() && value > 0) {
			buf[nd] = (size_t) value;
			length = nd;
			return;
		} else {
			PyErr_Clear();
		}
	}
	if (!PyList_Check(o) && !PyTuple_Check(o))
		goto fail;

	Py_ssize_t length = PySequence_Length(o);
	PyObject ** items = PySequence_Fast_ITEMS(o);
	for (int i = 0; i < length; ++i) {
		value = PyLong_AsSsize_t(items[i]);
		PYERR_CLEAR_GOTO_FAIL;
		buf[nd] = value;
		++nd;
	}
	while(nd != MAX_ND) {
		buf[nd] = 0;
		++nd;
	}
	cohere();
	PYERR_PRINT_GOTO_FAIL;
	return;

fail:
	PyErr_Format(PyExc_TypeError,
	             "Failed to parse the shape.");
	return;
}

Shape::Shape() {
	nd = 1;
	length = 1;
	buf[0] = 1;
	std::fill(buf + 1, buf + MAX_ND, 0);
}



void Shape::print(const char * message) {
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

size_t Shape::cohere() {
	int new_nd = 0;
	length = 1;
	if (buf[0] == 0) goto fail;
	while (buf[new_nd] != 0 && new_nd < MAX_ND) {
		length *= buf[new_nd];
		++new_nd;
	}
	int i = new_nd;
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

Filter Shape::broadcast_to(Shape other) {
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

NDVector Shape::strides(int depth_diff) {
	NDVector result;
	size_t acc = 1;
	for (int i = nd - 1; i >= 0; --i) {
		result.buf[i + depth_diff] = acc;
		acc *= buf[i];
	}
	for (int i = 0; i < depth_diff; ++i) {
		result.buf[i] = 0;
	}
	return result;
}

