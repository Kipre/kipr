
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

Shape::Shape(Py_ssize_t * input, int size) {
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

Shape::Shape(PyObject * o, size_t target_length) {
	nd = 0;
	length = 1;
	int wildcard = -1;
	Py_ssize_t value;
	if (!PyList_Check(o) && !PyTuple_Check(o))
		KERR_RETURN("Shape must be a list or a tuple.");
	Py_ssize_t seq_length = PySequence_Length(o);
	if (seq_length == 0)
		KERR_RETURN("Shape must be a list or a tuple with at least one element.");
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
	int i = MAX_ND - 1;
	int dim_diff = a.nd - b.nd;
	while (i >= 0) {
		if (i >= dim_diff &&
		        a[i] != b[i - dim_diff]) {
			if (a[i] == 1) {
				buf[i] = b[i - dim_diff];
			} else if (b[i - dim_diff] == 1) {
				buf[i] = a[i];
			} else {
				PyErr_Format(Karray_error,
				             "Shapes %s and %s are not compatible.", a.str(), b.str());
				return;
			}
		} else {
			buf[i] = a[i];
		}
		length *= max(1, buf[i]);
		--i;
	}
}

static std::tuple<Shape, NDVector, NDVector>
paired_strides(Shape a, Shape b) noexcept {
	Shape common = Shape(a, b);
	NDVector astr, bstr;
	size_t acc = 1, bcc = 1;
	while (a.nd > b.nd)
		b.insert_one(0);
	while (b.nd > a.nd)
		a.insert_one(0);

	for (int k = a.nd - 1; k >= 0; --k) {
		if (a[k] == common[k]) {
			astr.buf[k] = acc;
			acc *= a[k];
		} else {
			astr.buf[k] = 0;
		}
		if (b[k] == common[k]) {
			bstr.buf[k] = bcc;
			bcc *= b[k];
		} else {
			bstr.buf[k] = 0;
		}
	}
	return {common, astr, bstr};
}

std::tuple<NDVector, NDVector>
Shape::paired_strides(Shape b) noexcept {
	NDVector astr, bstr;
	size_t acc = 1, bcc = 1;
	while ((nd - b.nd) > 0)
		b.insert_one(0);

	for (int k = nd - 1; k >= 0; --k) {
		if (b[k] == buf[k]) {
			bstr.buf[k] = bcc;
			bcc *= b[k];
			astr.buf[k] = acc;
			acc *= buf[k];
		} else if (b[k] == 1) {
			astr.buf[k] = acc;
			acc *= buf[k];
			bstr.buf[k] = 0;
		} else {
			PyErr_Format(Karray_error,
			             "Shapes %s and %s not compatible for inplace binary op.",
			             str(), b.str());
			return {astr, bstr};
		}
	}
	return {astr, bstr};
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

// void Shape::set(int i, size_t val) {
// 	buf[i] = val;
// }

void Shape::set(int i, size_t val) {
	if (i < nd) {
		length /= buf[i];
		buf[i] = val;
		length *= val;
	} else {
		for (int k=nd; k < i; ++k) {
			buf[k] = 1;
		}
		buf[i] = val;
		length *= val;
		nd = i + 1;
	}
}

size_t Shape::nbmats() {
	if (nd == 1)
		return 0;
	if (nd == 2)
		return 1;
	return length / (buf[nd-1] * buf[nd-2]);
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



std::string Shape::str() const {
	std::string result("[");
	result += std::to_string(buf[0]);
	for (int i = 1; i < nd; ++i)
		result += ", " + std::to_string(buf[i]);
	return result + "]";
}

NDVector Shape::broadcast_to(Shape & other) {
	NDVector result;
	int dim_diff = other.nd - nd;
	if (dim_diff < 0) {
		PyErr_Format(Karray_error,
		             "Cannot broadcast shape %s to %s.", str(), other.str());
		return result;
	}
	int acc = 1;
	for (int i = other.nd - 1; i > dim_diff; --i) {
		if (buf[i - dim_diff] == 1 && other[i] != 1) {
			result.buf[i] = 0;
		} else if (buf[i - dim_diff] == other[i]) {
			result.buf[i] = acc;
			acc *= other[i];
		} else {
		PyErr_Format(Karray_error,
		             "Cannot broadcast shape %s to %s.", str(), other.str());
		return result;
		}
	}
	for (int i = 0; i < dim_diff; ++i) {
		result.buf[i] = 0;
	}
	return result;

}

size_t Shape::sum() {
	size_t result = 0;
	for (int i = 0; i < nd; ++i) {
		result += buf[i];
	}
	return result;
}

size_t Shape::operator[](int i) const {
	if (i >= 0 && i < MAX_ND) {
		return buf[i];
	} else if (i > - nd - 1 && i < 0) {
		return buf[nd + i];
	} else {
		throw std::exception("index out of range");
	}
}

size_t Shape::pop(int i) noexcept {
	if (abs(i) >= nd)
		throw std::exception("Shape::pop out of range");
	if (i == -1)
		i = nd - 1;
	size_t tmp = buf[i];
	if (i == 0 && nd == 1) {
		buf[0] = 1;
		length = 1;
		return tmp;
	}
	while (i != MAX_ND - 1) {
		buf[i] = buf[i + 1];
		++i;
	}
	buf[MAX_ND - 1] = 0;
	--nd;
	length /= tmp;
	return tmp;
}

NDVector Shape::strides(int depth_diff) const {
	NDVector result;
	size_t acc = 1;
	// printf("depth diff %i, nd %i\n", depth_diff, nd);
	for (int i = nd - 1; i >= 0; --i) {
		result.buf[i + depth_diff] = acc;
		acc *= buf[i];
	}
	for (int i = 0; i < depth_diff; ++i) {
		result.buf[i] = 0;
	}
	return result;
}

void Shape::insert_one(int i) {
	if (i < 0 || i > nd)
		KERR_RETURN("Cannot insert 1 into shape because index is out of bounds.");
	++nd;
	int k = MAX_ND - 1;
	while (k > i) {
		buf[k] = buf[k - 1];
		--k;
	}
	buf[i] = 1;
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
	if (abs(value) > nd - 1)
		KERR_RETURN_VAL("Axis is out of range.", 9);
	return (size_t) (value % nd + nd) % nd;

}

size_t Shape::axis(int ax) {
	if (abs(ax) > nd - 1)
		KERR_RETURN_VAL("Axis is out of range.", 9);
	return (size_t) (ax % nd + nd) % nd;

}

// static compatible_for_matmul(Shape & a, Shape & b) {
// 	if (a.nd < 2 || b.nd < 2)
// 		return {};
// 	if (a[a.nd - 1] != b[b.nd - 2])
// 		return {};
// 	Shape result;
// 	if (a.nd > b.nd)
// 		result = b
// 	else 
// 		result = a;


// 	return true;
// }

std::tuple<Shape, NDVector> Shape::transpose() const {
	Shape result;
	result.nd = nd;
	result.length = length;
	std::copy(buf, buf + MAX_ND, result.buf);
	NDVector strides_t = strides();
	result.buf[nd-1] = buf[nd-2];
	result.buf[nd-2] = buf[nd-1];
	size_t tmp = strides_t.buf[nd-1];
	strides_t.buf[nd-1] = strides_t.buf[nd-2];
	strides_t.buf[nd-2] = tmp;
	return {result, strides_t};
}

int Shape::last_axis() {
	int i = nd-1;
	while (buf[i] == 1 && i > 0)
		--i;
	return i;
}