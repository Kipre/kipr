

template<typename T>
class FastSequence
{
public:
	std::vector<T> elements;
	FastSequence(PyObject * o, bool accept_singleton = false);
	~FastSequence() = default;
	Shape to_shape() {
		auto shape = Shape(elements.data(), elements.size());
		shape.validate();
		return shape;
	}
};

template<typename T>
FastSequence<T>::FastSequence(PyObject * o, bool accept_singleton) {
	if (accept_singleton) {
		T value = (T) PyLong_AsSsize_t(o);
		if (!PyErr_Occurred()) {
			elements.push_back(value);
			return;
		} else {
			PyErr_Clear();
		}
	}
	if (!PyList_Check(o) &&
	        !PyTuple_Check(o))
		goto fail;
	Py_ssize_t length = PySequence_Length(o);
	elements.reserve(length);
	PyObject ** items = PySequence_Fast_ITEMS(o);
	for (int i = 0; i < length; ++i) {
		elements.push_back((T) PyLong_AsSsize_t(items[i]));
		PYERR_CLEAR_GOTO_FAIL;
	}

	return;
fail:
	PyErr_Format(PyExc_TypeError,
	             "Input must be a list or a tuple of %s.",
	             typeid(T).name());
	return;
}

class Int
{
public:
	int value;
	Int(PyObject * o) {
		if (!PyIndex_Check(o))
			goto fail;

		value = PyLong_AsSsize_t(o);
		return;
fail:
		PyErr_SetString(PyExc_ValueError, "");
		return;
	};

};

class Float
{
public:
	float value;
	Float(PyObject * o) {
		if (!PyNumber_Check(o))
			goto fail;

		value = (float) PyFloat_AsDouble(o);
		return;
fail:
		PyErr_SetString(PyExc_ValueError, "Failed to read a Float");
		return;
	};

	void print() {
		std::cout << value << ", ";
	}

};

class Axis
{
public:
	Py_ssize_t value;
	Axis(int nd, PyObject * o, Py_ssize_t default_value = -1) {
		if (o) {
			value = Int(o).value;
			if (Py_ABS(value) > nd - 1) {
				PyErr_SetString(PyExc_ValueError, "Axis out of range.");
				return;
			}
			value = (value % nd + nd) % nd;
		} else {
			value = default_value;
		}
	};
	~Axis() = default;
};

template<class T>
class NestedSequence {
public:
	std::vector<T> data;
	Shape shape;

	// NestedSequence();
	NestedSequence(PyObject * o);
	~NestedSequence() = default;

	bool parse_data(PyObject * o, int depth = 0);

	Karray to_Karray() {
		return Karray(shape, data);
	};

	void print() {
		std::cout << "Printing NestedSequence\n\t";
		shape.print();
		std::cout << '\n';
		for ( auto &i : data ) {
			std::cout << i << ", ";
		}
		std::cout << '\n';
	};
};

template<class T>
bool NestedSequence<T>::parse_data(PyObject * o, int depth) {
	if (PySequence_Check(o) && depth < MAX_ND) {
		Py_ssize_t length = PySequence_Length(o);
		for (int i = 0; i < length; ++i) {
			PyObject * item = PySequence_GetItem(o, i);
			if (!NestedSequence<T>::parse_data(item, depth + 1))
				goto fail;
			Py_DECREF(item);
		}
		return shape.assert_or_set((size_t) length, depth);
	} else if (PyNumber_Check(o)) {
		data.push_back((T) PyFloat_AsDouble(o));
		return true;
	}
fail:
	PyErr_SetString(PyExc_TypeError, "Failed to parse input value.");
	return false;

}


template<class T>
NestedSequence<T>::NestedSequence(PyObject * o) {
	size_t length;
	NestedSequence<T>::parse_data(o, 0);
	PYERR_PRINT_GOTO_FAIL;
	length = shape.validate();
	PYERR_PRINT_GOTO_FAIL;
	if (data.size() !=  length)
		goto fail;
	return;
fail:
	PyErr_SetString(PyExc_TypeError, "Failed to build a NestedSequence.");
	return;
}



