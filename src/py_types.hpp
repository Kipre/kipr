

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
	IF_ERROR_RETURN();
	length = shape.validate();
	IF_ERROR_RETURN();
	if (data.size() !=  length)
		PyErr_SetString(Karray_error, "Data does not match the expected size.");
	return;
}



