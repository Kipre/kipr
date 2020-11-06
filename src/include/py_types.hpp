#include <vector>
#include <iostream>
#include <string>


template<class T>
class FastSequence
{
public:
	std::vector<T> elements;
	FastSequence(PyObject * o, bool accept_singleton = false);
	// FastSequence(PyObject * o);
	~FastSequence() = default;
};

template<class T>
FastSequence<T>::FastSequence(PyObject * o, bool accept_singleton) {
	// DEBUG_Obj(o);
	if (accept_singleton) {
		T value = T(o);
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
	for (int i=0; i < length; ++i) {
		elements.push_back(T(items[i]));
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
	Py_ssize_t value;
	Int(PyObject * o) {
		if (!PyIndex_Check(o))
			goto fail;

		value = PyLong_AsSsize_t(o);
		return;
		fail:
			PyErr_SetString(PyExc_ValueError, "");
			return;
	};
	~Int() = default;
	
};

class Axis
{
public:
	Py_ssize_t value;
	Axis(int nd, PyObject * o, Py_ssize_t default_value = -1) {
		if (o) {
			value = Int(o).value;
			if (Py_ABS(value) > nd-1) {
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
