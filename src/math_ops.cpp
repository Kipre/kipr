PyObject *
Karray_binary_op(PyObject * self, PyObject * other, 
                void (*op_kernel)(float *, float*, Py_ssize_t)) {
    Karray *a, *b, *c;
    Py_ssize_t data_length;


    if (!is_Karray(self) || !is_Karray(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    a = reinterpret_cast<Karray *>(self);
    b = reinterpret_cast<Karray *>(other);
    c = new_Karray_as(a);
    Karray_copy(a, c);

    data_length = Karray_length(a);
    if (data_length == Karray_length(b)) {
        op_kernel(c->data, b->data, data_length);
    } else if (true) {

    } else {
        PyErr_SetString(PyExc_TypeError, "Data length does not match.");
        PyErr_Print();
        goto fail;
    }

    return reinterpret_cast<PyObject *>(c);

    fail:
        Py_XDECREF(a);
        Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "Failed to add elements.");
        return NULL;
}


PyObject *
Karray_inplace_binary_op(PyObject * self, PyObject * other, 
                         void (*op_kernel)(float *, float*, Py_ssize_t)) {
    Karray *a, *b;
    Py_ssize_t data_length;

    if (!is_Karray(self) || !is_Karray(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    a = reinterpret_cast<Karray *>(self);
    b = reinterpret_cast<Karray *>(other);
    
    data_length = Karray_length(a);
    if (data_length == Karray_length(b)) {
        op_kernel(a->data, b->data, data_length);
    } else {
        PyErr_SetString(PyExc_TypeError, "Data length does not match.");
        PyErr_Print();
        goto fail;
    }
    Py_INCREF(self);
    return self;

    fail:
        Py_XDECREF(a);
        Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "Failed to add elements.");
        return NULL;
}



PyObject *
Karray_add(PyObject * self, PyObject * other) {
    return Karray_binary_op(self, other, add_kernel);
}

PyObject *
Karray_inplace_add(PyObject * self, PyObject * other) {
    return Karray_inplace_binary_op(self, other, add_kernel);
}

PyObject *
Karray_sub(PyObject * self, PyObject * other) {
    return Karray_binary_op(self, other, sub_kernel);
}

PyObject *
Karray_inplace_sub(PyObject * self, PyObject * other) {
    return Karray_inplace_binary_op(self, other, sub_kernel);
}

PyObject *
Karray_mul(PyObject * self, PyObject * other) {
    return Karray_binary_op(self, other, mul_kernel);
}

PyObject *
Karray_inplace_mul(PyObject * self, PyObject * other) {
    return Karray_inplace_binary_op(self, other, mul_kernel);
}

PyObject *
Karray_div(PyObject * self, PyObject * other) {
    return Karray_binary_op(self, other, div_kernel);
}

PyObject *
Karray_inplace_div(PyObject * self, PyObject * other) {
    return Karray_inplace_binary_op(self, other, div_kernel);
}
