

void
Karray_dealloc(PyKarray *self) {
    printf("from python with refcount=%i\n", self->ob_base.ob_refcnt);
    self->arr.~Karray();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *
Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    return type->tp_alloc(type, 0);
}

PyKarray *
new_PyKarray() {
    return reinterpret_cast<PyKarray *>(KarrayType.tp_alloc(&KarrayType, 0));
}

int
Karray_init(PyKarray *self, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"data", "shape", NULL};
    PyObject *input = NULL, *shape = NULL;
    Karray candidate;
    Shape proposed_shape;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", kwlist,
                                     &input, &shape))
        return -1;

    if (shape) {
        proposed_shape = Shape(shape);
        PYERR_PRINT_GOTO_FAIL;
    }

    switch (py_type(input)) {
    case (KARRAY): {
        // printf("initializin from karray\n");
        Py_INCREF(input);
        PyKarray * karr = reinterpret_cast<PyKarray *>(input);
        candidate = karr->arr;
        break;
    }
    case (STRING): {
        auto mode = read_mode(input);
        PYERR_PRINT_GOTO_FAIL;
        candidate.from_mode(proposed_shape, mode);
        break;
    }
    case (NUMPY_ARRAY):
        Py_INCREF(input);
        candidate.from_numpy(input);
        if (PyErr_Occurred()) {
            PyErr_Clear();
        } else {
            break;
        }
    case (NUMBER):
    case (SEQUENCE): {
        NestedSequence<float> nest(input);
        PYERR_PRINT_GOTO_FAIL;
        candidate = nest.to_Karray();
        if (shape) {
            candidate.broadcast(proposed_shape);
        }
    }
    break;
    default:
        PyErr_SetString(PyExc_TypeError,
                        "Input object not understood.");
    }
    PYERR_PRINT_GOTO_FAIL;

    self->arr.swap(candidate);

    Py_DECREF(input);
    Py_XDECREF(shape);
    return 0;

fail:
    Py_DECREF(input);
    Py_XDECREF(shape);
    PyErr_SetString(PyExc_TypeError,
                    "Failed to initialize kipr.arr.");
    return -1;
}

PyObject *
execute_func(PyObject *self, PyObject * input) {
    DEBUG_Obj(input, "");


    Shape shape(input, (size_t) 120);
    shape.print();

    Py_RETURN_NONE;
}

PyObject *
Karray_str(PyKarray * self) {
    return PyUnicode_FromString(self->arr.str().c_str());
}

PyObject *
Karray_subscript(PyObject *here, PyObject * key) {
    auto self = reinterpret_cast<PyKarray *>(here);

    Py_INCREF(key);
    if (!PyTuple_Check(key))
        key = Py_BuildValue("(O)", key);

    auto result = new_PyKarray();
    result->arr = self->arr.subscript(key);
    PYERR_PRINT_GOTO_FAIL;
    Py_DECREF(key);
    return reinterpret_cast<PyObject *>(result);

fail:
    Py_DECREF(key);
    PyErr_SetString(PyExc_ValueError, "Failed to apply subscript.");
    return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_reshape(PyKarray * self, PyObject * shape) {
    Py_INCREF(reinterpret_cast<PyObject *>(self));
    Shape new_shape(shape, self->arr.shape.length);
    PYERR_RETURN_VAL(NULL);
    new_shape.print();
    self->arr.shape = new_shape;
    return reinterpret_cast<PyObject *>(self);
}

PyObject *
Karray_getshape(PyKarray *self, void *closure) {
    int nd = self->arr.shape.nd;
    PyObject * result = PyTuple_New(nd);
    for (int k = 0; k < nd; k++) {
        PyTuple_SET_ITEM(result, k, PyLong_FromSize_t(self->arr.shape[k]));
    }
    return result;
}

PyObject *
Karray_getrefcnt(PyKarray *self, void *closure) {
    Py_ssize_t refcnt = self->ob_base.ob_refcnt;
    return PyLong_FromSsize_t(refcnt);
}

PyObject *
Karray_numpy(PyKarray *self, PyObject *Py_UNUSED(ignored)) {
    int nd = self->arr.shape.nd;
    npy_intp * dims = new npy_intp[nd];
    for (int k = 0; k < nd; k++) {
        dims[k] = (npy_intp) self->arr.shape[k];
    }
    float * buffer = new float[self->arr.shape.length];
    std::copy(self->arr.data, self->arr.data + self->arr.shape.length, buffer);
    PyObject * result = PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT, buffer);
    PyArray_UpdateFlags(reinterpret_cast<PyArrayObject *>(result), NPY_ARRAY_OWNDATA);
    return result;
}

PyObject *
Karray_broadcast(PyKarray * self, PyObject * shape) {
    Py_INCREF(reinterpret_cast<PyObject *>(self));
    Shape new_shape(shape);
    PYERR_RETURN_VAL(NULL);
    new_shape.print();
    self->arr.broadcast(new_shape);
    return reinterpret_cast<PyObject *>(self);
}



PyObject *
Karray_sum(PyKarray * self, PyObject * shape) {
    Py_INCREF(reinterpret_cast<PyObject *>(self));
    Shape new_shape(shape);
    PYERR_RETURN_VAL(NULL);
    new_shape.print();
    self->arr.broadcast(new_shape);
    return reinterpret_cast<PyObject *>(self);
}


PyObject *
Karray_sum(PyKarray *here, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"axis", "weights", NULL};

    int axis = NO_AXIS;
    PyKarray * weights_obj = NULL;
    PyKarray * result = new_PyKarray();


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i$O!", kwlist,
                                     &axis, &KarrayType, &weights_obj))
        return NULL;

    PyKarray * self = reinterpret_cast<PyKarray *>(here);

    if (axis == NO_AXIS) {
        result->arr = self->arr.flat_sum();
    } else {
        size_t ax = self->arr.shape.axis(axis);
        PYERR_RETURN_VAL(NULL);
        if (weights_obj == NULL) {
            result->arr = self->arr.sum(ax, Karray(Shape(), 1.0));
        } else {
            result->arr = self->arr.sum(ax, weights_obj->arr);
            PYERR_RETURN_VAL(NULL);
        }
    }

    return reinterpret_cast<PyObject *>(result);
}


PyObject *
Karray_mean(PyKarray *here, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"axis", "weights", NULL};

    int axis = NO_AXIS;
    PyKarray * weights_obj = NULL;
    PyKarray * result = new_PyKarray();


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i$O!", kwlist,
                                     &axis, &KarrayType, &weights_obj))
        return NULL;

    PyKarray * self = reinterpret_cast<PyKarray *>(here);

    if (axis == NO_AXIS) {
        result->arr = self->arr.flat_sum(true);
    } else {
        size_t ax = self->arr.shape.axis(axis);
        PYERR_RETURN_VAL(NULL);
        if (weights_obj == NULL) {
            result->arr = self->arr.sum(ax, Karray(Shape(), 1.0), true);
        } else {
            result->arr = self->arr.sum(ax, weights_obj->arr, true);
            PYERR_RETURN_VAL(NULL);
        }
    }

    return reinterpret_cast<PyObject *>(result);
}

