

void
Karray_dealloc(PyKarray *self) {
    DEBUG_Obj((PyObject *) self, "deallocating from python")
    // self->arr.print("deallocating from python");
    self->arr.~Karray();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *
Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyKarray *self;
    self = reinterpret_cast<PyKarray *>(type->tp_alloc(type, 0));
    // if (self != NULL) {
    //     self->arr = Karray();
    // }
    return reinterpret_cast<PyObject *>(self);
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
        candidate.steal(nest.to_Karray());
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

    self->arr.steal(candidate);
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


    PyObject * out = Karray_new(&KarrayType, NULL, NULL);

    return out;
}

PyObject *
Karray_str(PyKarray * self) {
    return PyUnicode_FromString(self->arr.str().c_str());
}
