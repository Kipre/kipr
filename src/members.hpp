

void Karray_dealloc(PyKarray *self) {
    // printf("from python with refcount=%i\n", self->ob_base.ob_refcnt);
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

PyKarray *
new_PyKarray(Shape &shape) {
    PyKarray * result = reinterpret_cast<PyKarray *>(KarrayType.tp_alloc(&KarrayType, 0));
    result->arr = Karray(shape);
    return result;
}

PyKarray *
new_PyKarray(Shape &shape, float val) {
    PyKarray * result = reinterpret_cast<PyKarray *>(KarrayType.tp_alloc(&KarrayType, 0));
    result->arr = Karray(shape, val);
    return result;
}

PyKarray *
new_PyKarray(const Karray &arr) {
    PyKarray * result = reinterpret_cast<PyKarray *>(KarrayType.tp_alloc(&KarrayType, 0));
    result->arr = Karray(arr);
    return result;
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
        Py_INCREF(shape);
        proposed_shape = Shape(shape);
        IF_ERROR_RETURN(-1);
    }

    Py_INCREF(input);
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
        IF_ERROR_RETURN(-1);
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
        IF_ERROR_RETURN(-1);
        candidate = nest.to_Karray();
        if (shape) {
            candidate = candidate.broadcast(proposed_shape);
        }
    }
    break;
    default:
        PyErr_SetString(PyExc_TypeError,
                        "Input object not understood.");
    }
    IF_ERROR_RETURN(-1);

    self->arr.swap(candidate);

    Py_DECREF(input);
    Py_XDECREF(shape);
    return 0;

// fail:
//     Py_DECREF(input);
//     Py_XDECREF(shape);
//     PyErr_SetString(PyExc_TypeError,
//                     "Failed to initialize kipr.arr.");
//     return -1;
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
    IF_ERROR_RETURN(NULL);
    Py_DECREF(key);
    return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_reshape(PyKarray * self, PyObject * shape) {
    Shape new_shape(shape, self->arr.shape.length);
    IF_ERROR_RETURN(NULL);
    self->arr.shape = new_shape;

    auto result = reinterpret_cast<PyObject *>(self);
    Py_INCREF(result);
    return result;
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
    IF_ERROR_RETURN(NULL);
    auto result = new_PyKarray(self->arr.broadcast(new_shape));
    IF_ERROR_RETURN(NULL);
    return reinterpret_cast<PyObject *>(result);
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
        IF_ERROR_RETURN(NULL);
        if (weights_obj == NULL) {
            result->arr = self->arr.sum(ax, Karray(Shape(), 1.0));
        } else {
            result->arr = self->arr.sum(ax, weights_obj->arr);
            IF_ERROR_RETURN(NULL);
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
        IF_ERROR_RETURN(NULL);
        if (weights_obj == NULL) {
            result->arr = self->arr.sum(ax, Karray(Shape(), 1.0), true);
        } else {
            result->arr = self->arr.sum(ax, weights_obj->arr, true);
            IF_ERROR_RETURN(NULL);
        }
    }

    return reinterpret_cast<PyObject *>(result);
}

PyObject *Karray_transpose(PyObject *here, PyObject *Py_UNUSED(ignored)) {
    PyKarray * self = reinterpret_cast<PyKarray *>(here);
    auto [shape_t, strides_t] = self->arr.shape.transpose();
    PyKarray * result = new_PyKarray(shape_t);
    Positions pos {0, 0, 0};
    simple_transfer(self->arr.data, result->arr.data, &pos, shape_t, strides_t, 0);
    return reinterpret_cast<PyObject *>(result);
}
