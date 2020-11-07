
void
Karray_dealloc(Karray *self) {
    delete[] self->data;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *
Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Karray *self;
    self = reinterpret_cast<Karray *>(type->tp_alloc(type, 0));
    if (self != NULL) {
        self->nd = 1;
        reset_shape(self);
        self->data = new float[1];
        self->data[0] = 0.0;
    }
    return reinterpret_cast<PyObject *>(self);
}

int
Karray_init(Karray *self, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"data", "shape", NULL};
    PyObject *input = NULL, *shape = NULL;
    Py_ssize_t proposed_shape[MAX_NDIMS] = {0};
    bool random = false, range = false, value = false;
    float init_value;
    Py_ssize_t data_length = 0;
    unsigned int val;
    PyArrayObject* arr;
    float * arr_data;
    int nd;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", kwlist,
                                     &input, &shape))
        return -1;


    if (PyArray_Check(input) &&
        (nd = PyArray_NDIM((PyArrayObject *) input)) < MAX_NDIMS &&
        PyArray_TYPE((PyArrayObject *) input) == NPY_FLOAT) {
        arr = (PyArrayObject *) input;
        self->nd = nd;
        npy_intp * input_shape = PyArray_SHAPE(arr);
        for (int i=0; i<nd; ++i) {
            self->shape[i] = (Py_ssize_t) input_shape[i];
        }
        npy_intp length = PyArray_SIZE(arr);
        arr_data = (float *) PyArray_DATA(arr);
        delete[] self->data;
        self->data = new float[length];
        for (int k=0; k < length; ++k) {
            self->data[k] = arr_data[k];
        }
        data_length = (Py_ssize_t) length;
    } else if (PyUnicode_Check(input)) {
        if (PyUnicode_Compare(input, PyUnicode_FromString("random")) == 0) {
            random = true;
            self->data[0] = (float) rand() / (float) 32767;
        } else if (PyUnicode_Compare(input, PyUnicode_FromString("range")) == 0) {
            range = true;
        }
    } else if (PySequence_Check(input)) {
        data_length = Karray_init_from_data(self, input);
        Karray_IF_ERR_GOTO_FAIL;
    } else if (PyNumber_Check(input)) {
        PyObject *float_obj = PyNumber_Float(input);
        init_value = static_cast<float>(PyFloat_AsDouble(float_obj));
        Karray_IF_ERR_GOTO_FAIL;
        Py_DECREF(float_obj);
        self->data[0] = init_value;
        value = true;
    } else {
        PyErr_SetString(PyExc_TypeError, "Unsupported input data.");
        PyErr_Print();
        goto fail;
    }

    if (shape) {
        int proposed_nd = parse_shape(shape, proposed_shape);
        Karray_IF_ERR_GOTO_FAIL;

        Py_ssize_t proposed_length = product(proposed_shape, proposed_nd);

        // Check if the propsed makes sense with repect to data
        if (data_length != 0) {
            if (data_length == proposed_length) {
                self->nd = proposed_nd;
                set_shape(self, proposed_shape);
            } else {
                PyErr_SetString(PyExc_TypeError, 
                    "Proposed shape did not align with data.");
                PyErr_Print();
                goto fail;
            }
        } else {
            delete[] self->data;
            self->data = new float[proposed_length];
            if (value) {
                for (int k=0; k < proposed_length; ++k)
                    self->data[k] = init_value;
            } else if (range) {
                for (int k=0; k < proposed_length; ++k)
                    self->data[k] = (float) k;
            } else if (random) {
                for (int k=0; k < proposed_length; ++k) {
                    if (_rdrand32_step(&val) == 0) {
                        PyErr_SetString(PyExc_SystemError, 
                            "Could not generate a random value.");
                        PyErr_Print();
                        goto fail;
                    }
                    self->data[k] = (float)((double) val / (double) 4294967295);
                }
            }
            self->nd = proposed_nd;
            set_shape(self, proposed_shape);
        }
    } else if (range){
        PyErr_SetString(PyExc_TypeError, 
            "A shape must be provided when using \"range\" magic.");
        PyErr_Print();
        goto fail;
    }
    return 0;

    fail:
        Py_XDECREF(shape);
        Py_DECREF(input);
        PyErr_SetString(PyExc_TypeError, "Failed to build the array.");
        return -1;
}


PyObject *
Karray_str(Karray * self) {
    Py_ssize_t index[MAX_NDIMS] = {0};
    std::string result = "kipr.arr(" + str(self, index);
    result.pop_back();
    result += '\n';
    result += std::string(STR_OFFSET - 1, ' ') + "shape=" + shape_str(self);
    result += ')';
    return PyUnicode_FromString(result.c_str());
}

PyObject *
Karray_getshape(Karray *self, void *closure) {
    PyObject * result = PyTuple_New(self->nd);
    for (int k=0; k < self->nd; k++) {
        PyTuple_SET_ITEM(result, k, PyLong_FromSsize_t(self->shape[k]));
    }
    return result;
}

// int
// Karray_setshape(Karray *self, PyObject *value, void *closure)
// {
//     PyErr_SetString(PyExc_TypeError,
//         "Shape is not settable, use reshape instead.");
//     return -1;
// }


PyObject *
Karray_numpy(Karray *self, PyObject *Py_UNUSED(ignored)) {
    npy_intp * dims = new npy_intp[self->nd];
    for (int k=0; k < self->nd; k++) {
        dims[k] = (npy_intp) self->shape[k];
    }
    Py_INCREF(self);
    return PyArray_SimpleNewFromData(self->nd, dims, NPY_FLOAT, self->data);
}

PyObject *
Karray_reshape(Karray *self, PyObject *shape) {
    Py_ssize_t proposed_shape[MAX_NDIMS] = {1};

    int proposed_nd = parse_shape(shape, proposed_shape);
    if (proposed_nd < 1) {
        PyErr_Print();
        PyErr_SetString(PyExc_TypeError, "Failed to reshape array.");
        return NULL;
    }
    if (Karray_length(self) == product(proposed_shape, proposed_nd)) {
        set_shape(self, proposed_shape);
        self->nd = proposed_nd;
        Py_INCREF(self);
        return reinterpret_cast<PyObject *>(self);
    } else {
        PyErr_SetString(PyExc_TypeError,
            "Proposed shape doesn't align with data.");
        return NULL;
    }
}

PyObject * 
Karray_subscript(PyObject *o, PyObject *key) {
    Karray * self = reinterpret_cast<Karray *>(o);
    Karray * result = new_Karray();
    Py_ssize_t offsets[MAX_NDIMS] = {};
    Py_ssize_t result_length;

    Py_ssize_t nb_indices = sum(self->shape, self->nd);
    Py_ssize_t * filters = new Py_ssize_t[nb_indices];
    for (int k=0; k < nb_indices; k++) {
        filters[k] = -1;
    }

    Py_INCREF(key);
    if (!PyTuple_Check(key))
        key = Py_BuildValue("(O)", key);

    make_filter(key, self, result, filters);
    Karray_IF_ERR_GOTO_FAIL;

    // DEBUG_carr(filters, nb_indices, "filter");
    // DEBUG_Karr(result, "result");

    delete[] result->data;
    result_length = Karray_length(result);
    result->data = new float[result_length];


    filter_offsets(self->shape, offsets);

    transfer_data(self->nd, self->shape, self->data, 
                  result, filters, offsets);

    return reinterpret_cast<PyObject *>(result);

    fail:
        PyErr_SetString(PyExc_IndexError, "Failed to apply subscript.");
        return NULL;
}

PyObject * 
Karray_broadcast(Karray *self, PyObject *o) {
    Py_ssize_t shape[MAX_NDIMS] = {};
    Karray *result;
    parse_shape(o, shape);
    Karray_IF_ERR_GOTO_FAIL;

    result = broadcast(self, shape);
    Karray_IF_ERR_GOTO_FAIL;

    return reinterpret_cast<PyObject *>(result);

    fail:
        PyErr_SetString(PyExc_TypeError, 
            "Failed to apply broadcast, input shape is probably not coherent.");
        return NULL;
}


void
inline sum(float * self_data, float * result_data, float * weights_data,
           Py_ssize_t * self_shape, Py_ssize_t * strides,
           int axis, int depth = 0) {
    if (axis != depth) {
        for (int k=0; k < self_shape[depth]; ++k) {
            sum(self_data + strides[depth]*k, result_data + strides[depth]*k/self_shape[axis], 
                weights_data, self_shape, strides, axis, depth + 1);
        }
    } else {
        for (int i=0; i < self_shape[axis]; ++i) {
            for (int k=0; k < strides[axis]; ++k) {
                // printf("val and result: %f %f %i %i\n", self_data[strides[axis] * i + k], result_data[k], strides[axis] * i + k, i);
                result_data[k] += self_data[strides[axis] * i + k] * weights_data[i];
            }
        }
    }
}



PyObject *
Karray_mean(Karray *self, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"axis", "weights", NULL};

    PyObject *axis = NULL, *weights_obj = NULL;
    Py_ssize_t output_shape[MAX_NDIMS] = {};
    Py_ssize_t strides[MAX_NDIMS] = {};
    Py_ssize_t fake_shape[MAX_NDIMS] = {};
    Karray * result, * weights;
    Py_ssize_t reduction, ax, stride = 1;
    bool weights_passed;


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O$O", kwlist,
                                     &axis, &weights_obj))
        return NULL;

    Karray * karr = reinterpret_cast<Karray *>(self);

    ax = Axis(karr->nd, axis).value;
    PYERR_PRINT_GOTO_FAIL;

    if (ax == -1) {
        output_shape[0] = 1;
        reduction = Karray_length(karr);
        strides[0] = 1;
        fake_shape[0] = reduction;
        ax = 0;
    } else {
        copy_shape(karr->shape, output_shape);
        reduction = shape_pop(output_shape, ax);
        get_strides(karr->nd, karr->shape, strides);
        copy_shape(karr->shape, fake_shape);
    }

    result = new_Karray_from_shape(output_shape, 0);

    if (weights_obj) {
        weights_passed = true;
        if (!is_Karray(weights_obj)) {
            PyErr_SetString(PyExc_TypeError, 
                "Weights should be a <kipr.arr>.");
            goto fail;
        } else {
            weights = reinterpret_cast<Karray *>(weights_obj);
            if (Karray_length(weights) != reduction) {
                PyErr_SetString(PyExc_TypeError, 
                    "Weights not compatible with reduction.");
                goto fail;
            }

        } 
    } else {
        Py_ssize_t weights_shape[MAX_NDIMS] = {reduction};
        weights = new_Karray_from_shape(weights_shape, 1);
    }

    sum(karr->data, result->data, weights->data,
        fake_shape, strides, ax);


    for (int k=0; k < Karray_length(result); ++k) {
        result->data[k] /= (float) reduction;
    }


    if (!weights_passed)
        Py_XDECREF(weights);
    return reinterpret_cast<PyObject *>(result);

    fail:
        if (!weights_passed)
            Py_XDECREF(weights);
        Py_XDECREF(result);
        return NULL;
}



PyObject *
Karray_sum(Karray *self, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"axis", "weights", NULL};

    PyObject *axis = NULL, *weights_obj = NULL;
    Py_ssize_t output_shape[MAX_NDIMS] = {};
    Py_ssize_t strides[MAX_NDIMS] = {};
    Py_ssize_t fake_shape[MAX_NDIMS] = {};
    Karray * result, * weights;
    Py_ssize_t reduction, ax, stride = 1;
    bool weights_passed;


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O$O", kwlist,
                                     &axis, &weights_obj))
        return NULL;

    Karray * karr = reinterpret_cast<Karray *>(self);

    ax = Axis(karr->nd, axis).value;
    PYERR_PRINT_GOTO_FAIL;

    if (ax == -1) {
        output_shape[0] = 1;
        reduction = Karray_length(karr);
        strides[0] = 1;
        fake_shape[0] = reduction;
        ax = 0;
    } else {
        copy_shape(karr->shape, output_shape);
        reduction = shape_pop(output_shape, ax);
        get_strides(karr->nd, karr->shape, strides);
        copy_shape(karr->shape, fake_shape);
    }

    result = new_Karray_from_shape(output_shape, 0);

    if (weights_obj) {
        weights_passed = true;
        if (!is_Karray(weights_obj)) {
            PyErr_SetString(PyExc_TypeError, 
                "Weights should be a <kipr.arr>.");
            goto fail;
        } else {
            weights = reinterpret_cast<Karray *>(weights_obj);
            if (Karray_length(weights) != reduction) {
                PyErr_SetString(PyExc_TypeError, 
                    "Weights not compatible with reduction.");
                goto fail;
            }

        } 
    } else {
        Py_ssize_t weights_shape[MAX_NDIMS] = {reduction};
        weights = new_Karray_from_shape(weights_shape, 1);
    }

    sum(karr->data, result->data, weights->data,
        fake_shape, strides, ax);


    if (!weights_passed)
        Py_XDECREF(weights);
    return reinterpret_cast<PyObject *>(result);

    fail:
        if (!weights_passed)
            Py_XDECREF(weights);
        Py_XDECREF(result);
        return NULL;
}


PyObject *
Karray_val(Karray *self, PyObject *Py_UNUSED(ignored)) {
    if (Karray_length(self) != 1) {
        PyErr_SetString(PyExc_TypeError, 
            "Val method called on a non scalar array.");
        return NULL;
    }
    return PyFloat_FromDouble((double) self->data[0]);
}