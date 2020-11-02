PyObject *
execute_func(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    DEBUG_Obj(self);

    Py_ssize_t strides[MAX_NDIMS] = {};
    Karray * arr = reinterpret_cast<Karray *>(self);

    DEBUG_shape(arr->shape);

    float b = 0.0F;
    float a = 2.0F/b;

    printf("a: %i\n", a);

    Py_RETURN_NONE;
}



PyObject *
max_nd(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromLong(static_cast<long>(MAX_NDIMS));
}