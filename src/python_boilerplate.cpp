

static PyMemberDef Karray_members[] = {
    {"attr", T_INT, offsetof(Karray, attr), 0,
     "Arbitrary attribute."},
    {NULL}  /* Sentinel */
};

static PyGetSetDef Karray_getsetters[] = {
    {"shape", (getter) Karray_getshape, NULL,
     "Shape of the array.", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef Karray_methods[] = {
    {"reshape", (PyCFunction) Karray_reshape, METH_O,
     "Return the kipr.arr with the new shape."},
    {"numpy", (PyCFunction) Karray_numpy, METH_NOARGS,
     "Return a numpy representtion of the Karray."},    
    {"execute", (PyCFunction)  execute_func, METH_NOARGS,
     "Testing function to execute C code."},
    {NULL}  /* Sentinel */
};


static PyMethodDef arraymodule_methods[] = {
    {"max_nd", max_nd, METH_NOARGS,
     "Get maximum number of dimensions for a kipr.arr() array."},
    {"execute", execute_func, METH_NOARGS,
     "Testing function to execute C code."},
    {"internal", internal_test, METH_NOARGS,
     "Execute C code tests."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyModuleDef arraymodule = {
    PyModuleDef_HEAD_INIT,
    "kipr_array",
    "Array backend.",
    -1,
    arraymodule_methods
};



static PyNumberMethods Karray_as_number = {
    .nb_add = Karray_add,
    .nb_subtract = Karray_sub,
    .nb_multiply = Karray_mul,

    .nb_inplace_add = Karray_inplace_add,
    .nb_inplace_subtract = Karray_inplace_sub,
    .nb_inplace_multiply = Karray_inplace_mul,

    .nb_true_divide = Karray_div,
    .nb_inplace_true_divide = Karray_inplace_div
};

static PyMappingMethods Karray_as_mapping = {
    .mp_subscript = Karray_subscript
};

static PyTypeObject KarrayType = {
    Karray_HEAD_INIT
    .tp_name = "kipr.arr",
    .tp_basicsize = sizeof(Karray) - sizeof(float),
    .tp_itemsize = sizeof(float),
    .tp_dealloc = (destructor) Karray_dealloc,
    .tp_repr = (reprfunc) Karray_str, // Not ideal
    .tp_as_number = &Karray_as_number,
    .tp_as_mapping = &Karray_as_mapping,
    .tp_str = (reprfunc) Karray_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Array object from kipr.",
    .tp_methods = Karray_methods,
    .tp_members = Karray_members,
    .tp_getset = Karray_getsetters,
    .tp_init = (initproc) Karray_init,
    .tp_new = Karray_new,
};

PyMODINIT_FUNC
PyInit_kipr_array(void)
{
    import_array();
    PyObject *m;
    if (PyType_Ready(&KarrayType) < 0)
        return NULL;

    m = PyModule_Create(&arraymodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&KarrayType);
    if (PyModule_AddObject(m, "arr", (PyObject *) &KarrayType) < 0) {
        Py_DECREF(&KarrayType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

