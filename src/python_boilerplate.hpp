static PyMemberDef Karray_members[] = {
    // {"attr", T_INT, offsetof(PyKarray, attr), 0,
    //  "Arbitrary attribute."},
    {NULL}  /* Sentinel */
};

static PyGetSetDef Karray_getsetters[] = {
    {"refcnt", (getter) Karray_getrefcnt, NULL,
     "Python refcount of the object.", NULL},
    {"shape", (getter) Karray_getshape, NULL,
     "Shape of the array.", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef Karray_methods[] = {
    {"reshape", (PyCFunction) Karray_reshape, METH_O,
     "Return the kipr.arr with the new shape."},
    {"broadcast", (PyCFunction) Karray_broadcast, METH_O,
     "Return the kipr.arr with the breadcasted shape."},
    {"mean", (PyCFunction) Karray_mean, METH_VARARGS | METH_KEYWORDS,
     "Return the averaged array."},
    {"sum", (PyCFunction) Karray_sum, METH_VARARGS | METH_KEYWORDS,
     "Return the sum of the array along all or a particular dim."},
    {"numpy", (PyCFunction) Karray_numpy, METH_NOARGS,
     "Return a numpy representtion of the PyKarray."},
    {"execute", (PyCFunction)  execute_func, METH_O,
     "Testing function to execute C code."},
    {"transpose", (PyCFunction)  Karray_transpose, METH_NOARGS,
     "Get the transpose of <kipr.arr>."},
    {NULL}  /* Sentinel */
};



static PyMethodDef Graph_methods[] = {
    {"compile", (PyCFunction)  Graph_prepare,  METH_FASTCALL,
     "Compile graph."},
    {NULL}  /* Sentinel */
};


static PyGetSetDef Graph_getsetters[] = {
    {NULL}  /* Sentinel */
};

static PyMemberDef Graph_members[] = {
    // {"attr", T_INT, offsetof(PyKarray, attr), 0,
    //  "Arbitrary attribute."},
    {NULL}  /* Sentinel */
};

static PyMethodDef arraymodule_methods[] = {
    // {"max_nd", max_nd, METH_NOARGS,
    //  "Get maximum number of dimensions for a kipr.arr() array."},
    {"execute", execute_func, METH_O,
     "Testing function to execute C code."},
    {"function", function_decorator, METH_O,
     "Function decorator."},
    {"internal_test", internal_test, METH_NOARGS,
     "Execute C/C++ side tests."},
    {"relu", Karray_relu, METH_O,
     "ReLU function for <kipr.arr> arrays."},
    {"exp", Karray_exp, METH_O,
     "Exponential function for <kipr.arr> arrays."},
    {"softmax", (PyCFunction) Karray_softmax, METH_FASTCALL,
     "Softmax function for <kipr.arr> arrays, computes along the last axis."},
    {"log", Karray_log, METH_O,
     "Log function for <kipr.arr> arrays."},
    {"cache_info", cache_info, METH_NOARGS,
     "Function to query CPU info about cache configuration."},
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

    .nb_negative = Karray_negative,

    .nb_inplace_add = Karray_inplace_add,
    .nb_inplace_subtract = Karray_inplace_sub,
    .nb_inplace_multiply = Karray_inplace_mul,

    .nb_true_divide = Karray_div,
    .nb_inplace_true_divide = Karray_inplace_div,

    .nb_matrix_multiply = Karray_matmul
};

static PyMappingMethods Karray_as_mapping = {
    .mp_subscript = Karray_subscript
};

static PyTypeObject KarrayType = {
    Karray_HEAD_INIT
    .tp_name = KARRAY_NAME,
    .tp_basicsize = sizeof(PyKarray) - sizeof(float),
    .tp_itemsize = sizeof(float),
    .tp_dealloc = (destructor) Karray_dealloc,
    .tp_repr = (reprfunc) Karray_str, 
    .tp_as_number = &Karray_as_number,
    .tp_as_mapping = &Karray_as_mapping,
    .tp_str = (reprfunc) Karray_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Array object from kipr",
    .tp_methods = Karray_methods,
    .tp_members = Karray_members,
    .tp_getset = Karray_getsetters,
    .tp_init = (initproc) Karray_init,
    .tp_new = Karray_new,
};



static PyTypeObject GraphType = {
    Karray_HEAD_INIT
    .tp_name = "kipr.graph",
    .tp_basicsize = sizeof(PyGraph),
    .tp_dealloc = (destructor) Graph_dealloc,
    .tp_repr = (reprfunc) Graph_str, 
    .tp_str = (reprfunc) Graph_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Graph object from kipr",
    .tp_methods = Graph_methods,
    .tp_members = Graph_members,
    .tp_getset = Graph_getsetters,
    .tp_init = (initproc) Graph_init,
    .tp_new = Graph_new,
};

PyMODINIT_FUNC
PyInit_kipr_array(void)
{
    Karray_error = PyErr_NewException("kipr.KarrayError", NULL, NULL);
    import_array();
    PyObject *m;
    if (PyType_Ready(&KarrayType) < 0)
        return NULL;
    if (PyType_Ready(&GraphType) < 0)
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

    Py_INCREF(&GraphType);
    if (PyModule_AddObject(m, "graph", (PyObject *) &GraphType) < 0) {
        Py_DECREF(&GraphType);
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "KarrayError", Karray_error) < 0) {
        Py_DECREF(Karray_error);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}