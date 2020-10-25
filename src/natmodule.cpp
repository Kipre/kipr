#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *
test_impl(PyObject *self, PyObject *args)
{
    int a, b;

    if (!PyArg_ParseTuple(args, "ii", &a, &b))
        return NULL;
    
    return PyLong_FromLong(a + b);
}

static PyMethodDef NatMethods[] = {
    {"test",  test_impl, METH_VARARGS,
     "Bla bla bla."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef natmodule = {
    PyModuleDef_HEAD_INIT,
    "kipr_nat",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    NatMethods
};

PyMODINIT_FUNC
PyInit_kipr_nat(void)
{
    return PyModule_Create(&natmodule);
}