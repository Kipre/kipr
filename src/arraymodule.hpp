#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>
#include <new>
#include <string>
// #include <vector>
#include <immintrin.h>

// Breakpoint
#include <windows.h>
#include <debugapi.h>

// To avoid c++ mixed designated initializers error
#define Karray_HEAD_INIT \
    .ob_base={.ob_base={1, NULL }, .ob_size=0},

#define MAX_NDIMS 8

#define Karray_IF_ERR_GOTO_FAIL \
    if (PyErr_Occurred()) { \
        PyErr_Print(); \
        goto fail; \
    }


int MAX_PRINT_SIZE = 30;
int STR_OFFSET = 10;


typedef struct {
    PyObject_HEAD
    int nd;
    Py_ssize_t shape [MAX_NDIMS];
    float * data;
    int attr;
} Karray;

// utility functions
Py_ssize_t Karray_length(Karray * self);
bool is_Karray(PyObject * obj);

// member functions
static void Karray_dealloc(Karray *self);
static PyObject * Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int Karray_init(Karray *self, PyObject *args, PyObject *kwds);
static PyObject *  Karray_str(Karray * self);

// math member functions
static PyObject * Karray_add(PyObject * self, PyObject * other);

// mapping methods
static PyObject* Karray_subscript(PyObject *o, PyObject *key);

// getters and setters
static PyObject * Karray_getshape(Karray *self, void *closure);

// array methods
static PyObject * Karray_numpy(Karray *self, PyObject *Py_UNUSED(ignored));
static PyObject * Karray_reshape(Karray *self, PyObject *shape);

// independent methods
static PyObject * max_nd(PyObject *self, PyObject *Py_UNUSED(ignored));
static PyObject * execute_func(PyObject *self, PyObject *Py_UNUSED(ignored));

// python overhead
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

static PyNumberMethods Karray_as_number = {
    .nb_add = Karray_add
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

static PyMethodDef arraymodule_methods[] = {
    {"max_nd",  max_nd, METH_NOARGS,
     "Get maximum number of dimensions for a kipr.arr() array."},
    {"execute",  execute_func, METH_NOARGS,
     "Testing function to execute C code."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyModuleDef arraymodule = {
    PyModuleDef_HEAD_INIT,
    "kipr_array",
    "Array backend.",
    -1,
    arraymodule_methods
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

