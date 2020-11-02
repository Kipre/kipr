#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>

// #include <new>
#include <string>
// #include <iostream>
// #include <cstdarg>  // for varargs
#include <immintrin.h>

// debugging bullshit
#ifdef _WIN32 
 #include <windows.h> 
 #include <debugapi.h> 
#endif

// To avoid c++ mixed designated initializers error
#define Karray_HEAD_INIT \
    .ob_base={.ob_base={1, NULL }, .ob_size=0},

#define MAX_NDIMS 8
#define STR_OFFSET 10

#define Karray_IF_ERR_GOTO_FAIL \
    if (PyErr_Occurred()) { \
        PyErr_Print(); \
        goto fail; \
    }

typedef struct {
    PyObject_HEAD
    int nd;
    Py_ssize_t shape [MAX_NDIMS];
    float * data;
    int attr;
} Karray;

// utils
Py_ssize_t Karray_length(Karray *self);

// members
void Karray_dealloc(Karray *self);
int Karray_init(Karray *self, PyObject *args, PyObject *kwds);
PyObject * Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
PyObject * Karray_str(Karray * self);
PyObject * Karray_getshape(Karray *self, void *closure);
PyObject * Karray_numpy(Karray *self, PyObject *Py_UNUSED(ignored));
PyObject * Karray_reshape(Karray *self, PyObject *shape);
PyObject * Karray_subscript(PyObject *o, PyObject *key);

// math
PyObject * Karray_add(PyObject * self, PyObject * other);
PyObject * Karray_inplace_add(PyObject * self, PyObject * other);
PyObject * Karray_sub(PyObject * self, PyObject * other);
PyObject * Karray_inplace_sub(PyObject * self, PyObject * other);
PyObject * Karray_mul(PyObject * self, PyObject * other);
PyObject * Karray_inplace_mul(PyObject * self, PyObject * other);
PyObject * Karray_div(PyObject * self, PyObject * other);
PyObject * Karray_inplace_div(PyObject * self, PyObject * other);

// module functions
PyObject * internal_test(PyObject *self, PyObject *Py_UNUSED(ignored));
PyObject * execute_func(PyObject *self, PyObject *Py_UNUSED(ignored));
PyObject * max_nd(PyObject *self, PyObject *Py_UNUSED(ignored));