#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>

#include <immintrin.h>
#include <iostream>

// debugging bullshit
#ifdef _WIN32 
 #include <windows.h> 
 #include <debugapi.h> 
#endif

// To avoid c++ mixed designated initializers error
#define Karray_HEAD_INIT \
    .ob_base={.ob_base={1, NULL }, .ob_size=0},

const int MAX_ND = 8;
const char * KARRAY_NAME = "kipr.arr";

const int MAX_PRINT_SIZE = 30;
const int STR_OFFSET = 10;

#define Karray_IF_ERR_GOTO_FAIL \
    if (PyErr_Occurred()) { \
        PyErr_Print(); \
        goto fail; \
    }

#define PYERR_PRINT_GOTO_FAIL \
    if (PyErr_Occurred()) { \
        PyErr_Print(); \
        goto fail; \
    }

#define PYERR_CLEAR_GOTO_FAIL \
    if (PyErr_Occurred()) { \
        PyErr_Clear(); \
        goto fail; \
    }

#define PYERR_CLEAR_CONTINUE \
    if (PyErr_Occurred()) { \
        PyErr_Clear(); \
    }

typedef struct {
    PyObject_HEAD
    size_t shape [MAX_ND];
    float * data;
} PyKarray;

// members
void Karray_dealloc(PyKarray *self);
int Karray_init(PyKarray *self, PyObject *args, PyObject *kwds);
PyObject * Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
// PyObject * Karray_str(PyKarray * self);
// PyObject * Karray_getshape(PyKarray *self, void *closure);
// PyObject * Karray_subscript(PyObject *o, PyObject *key);

// member functions
PyObject * Karray_numpy(PyKarray *self, PyObject *Py_UNUSED(ignored));
// PyObject * Karray_val(PyKarray *self, PyObject *Py_UNUSED(ignored));
PyObject * Karray_reshape(PyKarray *self, PyObject *shape);
PyObject * Karray_broadcast(PyKarray *self, PyObject *o);
PyObject * Karray_mean(PyKarray *self, PyObject *args, PyObject *kwds);
PyObject * Karray_sum(PyKarray *self, PyObject *args, PyObject *kwds);

// math
// PyObject * Karray_add(PyObject * self, PyObject * other);
// PyObject * Karray_inplace_add(PyObject * self, PyObject * other);
// PyObject * Karray_sub(PyObject * self, PyObject * other);
// PyObject * Karray_inplace_sub(PyObject * self, PyObject * other);
// PyObject * Karray_mul(PyObject * self, PyObject * other);
// PyObject * Karray_inplace_mul(PyObject * self, PyObject * other);
// PyObject * Karray_div(PyObject * self, PyObject * other);
// PyObject * Karray_inplace_div(PyObject * self, PyObject * other);
// PyObject * Karray_matmul(PyObject * self, PyObject * other);
// PyObject * Karray_negative(PyObject * self);

// module functions
PyObject * internal_test(PyObject *self, PyObject *Py_UNUSED(ignored));
PyObject * execute_func(PyObject *self, PyObject *Py_UNUSED(ignored));
// PyObject * max_nd(PyObject *self, PyObject *Py_UNUSED(ignored));
// PyObject * Karray_relu(PyObject *self, PyObject * o);
// PyObject * Karray_exp(PyObject *self, PyObject * o);
// PyObject * Karray_softmax(PyObject *self, PyObject * o);
// PyObject * Karray_log(PyObject *self, PyObject * o);


#define DEBUG_Obj(o)   PyObject_Print(o, stdout, Py_PRINT_RAW); printf("\n");

// #include "include/py_types.hpp"