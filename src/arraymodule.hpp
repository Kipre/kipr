#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>

#include <immintrin.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

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

class Shape
{
public:
    int nd = 1;
    size_t length = 1;
    size_t values[MAX_ND] = {1};

    Shape() {};
    Shape(size_t * input) {
        nd = 0;
        length = 1;
        while(input[nd] != 0 && nd < MAX_ND) {
            length *= input[nd];
            values[nd] = input[nd];
            ++nd;
        }
        int i = nd;
        while(i < MAX_ND) {
            values[i++] = 0;
        }
    };

    ~Shape() = default;

    void print(const char * message = "") {
        std::cout << "Shape " << message << 
        " nd=" << nd << ", length=" << length << "\n\t";
        for (int k=0; k < MAX_ND; ++k) {
            std::cout << values[k] << ", ";
        }
        std::cout << '\n';
    }

    bool assert_or_set(size_t value, int dim) {
        if ((dim == 0 && values[0] == 1) ||
            values[dim] == 0) {
            values[dim] = value;
            return true;
        } else if (values[dim] == value) {
            return true;
        } else {
            return false;
        }
    }

    size_t cohere() {
        nd = 0;
        length = 1;
        while(values[nd] != 0 && nd < MAX_ND) {
            length *= values[nd];
            ++nd;
        }
        int i = nd;
        while(i < MAX_ND) {
            if (values[i++]) {
                PyErr_SetString(PyExc_ValueError, "Shape is corrupted.");
                return 0;
            }
        }
        return length;
    }

    void write(size_t * destination) {
        for (int i=0; i < MAX_ND; ++i) {
            destination[i] = values[i];
        }
    }
    
};



class Karray
{
public:
    bool owned;
    Shape shape;
    float * data;

    Karray() {
        owned = true;
        shape = Shape();
        data = new float[1];
        data[0] = 0;
    };

    Karray(PyObject * self) {
        PyKarray * karr = reinterpret_cast<PyKarray *>(self);
        shape = Shape(karr->shape);
        data = karr->data;
        owned = false;
    };

    Karray(Shape new_shape, std::vector<float> vec) {
        owned = false;
        shape = new_shape;
        data = vec.data();
    };

    void bind(PyKarray * self) {
        if (self->data)
            delete[] self->data;
        shape.write(self->shape);
        self->data = data;
        owned = false;
    }

    ~Karray() {
        if (owned)
            delete[] data;
    };

    void print(const char * message = "") {
        std::cout << "Printing Karray " << message << "\n\t";
        shape.print();
        std::cout << "\t";
        auto limit = min(shape.length, 30);
        for (int i=0; i < limit; ++i) {
            std::cout << data[i] << ", ";
        }
        std::cout << "\n";
    }
    
};

#include "include/py_types.hpp"