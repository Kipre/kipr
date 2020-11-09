#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>

#include <immintrin.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <random>

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

class Filter;
class NDVector;


class Shape
{
public:
    int nd;
    size_t length;

    Shape();
    template<typename T>
    Shape(T * input, int size = 8); 
    // Shape(size_t * input, int size = 8);
    Shape(PyObject * o, bool accept_singleton = false);
    void print(const char * message = "");
    bool assert_or_set(size_t value, int dim);
    size_t operator[](size_t i);
    size_t cohere();
    void write(size_t * destination);
    std::string str();
    size_t sum();
    NDVector strides(int depth_diff = 0);
    Filter broadcast_to(Shape other);

private:
    size_t buf[MAX_ND];

};

const size_t ERROR_MODE = 0;
const size_t RANDOM_UNIFORM = 1;
const size_t RANDOM_NORMAL = 2;
const size_t RANGE = 3;


const size_t NUMPY_ARRAY = 3;
const size_t STRING = 5;
const size_t NUMBER = 7;
const size_t SEQUENCE = 11;

class Karray
{
public:
    bool owned;
    int seed;
    Shape shape;
    float * data;

    Karray();
    Karray(Shape new_shape, std::vector<float> vec);
    Karray(Shape new_shape, float * new_data);
    // Karray(Shape new_shape, size_t mode) noexcept;
    void from_mode(Shape new_shape, size_t mode) noexcept;
    void from_numpy(PyObject * o) noexcept;
    ~Karray() noexcept;
    void print(const char * message = "");
    std::string str();
    void steal(Karray& other);
    void broadcast(Shape new_shape);
};

class Filter
{
public:
    bool failed = false;
    size_t offset[MAX_ND];
    size_t * buf;

    Filter() {};
    Filter(Shape& shape);
    ~Filter();
    void set_range_along_axis(int axis);
    void set_val_along_axis(int axis, size_t value);
    void print(const char * message = "");
    Filter& operator=(Filter && other) noexcept;
    Filter(Filter&& other) noexcept;
};

class NDVector
{
public:
    size_t buf[MAX_ND] = {};

    NDVector(size_t value) : buf{value} {};
    NDVector() {};

    size_t operator[](size_t i) {
        return buf[i];
    };

    void print(const char * message = "") {
        std::cout << "NDVector " << message << "\n\t";
        for (int k = 0; k < MAX_ND; ++k) {
            std::cout << buf[k] << ", ";
        }
        std::cout << '\n';
    };

    ~NDVector() {
        printf("destroying ndvector\n");
    };

    // NDVector(NDVector&& other) noexcept : buf{0} {
    //     *buf = *other.buf;
    //     *other.buf = nullptr;
    // };

    // NDVector& operator=(NDVector&& other) noexcept {
    //     if (this != &other) {
    //         *buf = *other.buf;
    //         *other.buf = nullptr;
    //     }
    //     return *this;
    // };
};

typedef struct {
    PyObject_HEAD
    Karray arr;
} PyKarray;

//utils
size_t read_mode(PyObject * o);

// members
void Karray_dealloc(PyKarray *self);
int Karray_init(PyKarray *self, PyObject *args, PyObject *kwds);
PyObject * Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
PyObject * Karray_str(PyKarray * self);
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


#define DEBUG_Obj(o, msg)  printf(msg); PyObject_Print(o, stdout, Py_PRINT_RAW); printf("\n");


#include "py_types.hpp"