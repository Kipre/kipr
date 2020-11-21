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
#include <numeric>
#include <map>
#include <tuple>

// debugging bullshit
#ifdef _WIN32
#include <windows.h>
#include <debugapi.h>
#endif

#include "internal_test.hpp"


// To avoid c++ mixed designated initializers error
#define Karray_HEAD_INIT \
    .ob_base={.ob_base={1, NULL }, .ob_size=0},

const int MAX_ND = 8;
const char * KARRAY_NAME = "kipr.arr";

const int MAX_PRINT_SIZE = 100;
const int STR_OFFSET = 10;

PyObject* Karray_error;

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

#define VALERR_PRINT_GOTO_FAIL(msg) \
    PyErr_SetString(PyExc_ValueError, msg);\
    PyErr_Print();\
    goto fail;\

#define KERR_GOTO_FAIL(msg) \
    {PyErr_SetString(Karray_errorr, msg);\
    goto fail;}

#define KERR_RETURN(msg) \
    {PyErr_SetString(Karray_error, msg);\
    return;}


#define KERR_FORMAT_RETURN(...) \
    {PyErr_Format(Karray_error, __VA_ARGS__);\
    return;}

#define KERR_RETURN_VAL(msg, val) \
    {PyErr_SetString(Karray_error, msg);\
    return val;}

#define PYERR_GOTO_FAIL \
    if (PyErr_Occurred()) \
        goto fail;

#define PYERR_RETURN \
    if (PyErr_Occurred()) \
        return;

#define PYERR_SET_RETURN(msg) \
    if (PyErr_Occurred()) {\
        PyErr_Print(); \
        PyErr_SetString(Karray_error, msg);\
        return;}

#define PYERR_RETURN_VAL(val) \
    if (PyErr_Occurred()) \
        return val;


typedef float(*binary_op)(float, float);
typedef void(*binary_kernel)(float *, float *, float *, ssize_t);
typedef void(*binary_val_kernel)(float *, float *, float, ssize_t);
typedef void(*unary_kernel)(float *, float *, ssize_t);



class Filter;
class NDVector;


class Shape
{
public:
    bool def = true;
    int nd;
    size_t length;

    Shape();
    Shape(int ndims...);
    Shape(size_t * input, int size = 8);
    Shape(Py_ssize_t * input, int size);
    Shape(PyObject * o, size_t target_length = 0);
    Shape(Shape a, Shape b) noexcept;
    void swap(Shape &other);
    void print(const char * message = "") const;
    void set(int i, size_t val);
    bool assert_or_set(size_t value, int dim);
    size_t operator[](int i) const;
    bool operator==(Shape &other);
    size_t validate();
    void write(size_t * destination);
    std::string str() const;
    size_t sum();
    NDVector strides(int depth_diff = 0) const;
    NDVector broadcast_to(Shape & other);
    void push_back(size_t dim);
    void insert_one(int i);
    size_t pop(int i = -1) noexcept;
    size_t axis(PyObject * o);
    size_t axis(int ax);
    bool compatible_for_matmul(Shape & other);
    std::tuple<NDVector, NDVector> paired_strides(Shape b) noexcept;
    std::tuple<Shape, NDVector> transpose() const;
    size_t nbmats();

private:
    size_t buf[MAX_ND];
};

const size_t ERROR_MODE = 0;
const size_t RANDOM_UNIFORM = 1;
const size_t RANDOM_NORMAL = 2;
const size_t RANGE = 3;

const int NO_AXIS = 9;


const size_t NUMPY_ARRAY = 3;
const size_t STRING = 5;
const size_t NUMBER = 7;
const size_t SEQUENCE = 11;
const size_t SLICE = 13;
const size_t KARRAY = 17;

class Karray
{
public:
    int seed;
    Shape shape;
    float * data = nullptr;

    // structor
    Karray();
    Karray(float value);
    Karray(Shape new_shape, std::vector<float> vec);
    Karray(Shape new_shape, float * new_data);
    Karray(Shape new_shape, float value);
    Karray(Shape new_shape);
    ~Karray() noexcept;

    // copy and move
    Karray(const Karray& other);
    Karray& operator=(const Karray&);
    Karray(Karray&& other);
    Karray& operator=(Karray&&);
    void swap(Karray& other);

    // math
    Karray& operator+=(const Karray& other);
    Karray& operator/=(const Karray& other);
    Karray& operator*=(const Karray& other);
    Karray& operator-=(const Karray& other);
    Karray operator+(const Karray& rhs);
    Karray operator/(const Karray& rhs);
    Karray operator-(const Karray& rhs);
    Karray operator*(const Karray& rhs);

    void from_mode(Shape new_shape, size_t mode) noexcept;
    void from_numpy(PyObject * o) noexcept;
    void print(const char * message = "");
    std::string str();
    Karray broadcast(Shape new_shape);
    Karray subscript(PyObject * key);
    Karray matmul(Karray & other);
    Karray flat_sum(bool mean = false);
    Karray sum(size_t axis, const Karray &weights, bool mean = false);
    Karray elementwise_binary_op(const Karray &other, binary_kernel kernel, binary_op op);
    void inplace_binary_op(const Karray  &rhs, binary_kernel kernel, binary_op op);
};

class Filter
{
public:
    size_t offset[MAX_ND + 1];
    std::vector<size_t> vec;

    Filter() {
        offset[0] = 0;
    };
    Filter(Shape& shape);
    void set_range_along_axis(int axis);
    void set_val_along_axis(int axis, size_t value);
    void print(const char * message = "");
    Filter& operator=(Filter && other) noexcept;
    Filter(Filter&& other) noexcept;
    Shape from_subscript(PyObject * subscript, Shape &current_shape);
    void push_back(size_t number, int index);
};

class NDVector
{
public:
    size_t buf[MAX_ND] = {0};

    NDVector(size_t value) : buf{value} {};
    NDVector() {};

    size_t operator[](size_t i) const {
        return buf[i];
    };

    void print(const char * message = "") {
        std::cout << "NDVector " << message
                  << " " << str() << "\n";
    };

    std::string str() {
        std::ostringstream ss;
        for (int k = 0; k < MAX_ND; ++k) {
            ss << buf[k] << ", ";
        }
        return ss.str();
    };

    // ~NDVector() {
    //     // printf("destroying ndvector %s\n", str().c_str());
    // };
};

typedef struct {
    PyObject_HEAD
    Karray arr;
} PyKarray;


struct Positions {
    size_t write;
    size_t left;
    size_t right;
};

// utils
size_t read_mode(PyObject * o);
std::vector<PyObject *> full_subscript(PyObject * tuple, Shape& current_shape);
void _sum(float * self_data, float * result_data, float * weights_data,
          Shape &self_shape, NDVector &strides, bool multiple_weights,
          bool mean, int axis, int depth);
static std::tuple<Shape, NDVector, NDVector> paired_strides(Shape a, Shape b) noexcept;
void transpose(float * from, float * to, Positions * pos,
               Shape & shape, const NDVector& strides, int depth);

// members
void Karray_dealloc(PyKarray *self);
int Karray_init(PyKarray *self, PyObject *args, PyObject *kwds);
PyObject * Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
PyObject * Karray_str(PyKarray * self);
PyObject * Karray_subscript(PyObject *o, PyObject *key);

// getters and setters
PyObject * Karray_getshape(PyKarray *self, void *closure);
PyObject * Karray_getrefcnt(PyKarray *self, void *closure);

// member functions
PyObject * Karray_numpy(PyKarray *self, PyObject *Py_UNUSED(ignored));
// PyObject * Karray_val(PyKarray *self, PyObject *Py_UNUSED(ignored));
PyObject * Karray_reshape(PyKarray *self, PyObject *shape);
PyObject * Karray_broadcast(PyKarray *self, PyObject *o);
PyObject * Karray_mean(PyKarray *self, PyObject *args, PyObject *kwds);
PyObject * Karray_sum(PyKarray *self, PyObject *args, PyObject *kwds);
PyObject * Karray_transpose(PyObject *here, PyObject *Py_UNUSED(ignored));

// math
PyObject * Karray_add(PyObject * self, PyObject * other);
PyObject * Karray_inplace_add(PyObject * self, PyObject * other);
PyObject * Karray_sub(PyObject * self, PyObject * other);
PyObject * Karray_inplace_sub(PyObject * self, PyObject * other);
PyObject * Karray_mul(PyObject * self, PyObject * other);
PyObject * Karray_inplace_mul(PyObject * self, PyObject * other);
PyObject * Karray_div(PyObject * self, PyObject * other);
PyObject * Karray_inplace_div(PyObject * self, PyObject * other);
PyObject * Karray_matmul(PyObject * self, PyObject * other);
PyObject * Karray_negative(PyObject * self);

// module functions
PyObject * internal_test(PyObject *self, PyObject *Py_UNUSED(ignored));
PyObject * execute_func(PyObject *self, PyObject *Py_UNUSED(ignored));
PyObject * function_decorator(PyObject *self, PyObject *func);
PyObject * Karray_relu(PyObject *self, PyObject * o);
PyObject * Karray_exp(PyObject *self, PyObject * o);
PyObject * Karray_softmax(PyObject *module, PyObject *const *args, Py_ssize_t nargs);
PyObject * Karray_log(PyObject *self, PyObject * o);

// other
PyObject * cache_info(PyObject *self, PyObject * input);

#define DEBUG_Obj(o, msg)  printf(msg); PyObject_Print(o, stdout, Py_PRINT_RAW); printf("\n");


#include "py_types.hpp"