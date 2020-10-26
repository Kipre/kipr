#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>
#include <new>
#include <string>
#include <immintrin.h>

#define MAX_NDIMS 8

int MAX_PRINT_SIZE = 30;
int STR_OFFSET = 10;

typedef struct {
    PyObject_HEAD
    int nd;
    int shape [MAX_NDIMS];
    float * data;
    int attr;
} Karray;

int product(int * arr, int len, int increment=0) {
    int result = 1;
    while (len >  0) result *= arr[--len] + increment;
    return result;
}

// Infer data length from shape
int data_theo_length(Karray *self) {
    return product(self->shape, self->nd);
}

void set_shape(Karray *self, int * shape) {
    for (int k=0; k<MAX_NDIMS; k++) {
        self->shape[k] = shape[k];
    }
}

void reset_shape(Karray *self) {
    int shape [MAX_NDIMS] = {1};
    set_shape(self, shape);
}

// DEBUG FUNCTION
void print_arr(Karray * self, char * message) {

    printf("%s\n", message);
    printf("\tnumber of dimensions: %i\n", self->nd);
    printf("\tshape: ");
    if (self->nd > 0 && self->nd < MAX_NDIMS) {
        for (int k=0; k<self->nd + 2; k++) {
            printf(" %i,", self->shape[k]);
        }
    }
    printf("\n");
    int length = data_theo_length(self);
    printf("\tdata theoretical length: %i\n", length);
    if (length < 1000) {
        printf("\tdata: ");
        for (int k=0; k<length; k++) {
            printf(" %f,", self->data[k]);
        }
        printf("\n");
    }
} 

static void
Karray_dealloc(Karray *self)
{
    delete[] self->data;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Karray *self;
    self = (Karray *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->nd = 1;
        reset_shape(self);
        self->data = new float[1];
        self->data[0] = 0.0;
    }
    return (PyObject *) self;
}

static int infer_shape(PyObject * input, int * shape, int depth=0) 
{
    if (depth > MAX_NDIMS) {
        PyErr_SetString(PyExc_TypeError, "Input data too deep.");
        return -1;
    }
    if (PySequence_Check(input)) {
        int length = PySequence_Length(input);
        if (length > 0) {
            shape[depth] = length;
            PyObject *item = PySequence_GetItem(input, 0);
            if (infer_shape(item, shape, depth + 1) < 0) {
                return -1;
            }
            Py_DECREF(item);
        } else {
            PyErr_SetString(PyExc_TypeError, "Failed to infer shape from data.");
            return -1;
        }
    }
    if (depth == 0) {
        int ndims = 0;
        while (shape[++ndims] != 0 && ndims<MAX_NDIMS);
        return ndims;
    }
    return 1;
}

static bool is_scalar(Karray * self) {
    return (self->nd == 1) && (self->shape[0] == 1);
}

static int copy_data(PyObject * input, int * shape, float * result, int depth=0, int position=0)
{   
    if (shape[depth] > 0 && PySequence_Check(input)) {
        for (int k=0; k<shape[depth]; k++) {
            PyObject *item = PySequence_GetItem(input, k);
            if (PyErr_Occurred()) {
                PyErr_SetString(PyExc_TypeError, "Input data should be well formed.");
                return -1;
            }
            position = copy_data(item, shape, result, depth + 1, position);
            if (position == -1) {
                return -1;
            }
            Py_DECREF(item);
        }
    } else if (PyNumber_Check(input)) {
        PyObject *float_obj = PyNumber_Float(input);
        float scalar = (float) PyFloat_AsDouble(float_obj);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Could not read input data as floats.");
            return -1;
        }
        result[position++] = scalar;
        Py_DECREF(float_obj);
    } else {
        PyErr_SetString(PyExc_TypeError, "Data should be numerical.");
        return -1;
    }
    return position;
}

static int parse_shape(PyObject * sequence, int * shape) {
    if (PySequence_Check(sequence)) {
        int nd = PySequence_Length(sequence);
        if (nd < 1) {
            PyErr_SetString(PyExc_TypeError, "Shape must have at least one element.");
            return -1;
        }
        for (int k=0; k<nd; k++) {
            PyObject * element = PySequence_GetItem(sequence, k);
            shape[k] = (int) PyLong_AsLong(element);
            if (PyErr_Occurred() || shape[k] == 0) {
                PyErr_SetString(PyExc_TypeError, "Shape must ba a sequence of non-zero integers.");
                return -1;
            }
            Py_DECREF(element);
        }
        return nd;
    } else {
        PyErr_SetString(PyExc_TypeError, "Shape must be a sequence.");
        return -1;
    } 
}

static int
Karray_init(Karray *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"data", "shape", NULL};
    PyObject *input = NULL, *shape = NULL;
    int data_length = 0, proposed_nd = 0;
    int proposed_shape[MAX_NDIMS] = {0};


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", kwlist,
                                     &input, &shape))
        return -1;

    
    int inferred_shape[MAX_NDIMS] = {1};
    self->nd = infer_shape(input, inferred_shape);
    if (PyErr_Occurred()) {
        PyErr_Print();
        goto fail;
    }
    set_shape(self, inferred_shape);
    data_length = data_theo_length(self);
    delete[] self->data;
    self->data = new float[data_length];
    int final_position = copy_data(input, self->shape, self->data);
    if (data_length != final_position) {
        if (PyErr_Occurred()) {
            PyErr_Print();
        } else {
            PyErr_SetString(PyExc_TypeError, "Data read failed.");
        }
        goto fail;
    }

    if (shape) {
        int proposed_nd = parse_shape(shape, proposed_shape);
        if (PyErr_Occurred()) {
            PyErr_Print(); 
            goto fail;
        }
        int proposed_length = product(proposed_shape, proposed_nd);

        // Check if the propsed makes sense with repect to data
        if (data_theo_length(self) != proposed_length) {
            // If it doesn't but data is scalar then we can replicate the value
            if (is_scalar(self)) {
                float current_value = self->data[0];
                delete[] self->data;
                self->data = new float [proposed_length];
                for (int k=0; k<proposed_length; k++) {
                    self->data[k] = current_value;
                }
            } else {
                PyErr_SetString(PyExc_TypeError, "Proposed shape doesn't align with data.");
                goto fail;
            }
        } else {
            self->nd = proposed_nd;
            set_shape(self, proposed_shape);   
        }        
    }
    return 0;

    fail:
        Py_XDECREF(shape);
        Py_DECREF(input);
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "Failed to build array.");
        return -1;
}

int offset(Karray * self, int * index) {
    int k=0;
    int result = 0;
    while (k<self->nd-1) {
        result = (result + index[k])*self->shape[k+1];
        ++k;
    }
    result += index[k];
    return result;
} 

std::string str(Karray * self, int * index, int depth=0, bool repr=false) 
{
    std::string result = "[";
    int current_offset = offset(self, index);

    if (depth < self->nd && (current_offset < MAX_PRINT_SIZE) && !repr) {
        for (int k=0; k<self->shape[depth]; k++) {
            index[depth] = k;
            if (k != 0 && depth != self->nd-1) {
                result += std::string(depth + STR_OFFSET, ' ');
            }
            std::string sub = str(self, index, depth + 1);
            result += sub;
            if (sub == "....") return result;
        } 
        //remove last newline and comma
        if (depth != self->nd) {
            result.pop_back(); result.pop_back();
        }
        return result + "],\n";
    } else if (current_offset < MAX_PRINT_SIZE && !repr) {
        return " " + std::to_string(self->data[current_offset]) + ",";
    } else {
        return std::string("....");
    }
}

std::string shape_str(Karray * self) {
    std::string result = "[";
    for (int k=0; k<self->nd; k++) {
        result += " " + std::to_string(self->shape[k]) + ",";
    }
    result.pop_back();
    return result + "]";
}

static PyObject * 
Karray_str(Karray * self) 
{   
    int index [MAX_NDIMS] = {0};
    std::string result = "kipr.arr(" + str(self, index);
    result.pop_back();
    result += '\n';
    result += std::string(STR_OFFSET, ' ') + "shape=" + shape_str(self);
    result += ')';
    return PyUnicode_FromString(result.c_str());
}


static PyMemberDef Karray_members[] = {
    {"attr", T_INT, offsetof(Karray, attr), 0,
     "Arbitrary attribute."},
    {NULL}  /* Sentinel */
};

static PyObject *
Karray_getshape(Karray *self, void *closure)
{
    PyObject * result = PyTuple_New(self->nd);
    for (int k=0; k<self->nd; k++) {
        PyTuple_SET_ITEM(result, k, PyLong_FromLong(self->shape[k]));
    }
    return result;
}

static int
Karray_setshape(Karray *self, PyObject *value, void *closure)
{
    PyErr_SetString(PyExc_TypeError, "Shape is not settable, use reshape instead.");
    return -1;
}

static PyGetSetDef Karray_getsetters[] = {
    {"shape", (getter) Karray_getshape, (setter) Karray_setshape,
     "Shape of the array.", NULL},
    {NULL}  /* Sentinel */
};

static PyObject *
Karray_numpy(Karray *self, PyObject *Py_UNUSED(ignored))
{   
    npy_intp * dims = new npy_intp[self->nd];
    for (int k=0; k<self->nd; k++) {
        dims[k] = (npy_intp) self->shape[k];
    }
    Py_INCREF(self);
    return PyArray_SimpleNewFromData(self->nd, dims, NPY_FLOAT, self->data);
}

static PyObject *
Karray_reshape(Karray *self, PyObject *args)
{
    PyObject *shape;
    int proposed_shape [MAX_NDIMS] = {1};

    if (!PyArg_ParseTuple(args, "O", &shape))
        return NULL;

    int proposed_nd = parse_shape(shape, proposed_shape);
    if (proposed_nd < 1) {
        return NULL;
    }
    if (data_theo_length(self) == product(proposed_shape, proposed_nd)) {
        set_shape(self, proposed_shape);
        return (PyObject *) self;
    } else {
        PyErr_SetString(PyExc_TypeError, "Proposed shape doesn't align with data.");
        return NULL;
    }
}

static PyObject *
max_nd(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromLong((long) MAX_NDIMS);
}

static PyMethodDef arraymodule_methods[] = {
    {"max_nd",  max_nd, METH_NOARGS,
     "Get maximum number of dimentions for a kipr.arr() array."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyMethodDef Karray_methods[] = {
    {"reshape", (PyCFunction) Karray_reshape, METH_VARARGS,
     "Return the kipr.arr with the new shape."},
    {"numpy", (PyCFunction) Karray_numpy, METH_NOARGS,
     "Return a numpy representtion of the Karray."
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject KarrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "kipr.arr",                     /* tp_name */
    sizeof(Karray),                 /* tp_basicsize */
    0,                              /* tp_itemsize */
    (destructor) Karray_dealloc,    /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    0,                              /* tp_repr */
    0,                              /* tp_as_number */
    0,                              /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    0,                              /* tp_hash */
    0,                              /* tp_call */
    (reprfunc) Karray_str,          /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,/* tp_flags */
    "Array object from kipr.",      /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    Karray_methods,                 /* tp_methods */
    Karray_members,                 /* tp_members */
    Karray_getsetters,              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc) Karray_init,         /* tp_init */
    0,                              /* tp_alloc */
    Karray_new,                     /* tp_new */
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