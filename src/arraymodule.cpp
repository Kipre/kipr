#include "arraymodule.hpp"

static PyMemberDef Karray_members[] = {
    // {"attr", T_INT, offsetof(PyKarray, attr), 0,
    //  "Arbitrary attribute."},
    {NULL}  /* Sentinel */
};

static PyGetSetDef Karray_getsetters[] = {
    // {"shape", (getter) Karray_getshape, NULL,
    //  "Shape of the array.", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef Karray_methods[] = {
    // {"reshape", (PyCFunction) Karray_reshape, METH_O,
    //  "Return the kipr.arr with the new shape."},
    // {"broadcast", (PyCFunction) Karray_broadcast, METH_O,
    //  "Return the kipr.arr with the breadcasted shape."},
    // {"mean", (PyCFunction) Karray_mean, METH_VARARGS | METH_KEYWORDS,
    //  "Return the averaged array."},
    // {"sum", (PyCFunction) Karray_sum, METH_VARARGS | METH_KEYWORDS,
    //  "Return the sum of the array along all or a particular dim."},
    // {"numpy", (PyCFunction) Karray_numpy, METH_NOARGS,
    //  "Return a numpy representtion of the PyKarray."},
    // {"val", (PyCFunction) Karray_val, METH_NOARGS,
    //  "Return the float value of a scalar <kipr.arr>."},    
    {"execute", (PyCFunction)  execute_func, METH_O,
     "Testing function to execute C code."},
    {NULL}  /* Sentinel */
};


static PyMethodDef arraymodule_methods[] = {
    // {"max_nd", max_nd, METH_NOARGS,
    //  "Get maximum number of dimensions for a kipr.arr() array."},
    {"execute", execute_func, METH_O,
     "Testing function to execute C code."},
    // {"internal", internal_test, METH_NOARGS,
    //  "Execute C code tests."},
    // {"relu", Karray_relu, METH_O,
    //  "ReLU function for <kipr.arr> arrays."},
    // {"exp", Karray_exp, METH_O,
    //  "Exponential function for <kipr.arr> arrays."},
    // {"softmax", Karray_softmax, METH_O,
    //  "Softmax function for <kipr.arr> arrays, computes along the last axis."},
    // {"ln", Karray_log, METH_O,
    //  "Log function for <kipr.arr> arrays."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyModuleDef arraymodule = {
    PyModuleDef_HEAD_INIT,
    "kipr_array",
    "Array backend.",
    -1,
    arraymodule_methods
};



// static PyNumberMethods Karray_as_number = {
//     .nb_add = Karray_add,
//     .nb_subtract = Karray_sub,
//     .nb_multiply = Karray_mul,

//     .nb_negative = Karray_negative,

//     .nb_inplace_add = Karray_inplace_add,
//     .nb_inplace_subtract = Karray_inplace_sub,
//     .nb_inplace_multiply = Karray_inplace_mul,

//     .nb_true_divide = Karray_div,
//     .nb_inplace_true_divide = Karray_inplace_div,

//     .nb_matrix_multiply = Karray_matmul
// };

// static PyMappingMethods Karray_as_mapping = {
//     .mp_subscript = Karray_subscript
// };

static PyTypeObject KarrayType = {
    Karray_HEAD_INIT
    .tp_name = KARRAY_NAME,
    .tp_basicsize = sizeof(PyKarray) - sizeof(float),
    .tp_itemsize = sizeof(float),
    .tp_dealloc = (destructor) Karray_dealloc,
    // .tp_repr = (reprfunc) Karray_str, 
    // .tp_as_number = &Karray_as_number,
    // .tp_as_mapping = &Karray_as_mapping,
    // .tp_str = (reprfunc) Karray_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Array object from kipr",
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


class Shape
{
public:
    int nd;
    size_t length;
    size_t values[MAX_ND];
    Shape();
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
    
};

class Karray
{
public:
    bool owned;
    Shape shape;
    float * data;
    Karray();
    Karray(PyObject * self) {
        PyKarray * karr = reinterpret_cast<PyKarray *>(self);
        shape = Shape(karr->shape);
        data = karr->data;
        owned = false;
    };
    ~Karray() {
        if (owned)
            delete[] data;
    };
    
};


void
Karray_dealloc(PyKarray *self) {
    delete[] self->data;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *
Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyKarray *self;
    self = reinterpret_cast<PyKarray *>(type->tp_alloc(type, 0));
    if (self != NULL) {
        self->shape[0] = 1;
        for (int k=1; k < MAX_ND; ++k) {
            self->shape[k] = 0;
        }
        self->data = new float[1];
        self->data[0] = 0.0;
    }
    return reinterpret_cast<PyObject *>(self);
}

int
Karray_init(PyKarray *self, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"data", "shape", NULL};
    PyObject *input = NULL, *shape = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", kwlist,
                                     &input, &shape))
        return -1;

    self->data[0] = 0.1f;

    return 0;
}


PyObject *
execute_func(PyObject *self, PyObject * input) {
    DEBUG_Obj(input);

    size_t shape[MAX_ND] = {3, 4, 5, 2, 0, 0, 0, 0};

    Shape trry(shape);

    trry.print();
    Py_RETURN_NONE;
}
