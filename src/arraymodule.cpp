#include "arraymodule.hpp" 


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
    {"broadcast", (PyCFunction) Karray_broadcast, METH_O,
     "Return the kipr.arr with the breadcasted shape."},
    {"mean", (PyCFunction) Karray_mean, METH_VARARGS | METH_KEYWORDS,
     "Return the averaged array."},
    {"sum", (PyCFunction) Karray_sum, METH_VARARGS | METH_KEYWORDS,
     "Return the sum of the array along all or a particular dim."},
    {"numpy", (PyCFunction) Karray_numpy, METH_NOARGS,
     "Return a numpy representtion of the Karray."},
    {"val", (PyCFunction) Karray_val, METH_NOARGS,
     "Return the float value of a scalar <kipr.arr>."},    
    {"execute", (PyCFunction)  execute_func, METH_O,
     "Testing function to execute C code."},
    {NULL}  /* Sentinel */
};


static PyMethodDef arraymodule_methods[] = {
    {"max_nd", max_nd, METH_NOARGS,
     "Get maximum number of dimensions for a kipr.arr() array."},
    {"execute", execute_func, METH_O,
     "Testing function to execute C code."},
    {"internal", internal_test, METH_NOARGS,
     "Execute C code tests."},
    {"relu", Karray_relu, METH_O,
     "ReLU function for <kipr.arr> arrays."},
    {"exp", Karray_exp, METH_O,
     "Exponential function for <kipr.arr> arrays."},
    {"softmax", Karray_softmax, METH_O,
     "Softmax function for <kipr.arr> arrays, computes along the last axis."},
    {"ln", Karray_log, METH_O,
     "Log function for <kipr.arr> arrays."},
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


template <typename T>
void DEBUG_carr(T * carr, int len, char const * message = "") {
    printf("%s\n", message);
    printf("\tprinting array, length: %i\n", len);
    printf("\telements: ");
    for (int k=0; k < len + 1; k++) {
        printf(" %i,", carr[k]);
    }
    printf("\n");
}

void DEBUG_shape(Py_ssize_t * carr, char const * message = "") {
    printf("%s\n", message);;
    printf("\tshape: ");
    for (int k=0; k < MAX_NDIMS; k++) {
        printf(" %I64i,", carr[k]);
    }
    printf("\n");
}

void DEBUG_Karr(Karray * self, char const * message = "") {
    printf("%s\n", message);
    printf("\tnumber of dimensions: %i\n", self->nd);
    printf("\tshape: ");
    for (int k=0; k < self->nd + 1; k++) {
        printf(" %I64i,", self->shape[k]);
    }
    printf("\n");
    Py_ssize_t length = Karray_length(self);
    printf("\tdata theoretical length: %Ii\n", length);
    if (length < 50) {
        printf("\tdata: ");
        for (int k=0; k < length; k++) {
            printf(" %f,", self->data[k]);
        }
        printf("\n");
    }
}


/************************************************
                Utility functions
************************************************/

template <typename T>
inline T product(T * arr, int len, int increment = 0, int depth = 0) {
    T result = 1;
    while (len >  depth) result *= arr[--len] + (T) increment;
    return result;
}

template <typename T>
T sum(T * arr, int len, int depth = 0) {
    T result = 0;
    while (len >  depth) result += arr[--len];
    return result;
}

/************************************************
            Member utility functions
************************************************/

void get_strides(int nd, Py_ssize_t * shape, Py_ssize_t * holder) {
    Py_ssize_t current_value = 1;
    int dim = nd - 1;

    while (dim >= 0) {
        holder[dim] = current_value;
        current_value *= shape[dim--];
    }
}

Py_ssize_t get_stride(Karray * self, int axis) {
    Py_ssize_t result = 1;
    int dim = self->nd - 1;
    while (dim > axis) {
        result *= self->shape[dim--];
    }
    return result;
}

int
inline num_dims(Py_ssize_t * shape) {
    int dim = 0;
    while (shape[dim] != 0 && dim != MAX_NDIMS)
        ++dim;
    return dim;
}

void filter_offsets(Py_ssize_t * shape, Py_ssize_t * offsets) {
    offsets[0] = 0;
    for (int k=1; k < MAX_NDIMS; ++k) {
        offsets[k] = offsets[k-1] + shape[k-1];
    }
}

Py_ssize_t Karray_length(Karray *self) {
    return product(self->shape, self->nd);
}

void set_shape(Karray *self, Py_ssize_t * shape) {
    for (int k=0; k < MAX_NDIMS; k++) {
        self->shape[k] = shape[k];
    }
}

void copy_shape(Py_ssize_t * source, Py_ssize_t * destination) {
    for (int k=0; k < MAX_NDIMS; k++) {
        destination[k] = source[k];
    }
}

Py_ssize_t shape_pop(Py_ssize_t * shape, int i = -1) {
    Py_ssize_t result;
    int nd = num_dims(shape);
    if (nd == 1) {
        result = shape[0];
        shape[0] = 1;
    } else {
        if (Py_ABS(i) > nd-1) throw std::runtime_error("Dim is out of range.");
        i = (i % nd + nd) % nd;
        result = shape[i];
        while (i < nd) {
            shape[i] = shape[i+1];
            ++i;
        }
    }
    return result;
}

void reset_shape(Karray *self) {
    Py_ssize_t shape[MAX_NDIMS] = {1};
    set_shape(self, shape);
}

bool is_scalar(Karray * self) {
    return (self->nd == 1) && (self->shape[0] == 1);
}

Py_ssize_t offset(Karray * self, Py_ssize_t * index) {
    int k = 0;
    Py_ssize_t result = 0;
    while (k < self->nd-1) {
        result = (result + index[k])*self->shape[k+1];
        ++k;
    }
    result += index[k];
    return result;
}

int infer_shape(PyObject * input, Py_ssize_t * shape, int depth = 0) {
    Py_ssize_t length;

    if (depth > MAX_NDIMS) {
        goto fail;
    }

    if (PySequence_Check(input) && (length = PySequence_Length(input)) > 0) {
        PyObject * item = PySequence_GetItem(input, 0);
        int full_depth = infer_shape(item, shape, depth + 1);
        if (full_depth < 0)
            goto fail;
        shape[depth] = length;
        Py_DECREF(item);
        return full_depth;
    } else if (PyNumber_Check(input)) {
        return Py_MAX(depth, 1);
    } else {
        fail:
            PyErr_SetString(PyExc_TypeError, "Failed to infer shape.");
            return -1;
    }
}

int copy_data(PyObject * input, Py_ssize_t * shape,
                     float * result, int depth = 0, int position = 0) {
    if (PySequence_Check(input)) {
        for (int k=0; k < shape[depth]; k++) {
            PyObject *item = PySequence_GetItem(input, k);
            Karray_IF_ERR_GOTO_FAIL;
            position = copy_data(item, shape, result, depth + 1, position);
            if (position == -1) {
                goto fail;
            }
            Py_DECREF(item);
        }
    } else if (PyNumber_Check(input)) {
        PyObject *float_obj = PyNumber_Float(input);
        float scalar = static_cast<float>(PyFloat_AsDouble(float_obj));
        Karray_IF_ERR_GOTO_FAIL;
        result[position++] = scalar;
        Py_DECREF(float_obj);
    } else {
        fail:
            PyErr_SetString(PyExc_TypeError, "Could not copy data.");
            return -1;
    }
    return position;
}

int parse_shape(PyObject * sequence, Py_ssize_t * shape) {
    // returns the nb of dimensions
    Py_INCREF(sequence);  // real
    if (PySequence_Check(sequence)) {
        Py_ssize_t nd = PySequence_Length(sequence);
        if (nd < 1 || nd > MAX_NDIMS) {
            PyErr_Format(PyExc_TypeError,
                "Shape must have between one and %i elements.", MAX_NDIMS);
            return -1;
        }
        for (int k=0; k < nd; k++) {
            PyObject * element = PySequence_GetItem(sequence, k);
            shape[k] = static_cast<int>(PyLong_AsLong(element));
            if (PyErr_Occurred() || shape[k] == 0) {
                PyErr_SetString(PyExc_TypeError,
                    "Shape must ba a sequence of non-zero integers.");
                return -1;
            }
            Py_DECREF(element);
        }
        return static_cast<int>(nd);
    } else {
        PyErr_SetString(PyExc_TypeError, "Shape must be a sequence.");
        return -1;
    }
}

Py_ssize_t
Karray_init_from_data(Karray * self, PyObject * sequence) {
    Py_ssize_t inferred_shape[MAX_NDIMS] = {1};
    Py_ssize_t data_length;
    int final_position;

    int nd = infer_shape(sequence, inferred_shape);
    Karray_IF_ERR_GOTO_FAIL;
    self->nd = nd;

    set_shape(self, inferred_shape);
    data_length = Karray_length(self);
    if (self->data)
        delete[] self->data;
    self->data = new float[data_length];

    final_position = copy_data(sequence, self->shape, self->data);
    Karray_IF_ERR_GOTO_FAIL;

    if (final_position != data_length) goto fail;

    return data_length;

    fail:
        PyErr_SetString(PyExc_TypeError, "Data read failed.");
        return -1;
}

std::string str(Karray * self, Py_ssize_t * index,
                int depth = 0, bool repr = false) {
    // TODO(me): rewrite function without index
    // DEBUG_Karr(self);
    // printf("%i\n", depth);
    std::string result = "[";
    Py_ssize_t current_offset = offset(self, index);

    if (depth < self->nd && (current_offset < MAX_PRINT_SIZE) && !repr) {
        for (int k=0; k < self->shape[depth]; k++) {
            index[depth] = k;
            if (k != 0 && depth != self->nd-1) {
                result += std::string(depth + STR_OFFSET, ' ');
            }
            std::string sub = str(self, index, depth + 1);
            result += sub;
            if (sub == "....") return result;
        }
        // remove last newline and comma
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
    for (int k=0; k < self->nd; k++) {
        result += " " + std::to_string(self->shape[k]) + ",";
    }
    result.pop_back();
    return result + "]";
}

bool is_Karray(PyObject * obj) {
    return (obj->ob_type == &KarrayType);
}

char char_type(PyObject * obj) {
    if (is_Karray(obj))
        return 'k';
    if (PyArray_Check(obj))
        return 'a';
    if (PyUnicode_Check(obj))
        return 'u';
    if (PyTuple_Check(obj))
        return 't';
    if (PySequence_Check(obj))
        return 's';
    if (PyNumber_Check(obj))
        return 'n';
    if (PyIndex_Check(obj))
        return 'i';
    if (PyMapping_Check(obj))
        return 'm';
    if (PyObject_CheckBuffer(obj))
        return 'b';
    if (PyIter_Check(obj))
        return '$';
    if (PySlice_Check(obj))
        return ':';
    return '?';
}



void
inline transfer(int from_nd, Py_ssize_t * from_shape,
                float * from_data, Karray * to,
                Py_ssize_t * filter, Py_ssize_t * strides,
                Py_ssize_t * positions, Py_ssize_t * offsets,
                int depth) {
    if (depth < from_nd) {
        if (filter[offsets[depth]] == -1) {
            for (int k=0; k < from_shape[depth]; ++k) {
                positions[1] += strides[depth]*(k != 0);
                transfer(from_nd, from_shape, from_data, to, filter, strides,
                         positions, offsets, depth + 1);
            }
            positions[1] -= strides[depth]*(from_shape[depth] - 1);
        } else {
            Py_ssize_t last_index = 0, current_index = 0, k = 0;
            for (; k < from_shape[depth]; ++k) {
                current_index = filter[offsets[depth] + k];
                if (current_index < 0) break;
                positions[1] += strides[depth] * (current_index - last_index);
                transfer(from_nd, from_shape, from_data, to, filter, strides,
                         positions, offsets, depth + 1);
                last_index = current_index;
            }
            positions[1] -= strides[depth]*last_index;
        }
    } else {
        to->data[positions[0]++] = from_data[positions[1]];
    }
}

Py_ssize_t
transfer_data(int from_nd, Py_ssize_t * from_shape,
              float * from_data, Karray * to,
              Py_ssize_t * filter, Py_ssize_t * offsets) {
    Py_ssize_t strides[MAX_NDIMS] = {};
    get_strides(from_nd, from_shape, strides);
    Py_ssize_t positions[2] = {0, 0};

    transfer(from_nd, from_shape, from_data, to, 
             filter, strides, positions, offsets, 0);

    return positions[0];
}

Py_ssize_t align_index(Karray * self, int axis, Py_ssize_t index) {
    Py_ssize_t length = self->shape[axis];
    if (axis > MAX_NDIMS) {
        PyErr_Format(PyExc_IndexError,
            "Subscript is probably too wide because the array can't have an axis %i.",
             axis);
        return -1;
    }
    if (index < length && index > -length) {
        return (length + index) % length;
    }

    PyErr_Format(PyExc_IndexError,
        "Index %i out of bounds on axis %i with length %i.",
        index, axis, length);
    return -1;
}

void
make_filter(PyObject * tuple, Karray * from,
            Karray * to, Py_ssize_t * filter) {
    Py_ssize_t position = 0, seq_length;
    int shape_count = 0, current_dim = 0, current_tup_item = 0;
    bool found_ellipsis = false;
    Py_ssize_t tup_length = PyTuple_Size(tuple);

    while (current_tup_item < tup_length) {
        PyObject * current_indices = PyTuple_GetItem(tuple, current_tup_item++);
        int index_position = -1;
        if (PySlice_Check(current_indices)) {
            Py_ssize_t start, stop, step, slicelength;
            PySlice_GetIndicesEx(current_indices, from->shape[current_dim],
                                 &start, &stop, &step, &slicelength);
            // printf("start, stop, step, slicelength: %i %i %i %i \n", start, stop, step, slicelength);
            if (start == stop) {
                filter[position] = start;
                position += from->shape[current_dim++];
            } else {
                for (int i=0; i < slicelength; ++i) {
                    filter[++index_position + position] = i*step + start;
                }
                position += from->shape[current_dim++];
                to->shape[shape_count++] = slicelength;
            }

        } else if (PyIndex_Check(current_indices)) {
            Py_ssize_t index = (Py_ssize_t) PyLong_AsLong(current_indices);
            index = align_index(from, current_dim, index);
            Karray_IF_ERR_GOTO_FAIL;
            filter[position] = index;
            position += from->shape[current_dim++];

        } else if (PySequence_Check(current_indices) &&
                   (seq_length = PySequence_Length(current_indices)) > 0 &&
                   seq_length <= from->shape[current_dim]) {
            for (int i=0; i < seq_length; i++) {
                PyObject * item = PySequence_GetItem(current_indices, i);
                filter[++index_position + position] =
                    align_index(from, current_dim,
                        static_cast<int>(PyLong_AsLong(item)));
                Karray_IF_ERR_GOTO_FAIL;
            }
            position += from->shape[current_dim++];
            to->shape[shape_count++] = seq_length;

        } else if (current_indices == Py_Ellipsis &&
                   !found_ellipsis) {
            Py_ssize_t nb_axes_to_elli = from->nd - tup_length + 1;
            int done_axes = current_dim;
            while (current_dim < nb_axes_to_elli + done_axes) {
                position += from->shape[current_dim];
                to->shape[shape_count++] = from->shape[current_dim++];
            }
            found_ellipsis = true;

        } else {
            goto fail;
        }
    }
    // finish index creation
    while (current_dim < from->nd) {
        position += from->shape[current_dim];
        to->shape[shape_count++] = from->shape[current_dim];
        ++current_dim;
    }
    // DEBUG_Obj(tuple);
    // printf("shape_count + (current_dim == 1):  %i, Py_MAX(shape_count, 1): %i\n", shape_count + (current_dim == 1), Py_MAX(shape_count, 1));
    to->nd = Py_MAX(1, shape_count);
    return;

    fail:
        PyErr_Format(PyExc_TypeError,
            "Could not understand item %i of the subscript.", current_tup_item - 1);
        return;
}

Karray *
new_Karray(Py_ssize_t size = 1) {
    PyTypeObject * type = &KarrayType;
    Karray *self;
    self = reinterpret_cast<Karray *>(type->tp_alloc(type, 0));
    if (self != NULL) {
        self->nd = 1;
        reset_shape(self);
        self->data = new float[size];
    }
    return self;
}

Karray *
new_Karray_as(Karray * other) {
    PyTypeObject * type = &KarrayType;
    Karray *self;
    self = reinterpret_cast<Karray *>(type->tp_alloc(type, 0));
    self->nd = other->nd;
    set_shape(self, other->shape);
    self->data = new float[Karray_length(other)];
    return self;
}

Karray *
new_Karray_from_shape(Py_ssize_t * shape, float fill_value = -1.01) {
    PyTypeObject * type = &KarrayType;
    int nd = num_dims(shape);
    Karray *self;
    self = reinterpret_cast<Karray *>(type->tp_alloc(type, 0));
    self->nd = nd;
    set_shape(self, shape);
    Py_ssize_t length = Karray_length(self);
    self->data = new float[length];
    if (fill_value != -1.01) {
        std::fill(self->data, self->data + length, fill_value);
    }
    return self;
}

void
Karray_copy(Karray * source, Karray * destination) {    
    destination->nd = source->nd;
    memcpy(destination->shape, source->shape, sizeof(Py_ssize_t)*MAX_NDIMS);
    Py_ssize_t length = Karray_length(source);
    if (destination->data)
        delete[] destination->data;
    destination->data = new float[length];
    memcpy(destination->data, source->data, sizeof(float)*length);
    return;
}


bool
safe_cast(PyObject * obj, Karray ** arr) {
    if (is_Karray(obj)) {
        Py_INCREF(obj);
        *arr = reinterpret_cast<Karray *>(obj);
        return false;
    } else {
        *arr = new_Karray();
        Karray_init_from_data(*arr, obj);
        Karray_IF_ERR_GOTO_FAIL;
        return true;
    }

    fail:
        PyErr_SetString(PyExc_TypeError,
            "Failed to cast the operand into <kipr.arr>.");
        return NULL;
}

Karray *
safe_copy(PyObject * obj) {
    Karray * target = new_Karray();
    if (is_Karray(obj)) {
        Karray_copy(reinterpret_cast<Karray *>(obj), target);
    } else {
        Karray_init_from_data(target, obj);
        Karray_IF_ERR_GOTO_FAIL;
    }
    return target;

    fail:
        PyErr_SetString(PyExc_TypeError,
            "Failed to copy the operand into <kipr.arr>.");
        return NULL;
}

Py_ssize_t *
common_shape(Karray * a, Karray * b) {
    Py_ssize_t dim_a, dim_b;
    Py_ssize_t * result = new Py_ssize_t[MAX_NDIMS];
    
    if (b->nd > a->nd) {
        std::swap(a, b);
    }

    int i_a = a->nd-1, i_b = b->nd-1;
    while (i_a >= 0) {
        if (i_b >= 0) {
            dim_a = a->shape[i_a];
            dim_b = b->shape[i_b];
            if (!(dim_a == dim_b || (dim_a == 1 || dim_b == 1))) {
                goto fail;
            }
            result[i_a] = Py_MAX(dim_a, dim_b);
            // printf("result[i_a], dim_a, dim_b: %I64i %I64i %I64i\n", result[i_a], dim_a, dim_b);
            --i_b;
        } else {
            result[i_a] = a->shape[i_a];
        }
        --i_a;
    }
    for (int i=a->nd; i < MAX_NDIMS; ++i) {
        result[i] = 0;
    }
    // DEBUG_shape(result);
    return result;

    fail:
        PyErr_SetString(PyExc_TypeError, 
            "Arrays are not broadcastable.");
        return NULL;
}

void
broadcast_filter(Karray * from, Py_ssize_t * to_shape, 
                 Py_ssize_t * filter, Py_ssize_t * offsets)  {
    int from_i = from->nd - 1;
    for (int i=MAX_NDIMS-1; i >= 0; --i) {
        for (int k=0; k < to_shape[i]; ++k) {
            if (from_i >= 0) {
                filter[offsets[i] + k] = k * (from->shape[from_i] != 1);
            } else {
                filter[offsets[i] + k] = 0;
            }
        }
        if (to_shape[i] > 0)
            --from_i;
    }
}

bool
broadcastable(Py_ssize_t * shape_a, Py_ssize_t * shape_b, 
              int dim_a = 0, int dim_b = 0) {
    // compute dimensions if not available
    if (dim_a == 0) {
        dim_a = num_dims(shape_a);
    }
    if (dim_b == 0) {
        dim_b = num_dims(shape_b);
    }
    // swap so that a is bigger
    if (dim_b > dim_a) {
        std::swap(shape_a, shape_b);
        std::swap(dim_a, dim_b);
    }
    // decrement dims so they become indexes
    --dim_a; --dim_b;

    while (dim_b >= 0) {
        if ((shape_a[dim_a] != shape_b[dim_b]) &&
            (shape_a[dim_a] != 1 &&
             shape_b[dim_b] != 1)) {
            return false;
        }
        --dim_a; --dim_b;
    }
    return true;
}

bool
broadcastable_to(Py_ssize_t * shape, Py_ssize_t * cast_shape, 
              int dim = 0, int cast_dim = 0) {
    // compute dimensions if not available
    if (dim == 0) {
        dim = num_dims(shape);
    }
    if (cast_dim == 0) {
        cast_dim = num_dims(cast_shape);
    }
    // cannot cast into a smaller array
    if (dim > cast_dim) {
        return false;
    }
    // decrement dims so they become indexes
    --dim; --cast_dim;

    while (dim >= 0) {
        if (shape[dim] != cast_shape[cast_dim] &&
            shape[dim] != 1) {
            return false;
        }
        --dim; --cast_dim;
    }
    return true;
}

Karray *
broadcast(Karray * self, Py_ssize_t * shape) {

    Karray * result = new_Karray_from_shape(shape);
    Py_ssize_t filter_size = sum(shape, MAX_NDIMS);
    Py_ssize_t * filter = new Py_ssize_t[filter_size];
    Py_ssize_t strides[MAX_NDIMS] = {};
    Py_ssize_t positions[2] = {0, 0};
    Py_ssize_t stride_shape[MAX_NDIMS] = {};
    Py_ssize_t offsets[MAX_NDIMS] = {};

    const int target_nd = num_dims(shape);
    const int nb_ones_to_pad = target_nd - self->nd;
    int nb_ones_left = nb_ones_to_pad;
    if (nb_ones_to_pad < 0) goto fail;

    // sanity check
    if (!broadcastable_to(self->shape, shape, self->nd, target_nd)) {
        PyErr_SetString(PyExc_TypeError, "Cannot perform one-sided broadcast.");
        PyErr_Print();
        goto fail;
    }

    for (int i=0; i < MAX_NDIMS; ++i) {
        if (nb_ones_left > 0) {
            stride_shape[i] = 1;
            --nb_ones_left;
        } else {
            stride_shape[i] = self->shape[i-nb_ones_to_pad];
        }
    }
    
    filter_offsets(shape, offsets);

    broadcast_filter(self, shape, filter, offsets);

    get_strides(target_nd, stride_shape, strides);

    transfer(target_nd, shape, self->data, result,
             filter, strides, positions, offsets, 0);

    if (positions[0] != product(shape, target_nd)) {
        goto fail;
    }


    delete[] filter;
    return result;

    fail:
        delete[] filter;
        Py_XDECREF(result);
        PyErr_SetString(PyExc_TypeError, 
            "Failed to broadcast arrays.");
        return NULL;
}

void
add_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
        __m256 v_a = _mm256_load_ps(&destination[k]);
        __m256 v_b = _mm256_load_ps(&other[k]);
        v_a = _mm256_add_ps(v_a, v_b);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] += other[k];
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        destination[k] += other[k];
    }
#endif
}

void
sub_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
        __m256 v_a = _mm256_load_ps(&destination[k]);
        __m256 v_b = _mm256_load_ps(&other[k]);
        v_a = _mm256_sub_ps(v_a, v_b);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] -= other[k];
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        destination[k] -= other[k];
    }
#endif
}


void
mul_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
        __m256 v_a = _mm256_load_ps(&destination[k]);
        __m256 v_b = _mm256_load_ps(&other[k]);
        v_a = _mm256_mul_ps(v_a, v_b);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] *= other[k];
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        destination[k] *= other[k];
    }
#endif
}


void
div_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
        __m256 v_a = _mm256_load_ps(&destination[k]);
        __m256 v_b = _mm256_load_ps(&other[k]);
        v_a = _mm256_div_ps(v_a, v_b);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] /= other[k];
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        if (other[k] == 0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "");
            PyErr_Print();
            PyErr_Clear();
        }
        destination[k] /= other[k];
    }
#endif
}


void
exp_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
        __m256 v_a = _mm256_load_ps(&other[k]);
        v_a = _mm256_exp_ps(v_a);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] = exp(other[k]);
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        destination[k] = exp(other[k]);
    }
#endif
}


void
log_kernel(float * destination, float * other, Py_ssize_t length) {
#if __AVX__
    int k;
    for (k=0; k < length-8; k += 8) {
        __m256 v_a = _mm256_load_ps(&other[k]);
        v_a = _mm256_log_ps(v_a);
        _mm256_store_ps(&destination[k], v_a);
    }
    while (k < length) {
        destination[k] = exp(other[k]);
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        destination[k] = exp(other[k]);
    }
#endif
}


void
val_mul_kernel(float * destination, float value, Py_ssize_t length) {
#if __AVX__
    int k;
    __m256 values, constant = _mm256_set_ps(value, value, value, value, value, value, value, value);
    for (k=0; k < length-8; k += 8) {
        values = _mm256_load_ps(&destination[k]);
        values = _mm256_mul_ps(values, constant);
        _mm256_store_ps(&destination[k], values);
    }
    while (k < length) {
        destination[k] *= value;
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        destination[k] *= value;
    }
#endif
}

void
max_val_kernel(float * destination, float value, Py_ssize_t length) {
    #if __AVX__
    int k;
    __m256 values, val = _mm256_set_ps (value, value, value, value, value, value, value, value);
    for (k=0; k < length-8; k += 8) {
        values = _mm256_load_ps(&destination[k]);
        values = _mm256_max_ps(values, val);
        _mm256_store_ps(&destination[k], values);
    }
    while (k < length) {
        destination[k] = Py_MAX(value, destination[k]);
        k++;
    }
#else
    for (int k=0; k < length; k++) {
        destination[k] = Py_MAX(value, destination[k]);
    }
#endif
}

void
Karray_dealloc(Karray *self) {
    delete[] self->data;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *
Karray_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Karray *self;
    self = reinterpret_cast<Karray *>(type->tp_alloc(type, 0));
    if (self != NULL) {
        self->nd = 1;
        reset_shape(self);
        self->data = new float[1];
        self->data[0] = 0.0;
    }
    return reinterpret_cast<PyObject *>(self);
}

int
Karray_init(Karray *self, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"data", "shape", NULL};
    PyObject *input = NULL, *shape = NULL;
    Py_ssize_t proposed_shape[MAX_NDIMS] = {0};
    bool random = false, range = false, value = false;
    float init_value;
    Py_ssize_t data_length = 0;
    unsigned int val;
    PyArrayObject* arr;
    float * arr_data;
    int nd;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", kwlist,
                                     &input, &shape))
        return -1;


    if (PyArray_Check(input) &&
        (nd = PyArray_NDIM((PyArrayObject *) input)) < MAX_NDIMS &&
        PyArray_TYPE((PyArrayObject *) input) == NPY_FLOAT) {
        arr = (PyArrayObject *) input;
        self->nd = nd;
        npy_intp * input_shape = PyArray_SHAPE(arr);
        for (int i=0; i<nd; ++i) {
            self->shape[i] = (Py_ssize_t) input_shape[i];
        }
        npy_intp length = PyArray_SIZE(arr);
        arr_data = (float *) PyArray_DATA(arr);
        delete[] self->data;
        self->data = new float[length];
        for (int k=0; k < length; ++k) {
            self->data[k] = arr_data[k];
        }
        data_length = (Py_ssize_t) length;
    } else if (PyUnicode_Check(input)) {
        if (PyUnicode_Compare(input, PyUnicode_FromString("random")) == 0) {
            random = true;
            self->data[0] = (float) rand() / (float) 32767;
        } else if (PyUnicode_Compare(input, PyUnicode_FromString("range")) == 0) {
            range = true;
        }
    } else if (PySequence_Check(input)) {
        data_length = Karray_init_from_data(self, input);
        Karray_IF_ERR_GOTO_FAIL;
    } else if (PyNumber_Check(input)) {
        PyObject *float_obj = PyNumber_Float(input);
        init_value = static_cast<float>(PyFloat_AsDouble(float_obj));
        Karray_IF_ERR_GOTO_FAIL;
        Py_DECREF(float_obj);
        self->data[0] = init_value;
        value = true;
    } else {
        PyErr_SetString(PyExc_TypeError, "Unsupported input data.");
        PyErr_Print();
        goto fail;
    }

    if (shape) {
        int proposed_nd = parse_shape(shape, proposed_shape);
        Karray_IF_ERR_GOTO_FAIL;

        Py_ssize_t proposed_length = product(proposed_shape, proposed_nd);

        // Check if the propsed makes sense with repect to data
        if (data_length != 0) {
            if (data_length == proposed_length) {
                self->nd = proposed_nd;
                set_shape(self, proposed_shape);
            } else {
                PyErr_SetString(PyExc_TypeError, 
                    "Proposed shape did not align with data.");
                PyErr_Print();
                goto fail;
            }
        } else {
            delete[] self->data;
            self->data = new float[proposed_length];
            if (value) {
                for (int k=0; k < proposed_length; ++k)
                    self->data[k] = init_value;
            } else if (range) {
                for (int k=0; k < proposed_length; ++k)
                    self->data[k] = (float) k;
            } else if (random) {
                for (int k=0; k < proposed_length; ++k) {
                    if (_rdrand32_step(&val) == 0) {
                        PyErr_SetString(PyExc_SystemError, 
                            "Could not generate a random value.");
                        PyErr_Print();
                        goto fail;
                    }
                    self->data[k] = (float)((double) val / (double) 4294967295);
                }
            }
            self->nd = proposed_nd;
            set_shape(self, proposed_shape);
        }
    } else if (range){
        PyErr_SetString(PyExc_TypeError, 
            "A shape must be provided when using \"range\" magic.");
        PyErr_Print();
        goto fail;
    }
    return 0;

    fail:
        Py_XDECREF(shape);
        Py_DECREF(input);
        PyErr_SetString(PyExc_TypeError, "Failed to build the array.");
        return -1;
}


PyObject *
Karray_str(Karray * self) {
    Py_ssize_t index[MAX_NDIMS] = {0};
    std::string result = "kipr.arr(" + str(self, index);
    result.pop_back();
    result += '\n';
    result += std::string(STR_OFFSET - 1, ' ') + "shape=" + shape_str(self);
    result += ')';
    return PyUnicode_FromString(result.c_str());
}

PyObject *
Karray_getshape(Karray *self, void *closure) {
    PyObject * result = PyTuple_New(self->nd);
    for (int k=0; k < self->nd; k++) {
        PyTuple_SET_ITEM(result, k, PyLong_FromSsize_t(self->shape[k]));
    }
    return result;
}

// int
// Karray_setshape(Karray *self, PyObject *value, void *closure)
// {
//     PyErr_SetString(PyExc_TypeError,
//         "Shape is not settable, use reshape instead.");
//     return -1;
// }


PyObject *
Karray_numpy(Karray *self, PyObject *Py_UNUSED(ignored)) {
    npy_intp * dims = new npy_intp[self->nd];
    for (int k=0; k < self->nd; k++) {
        dims[k] = (npy_intp) self->shape[k];
    }
    Py_INCREF(self);
    return PyArray_SimpleNewFromData(self->nd, dims, NPY_FLOAT, self->data);
}

PyObject *
Karray_reshape(Karray *self, PyObject *shape) {
    Py_ssize_t proposed_shape[MAX_NDIMS] = {1};

    int proposed_nd = parse_shape(shape, proposed_shape);
    if (proposed_nd < 1) {
        PyErr_Print();
        PyErr_SetString(PyExc_TypeError, "Failed to reshape array.");
        return NULL;
    }
    if (Karray_length(self) == product(proposed_shape, proposed_nd)) {
        set_shape(self, proposed_shape);
        self->nd = proposed_nd;
        Py_INCREF(self);
        return reinterpret_cast<PyObject *>(self);
    } else {
        PyErr_SetString(PyExc_TypeError,
            "Proposed shape doesn't align with data.");
        return NULL;
    }
}

PyObject * 
Karray_subscript(PyObject *o, PyObject *key) {
    Karray * self = reinterpret_cast<Karray *>(o);
    Karray * result = new_Karray();
    Py_ssize_t offsets[MAX_NDIMS] = {};
    Py_ssize_t result_length;

    Py_ssize_t nb_indices = sum(self->shape, self->nd);
    Py_ssize_t * filters = new Py_ssize_t[nb_indices];
    for (int k=0; k < nb_indices; k++) {
        filters[k] = -1;
    }

    Py_INCREF(key);
    if (!PyTuple_Check(key))
        key = Py_BuildValue("(O)", key);

    make_filter(key, self, result, filters);
    Karray_IF_ERR_GOTO_FAIL;

    // DEBUG_carr(filters, nb_indices, "filter");
    // DEBUG_Karr(result, "result");

    delete[] result->data;
    result_length = Karray_length(result);
    result->data = new float[result_length];


    filter_offsets(self->shape, offsets);

    transfer_data(self->nd, self->shape, self->data, 
                  result, filters, offsets);

    return reinterpret_cast<PyObject *>(result);

    fail:
        PyErr_SetString(PyExc_IndexError, "Failed to apply subscript.");
        return NULL;
}

PyObject * 
Karray_broadcast(Karray *self, PyObject *o) {
    Py_ssize_t shape[MAX_NDIMS] = {};
    Karray *result;
    parse_shape(o, shape);
    Karray_IF_ERR_GOTO_FAIL;

    result = broadcast(self, shape);
    Karray_IF_ERR_GOTO_FAIL;

    return reinterpret_cast<PyObject *>(result);

    fail:
        PyErr_SetString(PyExc_TypeError, 
            "Failed to apply broadcast, input shape is probably not coherent.");
        return NULL;
}


void
inline sum(float * self_data, float * result_data, float * weights_data,
           Py_ssize_t * self_shape, Py_ssize_t * strides,
           int axis, int depth = 0) {
    if (axis != depth) {
        for (int k=0; k < self_shape[depth]; ++k) {
            sum(self_data + strides[depth]*k, result_data + strides[depth]*k/self_shape[axis], 
                weights_data, self_shape, strides, axis, depth + 1);
        }
    } else {
        for (int i=0; i < self_shape[axis]; ++i) {
            for (int k=0; k < strides[axis]; ++k) {
                // printf("val and result: %f %f %i %i\n", self_data[strides[axis] * i + k], result_data[k], strides[axis] * i + k, i);
                result_data[k] += self_data[strides[axis] * i + k] * weights_data[i];
            }
        }
    }
}



PyObject *
Karray_mean(Karray *self, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"axis", "weights", NULL};

    PyObject *axis = NULL, *weights_obj = NULL;
    Py_ssize_t output_shape[MAX_NDIMS] = {};
    Py_ssize_t strides[MAX_NDIMS] = {};
    Py_ssize_t fake_shape[MAX_NDIMS] = {};
    Karray * result, * weights;
    Py_ssize_t reduction, ax, stride = 1;
    bool weights_passed;


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O$O", kwlist,
                                     &axis, &weights_obj))
        return NULL;

    Karray * karr = reinterpret_cast<Karray *>(self);

    ax = Axis(karr->nd, axis).value;
    PYERR_PRINT_GOTO_FAIL;

    if (ax == -1) {
        output_shape[0] = 1;
        reduction = Karray_length(karr);
        strides[0] = 1;
        fake_shape[0] = reduction;
        ax = 0;
    } else {
        copy_shape(karr->shape, output_shape);
        reduction = shape_pop(output_shape, ax);
        get_strides(karr->nd, karr->shape, strides);
        copy_shape(karr->shape, fake_shape);
    }

    result = new_Karray_from_shape(output_shape, 0);

    if (weights_obj) {
        weights_passed = true;
        if (!is_Karray(weights_obj)) {
            PyErr_SetString(PyExc_TypeError, 
                "Weights should be a <kipr.arr>.");
            goto fail;
        } else {
            weights = reinterpret_cast<Karray *>(weights_obj);
            if (Karray_length(weights) != reduction) {
                PyErr_SetString(PyExc_TypeError, 
                    "Weights not compatible with reduction.");
                goto fail;
            }

        } 
    } else {
        Py_ssize_t weights_shape[MAX_NDIMS] = {reduction};
        weights = new_Karray_from_shape(weights_shape, 1);
    }

    sum(karr->data, result->data, weights->data,
        fake_shape, strides, ax);


    for (int k=0; k < Karray_length(result); ++k) {
        result->data[k] /= (float) reduction;
    }


    if (!weights_passed)
        Py_XDECREF(weights);
    return reinterpret_cast<PyObject *>(result);

    fail:
        if (!weights_passed)
            Py_XDECREF(weights);
        Py_XDECREF(result);
        return NULL;
}



PyObject *
Karray_sum(Karray *self, PyObject *args, PyObject *kwds) {
    char *kwlist[] = {"axis", "weights", NULL};

    PyObject *axis = NULL, *weights_obj = NULL;
    Py_ssize_t output_shape[MAX_NDIMS] = {};
    Py_ssize_t strides[MAX_NDIMS] = {};
    Py_ssize_t fake_shape[MAX_NDIMS] = {};
    Karray * result, * weights;
    Py_ssize_t reduction, ax, stride = 1;
    bool weights_passed;


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O$O", kwlist,
                                     &axis, &weights_obj))
        return NULL;

    Karray * karr = reinterpret_cast<Karray *>(self);

    ax = Axis(karr->nd, axis).value;
    PYERR_PRINT_GOTO_FAIL;

    if (ax == -1) {
        output_shape[0] = 1;
        reduction = Karray_length(karr);
        strides[0] = 1;
        fake_shape[0] = reduction;
        ax = 0;
    } else {
        copy_shape(karr->shape, output_shape);
        reduction = shape_pop(output_shape, ax);
        get_strides(karr->nd, karr->shape, strides);
        copy_shape(karr->shape, fake_shape);
    }

    result = new_Karray_from_shape(output_shape, 0);

    if (weights_obj) {
        weights_passed = true;
        if (!is_Karray(weights_obj)) {
            PyErr_SetString(PyExc_TypeError, 
                "Weights should be a <kipr.arr>.");
            goto fail;
        } else {
            weights = reinterpret_cast<Karray *>(weights_obj);
            if (Karray_length(weights) != reduction) {
                PyErr_SetString(PyExc_TypeError, 
                    "Weights not compatible with reduction.");
                goto fail;
            }

        } 
    } else {
        Py_ssize_t weights_shape[MAX_NDIMS] = {reduction};
        weights = new_Karray_from_shape(weights_shape, 1);
    }

    sum(karr->data, result->data, weights->data,
        fake_shape, strides, ax);


    if (!weights_passed)
        Py_XDECREF(weights);
    return reinterpret_cast<PyObject *>(result);

    fail:
        if (!weights_passed)
            Py_XDECREF(weights);
        Py_XDECREF(result);
        return NULL;
}


PyObject *
Karray_val(Karray *self, PyObject *Py_UNUSED(ignored)) {
    if (Karray_length(self) != 1) {
        PyErr_SetString(PyExc_TypeError, 
            "Val method called on a non scalar array.");
        return NULL;
    }
    return PyFloat_FromDouble((double) self->data[0]);
}
PyObject *
Karray_binary_op(PyObject * self, PyObject * other, 
                void (*op_kernel)(float *, float*, Py_ssize_t)) {
    Karray *a, *b, *c;
    Py_ssize_t data_length, *cmn_shape;
    bool a_owned = false, b_owned = false;


    if (!is_Karray(self) || !is_Karray(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    a = reinterpret_cast<Karray *>(self);
    b = reinterpret_cast<Karray *>(other);
    
    data_length = Karray_length(a);
    if (Karray_length(b) != data_length) {
        cmn_shape = common_shape(a, b);
        Karray_IF_ERR_GOTO_FAIL;
        a = broadcast(a, cmn_shape);
        a_owned = true;
        Karray_IF_ERR_GOTO_FAIL;
        b = broadcast(b, cmn_shape);
        b_owned = true;
        Karray_IF_ERR_GOTO_FAIL;
        data_length = Karray_length(a);
    } else {
        c = new_Karray_as(a);
        Karray_copy(a, c);
        a = c;

    }

    op_kernel(a->data, b->data, data_length);
    
    // Py_INCREF(a);

    
    if (b_owned)
        Py_DECREF(b);

    return reinterpret_cast<PyObject *>(a);

    fail:
        Py_XDECREF(a);
        Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, 
            "Failed to apply binary operation.");
        return NULL;
}


PyObject *
Karray_inplace_binary_op(PyObject * self, PyObject * other, 
                         void (*op_kernel)(float *, float*, Py_ssize_t)) {
    Karray *a, *b;
    Py_ssize_t data_length;

    if (!is_Karray(self) || !is_Karray(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    a = reinterpret_cast<Karray *>(self);
    b = reinterpret_cast<Karray *>(other);
    
    data_length = Karray_length(a);
    if (Karray_length(b) != data_length) {
        b = broadcast(b, a->shape);
        Karray_IF_ERR_GOTO_FAIL;
    }

    op_kernel(a->data, b->data, data_length);

    Py_INCREF(self);
    return self;

    fail:
        Py_XDECREF(a);
        Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, 
            "Failed to apply binary operation.");
        return NULL;
}



PyObject *
Karray_add(PyObject * self, PyObject * other) {
    return Karray_binary_op(self, other, add_kernel);
}

PyObject *
Karray_inplace_add(PyObject * self, PyObject * other) {
    return Karray_inplace_binary_op(self, other, add_kernel);
}

PyObject *
Karray_sub(PyObject * self, PyObject * other) {
    return Karray_binary_op(self, other, sub_kernel);
}

PyObject *
Karray_inplace_sub(PyObject * self, PyObject * other) {
    return Karray_inplace_binary_op(self, other, sub_kernel);
}

PyObject *
Karray_mul(PyObject * self, PyObject * other) {
    return Karray_binary_op(self, other, mul_kernel);
}

PyObject *
Karray_inplace_mul(PyObject * self, PyObject * other) {
    return Karray_inplace_binary_op(self, other, mul_kernel);
}

PyObject *
Karray_div(PyObject * self, PyObject * other) {
    return Karray_binary_op(self, other, div_kernel);
}

PyObject *
Karray_inplace_div(PyObject * self, PyObject * other) {
    return Karray_inplace_binary_op(self, other, div_kernel);
}


PyObject *
Karray_matmul(PyObject * self, PyObject * other) {
    Karray *a, *b, *c;
    Py_ssize_t left_dim, mid_dim, right_dim, 
               nb_mat_a, nb_mat_b,
               pos_a = 0, pos_b = 0, pos_c = 0;
    Py_ssize_t result_shape[MAX_NDIMS] = {};

    if (!is_Karray(self) || !is_Karray(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    a = reinterpret_cast<Karray *>(self);
    b = reinterpret_cast<Karray *>(other);

    if (a->nd < 2 || b->nd < 2) {
        PyErr_SetString(PyExc_TypeError, 
            "MatMul works on at least 2-dimensional arrays.");
        PyErr_Print();
        goto fail;
    }

    if (a->shape[a->nd - 1] != b->shape[b->nd - 2]) {
        PyErr_SetString(PyExc_TypeError, 
            "Arrays not compatible for MatMul.");
        PyErr_Print();
        goto fail;
    }

    left_dim = a->shape[a->nd - 2];
    mid_dim = a->shape[a->nd - 1];
    right_dim = b->shape[b->nd - 1];

    nb_mat_a = product(a->shape, a->nd-2);
    nb_mat_b = product(b->shape, b->nd-2);

    // printf("nb_mat_a, nb_mat_b: %i %i\n", nb_mat_a, nb_mat_b);

    if (nb_mat_a == nb_mat_b ||
        nb_mat_a == 1 ||
        nb_mat_b == 1) {
        result_shape[0] = Py_MAX(nb_mat_a, nb_mat_b);
        result_shape[1] = left_dim;
        result_shape[2] = right_dim;
    } else {
        PyErr_SetString(PyExc_TypeError, 
            "Arrays not compatible for MatMul.");
        PyErr_Print();
        goto fail;
    }

    c = new_Karray_from_shape(result_shape);

    for (int m=0; m < result_shape[0]; ++m) {
        pos_a = (m % nb_mat_a) * left_dim*mid_dim;
        pos_b = (m % nb_mat_b) * mid_dim*right_dim;
        for (int i=0; i < left_dim; ++i) {
            for (int j=0; j < right_dim; ++j) {
                c->data[pos_c] = 0;
                for (int k=0; k < mid_dim; ++k) {
                    // printf("indexes: %i %i\n", pos_a + k + mid_dim*i, pos_b + j + k*right_dim);
                    c->data[pos_c] += a->data[pos_a + k + mid_dim*i] * b->data[pos_b + j + k*right_dim]; 
                }
                ++pos_c;
            }
        }
    }

    // risky
    if (nb_mat_a >= nb_mat_b) {
        c->nd = a->nd;
        set_shape(c, a->shape);
        c->shape[c->nd-1] = right_dim;
        c->shape[c->nd-2] = left_dim;
    } else {
        c->nd = b->nd;
        set_shape(c, b->shape);
        c->shape[c->nd-1] = right_dim;
        c->shape[c->nd-2] = left_dim;
    }

    
    // Py_INCREF(c);
    return reinterpret_cast<PyObject *>(c);

    fail:
        PyErr_SetString(PyExc_TypeError, 
            "Failed to mat-mutiply arrays.");
        return NULL;
}

PyObject *
Karray_negative(PyObject * self) {
    Karray * result = new_Karray();
    Karray_copy(reinterpret_cast<Karray *>(self), result);

    val_mul_kernel(result->data, -1, Karray_length(result));

    return reinterpret_cast<PyObject *>(result);
}

PyObject *
execute_func(PyObject *self, PyObject * input) {
    DEBUG_Obj(input);

    auto values = FastSequence<Int>(input, true);
    if (PyErr_Occurred()) { 
        PyErr_Print(); 
        Py_RETURN_NONE; 
    }


    for(std::vector<Int>::iterator it = values.elements.begin(); it != values.elements.end(); ++it) {
     	std::cout << (*it).value << " ";
	}
	std::cout << std::endl;
    Py_RETURN_NONE;
}



PyObject *
max_nd(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromLong(static_cast<long>(MAX_NDIMS));
}



PyObject *
Karray_relu(PyObject *self, PyObject * o) {

	if (!is_Karray(o)) {
		Py_RETURN_NOTIMPLEMENTED;
	}
	
	Karray * result = new_Karray();
	Karray * arr = reinterpret_cast<Karray *>(o);
	Karray_copy(arr, result);

	Py_ssize_t length = Karray_length(arr);
	max_val_kernel(result->data, 0, Karray_length(result));

    return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_exp(PyObject *self, PyObject * o) {

	if (!is_Karray(o)) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	Karray * arr = reinterpret_cast<Karray *>(o);
	Karray * result = new_Karray_from_shape(arr->shape);

	Py_ssize_t length = Karray_length(arr);

	exp_kernel(result->data, arr->data, Karray_length(arr));

    return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_softmax(PyObject *self, PyObject * o) {

	if (!is_Karray(o)) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	Py_ssize_t reduction, nb_sums, sum_shape[MAX_NDIMS] = {};
	Karray * arr = reinterpret_cast<Karray *>(o);
	Karray * result = new_Karray_from_shape(arr->shape);

	copy_shape(arr->shape, sum_shape);
	reduction = shape_pop(sum_shape);
	nb_sums = product(sum_shape, arr->nd-1);

	float * tmp_sums = new float[nb_sums];
	std::fill(tmp_sums, tmp_sums+nb_sums, 0);

	Py_ssize_t length = Karray_length(arr);

	exp_kernel(result->data, arr->data, Karray_length(arr));

	for (int i=0; i < nb_sums; ++i) {
		for (int k=0; k < reduction; ++k) {
			tmp_sums[i] += result->data[k + i*reduction];
		}

		for (int k=0; k < reduction; ++k) {
			result->data[k + i*reduction] /= tmp_sums[i];
		}
	}

	delete[] tmp_sums;

    return reinterpret_cast<PyObject *>(result);
}

PyObject *
Karray_log(PyObject *self, PyObject * o) {

	if (!is_Karray(o)) {
		Py_RETURN_NOTIMPLEMENTED;
	}

	Karray * arr = reinterpret_cast<Karray *>(o);
	Karray * result = new_Karray_from_shape(arr->shape);

	Py_ssize_t length = Karray_length(arr);

	log_kernel(result->data, arr->data, Karray_length(arr));

    return reinterpret_cast<PyObject *>(result);
}

#include "test.hpp" 
