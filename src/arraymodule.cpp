#include "arraymodule.hpp"


/************************************************
                Debug functions
************************************************/
    // DEBUG

void DEBUG_print_int_carr(int * carr, int len, char * message="") 
{
    printf("%s\n", message);
    printf("\tprinting array, length: %k\n", len);
    printf("\telements: ");
    for (int k=0; k<len + 1; k++) {
        printf(" %i,", carr[k]);
    }
    printf("\n");
}

void DEBUG_print_arr(Karray * self, char * message="")
{
    printf("%s\n", message);
    printf("\tnumber of dimensions: %i\n", self->nd);
    printf("\tshape: ");
    for (int k=0; k<self->nd + 1; k++) {
        printf(" %i,", self->shape[k]);
    }
    printf("\n");
    int length = Karray_length(self);
    printf("\tdata theoretical length: %i\n", length);
    if (length < 50) {
        printf("\tdata: ");
        for (int k=0; k<length; k++) {
            printf(" %f,", self->data[k]);
        }
        printf("\n");
    }
}  

void DEBUG_print_type(PyObject * obj, char * message="") 
{
    printf("type check in %s\n", message);
    printf("\tis sequence %i \n", PySequence_Check(obj)); 
    printf("\tis array %i \n", PyArray_Check(obj)); 
    printf("\tis buffer %i \n", PyObject_CheckBuffer(obj)); 
    printf("\tis iter %i \n", PyIter_Check(obj)); 
    printf("\tis number %i \n", PyNumber_Check(obj)); 
    printf("\tis Karray %i \n", is_Karray(obj));
    printf("\tis mapping %i \n", PyMapping_Check(obj));
    printf("\tis index %i \n", PyIndex_Check(obj));
    printf("\tis slice %i \n", PySlice_Check(obj));
}

#define DEBUG_P(o)   PyObject_Print(o, stdout, Py_PRINT_RAW); printf("\n")

/************************************************
                Utility functions
************************************************/

int product(int * arr, int len, int increment=0, int depth=0) {
    int result = 1;
    while (len >  depth) result *= arr[--len] + increment;
    return result;
}

int sum(int * arr, int len, int depth=0) {
    int result = 0;
    while (len >  depth) result += arr[--len];
    return result;
}

/************************************************
            Member utility functions
************************************************/

int Karray_length(Karray *self) {
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

static bool is_scalar(Karray * self) {
    return (self->nd == 1) && (self->shape[0] == 1);
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

static int infer_shape(PyObject * input, int * shape, int depth=0) {
    int length;

    if (depth > MAX_NDIMS)
        return -1;

    if (PySequence_Check(input) && (length = PySequence_Length(input)) > 0 ) {
        PyObject * item = PySequence_GetItem(input, 0);
        int full_depth = infer_shape(item, shape, depth + 1);
        if (full_depth < 0)
            return -1;
        shape[depth] = length;
        Py_DECREF(item);
        return full_depth;
    } else if (PyNumber_Check(input)) {
        return Py_MAX(depth, 1);
    } else {
        return -1;
    }
}

static int copy_data(PyObject * input, int * shape, float * result, int depth=0, int position=0)
{   
    if (PySequence_Check(input)) {
        for (int k=0; k<shape[depth]; k++) {
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
        float scalar = (float) PyFloat_AsDouble(float_obj);
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

static int parse_shape(PyObject * sequence, int * shape) {
    if (PySequence_Check(sequence)) {
        int nd = PySequence_Length(sequence);
        if (nd < 1 || nd > MAX_NDIMS) {
            PyErr_SetString(PyExc_TypeError, "Shape must have between one and kipr.max_nd() elements.");
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
Karray_init_from_data(Karray * self, PyObject * sequence) 
{
    int inferred_shape[MAX_NDIMS] = {1};
    // int depth_shape[MAX_NDIMS] = {1};
    // int depth = 2;


    // printf("depth check \n");
    // printf("\tdepth %i \n", depth = depth_check(sequence, depth_shape)); 
    // DEBUG_print_int_carr(depth_shape, depth, "depth checked shape");

    int nd = infer_shape(sequence, inferred_shape);
    if (nd < 1) {
        PyErr_SetString(PyExc_TypeError, "Shape inference failed.");
        goto fail;
    }
    self->nd = nd;
    // printf("after infer %s\n", "");
    // DEBUG_print_int_carr(inferred_shape, self->nd, "inferred shape...");

    Py_INCREF(self);
    set_shape(self, inferred_shape);
    int data_length = Karray_length(self);
    if (self->data)
        delete[] self->data;
    self->data = new float[data_length];
    Py_INCREF(sequence);
    int final_position = copy_data(sequence, self->shape, self->data);
    Karray_IF_ERR_GOTO_FAIL;
    if (final_position != data_length) goto fail;

    // DEBUG_print_arr(self, "self in kinit");
    Py_INCREF(sequence);
    Py_INCREF(self);
    // DebugBreak();
    return data_length;

    fail:
        if (PyErr_Occurred())
            PyErr_Print();
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "Data read failed.");
        return -1;
}

std::string str(Karray * self, int * index, int depth=0, bool repr=false) 
{
    // DEBUG_print_arr(self);
    // printf("%i\n", depth);
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

bool is_Karray(PyObject * obj) 
{
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


/************************************************
               Member functions
************************************************/

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

static int
Karray_init(Karray *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"data", "shape", NULL};
    PyObject *input = NULL, *shape = NULL;
    int proposed_nd = 0, proposed_shape[MAX_NDIMS] = {0};


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", kwlist,
                                     &input, &shape))
        return -1;

    Py_INCREF(input);
    Py_INCREF(self);
    int data_length = Karray_init_from_data(self, input);
    Karray_IF_ERR_GOTO_FAIL;

    if (shape) {
        Py_INCREF(shape);
        int proposed_nd = parse_shape(shape, proposed_shape);
        Karray_IF_ERR_GOTO_FAIL;

        int proposed_length = product(proposed_shape, proposed_nd);
        // DEBUG_print_int_carr(proposed_shape, proposed_nd, "the shape...");

        // Check if the propsed makes sense with repect to data
        Py_INCREF(self);
        if (data_length != proposed_length && is_scalar(self)) {

            float current_value = self->data[0];
            delete[] self->data;
            self->data = new float [proposed_length];
            for (int k=0; k<proposed_length; k++) {
                self->data[k] = current_value;
                // printf("current_value %i\n", current_value);
            }
            self->nd = proposed_nd;
            Py_INCREF(self);
            set_shape(self, proposed_shape); 
            
        } else {
            self->nd = proposed_nd;
            Py_INCREF(self);
            set_shape(self, proposed_shape);   
        }        
    }

    Py_XDECREF(shape);
    // DebugBreak();
    Py_INCREF(self);
    return 0;

    fail:
        Py_XDECREF(shape);
        Py_DECREF(input);
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "Failed to build array.");
        return -1;
}


static PyObject * 
Karray_str(Karray * self) 
{   
    int index [MAX_NDIMS] = {0};
    std::string result = "kipr.arr(" + str(self, index);
    result.pop_back();
    result += '\n';
    result += std::string(STR_OFFSET - 1, ' ') + "shape=" + shape_str(self);
    result += ')';
    return PyUnicode_FromString(result.c_str());
}

static PyObject *
Karray_getshape(Karray *self, void *closure)
{
    PyObject * result = PyTuple_New(self->nd);
    for (int k=0; k<self->nd; k++) {
        PyTuple_SET_ITEM(result, k, PyLong_FromLong(self->shape[k]));
    }
    return result;
}

// static int
// Karray_setshape(Karray *self, PyObject *value, void *closure)
// {
//     PyErr_SetString(PyExc_TypeError, "Shape is not settable, use reshape instead.");
//     return -1;
// }


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
Karray_reshape(Karray *self, PyObject *shape)
{
    int proposed_shape [MAX_NDIMS] = {1};

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

static PyObject *
execute_func(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    
    DEBUG_print_type(Karray_new(&KarrayType, NULL, NULL));
    Py_RETURN_NONE;
}

static PyObject *
Karray_add(PyObject * self, PyObject * other) {
    Karray * b = (Karray *) Karray_new(&KarrayType, NULL, NULL);
    Karray * a = (Karray *) Karray_new(&KarrayType, NULL, NULL);

    // DEBUG_print_type(self, "KArray add");
    // printf("PyArray_NDIM %i\n", PyArray_NDIM((PyArrayObject *)self));
    // DebugBreak();

    Py_INCREF(self);
    if (is_Karray(self)) {
        a = (Karray *) self;
    } else {
        Py_INCREF(self);
        Karray_init_from_data(a, self);
        Karray_IF_ERR_GOTO_FAIL;
    }
    // DEBUG_print_arr(a, "a");
    // DebugBreak();

    Py_INCREF(other);
    if (is_Karray(other)) {
        b = (Karray *) other;
    } else {
        Karray_init_from_data(b, other);
        Karray_IF_ERR_GOTO_FAIL;
    }

    // DEBUG_print_arr(b, "b");

    int data_length = Karray_length(a);
    if (data_length == Karray_length(b)) {
        int k;
        for (k=0; k<data_length-8; k += 8) {
            __m256 v_a = _mm256_load_ps(&a->data[k]);
            __m256 v_b = _mm256_load_ps(&b->data[k]);
            v_a = _mm256_add_ps(v_a, v_b);
            _mm256_store_ps(&a->data[k], v_a);
        }
        while (k < data_length) {
            a->data[k] += b->data[k];
            k++;
        }

        // for (int k=0; k<data_length; k++) {
        //     a->data[k] += b->data[k];
        // }

    } else {
        PyErr_SetString(PyExc_TypeError, "Data length does not match.");
        PyErr_Print();
        goto fail;
    }
    Py_DECREF(b);
    Py_INCREF(a);
    return (PyObject *) a;

    fail:
        Py_XDECREF(a);
        Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "Failed to add elements.");
        return NULL;
}

static int transfer_data(Karray * from, Karray * to, int * filter, int * index,
                   int depth = 0, int position = -1) {
    if (depth < from->nd) {
        int depth_offset = sum(from->shape, depth);
        int ind;
        bool bypass = (filter[depth_offset] == -1);
        for (int k=0; k<from->shape[depth]; ++k) {
            if (bypass) {
                index[depth] = k;
                position = transfer_data(from, to, filter, index, depth + 1, position);
            } else if ((ind = filter[depth_offset + k]) >= 0) {
                index[depth] = ind;
                position = transfer_data(from, to, filter, index, depth + 1, position);
            }
        }
        return position;
    } else {
        to->data[++position] = from->data[offset(from, index)];
        return position;
    }
}

static int align_index(Karray * self, int axis, int index) {
    int length = self->shape[axis];
    if (index < length && index > -length) {
        return (length + index) % length;
    }

    PyErr_Format(PyExc_IndexError, "Index %i out of bounds on axis %i with length %i.", index, axis, length);
    // PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
    return 0;
}


static PyObject * Karray_subscript(PyObject *o, PyObject *key) 
{   
    // Py_INCREF(o);
    Karray * self = (Karray *) o;
    Karray * result = (Karray *) Karray_new(&KarrayType, NULL, NULL);

    int nb_indices = sum(self->shape, self->nd);
    int * filters = new int[nb_indices];
    for (int k=0; k<nb_indices; k++) {
        filters[k] = -1;
    }

    

    Py_INCREF(key);
    if (!PyTuple_Check(key))
        key = Py_BuildValue("(O)", key);


    int position=0, seq_length, shape_count=0, current_dim=0, current_tup_item=0;
    bool found_ellipsis = false;
    int tup_length = PyTuple_Size(key);

    printf("tuple seq_length %i\n", tup_length);
    while (current_tup_item<tup_length) {
        PyObject * current_indices = PyTuple_GetItem(key, current_tup_item++);
        // printf("current_tup_item %i, current_dim %i, shape_count %i, \n", current_tup_item, current_dim, shape_count);
        // DEBUG_P(current_indices);
        int index_position = -1;
        if (PySlice_Check(current_indices)) {
            Py_ssize_t start, stop, step, slicelength;
            PySlice_GetIndicesEx(current_indices, self->shape[current_dim], 
                                 &start, &stop, &step, 
                                 &slicelength);
            printf("start, stop, step, slicelength %i, %i, %i, %i, \n", start, stop, step, slicelength);
            if (start == stop) {
                filters[position] = start;
                position += self->shape[current_dim++];
            } else {
                for (int i=start; i<stop; i+=step) {
                    filters[++index_position + position] = i;
                }
                position += self->shape[current_dim++];
                result->shape[shape_count++] = slicelength;
            }
        } else if (PyNumber_Check(current_indices)) {
            int index = (int) PyLong_AsLong(current_indices);
            index = align_index(self, current_dim, index);
            Karray_IF_ERR_GOTO_FAIL;
            filters[position] = index;
            position += self->shape[current_dim++];
        } else if (PySequence_Check(current_indices) &&
                   (seq_length = PySequence_Length(current_indices)) > 0 &&
                   seq_length < self->shape[current_dim]) {
            for (int i=0; i<seq_length; i++) {
                PyObject * item = PySequence_GetItem(current_indices, i);
                filters[++index_position + position] = align_index(self, current_dim, (int) PyLong_AsLong(item));
                Karray_IF_ERR_GOTO_FAIL;
            }
            position += self->shape[current_dim++];
            result->shape[shape_count++] = seq_length;
        } else if (current_indices == Py_Ellipsis &&
                   !found_ellipsis) {
            printf("current_dim before elli, %i \n", current_dim);
            int nb_axes_to_elli = self->nd - tup_length + 1;
            int done_axes = current_dim;
            while (current_dim < nb_axes_to_elli + done_axes) {
                position += self->shape[current_dim];
                result->shape[shape_count++] = self->shape[current_dim++];            
            }
            printf("after nb_axes_to_elli %i, %i, %i \n", nb_axes_to_elli, current_dim, shape_count);

            found_ellipsis = true;
        } else {
            goto fail;
        }
    }
    // finish index creation
    while (current_dim < self->nd) {
        // for (int j=0; j<self->shape[current_dim]; ++j)
        //     filters[j + position] = j;
        position += self->shape[current_dim];
        result->shape[shape_count++] = self->shape[current_dim];
        ++current_dim;
    }
    result->nd = shape_count + (current_dim == 1);


    // DEBUG_print_arr(result, "result");
    DEBUG_print_int_carr(filters, nb_indices, "filters");

    // printf("shape count %i\n", shape_count);
    // printf("result length %i\n", Karray_length(result));
    delete[] result->data;
    int result_length = Karray_length(result);
    result->data = new float[result_length];
    int index[MAX_NDIMS] = {};

    if (transfer_data(self, result, filters, index) != result_length - 1)
        goto fail;

    return (PyObject *) result;

    fail:
        PyErr_SetString(PyExc_IndexError, "Failed to understand subscript.");
        return NULL;
}


