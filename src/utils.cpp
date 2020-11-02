
int MAX_PRINT_SIZE = 30;


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

void get_strides(Karray * self, Py_ssize_t * holder) {
    Py_ssize_t current_value = 1;
    int dim = self->nd - 1;

    while (dim >= 0) {
        holder[dim] = current_value;
        current_value *= self->shape[dim--];
    }
}

int
inline num_dims(Py_ssize_t * shape) {
    int dim = 0;
    while (shape[dim] != 0 && dim != MAX_NDIMS)
        ++dim;
    return dim;
}

void filter_offsets(Karray * origin, Py_ssize_t * offsets) {
    offsets[0] = 0;
    for (int k=1; k < origin->nd; ++k) {
        offsets[k] = offsets[k-1] + origin->shape[k-1];
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
inline transfer(Karray * from, Karray * to,
                Py_ssize_t * filter, Py_ssize_t * strides,
                Py_ssize_t * positions, Py_ssize_t * offsets,
                int depth) {
    if (depth < from->nd) {
        if (filter[offsets[depth]] == -1) {
            for (int k=0; k < from->shape[depth]; ++k) {
                positions[1] += strides[depth]*(k != 0);
                transfer(from, to, filter, strides,
                         positions, offsets, depth + 1);
            }
            positions[1] -= strides[depth]*(from->shape[depth] - 1);
        } else {
            Py_ssize_t last_index = 0, current_index = 0, k = 0;
            for (; k < from->shape[depth]; ++k) {
                current_index = filter[offsets[depth] + k];
                if (current_index < 0) break;
                positions[1] += strides[depth] * (current_index - last_index);
                transfer(from, to, filter, strides,
                         positions, offsets, depth + 1);
                last_index = current_index;
            }
            positions[1] -= strides[depth]*last_index;
        }
    } else {
        to->data[positions[0]++] = from->data[positions[1]];
    }
}

Py_ssize_t
transfer_data(Karray * from, Karray * to,
              Py_ssize_t * filter, Py_ssize_t * offsets) {
    Py_ssize_t strides[MAX_NDIMS] = {};
    get_strides(from, strides);
    Py_ssize_t positions[2] = {0, 0};

    transfer(from, to, filter, strides, positions, offsets, 0);

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
new_Karray_from_shape(Py_ssize_t * shape) {
    PyTypeObject * type = &KarrayType;
    int nd = num_dims(shape);
    Karray *self;
    self = reinterpret_cast<Karray *>(type->tp_alloc(type, 0));
    self->nd = nd;
    set_shape(self, shape);
    self->data = new float[Karray_length(self)];
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

Karray *
broadcast(Karray * self, Py_ssize_t * shape) {
    Karray * result = new_Karray_from_shape(shape);

    fail:
        PyErr_SetString(PyExc_TypeError, 
            "Failed to broadcast arrays.");
        return NULL;
}
