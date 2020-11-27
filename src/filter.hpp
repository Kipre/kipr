Filter::Filter(Shape& shape) {
    size_t total = 0;
    int i;
    offset[0] = 0;
    for (i = 0; i < shape.nd;) {
        total += shape[i];
        offset[++i] = total;
    }
    while (i != MAX_ND) {
        offset[i++] = total;
    }
    vec.reserve(total);
    // std::fill(vec, vec + total, -1);
}

Filter::Filter(Filter&& other) noexcept : vec{}, offset{0} {
    vec = std::move(other.vec);
    for (int i = 0; i < MAX_ND; ++i) {
        offset[i] = other.offset[i];
        other.offset[i] = 0;
    }
}

void Filter::set_val_along_axis(int axis , size_t value) {
    // printf("writing val from %i to %i on axis %i\n", offset[axis], offset[axis + 1], axis);
    std::fill(vec.begin() + offset[axis], vec.begin() + offset[axis + 1], value);
}

void Filter::set_range_along_axis(int axis) {
    // printf("writing range from %i to %i on axis %i\n", offset[axis], offset[axis + 1], axis);
    std::iota (vec.begin() + offset[axis], vec.begin() + offset[axis + 1], 0);
}

void Filter::print(const char * message) {
    std::cout << "Filter " << message << "\n\t";
    int o = 1;
    for (int k = 0; k < vec.size(); ++k) {
        if (k == offset[o]) {
            std::cout << "| ";
            ++o;
        }
        std::cout << vec[k] << ", ";
    }
    std::cout << "\n\toffsets:";
    for (int k = 0; k < MAX_ND + 1; ++k) {
        std::cout << offset[k] << ", ";
    }
    std::cout << '\n';
}

Filter& Filter::operator=(Filter&& other) noexcept {
    if (this != &other) {
        vec = std::move(other.vec);
        for (int i = 0; i < MAX_ND; ++i) {
            offset[i] = other.offset[i];
            other.offset[i] = 0;
        }
    }
    return *this;
}

void Filter::push_back(size_t number, int index) {
    vec.push_back(number);
    offset[index + 1] = vec.size();
}

Shape Filter::from_subscript(PyObject * key, Shape &current_shape) {

    Shape new_shape;
    size_t ind;
    size_t nd = 0;
    int rest;

    std::vector<PyObject *> subs = full_subscript(key, current_shape);
    IF_ERROR_RETURN({});
    for (int i = 0; i < subs.size(); ++i) {
        switch (subscript_type(subs[i])) {
        case (NUMBER):
            ind = align_index(PyLong_AsSsize_t(subs[i]), current_shape[i]);
            push_back(ind, i);
            break;
        case (SLICE): {
            Py_ssize_t start, stop, step, slicelength;
            PySlice_GetIndicesEx(subs[i], current_shape[i],
                                 &start, &stop, &step, &slicelength);
            if (start == stop) {
                push_back((size_t) start, i);
            } else {
                for (int k = 0; k < slicelength; ++k) {
                    push_back(k * step + start, i);
                }
                new_shape.set(nd++, slicelength);
            }
        }
        break;
        case (SEQUENCE): {
            Py_ssize_t length = PySequence_Length(subs[i]);
            PyObject ** items = PySequence_Fast_ITEMS(subs[i]);
            // printf("seq length: %i\n", length);
            for (int k = 0; k < length; ++k) {
                ind = align_index(PyLong_AsSsize_t(items[k]), current_shape[i]);
                IF_ERROR_RETURN({});
                push_back(ind, i);
            }
            new_shape.set(nd++, length);
        }
        }
    }
    rest = subs.size();
    offset[rest] = vec.size();
    while (rest < MAX_ND) {
        ++rest;
        offset[rest] = offset[rest - 1];
    }
    return new_shape;

}

std::vector<PyObject *> full_subscript(PyObject * tuple, Shape& current_shape) {
    std::vector<PyObject *> elements;
    elements.reserve(current_shape.nd);
    Py_ssize_t tup_length = PySequence_Length(tuple);
    bool found_ellipsis = false;
    int missing_dims;

    if (tup_length > current_shape.nd) {
        PyErr_SetString(Karray_error,
                        "Subscript has too much elements.");
        return {};
    } else {

        PyObject * full_slice = PySlice_New(NULL, NULL, NULL);
        // Py_INCREF(full_slice);
        PyObject ** items = PySequence_Fast_ITEMS(tuple);

        for (int i = 0; i < tup_length; ++i) {
            if (items[i] == Py_Ellipsis && !found_ellipsis) {
                for (int k = 0; k < current_shape.nd - (tup_length - 1); ++k)
                    elements.push_back(full_slice);
                found_ellipsis = true;
            } else if (items[i] == Py_Ellipsis && found_ellipsis) {
                PyErr_SetString(Karray_error, "Ellipsis cannot appear twice in subscript.");
                return {};
            } else {
                // Py_INCREF(items[i]);
                elements.push_back(items[i]);
            }
        }
        missing_dims = current_shape.nd - elements.size();
        for (int i = 0; i < missing_dims; ++i)
            elements.push_back(full_slice);

        return elements;
    }
}