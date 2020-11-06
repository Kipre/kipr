
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
