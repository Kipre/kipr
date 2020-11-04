
#include "internal_test.h"

Karray *
new_Karray_by_hand(int nd...) {
    va_list args;
    va_start(args, nd);
    Karray * result = new_Karray();
    Py_ssize_t data_length;
    int nb_init_values = 0;

    if (nd > MAX_NDIMS) {
        return NULL;
    }

    for (int i = 0; i < nd; ++i) {
        result->shape[i] = (Py_ssize_t) va_arg(args, int);
    }
    va_end(args);

    result->nd = nd;
    data_length = Karray_length(result);
    delete[] result->data;
    result->data = new float[data_length];


    for (int i = 0; i < data_length; ++i) {
        result->data[i] = (float) i + 1;
    }
    return result;
}


Karray *
full_by_hand(int nd...) {
    va_list args;
    va_start(args, nd);
    Karray * result = new_Karray();
    Py_ssize_t data_length;
    int nb_init_values = 0;

    if (nd > MAX_NDIMS) {
        return NULL;
    }
    result->nd = nd;

    for (int i = 0; i < nd; ++i) {
        result->shape[i] = (Py_ssize_t) va_arg(args, int);
    }
    data_length = Karray_length(result);
    delete[] result->data;
    result->data = new float[data_length];

    for (int i = 0; i < data_length; ++i) {
        result->data[i] = (float) va_arg(args, double);
    }

    va_end(args);

    return result;
}

Py_ssize_t *
shape_by_hand(int nd...) {
    Py_ssize_t * shape = new Py_ssize_t[MAX_NDIMS];
    va_list args;
    va_start(args, nd);

    for (int i = 0; i < MAX_NDIMS; ++i) {
        if (i < nd) {
            shape[i] = (Py_ssize_t) va_arg(args, int);
        } else {
            shape[i] = 0;
        }
    }
    va_end(args);
    return shape;
}

#define KA(...) new_Karray_by_hand(__VA_ARGS__)


///////////////////////////
//         TEST          //
///////////////////////////

PyObject *
internal_test(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    SETUP_KTEST;

    Karray * a1 = KA(1, 1);
    Karray * a2 = KA(1, 2);
    Karray * a3 = KA(1, 3);
    Karray * a11 = KA(2, 1, 1);
    Karray * a12 = KA(2, 1, 2);
    Karray * a13 = KA(2, 1, 3);
    Karray * a21 = KA(2, 2, 1);
    Karray * a141 = KA(3, 1, 4, 1);
    Karray * a142 = KA(3, 1, 4, 2);
    Karray * a315 = KA(3, 3, 1, 5);
    Karray * a343 = KA(3, 3, 4, 3);

    Karray * b1 = full_by_hand(1, 1, 1.);
    Karray * b12 = full_by_hand(2, 1, 2, 1., 2.);

    Karray * c23 = full_by_hand(2, 2, 3, 1., 1., 1., 2., 2., 2.);

    Karray * c325 = full_by_hand(3, 3, 2, 5, 1., 2., 3., 4., 5.,
                                             1., 2., 3., 4., 5.,
                                             6., 7., 8., 9., 10.,
                                             6., 7., 8., 9., 10.,
                                             11., 12., 13., 14., 15.,
                                             11., 12., 13., 14., 15.);

    TEST(CommonShape) {

        ASSERT_CARR_EQ(common_shape(a1, a2), shape_by_hand(1, 2), 8);

        ASSERT_CARR_EQ(common_shape(a13, a21), shape_by_hand(2, 2, 3), 8);
        ASSERT_CARR_EQ(common_shape(a141, a315), shape_by_hand(3, 3, 4, 5), 8);

        ASSERT_NULL_AND_ERROR(common_shape(a21, a142));
        ASSERT_NULL_AND_ERROR(common_shape(a315, a142));

    };

    TEST(NumDims) {

        ASSERT_EQ(num_dims(shape_by_hand(3, 2, 1, 4)), 3);
        ASSERT_EQ(num_dims(shape_by_hand(2, 2, 4)), 2);
        ASSERT_EQ(num_dims(shape_by_hand(6, 2, 4, 2, 5, 7, 4)), 6);
    };



    TEST(NewFromShape) {

        ASSERT_SHAPE_EQ(new_Karray_from_shape(shape_by_hand(1, 1)), KA(1, 1));
        ASSERT_SHAPE_EQ(new_Karray_from_shape(shape_by_hand(2, 1, 2)), KA(2, 1, 2));
        ASSERT_SHAPE_EQ(new_Karray_from_shape(shape_by_hand(2, 3, 3)), KA(2, 3, 3));
    };

    TEST(FullByHand) {
        ASSERT_KARR_EQ(a1, b1);
        ASSERT_KARR_EQ(a12, b12);
    };

    TEST(BroadcastFilter) {
        Py_ssize_t * shape = shape_by_hand(3, 3, 4, 3);
        Py_ssize_t filter_size = sum(shape, MAX_NDIMS);
        Py_ssize_t * filter = new Py_ssize_t[filter_size];

        Py_ssize_t offsets[MAX_NDIMS] = {};
        filter_offsets(shape, offsets);
        
        broadcast_filter(a13, shape, filter, offsets);

        Py_ssize_t expected[10] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 2};

        ASSERT_EQ(10, filter_size);

        ASSERT_CARR_EQ(filter, expected, 10);

        delete[] filter;
    };



    TEST(Broadcastable) {
        ASSERT(broadcastable(a1->shape, a2->shape));
        ASSERT(broadcastable(a12->shape, a2->shape));
        ASSERT(broadcastable(a12->shape, a21->shape));
        ASSERT(broadcastable(a141->shape, a12->shape));
        ASSERT(broadcastable(a142->shape, a12->shape));
        ASSERT(broadcastable(a141->shape, a315->shape));

        ASSERT_FALSE(broadcastable(a12->shape, a13->shape));
        ASSERT_FALSE(broadcastable(a2->shape, a3->shape));
        ASSERT_FALSE(broadcastable(a13->shape, a2->shape));
        ASSERT_FALSE(broadcastable(a142->shape, a21->shape));
        ASSERT_FALSE(broadcastable(a142->shape, a315->shape));
    };



    TEST(BroadcastableTo) {
        ASSERT(broadcastable_to(a1->shape, a2->shape));
        ASSERT(broadcastable_to(a141->shape, a142->shape));

        ASSERT_FALSE(broadcastable_to(a12->shape, a21->shape));
        ASSERT_FALSE(broadcastable_to(a141->shape, a315->shape));
    };



    TEST(Broadcast) {
        Karray * res;
        ASSERT_NO_ERROR(res = broadcast(a21, shape_by_hand(2, 2, 3)));

        ASSERT_KARR_EQ(res, c23);
        Karray_dealloc(res);

        ASSERT_NO_ERROR(res = broadcast(a315, shape_by_hand(3, 3, 2, 5)));

        ASSERT_KARR_EQ(res, c325);
        Karray_dealloc(res);

        ASSERT_ERROR(broadcast(a315, shape_by_hand(3, 3, 5, 4)));
    };



    RUN_KTEST;
}

