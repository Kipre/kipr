#include "internal_test.h"

#define KA(...) new_Karray_by_hand(__VA_ARGS__)

static PyObject *
internal_test(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    SETUP_KTEST;

    Karray* a1 = KA(1, 1),
            * a2 = KA(1, 2),
              * a3 = KA(1, 3),
                * a11 = KA(2, 1, 1),
                  * a12 = KA(2, 1, 2),
                    * a13 = KA(2, 1, 3),
                      * a21 = KA(2, 2, 1),
                        * a141 = KA(3, 1, 4, 1),
                          * a142 = KA(3, 1, 4, 2),
                            * a315 = KA(3, 3, 1, 5);

    TEST(TestCommonShape, {

        ASSERT_CARR_EQ(common_shape(a1, a2), shape_by_hand(1, 2), 8);
        ASSERT_CARR_EQ(common_shape(a13, a21), shape_by_hand(2, 2, 3), 8);
        ASSERT_CARR_EQ(common_shape(a141, a315), shape_by_hand(3, 3, 4, 5), 8);

        ASSERT_NULL(common_shape(a21, a142));
        ASSERT_NULL(common_shape(a315, a142));

    });

    TEST(TestNumDims, {

        ASSERT_EQ(num_dims(shape_by_hand(3, 2, 1, 4)), 3);
        ASSERT_EQ(num_dims(shape_by_hand(2, 2, 4)), 2);
        ASSERT_EQ(num_dims(shape_by_hand(6, 2, 4, 2, 5, 7, 4)), 6);
    });



    TEST(TestNewFromShape, {

        ASSERT_SHAPE_EQ(new_Karray_from_shape(shape_by_hand(1, 1)), KA(1, 1));
        ASSERT_SHAPE_EQ(new_Karray_from_shape(shape_by_hand(2, 1, 2)), KA(2, 1, 2));
        ASSERT_SHAPE_EQ(new_Karray_from_shape(shape_by_hand(2, 3, 3)), KA(2, 3, 3));
    });

    RUN_KTEST;
}


