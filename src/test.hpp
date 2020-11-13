
PyObject *
internal_test(PyObject *self, PyObject *Py_UNUSED(ignored)) {

	TEST(ShapePop) {
		Shape shape1(5, 3, 5, 6, 7, 3);
		Shape shape2(4, 3, 5, 6, 7);
		Shape shape3(3, 3, 6, 7);
		Shape shape4(2, 6, 7);

		ASSERT(3 == shape1.pop());
		ASSERT_SHAPE_EQ(shape1, shape2);

		ASSERT(5 == shape1.pop(1));
		ASSERT_SHAPE_EQ(shape1, shape3);

		ASSERT(3 == shape1.pop(0));
		ASSERT_SHAPE_EQ(shape1, shape4);
	};

	TEST(ShapeAxis) {
		Shape shape1(3, 4, 5, 2);
		PyObject * zero = Py_BuildValue("i", 0);
		PyObject * one = Py_BuildValue("i", 1);
		PyObject * two = Py_BuildValue("i", 2);
		PyObject * three = Py_BuildValue("i", 3);
		PyObject * minusone = Py_BuildValue("i", -1);
		PyObject * fl = Py_BuildValue("f", 1.0);

		ASSERT(shape1.axis(zero) == 0);
		ASSERT(shape1.axis(one) == 1);
		ASSERT(shape1.axis(minusone) == 2);
		ASSERT(shape1.axis(two) == 2);
		ASSERT(shape1.axis(NULL) == 3);
		ASSERT_ERROR(shape1.axis(three));
		ASSERT_ERROR(shape1.axis(fl));
	};


    RUN_KTEST;
}

