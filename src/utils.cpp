
void transfer(float * from, float * to, size_t * positions, size_t * strides,
              const Filter & filter, int nd, int depth) {
	if (depth < nd) {
		size_t current_value, last_value = 0;
		for (int k = filter.offset[depth]; k < filter.offset[depth + 1]; ++k) {
			current_value = filter.vec[k];
			positions[1] += (current_value - last_value) * strides[depth];
			last_value = current_value;
			transfer(from, to, positions, strides, filter, nd, depth + 1);
		}
		positions[1] -= last_value * strides[depth];
	} else {
		// printf("writing from %i to %i\n", positions[1], positions[0]);
		to[positions[0]++] = from[positions[1]];
	}
}

size_t read_mode(PyObject * o) {
	if (!PyUnicode_Check(o))
		return ERROR_MODE;
	if (PyUnicode_Compare(o, PyUnicode_FromString("rand")) == 0 ||
	        PyUnicode_Compare(o, PyUnicode_FromString("random")) == 0) {
		return RANDOM_UNIFORM;
	}
	if (PyUnicode_Compare(o, PyUnicode_FromString("randn")) == 0) {
		return RANDOM_NORMAL;
	}
	if (PyUnicode_Compare(o, PyUnicode_FromString("range")) == 0) {
		return RANGE;
	}
	PyErr_Format(PyExc_ValueError,
	             "String magic %s not understood.", PyUnicode_AsUTF8(o));
	return ERROR_MODE;
}

size_t py_type(PyObject * o) {
	if ((PyTypeObject *) PyObject_Type(o) == &KarrayType)
		return KARRAY;
	if (PyArray_Check(o))
		return NUMPY_ARRAY;
	if (PyUnicode_Check(o))
		return STRING;
	if (PyNumber_Check(o))
		return NUMBER;
	if (PySequence_Check(o))
		return SEQUENCE;
	if (PySlice_Check(o))
		return SLICE;
	return 0;
}

size_t subscript_type(PyObject * o) {
	if (PyNumber_Check(o))
		return NUMBER;
	if (PySlice_Check(o))
		return SLICE;
	if (PySequence_Check(o))
		return SEQUENCE;
	return 0;
}

size_t align_index(Py_ssize_t i, size_t dim_length) {
	if (abs(i) >= dim_length) {
		PyErr_Format(PyExc_ValueError,
		             "Index %i out of range on axis with length %i.",
		             i, dim_length);
		return 0;
	} else {
		return (size_t) (i % dim_length + dim_length) % dim_length;
	}
}


std::map<int, std::string> op_name() {
	std::map<int, std::string> correspondance;
	correspondance[  0 ] = std::string("STOP_CODE");
	correspondance[  1 ] = std::string("POP_TOP");
	correspondance[  2 ] = std::string("ROT_TWO");
	correspondance[  3 ] = std::string("ROT_THREE");
	correspondance[  4 ] = std::string("DUP_TOP");
	correspondance[  5 ] = std::string("ROT_FOUR");
	correspondance[  9 ] = std::string("NOP");
	correspondance[ 10 ] = std::string("UNARY_POSITIVE");
	correspondance[ 11 ] = std::string("UNARY_NEGATIVE");
	correspondance[ 12 ] = std::string("UNARY_NOT");
	correspondance[ 13 ] = std::string("UNARY_CONVERT");
	correspondance[ 15 ] = std::string("UNARY_INVERT");
	correspondance[ 19 ] = std::string("BINARY_POWER");
	correspondance[ 20 ] = std::string("BINARY_MULTIPLY");
	correspondance[ 21 ] = std::string("BINARY_DIVIDE");
	correspondance[ 22 ] = std::string("BINARY_MODULO");
	correspondance[ 23 ] = std::string("BINARY_ADD");
	correspondance[ 24 ] = std::string("BINARY_SUBTRACT");
	correspondance[ 25 ] = std::string("BINARY_SUBSCR");
	correspondance[ 26 ] = std::string("BINARY_FLOOR_DIVIDE");
	correspondance[ 27 ] = std::string("BINARY_TRUE_DIVIDE");
	correspondance[ 28 ] = std::string("INPLACE_FLOOR_DIVIDE");
	correspondance[ 29 ] = std::string("INPLACE_TRUE_DIVIDE");
	correspondance[ 30 ] = std::string("SLICE+0");
	correspondance[ 31 ] = std::string("SLICE+1");
	correspondance[ 32 ] = std::string("SLICE+2");
	correspondance[ 33 ] = std::string("SLICE+3");
	correspondance[ 40 ] = std::string("STORE_SLICE+0");
	correspondance[ 41 ] = std::string("STORE_SLICE+1");
	correspondance[ 42 ] = std::string("STORE_SLICE+2");
	correspondance[ 43 ] = std::string("STORE_SLICE+3");
	correspondance[ 50 ] = std::string("DELETE_SLICE+0");
	correspondance[ 51 ] = std::string("DELETE_SLICE+1");
	correspondance[ 52 ] = std::string("DELETE_SLICE+2");
	correspondance[ 53 ] = std::string("DELETE_SLICE+3");
	correspondance[ 54 ] = std::string("STORE_MAP");
	correspondance[ 55 ] = std::string("INPLACE_ADD");
	correspondance[ 56 ] = std::string("INPLACE_SUBTRACT");
	correspondance[ 57 ] = std::string("INPLACE_MULTIPLY");
	correspondance[ 58 ] = std::string("INPLACE_DIVIDE");
	correspondance[ 59 ] = std::string("INPLACE_MODULO");
	correspondance[ 60 ] = std::string("STORE_SUBSCR");
	correspondance[ 61 ] = std::string("DELETE_SUBSCR");
	correspondance[ 62 ] = std::string("BINARY_LSHIFT");
	correspondance[ 63 ] = std::string("BINARY_RSHIFT");
	correspondance[ 64 ] = std::string("BINARY_AND");
	correspondance[ 65 ] = std::string("BINARY_XOR");
	correspondance[ 66 ] = std::string("BINARY_OR");
	correspondance[ 67 ] = std::string("INPLACE_POWER");
	correspondance[ 68 ] = std::string("GET_ITER");
	correspondance[ 70 ] = std::string("PRINT_EXPR");
	correspondance[ 71 ] = std::string("PRINT_ITEM");
	correspondance[ 72 ] = std::string("PRINT_NEWLINE");
	correspondance[ 73 ] = std::string("PRINT_ITEM_TO");
	correspondance[ 74 ] = std::string("PRINT_NEWLINE_TO");
	correspondance[ 75 ] = std::string("INPLACE_LSHIFT");
	correspondance[ 76 ] = std::string("INPLACE_RSHIFT");
	correspondance[ 77 ] = std::string("INPLACE_AND");
	correspondance[ 78 ] = std::string("INPLACE_XOR");
	correspondance[ 79 ] = std::string("INPLACE_OR");
	correspondance[ 80 ] = std::string("BREAK_LOOP");
	correspondance[ 81 ] = std::string("WITH_CLEANUP");
	correspondance[ 82 ] = std::string("LOAD_LOCALS");
	correspondance[ 83 ] = std::string("RETURN_VALUE");
	correspondance[ 84 ] = std::string("IMPORT_STAR");
	correspondance[ 85 ] = std::string("EXEC_STMT");
	correspondance[ 86 ] = std::string("YIELD_VALUE");
	correspondance[ 87 ] = std::string("POP_BLOCK");
	correspondance[ 88 ] = std::string("END_FINALLY");
	correspondance[ 89 ] = std::string("BUILD_CLASS");
	correspondance[ 90 ] = std::string("STORE_NAME");       // Index in name list
	correspondance[ 91 ] = std::string("DELETE_NAME");      // ""
	correspondance[ 92 ] = std::string("UNPACK_SEQUENCE");   // Number of tuple items
	correspondance[ 93 ] = std::string("FOR_ITER");
	correspondance[ 94 ] = std::string("LIST_APEND");
	correspondance[ 95 ] = std::string("STORE_ATTR");       // Index in name list
	correspondance[ 96 ] = std::string("DELETE_ATTR");      // ""
	correspondance[ 97 ] = std::string("STORE_GLOBAL");     // ""
	correspondance[ 98 ] = std::string("DELETE_GLOBAL");    // ""
	correspondance[ 99 ] = std::string("DUP_TOPX");          // number of items to duplicate
	correspondance[100 ] = std::string("LOAD_CONST");       // Index in const list
	correspondance[101 ] = std::string("LOAD_NAME");       // Index in name list
	correspondance[102 ] = std::string("BUILD_TUPLE");      // Number of tuple items
	correspondance[103 ] = std::string("BUILD_LIST");       // Number of list items
	correspondance[104 ] = std::string("BUILD_SET");        // Number of set items
	correspondance[105 ] = std::string("BUILD_MAP");        // Number of dict entries (upto 255);
	correspondance[106 ] = std::string("LOAD_ATTR");       // Index in name list
	correspondance[107 ] = std::string("COMPARE_OP");       // Comparison operator
	correspondance[108 ] = std::string("IMPORT_NAME");     // Index in name list
	correspondance[109 ] = std::string("IMPORT_FROM");     // Index in name list
	correspondance[110 ] = std::string("JUMP_FORWARD");    // Number of bytes to skip
	correspondance[111 ] = std::string("JUMP_IF_FALSE_OR_POP"); // Target byte offset from beginning of code
	correspondance[112 ] = std::string("JUMP_IF_TRUE_OR_POP");  // ""
	correspondance[113 ] = std::string("JUMP_ABSOLUTE");        // ""
	correspondance[114 ] = std::string("POP_JUMP_IF_FALSE");    // ""
	correspondance[115 ] = std::string("POP_JUMP_IF_TRUE");     // ""
	correspondance[116 ] = std::string("LOAD_GLOBAL");     // Index in name list
	correspondance[119 ] = std::string("CONTINUE_LOOP");   // Target address
	correspondance[120 ] = std::string("SETUP_LOOP");      // Distance to target address
	correspondance[121 ] = std::string("SETUP_EXCEPT");    // ""
	correspondance[122 ] = std::string("SETUP_FINALLY");   // ""
	correspondance[124 ] = std::string("LOAD_FAST");        // Local variable number
	correspondance[125 ] = std::string("STORE_FAST");       // Local variable number
	correspondance[126 ] = std::string("DELETE_FAST");      // Local variable number
	correspondance[130 ] = std::string("RAISE_VARARGS");    // Number of raise arguments (1, or 3);
	correspondance[131 ] = std::string("CALL_FUNCTION");    // //args + (//kwargs << 8);
	correspondance[132 ] = std::string("MAKE_FUNCTION");    // Number of args with default values
	correspondance[133 ] = std::string("BUILD_SLICE");      // Number of items
	correspondance[134 ] = std::string("MAKE_CLOSURE");
	correspondance[135 ] = std::string("LOAD_CLOSURE");
	correspondance[136 ] = std::string("LOAD_DEREF");
	correspondance[137 ] = std::string("STORE_DEREF");
	correspondance[140 ] = std::string("CALL_FUNCTION_VAR");     // //args + (//kwargs << 8);
	correspondance[141 ] = std::string("CALL_FUNCTION_KW");      // //args + (//kwargs << 8);
	correspondance[142 ] = std::string("CALL_FUNCTION_VAR_KW");  // //args + (//kwargs << 8);
	correspondance[143 ] = std::string("SETUP_WITH");
	correspondance[145 ] = std::string("EXTENDED_ARG");
	correspondance[146 ] = std::string("SET_ADD");
	correspondance[147 ] = std::string("MAP_ADD");

	return correspondance;
}
