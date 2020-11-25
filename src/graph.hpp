
void Graph::print(const char * msg) {
	std::cout << "graph: " << msg << "\n";
	for (auto op : ops)
		std::cout << "\t" << op.str() << "\n";
}

std::string Graph::str() const {
	std::ostringstream ss;
	ss << "graph\n";
	for (auto op : ops)
		ss << "\t" << op.str() << "\n";
	return ss.str();
}

void Graph_dealloc(PyGraph *self) {
	// printf("from python with refcount=%i\n", self->ob_base.ob_refcnt);
	self->g.~Graph();
	Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *
Graph_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
	return type->tp_alloc(type, 0);
}



template<class T>
std::vector<T> fast_to_vector(PyObject * o) {
	printf("fast_to_vector is not implemented for this type\n");
}

template<>
std::vector<std::string> fast_to_vector<std::string>(PyObject * o) {
	Py_ssize_t len = PySequence_Fast_GET_SIZE(o);
	std::vector<std::string> result(len);
	PyObject ** items = PySequence_Fast_ITEMS(o);
	for (int i = 0; i < len; ++i)
		result[i] = std::string(PyUnicode_AsUTF8(items[i]));
	return result;
}

const int KIPR_MODULE = -2;
const int RELU_FUNCTION = -12;
const int SOFTMAX_FUNCTION = -13;

int karray_function_code(std::string & name) {
	if (name == "relu")
		return RELU_FUNCTION;
	if (name == "softmax")
		return SOFTMAX_FUNCTION;
	return 0;
}

void print(std::stack<int> s) {
	std::cout << "stack [";
	while (!s.empty()) {
		std::cout << s.top() << ", ";
		s.pop();
	}
	std::cout << "]\n";
}

Op func_to_op(int function) {
	if (function == RELU_FUNCTION)
		return ElementwiseUnaryOp(relu_kernel, "relu");
	if (function == SOFTMAX_FUNCTION)
		return ElementwiseUnaryOp(exp_kernel, "softmax");
	if (function == BINARY_ADD)
		return ElementwiseBinaryOp(add_kernel, "add");
	if (function == BINARY_MATRIX_MULTIPLY)
		return MatMulOp();
	if (function == UNARY_NEGATIVE)
		return ElementwiseUnaryOp(exp_kernel, "negative");
	return Op();
}

int Graph_init(PyGraph *self, PyObject *args, PyObject *kwds) {
	char *kwlist[] = {"function", "trainable", NULL};
	PyObject *function = NULL, *trainable = NULL;

	PyObject * code;
	PyCodeObject * code_ob;
	Graph * g = &self->g;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", kwlist,
	                                 &function, &trainable))
		return -1;

	if (trainable) {
		Py_INCREF(trainable);
		DEBUG_Obj(trainable, "trainable");
	}

	Py_INCREF(function);
	if (!PyFunction_Check(function))
		return -1;


	code = PyFunction_GetCode(function);
	code_ob = reinterpret_cast<PyCodeObject *>(code);
	Py_ssize_t len = PyBytes_Size(code_ob->co_code);
	char* items =  PyBytes_AsString(code_ob->co_code);

	PyObject * globals = PyEval_GetGlobals();
	auto names = fast_to_vector<std::string>(code_ob->co_names);
	auto varnames = fast_to_vector<std::string>(code_ob->co_varnames);
	std::stack<int> stack;
	std::map<int, int> local;
	std::map<int, int> global;

	int op, arg;
	for (int i = 0; i < len; i += 2) {
		op = (int) (unsigned char) items[i];
		arg = (int) (unsigned char) items[i + 1];
		// printf("opcode %i, arg %i\n", op, arg);
		switch (op) {

		case (LOAD_FAST):
			if (!local.contains(arg)) {
				local[arg] = g->ops.size();
				g->inputs.push_back(g->ops.size());
				g->ops.push_back(LoadInput(varnames[arg]));
			}
			stack.push(local[arg]);
			break;

		case (LOAD_GLOBAL): {
			PyObject * var = PyDict_GetItemString(globals, names[arg].c_str());
			if (var == 0) {
				PyErr_Format(Karray_error,
				             "Global variable %s is not defined.",
				             names[arg].c_str());
				return -1;
			}
			if (PyModule_Check(var)) {
				std::string module_name(PyModule_GetName(var));
				if (module_name == std::string("kipr")) {
					stack.push(KIPR_MODULE);
				} else {
					PyErr_Format(Karray_error,
					             "Only 'kipr' modle can be used within kipr.graph functions but %s was used.",
					             module_name);
					return -1;
				}
			} else if (py_type(var) == KARRAY) {
				if (!global.contains(arg)) {
					global[arg] = g->ops.size();
					g->ops.push_back(LoadGlobal(names[arg]));
				}
				stack.push(global[arg]);
			} else {
				PyErr_Format(Karray_error, "Unknown global variable %s is used in function.", names[arg]);
				return -1;
			}
		}
		break;

		case (LOAD_METHOD):
			if (stack.top() != KIPR_MODULE) {
				PyErr_SetString(Karray_error, "Expected stack to have KIPR_MODULE on top.");
				return -1;
			}
			stack.pop();
			stack.push(karray_function_code(names[arg]));
			break;

		case (CALL_METHOD): {
			if (arg != 1) {
				PyErr_SetString(Karray_error, "At the moment, only one argument per function call is supported.");
				return -1;
			}
			int argument = stack.top(); stack.pop();
			int function = stack.top(); stack.pop();
			Op oper = func_to_op(function);
			oper.operands.push_back(argument);
			g->ops[argument].add_child(g->ops.size());
			stack.push(g->ops.size());
			g->ops.push_back(oper);
		}
		break;

		case (BINARY_MATRIX_MULTIPLY):
		case (BINARY_ADD): {
			int a = stack.top(); stack.pop();
			int b = stack.top(); stack.pop();
			Op oper = func_to_op(op);
			oper.operands.push_back(a);
			oper.operands.push_back(b);
			g->ops[a].add_child(g->ops.size());
			g->ops[b].add_child(g->ops.size());
			stack.push(g->ops.size());
			g->ops.push_back(oper);
		}
		break;

		case (UNARY_NEGATIVE): {
			int a = stack.top(); stack.pop();
			Op oper = func_to_op(op);
			oper.operands.push_back(a);
			g->ops[a].add_child(g->ops.size());
			stack.push(g->ops.size());
			g->ops.push_back(oper);
		}
		break;

		case (STORE_FAST):
			local[arg] = stack.top(); stack.pop();
			break;

		case (RETURN_VALUE):
			if (stack.top() != g->ops.size() - 1) {
				PyErr_SetString(Karray_error, "Expected the last line to compute the return value.");
				return -1;
			}
			break;

		default:
			PyErr_Format(Karray_error, "Unknown opcode %i.", op);
			return -1;
		}
		// print(stack);

	}

	g->instance = std::vector<Karray *>(g->ops.size());

	Py_DECREF(function);
	Py_XDECREF(trainable);
	return 0;

fail:
	Py_DECREF(function);
	Py_XDECREF(trainable);
	PyErr_SetString(PyExc_TypeError,
	                "Failed to initialize <kipr.graph>.");
	return -1;
}



PyObject *
Graph_str(PyGraph * self) {
	return PyUnicode_FromString(self->g.str().c_str());
}


// PyObject * Graph_call(PyGraph *self, PyObject *args, PyObject *kwds) {
// 	char *kwlist[] = {"a", "b", "c", "d", "e", "f", NULL};
// 	PyKarray *a = NULL, *b = NULL, *c = NULL,
// 	         *d = NULL, *e = NULL, *f = NULL;

// 	Graph * g = &self->g;

// 	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|O!O!O!O!O!", kwlist,
// 	                                 &KarrayType, &a,
// 	                                 &KarrayType, &b,
// 	                                 &KarrayType, &c,
// 	                                 &KarrayType, &d,
// 	                                 &KarrayType, &e,
// 	                                 &KarrayType, &f))
// 		return NULL;

//     std::vector<PyKarray *> inputs = {a, b, c, d, e, f, g};

//     int i = 0;
// 	for (auto v : g->inputs) {
//         g->instance[v] = &inputs[i++]->arr;
//     }

//     for (i=0; i < g->ops.size(); ++i) {
//     	g->ops[i].execute(g->instance, i);
//     }
// 	return reinterpret_cast<PyObject *>(g->instance[g->instance.size() - 1]);
// }

PyObject * Graph_prepare(PyGraph *self, PyObject *const *args, Py_ssize_t nargs) {
	Graph * g = &self->g;
	if (nargs != g->inputs.size()) {
		PyErr_Format(Karray_error,
		             "Wrong number of arguments, expecting %i but got %i.",
		             g->inputs.size(), nargs);
		return NULL;
	}
	for (int k = 0; k < nargs; ++k) {
		if (py_type(args[k]) != KARRAY) {
			PyErr_SetString(Karray_error,
			             "Only kipr.arr is a valid input.");
			return NULL;
		}
		g->instance[g->inputs[k]] = &reinterpret_cast<PyKarray *>(args[k])->arr;
	}
	Py_RETURN_NONE;
}