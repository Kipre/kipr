
void Graph::print(const char * msg) {
	std::cout << "graph: " << msg << "\n";
	for (auto op : ops)
		std::cout << "\t" << op->str() << "\n";
}

std::string Graph::str() const {
	std::ostringstream ss;
	ss << "graph\n";
	for (auto op : ops)
		ss << "\t" << op->str() << "\n";
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

int karray_function_code(std::string & name) {
	if (name == "relu")
		return RELU_FUNCTION;
	if (name == "softmax")
		return SOFTMAX_FUNCTION;
	return ERROR_CODE;
}

void print(std::stack<int> s) {
	std::cout << "stack [";
	while (!s.empty()) {
		std::cout << s.top() << ", ";
		s.pop();
	}
	std::cout << "]\n";
}

Op * func_to_op(size_t function) {
	if (function == RELU_FUNCTION)
		return new ElementwiseUnaryOp(relu_kernel, "relu");
	if (function == SOFTMAX_FUNCTION)
		return new ElementwiseUnaryOp(exp_kernel, "softmax");
	if (function == BINARY_ADD)
		return new EWBinaryOp<Add>();
	if (function == BINARY_SUBTRACT)
		return new EWBinaryOp<Sub>();
	if (function == BINARY_MULTIPLY)
		return new EWBinaryOp<Mul>();
	if (function == BINARY_TRUE_DIVIDE)
		return new EWBinaryOp<Div>();
	if (function == BINARY_MATRIX_MULTIPLY)
		return new MatMulOp {};
	if (function == UNARY_NEGATIVE)
		return new ElementwiseUnaryOp(exp_kernel, "negative");
	return new Op();
}

int Graph_init(PyGraph *self, PyObject *args, PyObject *kwds) {
	char *kwlist[] = {"function", NULL};
	PyObject *function = NULL;

	PyObject * code;
	PyCodeObject * code_ob;
	Graph * g = &self->g;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &function))
		return -1;

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
				g->ops.emplace_back(new LoadInput(varnames[arg]));
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
			switch (py_type(var)) {
			case (MODULE): {
				std::string module_name(PyModule_GetName(var));
				if (module_name == std::string("kipr")) {
					stack.push(KIPR_MODULE);
				} else {
					PyErr_Format(Karray_error,
					             "Only 'kipr' module can be used within kipr.graph functions but %s was used.",
					             module_name);
					return -1;
				}
			} break;
			case (KARRAY): {
				if (!global.contains(arg)) {
					global[arg] = g->ops.size();
					g->ops.emplace_back(new LoadGlobal(names[arg]));
				}
				stack.push(global[arg]);
			} break;
			default:
				PyErr_Format(Karray_error, "Unknown global variable %s is used in function.", names[arg]);
				return -1;
			}
		} break;

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
			size_t argument = stack.top(); stack.pop();
			size_t function = stack.top(); stack.pop();
			Op * oper = func_to_op(function);
			oper->operands.push_back(argument);
			g->ops[argument]->add_child(g->ops.size());
			stack.push(g->ops.size());
			g->ops.push_back(oper);
		}
		break;

		case (BINARY_MATRIX_MULTIPLY):
		case (BINARY_ADD):
		case (BINARY_MULTIPLY):
		case (BINARY_SUBTRACT):
		case (BINARY_TRUE_DIVIDE): {
			size_t b = stack.top(); stack.pop();
			size_t a = stack.top(); stack.pop();
			Op * oper = func_to_op(op);
			oper->operands.push_back(a);
			oper->operands.push_back(b);
			g->ops[a]->add_child(g->ops.size());
			g->ops[b]->add_child(g->ops.size());
			stack.push(g->ops.size());
			g->ops.push_back(oper);
		}
		break;

		case (UNARY_NEGATIVE): {
			size_t a = stack.top(); stack.pop();
			Op * oper = func_to_op(op);
			oper->operands.push_back(a);
			g->ops[a]->add_child(g->ops.size());
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

	g->instance = std::move(std::vector<Karray>(g->ops.size()));

	Py_DECREF(function);
	return 0;
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
		// DEBUG_Obj(args[k], "");
		g->instance[g->inputs[k]] = reinterpret_cast<PyKarray *>(args[k])->arr;
	}
	for (int k=0; k < g->ops.size(); ++k) {
		g->ops[k]->prepare(g->instance, k);
		IF_ERROR_RETURN(NULL);
		g->ops[k]->run(g->instance, k);
	}
	return (PyObject *) new_PyKarray(g->instance[g->ops.size()-1]);
	// Py_RETURN_NONE;
}






PyObject *
Graph_shapes(PyGraph *graph, PyObject *Py_UNUSED(ignored)) {
	Graph * g = &graph->g;
    for (auto k : g->instance)
    	k.shape.print();
    Py_RETURN_NONE;
}

PyObject *
Graph_values(PyGraph *graph, PyObject *Py_UNUSED(ignored)) {
	Graph * g = &graph->g;
    for (auto k : g->instance)
    	k.print();
    Py_RETURN_NONE;
}