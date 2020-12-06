
void Graph::print(const char * msg) {
	std::cout << "graph: " << msg << "\n";
	for (auto op : ops)
		std::cout << "\t" << op->str() << "\n";
}

void Graph::run() {
	for (int k = 0; k < ops.size(); ++k) {
		ops[k]->run(instance, k);
	}
}

void Graph::load(PyObject *const *args, Py_ssize_t nargs, bool check_shapes) {
	if (nargs != inputs.size()) {
		PyErr_Format(Karray_error,
		             "Wrong number of arguments, expecting %i but got %i.",
		             inputs.size(), nargs);
		return;
	}
	for (int k = 0; k < nargs; ++k) {
		if (py_type(args[k]) != KARRAY) {
			PyErr_SetString(Karray_error,
			                "Only kipr.arr is a valid input.");
			return;
		}
		auto karr = reinterpret_cast<PyKarray *>(args[k]);
		if (check_shapes && instance[inputs[k]].shape != karr->arr.shape) {
			PyErr_Format(Karray_error,
			             "Shapes of input %i %s does not match instantiated shape %s.",
			             k, karr->arr.shape.str().c_str(), instance[inputs[k]].shape.str().c_str());
			return;
		}
		instance[inputs[k]] = karr->arr;
	}
}

void Graph::back() {
	for (int k = ops.size() - 1; k >= 0; --k) {
		ops[k]->back(instance, ops, k);
	}
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
		return new ReluOp {};
	if (function == SOFTMAX_FUNCTION)
		return new SoftmaxOp {};
	if (function == BINARY_ADD || function == INPLACE_ADD)
		return new EWBinaryOp<Add> {};
	if (function == BINARY_SUBTRACT || function == INPLACE_SUBTRACT)
		return new EWBinaryOp<Sub> {};
	if (function == BINARY_MULTIPLY || function == INPLACE_MULTIPLY)
		return new EWBinaryOp<Mul> {};
	if (function == BINARY_TRUE_DIVIDE || function == INPLACE_TRUE_DIVIDE)
		return new EWBinaryOp<Div> {};
	if (function == BINARY_MATRIX_MULTIPLY || function == INPLACE_MATRIX_MULTIPLY)
		return new MatMulOp {};
	if (function == UNARY_NEGATIVE)
		return new UnaryNegative {};
	return new Op {};
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
	auto freevars = fast_to_vector<std::string>(code_ob->co_freevars);
	std::stack<size_t> stack;
	std::map<int, size_t> local;
	std::map<int, size_t> global;

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

		case (INPLACE_MATRIX_MULTIPLY):
		case (INPLACE_ADD):
		case (INPLACE_MULTIPLY):
		case (INPLACE_SUBTRACT):
		case (INPLACE_TRUE_DIVIDE): {
			size_t b = stack.top(); stack.pop();
			size_t a = stack.top(); stack.pop();
			Op * oper = func_to_op(op);
			oper->operands.push_back(a);
			oper->operands.push_back(b);
			stack.push(g->ops.size());
			g->ops[a]->add_child(stack.top());
			g->ops[b]->add_child(stack.top());
			g->ops.push_back(oper);
			local[a] = stack.top();
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

		case (LOAD_DEREF):
			PyErr_Format(Karray_error, 
				"Only global and local variables are not supported inside the graph. %s is a free variable.",
				freevars[arg].c_str());
			return -1;

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

PyObject * Graph_prepare(PyGraph *self, PyObject *const *args, Py_ssize_t nargs) {
	Graph * g = &self->g;

	g->load(args, nargs, false);
	IF_ERROR_RETURN(NULL);

	for (int k = 0; k < g->ops.size(); ++k) {
		g->ops[k]->prepare(g->instance, k);
		IF_ERROR_RETURN(NULL);
		g->ops[k]->run(g->instance, k);
	}
	return (PyObject *) new_PyKarray(g->instance[g->ops.size() - 1]);
}

PyObject * Graph_run(PyGraph *self, PyObject *const *args, Py_ssize_t nargs) {
	Graph * g = &self->g;

	g->load(args, nargs, true);
	IF_ERROR_RETURN(NULL);
	g->run();

	return (PyObject *) new_PyKarray(g->instance[g->ops.size() - 1]);
}

PyObject * Graph_backprop(PyGraph *self, PyObject *const *args, Py_ssize_t nargs) {
	Graph * g = &self->g;
	g->load(args, nargs - 1, true);
	IF_ERROR_RETURN(NULL);
	g->run();

	Karray * ret = &g->instance[g->inputs.size() - 1];

	if (py_type(args[nargs - 1]) != KARRAY) {
		PyErr_SetString(Karray_error,
		                "Only kipr.arr is a valid input for the target.");
		return NULL;
	}
	Karray * target = &reinterpret_cast<PyKarray *>(args[nargs - 1])->arr;
	size_t length;
	if ((length = target->shape.length) != ret->shape.length) {
		PyErr_Format(Karray_error,
		             "Target size does not match graph output size. Shapes are %s and %s.",
		             target->shape.str().c_str(),
		             ret->shape.str().c_str());
		return NULL;
	}
	sub_kernel(ret->data, ret->data, target->data, length);

	g->back();

	Py_RETURN_NONE;
}






PyObject *
Graph_shapes(PyGraph *graph, void *closure) {
	Graph * g = &graph->g;
	PyObject * result = PyList_New(g->ops.size());
	for (int k = 0; k < g->ops.size(); ++k) {
		PyList_SetItem(result, k, g->instance[k].shape.as_tuple());
	}
	// Py_INCREF(result);
	return result;
}

PyObject *
Graph_values(PyGraph *graph, PyObject *Py_UNUSED(ignored)) {
	Graph * g = &graph->g;
	for (auto k : g->instance)
		k.print();
	Py_RETURN_NONE;
}


PyObject *
Graph_instance(PyGraph *graph, void *closure) {
	Graph * g = &graph->g;
	PyObject * result = PyList_New(g->ops.size());
	for (int k = 0; k < g->ops.size(); ++k) {
		PyKarray * res = new_PyKarray(g->instance[k]);
		PyList_SetItem(result, k, reinterpret_cast<PyObject *>(res));
	}
	// Py_INCREF(result);
	return result;
}