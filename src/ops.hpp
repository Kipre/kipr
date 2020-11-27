typedef struct ElementwiseOperation {
	char name[4];
	binary_kernel kernel;
	binary_op op;
};

constexpr ElementwiseOperation Add {"add", add_kernel, _add};
constexpr ElementwiseOperation Sub {"sub", sub_kernel, _sub};
constexpr ElementwiseOperation Mul {"mul", mul_kernel, _mul};
constexpr ElementwiseOperation Div {"div", div_kernel, _div};





class Op {
public:
	std::string name = "op";
	std::vector<int> children;
	std::vector<int> operands;

	virtual void execute(std::vector<Karray> & v, size_t pos) {};
	virtual void run(std::vector<Karray> & v, size_t pos) {
		printf("default run\n");
	};
	virtual void prepare(std::vector<Karray> & v, size_t pos) {
		printf("default prepare\n");
	};

	void add_child(size_t i) {
		children.push_back((int) i);
	};

	std::string str() {
		std::ostringstream ss;
		ss << '[';
		for (auto a : operands)
			ss << a << ", ";
		ss << "] -> " << name << " -> [";
		for (auto a : children)
			ss << a << ", ";
		ss << "]";
		return ss.str();
	};
};


template<ElementwiseOperation ope>
class EWBinaryOp: public Op {
public:
	size_t length {0};
	bool simple = true;
	NDVector rstr {};
	NDVector lstr {};
	Shape common {};


	EWBinaryOp() {
		name = ope.name;
	};

	void prepare(std::vector<Karray> & v, size_t pos) {
		Karray * rhs = &v[operands[0]];
		Karray * lhs = &v[operands[1]];
		Karray * dest = &v[pos];

		printf("%s\n", name);
		rhs->shape.print("rhs");
		lhs->shape.print("lhs");

		if (rhs->shape.length == lhs->shape.length) {
			simple = true;
			length = rhs->shape.length;
			dest->reset(rhs->shape);
		} else {
			simple = false;
			auto [new_common, new_rstr, new_lstr] = paired_strides(rhs->shape, lhs->shape);
			IF_ERROR_RETURN();
			common = new_common;
			rstr = new_rstr;
			lstr = new_lstr;
			dest->reset(common);
		}
	};

	void run(std::vector<Karray> & v, size_t pos) {
		if (simple) {
			ope.kernel(v[pos].data, v[operands[0]].data, v[operands[1]].data, length);
		} else {
			Positions posi {0, 0, 0};
			rec_binary_op(v[pos].data, v[operands[0]].data, v[operands[1]].data,
			              common, rstr, lstr, &posi, ope.op, 0);
		}
	};
};

class ElementwiseUnaryOp: public Op {
public:
	unary_kernel kernel;
	size_t length;

	ElementwiseUnaryOp(unary_kernel ker, std::string opname = "unary op") :
		length {0}, kernel {ker} {
		name = opname;
	};

	void execute(std::vector<Karray *> & instance, size_t pos) {
		kernel(instance[pos]->data,
		       instance[operands[0]]->data,
		       length);
	};
};

class SoftmaxOp: public Op {
public:
	size_t length;

	SoftmaxOp() :
		length{0} {
		name = "softmax";
	};

	// void execute(std::vector<Karray *> & instance, size_t pos) {
	// 	int ax = instance[operands[0]]->shape.last_axis();
	// 	exp_kernel(instance[pos]->data, instance[operands[0]]->data, length);
	// 	Karray summed_exp = instance[pos]->sum(ax, Karray(1.), false);
	// 	summed_exp.shape.insert_one(ax);
	// 	instance[operands[0]]->operator/=(summed_exp);
	// };
};

class MatMulOp: public Op {
public:
	size_t M;
	size_t N;
	size_t I;
	size_t J;
	size_t K;

	MatMulOp() :
		M{0}, N{0}, I{0}, J{0}, K{0} {
		name = "matmul";
	};
};

class LoadGlobal: public Op {
public:

	LoadGlobal(std::string & inp) {
		name = inp;
	};

	void prepare(std::vector<Karray> & v, size_t pos) {
		PyObject * globals = PyEval_GetGlobals();
		PyObject * karr = PyDict_GetItemString(globals, name.c_str());
		v[pos] = reinterpret_cast<PyKarray *>(karr)->arr;
	};
};

class LoadInput: public Op {
public:

	LoadInput(std::string & inp) {
		name = inp;
	};
};