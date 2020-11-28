struct ElementwiseOperation {
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
	std::vector<size_t> children;
	std::vector<size_t> operands;

	virtual void execute(std::vector<Karray> & v, size_t pos) {};
	virtual void run(std::vector<Karray> & v, size_t pos) {
		// printf("default run\n");
	};
	virtual void prepare(std::vector<Karray> & v, size_t pos) {
		// printf("default prepare\n");
	};
	virtual void back(std::vector<Karray> & v, size_t pos) {};

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
		Karray * lhs = &v[operands[0]];
		Karray * rhs = &v[operands[1]];
		Karray * dest = &v[pos];

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
			              common, lstr, rstr, &posi, ope.op, 0);
		}
	};
};

class ReluOp: public Op {
public:
	size_t length;

	ReluOp() {
		name = "relu";
	};

	void prepare(std::vector<Karray> & v, size_t pos) {
		Karray * arg = &v[operands[0]];
		v[pos].reset(arg->shape);
		length = arg->shape.length;
	};

	void run(std::vector<Karray> & v, size_t pos) {
		val_max_kernel(v[pos].data, v[operands[0]].data, 0, length);
	};

	void back(std::vector<Karray> & v, size_t pos) {
		Karray * arg = &v[operands[0]];

		size_t i = 0;
		while(i < length) {
			arg->data[i] = v[pos].data[i] * (arg->data[i] > 0);
		}
	};
};



class UnaryNegative: public Op {
public:
	size_t length;

	UnaryNegative() {
		name = "negative";
	};

	void prepare(std::vector<Karray> & v, size_t pos) {
		Karray * arg = &v[operands[0]];
		v[pos].reset(arg->shape);
		length = arg->shape.length;
	};

	void run(std::vector<Karray> & v, size_t pos) {
		val_mul_kernel(v[pos].data, v[operands[0]].data, -1, length);
	};
};




class SoftmaxOp: public Op {
public:
	size_t length {};
	size_t ax {};

	SoftmaxOp() {
		name = "softmax";
	};

	void prepare(std::vector<Karray> & v, size_t pos) {
		Karray * arg = &v[operands[0]];
		v[pos].reset(arg->shape);
		length = arg->shape.length;

		ax = arg->shape.last_axis();
	};

	void run(std::vector<Karray> & v, size_t pos) {
		Karray * arg = &v[operands[0]];
		
		exp_kernel(v[pos].data, arg->data, length);
		Karray summed_exp = v[pos].sum(ax, Karray(1.), false);
		summed_exp.shape.insert_one((int) ax);
		// summed_exp.print();
		// summed_exp = summed_exp.broadcast(arg->shape);
		// summed_exp.print();
		// div_kernel(v[pos].data,  v[pos].data, summed_exp.data, length);
		Positions posi {0, 0, 0};
		auto [lstr, rstr] = arg->shape.paired_strides(summed_exp.shape);
		rec_binary_op(v[pos].data, v[pos].data, summed_exp.data, arg->shape,
                          lstr, rstr, &posi, _div, 0);
	};
};

class MatMulOp: public Op {
public:
	size_t M {};
	size_t N {};
	size_t I {};
	size_t J {};
	size_t K {};

	MatMulOp() {
		name = "matmul";
	};

	void prepare(std::vector<Karray> & v, size_t pos) {
		Karray * lhs = &v[operands[0]];
		Karray * rhs = &v[operands[1]];
		Karray * dest = &v[pos];

		if (lhs->shape.nd < 2 && rhs->shape.nd < 2) {
			PyErr_SetString(Karray_error, "Both arrays must be at least two-dimensional for matmul.");
			return;
		}

		I = lhs->shape[-2];
		K = lhs->shape[-1];
		J = rhs->shape[-1];

		M = lhs->shape.nbmats();
		N = rhs->shape.nbmats();

		if (K != rhs->shape[-2] || (M % N != 0 && N % M != 0)) {
			PyErr_Format(Karray_error,
			             "Matmul not possible with shapes %s and %s.",
			             lhs->shape.str(), rhs->shape.str());
			return;
		}

		Shape new_shape((M > N) ? lhs->shape : rhs->shape);
		new_shape.set(new_shape.nd - 2, I);
		new_shape.set(new_shape.nd - 1, J);

		dest->reset(new_shape);
	};

	void run(std::vector<Karray> & v, size_t pos) {
		Karray * lhs = &v[operands[0]];
		Karray * rhs = &v[operands[1]];
		Karray * dest = &v[pos];

		for (int m = 0; m < max(M, N); ++m) {
			int ia = m % M;
			int ib = m % N;

			general_matmul(dest->data + m * I * J,
			               lhs->data + ia * I * K,
			               rhs->data + ib * K * J,
			               I, J, K);
		}
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