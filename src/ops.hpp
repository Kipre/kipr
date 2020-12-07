struct ElementwiseOperation {
	char name[4];
	binary_kernel kernel;
	binary_kernel dkernel;
	binary_op op;
};

constexpr ElementwiseOperation Add {"add", add_kernel, add_dkernel, _add};
constexpr ElementwiseOperation Sub {"sub", sub_kernel, sub_dkernel, _sub};
constexpr ElementwiseOperation Mul {"mul", mul_kernel, mul_dkernel, _mul};
constexpr ElementwiseOperation Div {"div", div_kernel, div_dkernel, _div};



class Op {
public:
	size_t id;
	std::string name = "op";
	std::vector<Op *> children;
	std::vector<Op *> operands;
	Karray arr;

	virtual void run() {};
	virtual void prepare() {};
	virtual void back() {};

	void add_child(Op * child) {
		children.push_back(child);
	};

	std::string str() {
		std::ostringstream ss;
		ss << id << " [";
		for (auto a : operands)
			ss << a->id << ", ";
		ss << "] -> " << name << " -> [";
		for (auto a : children)
			ss << a->id << ", ";
		ss << "]";
		return ss.str();
	};

	void print() {
		arr.print(name.c_str());
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


	EWBinaryOp(size_t new_id) {
		id = new_id;
		name = ope.name;
	};

	void prepare() {
		Karray * lhs = &operands[0]->arr;
		Karray * rhs = &operands[1]->arr;

		if (rhs->shape.length == lhs->shape.length) {
			simple = true;
			length = rhs->shape.length;
			arr.reset(rhs->shape);
		} else {
			simple = false;
			auto [new_common, new_rstr, new_lstr] = paired_strides(rhs->shape, lhs->shape);
			IF_ERROR_RETURN();
			common = new_common;
			rstr = new_rstr;
			lstr = new_lstr;
			arr.reset(common);
		}
	};

	void run() {
		if (simple) {
			ope.kernel(arr.data, operands[0]->arr.data, operands[1]->arr.data, length);
		} else {
			Positions posi {0, 0, 0};
			rec_binary_op(arr.data, operands[0]->arr.data, operands[1]->arr.data,
			              common, lstr, rstr, &posi, ope.op, 0);
		}
	};

	void back() {
		Karray * lhs = &operands[0]->arr;
		Karray * rhs = &operands[1]->arr;

		if (simple) {
			ope.dkernel(arr.data, lhs->data, rhs->data, length);
		} else {
			PyErr_Format(PyExc_NotImplementedError,
			             "not 'simple' %s is not implemented",
			             name);
		}
	};
};

class ReluOp: public Op {
public:
	size_t length;

	ReluOp(size_t new_id) {
		id = new_id;
		name = "relu";
	};

	void prepare() {
		arr.reset(operands[0]->arr.shape);
		length = operands[0]->arr.shape.length;
	};

	void run() {
		val_max_kernel(arr.data, operands[0]->arr.data, 0, length);
	};

	void back() {};
};



class UnaryNegative: public Op {
public:
	size_t length;

	UnaryNegative(size_t new_id) {
		id = new_id;
		name = "negative";
	};

	void prepare() {
		Karray * arg = &operands[0]->arr;
		arr.reset(arg->shape);
		length = arg->shape.length;
	};

	void run() {
		val_mul_kernel(arr.data, operands[0]->arr.data, -1, length);
	};
};




class SoftmaxOp: public Op {
public:
	size_t length {};
	size_t ax {};

	SoftmaxOp(size_t new_id) {
		id = new_id;
		name = "softmax";
	};

	void prepare() {
		Karray * arg = &operands[0]->arr;
		arr.reset(arg->shape);
		length = arg->shape.length;

		ax = arg->shape.last_axis();
	};

	void run() {
		Karray * arg = &operands[0]->arr;

		exp_kernel(arr.data, arg->data, length);
		Karray summed_exp = arr.sum(ax, Karray(1.), false);
		summed_exp.shape.insert_one((int) ax);
		Positions posi {0, 0, 0};
		auto [lstr, rstr] = arg->shape.paired_strides(summed_exp.shape);
		rec_binary_op(arr.data, arr.data, summed_exp.data, arg->shape,
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

	MatMulOp(size_t new_id) {
		id = new_id;
		name = "matmul";
	};

	void prepare() {
		Karray * lhs = &operands[0]->arr;
		Karray * rhs = &operands[1]->arr;

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

		arr.reset(new_shape);
	};

	void run() {
		Karray * lhs = &operands[0]->arr;
		Karray * rhs = &operands[1]->arr;

		for (int m = 0; m < max(M, N); ++m) {
			int ia = m % M;
			int ib = m % N;

			general_matmul(arr.data + m * I * J,
			               lhs->data + ia * I * K,
			               rhs->data + ib * K * J,
			               I, J, K);
		}
	};
};

class LoadGlobal: public Op {
public:

	LoadGlobal(std::string & inp, size_t new_id) {
		id = new_id;
		name = inp;
	};

	void prepare() {
		PyObject * globals = PyEval_GetGlobals();
		PyObject * karr = PyDict_GetItemString(globals, name.c_str());
		arr = reinterpret_cast<PyKarray *>(karr)->arr;
	};
};

class LoadInput: public Op {
public:

	LoadInput(std::string & inp, size_t new_id) {
		id = new_id;
		name = inp;
	};
};