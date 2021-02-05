
template<binary_op op>
inline void rec_binary_op(float * dest, float * lhs, float * rhs, Shape &shape,
                          NDVector &l_strides, NDVector &r_strides, 
                          Positions * pos, int depth) {
	if (depth < shape.nd - 1) {
		for (int k = 0; k < shape[depth]; ++k) {
			rec_binary_op<op>(dest, lhs, rhs, shape, l_strides, r_strides, pos, depth + 1);
			pos->right += l_strides[depth];
			pos->left += r_strides[depth];
		}
		pos->right -= l_strides[depth] * shape[depth];
		pos->left -= r_strides[depth] * shape[depth];
	} else {
		for (int k = 0; k < shape[depth]; ++k) {
			dest[pos->write++] = op(lhs[pos->right + l_strides[depth] * k],
			                        rhs[pos->left + r_strides[depth] * k]);
		}
	}
}

class Op {
public:
	size_t id;
	std::string name = "op";
	std::vector<Op *> children;
	bool overwrite_gradients = true;

	Karray arr;
	Karray grad;

	virtual void run() {
		overwrite_gradients = true;
	};
	virtual void prepare() {};
	virtual void back() {};
	virtual std::vector<Op *> operands() { return {}; };

	void reset(Shape & new_shape) {
		arr.reset(new_shape);
		grad.reset(new_shape);
	}

	void add_child(Op * child) {
		children.push_back(child);
	};

	std::string str() {
		std::ostringstream ss;
		ss << id << " [";
		for (auto a : operands())
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

template<binary_kernel kernel, binary_kernel dkernel, binary_op op>
class EWBinaryOp: public Op {
public:
	size_t length {0};
	bool simple = true;
	Op * larg; // left hand side...
	Op * rarg;
	NDVector lstr {};
	NDVector rstr {};
	Shape common {};

	EWBinaryOp(size_t new_id, Op * new_lop, Op * new_rop) :
		larg {new_lop}, rarg {new_rop} {
		id = new_id;
		name = name;
	};

	void prepare() {
		Karray * lhs = &larg->arr, * rhs = &rarg->arr;

		if (rhs->shape.length == lhs->shape.length) {
			simple = true;
			length = rhs->shape.length;
			reset(rhs->shape);
		} else {
			simple = false;
			auto [new_common, new_rstr, new_lstr] = paired_strides(rhs->shape, lhs->shape);
			IF_ERROR_RETURN();
			common = new_common;
			rstr = new_rstr;
			lstr = new_lstr;
			reset(common);
		}
	};

	void run() {
		Op::run();
		Karray * lhs = &larg->arr, * rhs = &rarg->arr;

		if (simple) {
			kernel(arr.data, lhs->data, rhs->data, length);
		} else {
			Positions posi {0, 0, 0};
			rec_binary_op<op>(arr.data, lhs->data, rhs->data,
			              common, lstr, rstr, &posi, 0);
		}
	};

	void back() {
		// if (larg->overwrite_gradients) {
		// 	larg->overwrite_gradients = false;

		// }
		// if (simple) {
		// 	ope.dkernel(arr.data, lhs->data, rhs->data, length);
		// } else {
		// 	PyErr_Format(PyExc_NotImplementedError,
		// 	             "not 'simple' %s is not implemented",
		// 	             name);
		// }
	};

	std::vector<Op *> operands() {
		return std::vector<Op *> {larg, rarg};
	};
};

class ReluOp: public Op {
public:
	size_t length;
	Op * arg;

	ReluOp(size_t new_id, Op * new_arg) :
		arg {new_arg} {
		id = new_id;
		name = "relu";
	};

	void prepare() {
		arr.reset(arg->arr.shape);
		length = arg->arr.shape.length;
	};

	void run() {
		val_max_kernel(arr.data, arg->arr.data, 0, length);
	};

	void back() {
		
	};
};



class UnaryNegative: public Op {
public:
	size_t length;
	Op * arg;

	UnaryNegative(size_t new_id, Op * new_arg) :
		arg {new_arg} {
		id = new_id;
		name = "negative";
	};

	void prepare() {
		arr.reset(arg->arr.shape);
		length = arg->arr.shape.length;
	};

	void run() {
		val_mul_kernel(arr.data, arg->arr.data, -1, length);
	};
};




class SoftmaxOp: public Op {
public:
	size_t length {};
	size_t ax {};
	Op * arg;

	SoftmaxOp(size_t new_id, Op * new_arg) :
		arg {new_arg} {
		id = new_id;
		name = "softmax";
	};

	void prepare() {
		arr.reset(arg->arr.shape);
		length = arg->arr.shape.length;

		ax = arg->arr.shape.last_axis();
	};

	void run() {

		exp_kernel(arr.data, arg->arr.data, length);
		Karray summed_exp = arr.sum(ax, Karray(1.), false);
		summed_exp.shape.insert_one((int) ax);
		Positions posi {0, 0, 0};
		auto [lstr, rstr] = arg->arr.shape.paired_strides(summed_exp.shape);
		rec_binary_op(arr.data, arr.data, summed_exp.data, arg->arr.shape,
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
	Op * larg;
	Op * rarg;

	MatMulOp(size_t new_id, Op * new_lop, Op * new_rop) :
		larg {new_lop}, rarg {new_rop} {
		id = new_id;
		name = "matmul";
	};

	void prepare() {
		Karray * lhs = &larg->arr;
		Karray * rhs = &rarg->arr;

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
		for (int m = 0; m < std::max(M, N); ++m) {
			int ia = m % M;
			int ib = m % N;

			general_matmul(arr.data + m * I * J,
			               larg->arr.data + ia * I * K,
			               rarg->arr.data + ib * K * J,
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