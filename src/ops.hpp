
class Op {
public:
	std::string name = "op";
	std::vector<int> children;
	std::vector<int> operands;

	virtual void execute(std::vector<Karray *> & instance, size_t pos) {};

	void add_child(int i) {
		children.push_back(i);
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

class ElementwiseBinaryOp: public Op {
public:
	binary_kernel kernel;
	size_t length;

	ElementwiseBinaryOp(binary_kernel ker, std::string opname = "binary op") :
		length {0}, kernel {ker} {
		name = opname;
	};

	void execute(std::vector<Karray *> & instance, size_t pos) {
		kernel(instance[pos]->data,
		       instance[operands[0]]->data,
		       instance[operands[1]]->data,
		       length);
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
};

class LoadInput: public Op {
public:

	LoadInput(std::string & inp) {
		name = inp;
	};
};