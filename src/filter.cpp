
Filter::Filter(Shape& shape) {
	size_t total = 0;
	int i;
	offset[0] = 0;
	for (i = 0; i < shape.nd;) {
		total += shape[i];
		offset[++i] = total;
	}
	while (i != MAX_ND) {
		offset[i++] = total;
	}
	buf = new size_t[total];
	// std::fill(buf, buf + total, -1);
}

Filter::Filter(Filter&& other) noexcept : buf(nullptr), offset{0} {
	buf = other.buf;
	other.buf = nullptr;
	for (int i = 0; i < MAX_ND; ++i) {
		offset[i] = other.offset[i];
		other.offset[i] = 0;
	}
}

void Filter::set_val_along_axis(int axis , size_t value) {
	// printf("writing val from %i to %i on axis %i\n", offset[axis], offset[axis + 1], axis);
	std::fill(buf + offset[axis], buf + offset[axis + 1], value);
}

void Filter::set_range_along_axis(int axis) {
	// printf("writing range from %i to %i on axis %i\n", offset[axis], offset[axis + 1], axis);
	for (int k = offset[axis]; k < offset[axis + 1]; ++k) {
		buf[k] = k - offset[axis];
	}
}

void Filter::print(const char * message) {
	std::cout << "Filter " << message << "\n\t";
	int o = 1;
	for (int k = 0; k < offset[MAX_ND - 1]; ++k) {
		if (k == offset[o]) {
			std::cout << "| ";
			++o;
		}
		std::cout << buf[k] << ", ";
	}
	std::cout << "\n\toffsets:";
	for (int k = 0; k < MAX_ND; ++k) {
		std::cout << offset[k] << ", ";
	}
	std::cout << '\n';
}

Filter::~Filter() {
	// printf("freeing filter\n");
	delete[] buf;
}

Filter& Filter::operator=(Filter&& other) noexcept {
	if (this != &other) {
		delete[] buf;

		buf = other.buf;
		other.buf = nullptr;
		for (int i = 0; i < MAX_ND; ++i) {
			offset[i] = other.offset[i];
			other.offset[i] = 0;
		}
	}
	return *this;
}

// Filter& Filter::operator=(Filter && other) {
// 	for (int i = 0; i < MAX_ND; ++i) {
// 		offset[i] = other.offset[i];
// 		other.offset[i] = 0;
// 	}
// 	buf = other.buf;
// 	other.buf = nullptr;
// }

// bool Filter::test() {
// 	TEST(FilterInitialization) {
// 		size_t a[MAX_ND] = {3, 4, 5, 0, 0, 0, 0, 0};
// 		Shape shape(a);
// 		Filter f(shape);
// 	}
// }