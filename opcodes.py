import dis

additional = ['ERROR_CODE',
              'RANDOM_UNIFORM',
              'RANDOM_NORMAL',
              'RANGE',
              'NO_AXIS',
              'NUMPY_ARRAY',
              'STRING',
              'NUMBER',
              'SEQUENCE',
              'SLICE',
              'KARRAY',
              'KIPR_MODULE',
              'MODULE',
              'RELU_FUNCTION',
              'SOFTMAX_FUNCTION']

def generate_opcodes():
	result = "\n"
	for name, code in dis.opmap.items():
		result += f"const size_t {name} = {code};\n"
	result += "\n"
	for code, name in enumerate(additional):
		result += f"const size_t {name} = {code + 256};\n"
	return result

with open('src/opcodes.hpp', 'w') as f:
	f.write(generate_opcodes())
