import dis

def generate_opcodes():
	result = "\n"
	for name, code in dis.opmap.items():
		result += f"const int OP_{name} = {code};\n"
	return result

with open('opcodes.cpp', 'w') as f:
	f.write(generate_opcodes())
