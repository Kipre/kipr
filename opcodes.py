import dis

def generate_opcodes():
	result = "\n"
	for name, code in dis.opmap.items():
		result += f"const int {name} = {code};\n"
	return result

with open('opcodes.hpp', 'w') as f:
	f.write(generate_opcodes())
