import kipr as kp
import matplotlib.pyplot as plt
import dis
from types import ModuleType


style = {
    "rect_width": 20
}


class node:

    def __init__(self, name, operands=[]):
        self.name = name
        self.operands = operands
        self.children = []
        self.x = 0
        self.y = 0

    def is_binary(self):
        return len(self.operands) == 2

    def is_unary(self):
        return len(self.operands) == 1

    def no_args(self):
        return not isinstance(self.operands, list) == 0

    def add_child(self, child):
        self.children.append(child)

    def __str__(self):
        return f'{self.operands} -> {self.name} -> {self.children}, ({self.x}, {self.y})'

    __repr__ = __str__


class weak_graph:

    def __init__(self, func):
        self.ops = []
        self.inputs = []
        glob = globals()
        stack = []
        place = {}
        bytecode = dis.Bytecode(func)
        bytes = bytecode.codeobj.co_code
        names = bytecode.codeobj.co_names
        varnames = bytecode.codeobj.co_varnames
        for opbyte, arg in zip(bytes[0::2], bytes[1::2]):
            op, arg = int(opbyte), int(arg)
            # print(dis.opname[op], arg)
            if dis.opname[op] == 'LOAD_FAST':
                name = varnames[arg]
                if name not in place.keys():
                    place[name] = len(self.ops)
                    self.ops.append(node(name))
                    self.inputs.append(place[name])
                stack.append(place[name])

            elif dis.opname[op] == 'LOAD_GLOBAL':
                name = names[arg]
                var = glob[name]
                if isinstance(var, ModuleType) and var.__name__ == 'kipr':
                    stack.append('kipr')
                elif isinstance(var, kp.arr):
                    if name not in place.keys():
                        place[name] = len(self.ops)
                        self.ops.append(node(name))
                    stack.append(place[name])
                else:
                    raise Exception(f'Unknown external {name} was used.')

            elif dis.opname[op] == 'LOAD_METHOD':
                assert(stack.pop() == 'kipr'), "expected to have kipr on top of stack."
                stack.append("kp." + names[arg])

            elif dis.opname[op] == 'CALL_METHOD':
                assert(arg == 1), 'CALL_METHOD supports only calls with 1 argument.'
                arg = stack.pop()
                call = stack.pop()
                id = len(self.ops)
                stack.append(id)
                self.ops.append(node(call, [arg]))
                self.ops[arg].add_child(id)

            elif 'BINARY' in dis.opname[op]:
                b = stack.pop()
                a = stack.pop()
                id = len(self.ops)
                stack.append(id)
                self.ops.append(node(dis.opname[op], [a, b]))
                self.ops[a].add_child(id)
                self.ops[b].add_child(id)

            elif 'UNARY' in dis.opname[op]:
                a = stack.pop()
                id = len(self.ops)
                stack.append(id)
                self.ops.append(node(dis.opname[op], [a]))
                self.ops[a].add_child(id)

            elif dis.opname[op] == 'STORE_FAST':
                place[varnames[arg]] = stack.pop()

            elif dis.opname[op] == 'RETURN_VALUE':
                self.ret = stack.pop()
                assert(self.ret == len(self.ops) - 1), "Last element in ops should be the return value."

            else:
                raise Exception(f'Unknown op {op}: {dis.opname[op]}.')

            # print(stack)
        self.compute_positions()

    def compute_positions(self):
        def rec_traverse(node, height):
            node.y = height + 1
            result = node.y
            for i in node.operands:
                result = max(rec_traverse(self.ops[i], height + 1), result)
            return result

        self.depth = rec_traverse(self.ops[self.ret], 0)
        self.width = 0
        self.level_width = []
        for i in range(1, self.depth + 1):
            layer = [op for op in self.ops if op.y == i]
            total_width = sum([len(op.name) + 20 for op in layer]) + 20
            width = 0
            for i, op in enumerate(layer):
                width += 10
                op.x = width/total_width
                width += 10 + len(op.name)
            self.width = max(self.width, width)


    def _repr_svg_(self):
        svg_width = 6 * self.width
        result = f'<svg width="{svg_width}" height="{100 * self.depth}">'
        result += """<marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5"
                      markerWidth="6" markerHeight="6"
                      orient="auto-start-reverse">
                      <path d="M 0 0 L 10 5 L 0 10 z" />
                     </marker>"""
        for op in self.ops:
            x, y = op.x * svg_width, (self.depth - op.y) * 100 + 10
            width = (len(op.name) + 20) * 4
            for c in op.children:
                x1, y1 = self.ops[c].x * svg_width, (self.depth - self.ops[c].y) * 100 + 5
                result += f'<line x1="{x + 50}" y1="{y}" x2="{x1 + 50}" y2="{y1}" stroke="black" marker-end="url(#arrow)"/>'
            result += f'<rect x="{x}" y="{y}" width="{width}" height="50" rx="15" ry="15" stroke="black" fill="#bdd6ff"/>'
            result += f'<text x="{x + width // 2}" y="{y + 25}" text-anchor="middle">{op.name}</text>'
        return result + '</svg>'




W1 = kp.arr(1)
W2 = kp.arr(1)
b = kp.arr(1)

@weak_graph
def f(x, p):
    y = kp.relu(W1 @ x + b)
    y = kp.softmax(W2 @ x + p + y)
    return y

f

# for n in G.nodes(data=True):
#     print(n)

dis.dis(f)
bytecode = dis.Bytecode(f)
print(bytecode.codeobj)
for byte in bytecode:
    print(byte.opname, byte.arg)

for a in dir(bytecode.codeobj):
    if '__' not in a:
        print(a, ": ", getattr(bytecode.codeobj, a))
