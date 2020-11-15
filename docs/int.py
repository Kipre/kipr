import kipr as kp
import matplotlib.pyplot as plt
import networkx as nx
import dis

style = {
    'with_labels': True,
    'font_weight': 'bold',
    'arrowsize': 20,
    'node_size': 1000,
    'node_color': '#faa19b'
}

def f(x, W1, W2, b, y):
    x = relu(W1 @ x + b)
    x = softmax(W2 @ x)
    # loss = -sum(y*log(x))
    return x

a = [2, 4, 1, 5]
b = [5, 2, 4, 4, 5]

def get_strides(shape):
    acc = 1
    result = []
    for k in shape[-1::-1]:
        result.insert(0, acc)
        acc *= k
    return result

def pos(index, strides):
    assert(len(index) == len(strides))
    result = [i * s for i, s in zip(index, strides)]
    return sum(result)

def view_pos(index, strides):
    assert(len(index) == len(strides))
    result = [i * s for i, s in zip(index, strides)]
    return sum(result)

a_strides = get_strides(a)
b_strides = get_strides(b)
c = pos((2, 0, 2, 3, 3),  [0, 20, 5, 5, 1])
pos((0, 2, 0, 3), a_strides), c, c // 20

ar = kp.arr('range', shape=a)

def computation_graph(f):
    G = nx.DiGraph()
    bytecode = dis.Bytecode(f)
    stack = []
    mapping = {a: a for a in bytecode.codeobj.co_varnames}
    dis.dis(f)
    for byte in bytecode:
        # print(byte)
        if byte.opname in ['LOAD_FAST']:
            stack.append(mapping[byte.argval])
        elif byte.opname in ['LOAD_CONST']:
            stack.append(byte.argval)
        elif byte.opname in ['LOAD_GLOBAL']:
            stack.append(byte.argval)
        elif byte.opname in ['BINARY_ADD', 'BINARY_POWER',
                             'BINARY_MULTIPLY', 'BINARY_MATRIX_MULTIPLY']:
            a = stack.pop()
            b = stack.pop()
            val = byte.opname + "_" + str(byte.offset)
            stack.append(val)
            G.add_edge(a, val)
            G.add_edge(b, val)
        elif byte.opname in ['UNARY_NEGATIVE']:
            operand = stack.pop()
            val = byte.opname + "_" + str(byte.offset)
            stack.append(val)
            G.add_edge(operand, val)
        elif byte.opname in ['STORE_FAST']:
            mapping[byte.argval] = stack.pop()
        elif byte.opname in ['CALL_FUNCTION']:
            arg = stack.pop()
            f = stack.pop()
            val = f + "_" + str(byte.offset)
            stack.append(val)
            G.add_edge(arg, val)
        elif byte.opname in ['RETURN_VALUE']:
            arg = stack.pop()
            G.add_edge(arg, byte.opname)
        else:
            raise Exception(f"Unknown operation {byte.opname}")
        # print(stack)
    def group(graph, node, depth):
        graph.nodes[node]['group'] = depth
        for n in graph.predecessors(node):
            group(graph, n, depth + 1)
    group(G, 'RETURN_VALUE', 0)
    return G


G = computation_graph(f)

nx.draw(G,
        pos = nx.multipartite_layout(G, align='horizontal', subset_key="group"),
        **style)

# for n in G.nodes(data=True):
#     print(n)
