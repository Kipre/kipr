py
import kipr as kp
W1 = kp.arr('randn', shape=[20, 30])
W2 = kp.arr('randn', shape=[3, 20])
b = kp.arr('randn', shape=[20, 1])
@kp.graph
def f(x):
    x = kp.relu(W1 @ x + b)
    return kp.softmax(W2 @ x)

f.compile(kp.arr('randn', shape=[12, 30, 1]))
exit()

py
import kipr as kp
v = kp.arr('randn', shape=(2,))
w = kp.arr('randn', shape=(2,))
y = kp.arr('randn', shape=(2,))
z = kp.arr('randn', shape=(2,))
d = kp.arr([[1, .3], [0.5, -1]])

def f(x):
    x = x @ ((d + v - w)* y / z)
    return kp.relu(-x)

g = kp.graph(f)
g.compile(kp.arr(0.2, shape=[3, 2]))
g.shapes
exit()

py
import kipr as kp
@kp.graph
def f(x):
    return kp.relu(x)

f.compile(kp.arr('randn', shape=[2]))
f.backprop(kp.arr([-0.2, 0.4]), kp.arr([1, 1.001]))
f.values()
exit()

py
import kipr as kp
@kp.graph
def f(x):
    return kp.relu(x + x)

f.compile(kp.arr('randn', shape=[2]))
f.backprop(kp.arr([-0.2, 0.4]), kp.arr([1, 1.001]))
f.values()
exit()