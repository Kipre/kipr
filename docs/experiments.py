import kipr as kp
import numpy as np


def pos(index, strides):
    assert(len(index) == len(strides))
    result = [i * s for i, s in zip(index, strides)]
    return sum(result)

def get_strides(shape):
    acc = 1
    result = []
    for k in shape[-1::-1]:
        result.insert(0, acc)
        acc *= k
    return result

a = np.random.randn(4, 5)
b = np.random.randn(5, 3)

astr = get_strides(a.shape)
bstr = get_strides(b.shape)

I, K, J = a.shape[-2], a.shape[-1], b.shape[-1]
I, J, K
cstr = get_strides([I, J])

astr, bstr, cstr

astr, bstr, cstr = [K, 0, 1], [0, 1, J], [J, 1, 0]

astr, bstr, cstr # = [5, 0, 1], [0, 1, 3], [3, 1, 0]

assert(a.shape[-1] == b.shape[-2])
I, K, J = a.shape[-2], a.shape[-1], b.shape[-1]
a_flat = a.flatten()
b_flat = b.flatten()
c = np.zeros((I*J))
for i in range(I):
    for j in range(J):
        for k in range(K):
            assert(pos((i, j, k), astr) ==  i * K + k)
            assert(pos((i, j, k), bstr) == k * J + j)
            assert(pos((i, j, k), cstr) ==  i * J + j)
            c[pos((i, j, k), cstr)] += a_flat[pos((i, j, k), astr)] * b_flat[pos((i, j, k), bstr)]
            # print(i, j, k)
            print(pos((i, j, k), astr), pos((i, j, k), bstr), " -> ", pos((i, j, k), cstr))
            print(i * K + k, k * J + j, " -> ", i * J + j)
c.reshape(I, J)

a @ b - c.reshape(I, J)





for i in range(I):
    for j in range(J):
        c[i * J + j] = 0
        for k in range(K):
            c[i * J + j] += a[i * K + k] * b[k * J + j]
            print(i * J + j, " <- ",i * K + k, " * ", k * J + j)

for kb in range(0, K, BS):
    for jb in range (0, J, BS):
        for i in range(I):
            for k in range(kb, kb + BS):
                acc1 = a[i * K + k]
                for j in range(jb, jb + BS):
                    c[i * J + j] += acc1 * b[k * J + j]
                    # print(i * J + j, " <- ", "acc1", " * ", k * J + j)
                    # print(kb, jb, i, j, k)



c.reshape(I, J)
np.allclose(, a.reshape(I, K) @ b.reshape(K, J))




class simd:

    width = 8

    def __init__(self, arr=None):
        if arr is not None:
            arr = np.array(arr)
            assert(len(arr.shape) == 1 and len(arr) == self.width)
            self.data = arr
        else:
            self.data = np.random.randn(self.width)

    def _load(self, arr, index):
        self.data = np.zeros(self.width)
        for k in range(self.width):
            self.data[k] = arr[index + k]
        return self

    def _set1(self, val):
        self.data[:] = val
        return self

    @staticmethod
    def set1(val):
        result = simd()
        return result._set1(val)

    @staticmethod
    def load(arr, index):
        result = simd()
        return result._load(arr, index)

    def store(self, arr, index):
        arr[index: index+self.width] = self.data

    def __add__(self, other):
        return simd(self.data + other.data)

    def __mul__(self, other):
        return simd(self.data * other.data)

    def __str__(self):
        result = "simd "
        for k in range(self.width):
            result += str(self.data[k]) + ", "
        return result

    __repr__ = __str__

    @staticmethod
    def fma(a, b, c):
        return a * b + c

regsA, regsB = 3, 4          # blocksizes
I, K, J = regsA * 3, 10, regsB * 3 * simd.width

a = np.random.randn(I * K)
b = np.random.randn(K * J)

c = np.zeros((I * J))
for ii in range(0, I, regsA):
    for jj in range (0, J // simd.width, regsB):
        csum = [[simd.set1(0) for u in range(regsB)] for v in range(regsA)]
        # print(f"{ii = }, {jj = }")
        for k in range(K):
            for j in range(regsB):
                bb = simd.load(b, k * J + (j + jj) * simd.width)
                # print(f"{bb = }")
                for i in range(regsA):
                    aa = simd.set1(a[(i + ii) * K + k])
                    # print(f"{aa = }")
                    # print(i, j, k)
                    csum[i][j] = simd.fma(aa, bb, csum[i][j])
                    # print(i * J + j, " <- ", i * K + k, k * J + j)

        for j in range(regsB):
            for i in range(regsA):
                csum[i][j].store(c, (i + ii) * J + (j + jj) * simd.width)

c.reshape(I, J)

a.reshape(I, K) @ b.reshape(K, J)
