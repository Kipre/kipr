import unittest
import kipr as kp
import numpy as np

max_nd = 8
nb_random_checks = 5

def rand_shape(nd=None):
    if not nd:
        nd = np.random.randint(1, max_nd + 1)
    shape = np.random.randint(1,5, size=(nd))
    return shape.tolist()

def rand_shape_and_size(nd=None):
    if not nd:
        nd = np.random.randint(1, max_nd + 1)
    shape = np.random.randint(1,5, size=(nd))
    return shape.tolist(), int(shape.prod())




class TestKarrayInternals(unittest.TestCase):

    def test_internals(self):
        print('test_internals')
        self.assertTrue(kp.internal_test())

class TestModuleFunctions(unittest.TestCase):

    def test_relu(self):
        print('test_relu')

        for k in range(nb_random_checks):
            shape = rand_shape()

            ka = kp.arr('rand', shape=shape)
            na = ka.numpy()

            np.testing.assert_almost_equal(
                kp.relu(ka).numpy(), 
                na * (na > 0),
                err_msg=f"{shape = }"
            )




class TestKarrayObject(unittest.TestCase):

    def test_init(self):
        print('test_init')

        # at least one argument
        with self.assertRaises(TypeError):
            kp.arr()

        # too deep
        with self.assertRaises(TypeError):
            kp.arr([[[[[[[[[[1]]]]]]]]]])

        # data must be a sequence of numbers
        with self.assertRaises(TypeError):
            kp.arr(['a', 'b'])
        
        # shouldn't be any zeros in shape
        with self.assertRaises(kp.KarrayError):
            kp.arr(1, shape=[0, 1])
        
        # shape len must be > 1
        with self.assertRaises(kp.KarrayError):
            kp.arr(1, shape=[])
        
        # shape must be a sequence
        with self.assertRaises(kp.KarrayError):
            kp.arr(1, shape=2)
        
        # shape must be a sequence
        with self.assertRaises(kp.KarrayError):
            kp.arr(1, shape=np.array([]))

        # BUG
        # self.assertTrue(kp.arr(1, shape=(1)))
        
        # sanity check
        self.assertTrue(kp.arr(1))
        
        np.testing.assert_almost_equal(
            kp.arr(1).numpy(), 
            np.array([1]),
            err_msg="simple 1"
        )

        np.testing.assert_almost_equal(
            kp.arr(range(2), shape=[1, 1, 1, 2]).numpy(), 
            np.arange(2).reshape([1, 1, 1, 2]),
            err_msg="simple 2"
        )


        for k in range(nb_random_checks):
            shape = rand_shape()

            np.testing.assert_almost_equal(
                kp.arr('range', shape=shape).numpy(), 
                np.array(range(np.array(shape).prod())).reshape(shape),
                err_msg=f'{shape = }'
            )

        for k in range(nb_random_checks):
            shape = rand_shape()

            a = np.random.rand(*shape)
            np.testing.assert_almost_equal(
                kp.arr(a).numpy(), 
                a,
                err_msg=f'{shape = }'
            )

    def test_reshape(self):
        print('test_reshape')

        with self.assertRaises(kp.KarrayError):
            kp.arr(1).reshape([2])

        with self.assertRaises(kp.KarrayError):
            kp.arr(1).reshape([])

        with self.assertRaises(kp.KarrayError):
            kp.arr(1).reshape(np.array([]))

        for k in range(nb_random_checks):
            shape, size = rand_shape_and_size()
            np.testing.assert_almost_equal(
                kp.arr('range', shape=[size]).reshape(shape).numpy(), 
                np.array(range(size)).reshape(shape),
                err_msg=f'reshape to {shape}'
            )

    def test_print(self):

        print('test_print')
        print(kp.arr(1, shape=[2, 3]))

    def test_shape_attr(self):
        print('test_shape_attr')

        shape = [3, 4, 5]

        
        with self.assertRaises(AttributeError):
            kp.arr(1).shape = [1, 2]

        self.assertEqual(
            kp.arr(1, shape=shape).shape,
            tuple(shape)
        )


    def test_subscript(self):
        print('test_subscript')

        a = kp.arr('range', shape=[5, 5, 5, 5, 5])
        b = a.numpy()

        
        with self.assertRaises(kp.KarrayError):
            a[..., ...]

        with self.assertRaises(ValueError):
            a[5]

        with self.assertRaises(TypeError):
            a[1.3]

        with self.assertRaises(ValueError):
            print(a[-7])

        with self.assertRaises(ValueError):
            a[..., 5]


        a[1:1]

        c = kp.arr([1, 2])
        c[[1, 0]]

        subscripts = [(1),
                      (-1),
                      (-3),
                      (-3, -3, slice(-1, 0, -1)),
                      (0, ...),
                      (..., 0),
                      (..., 4),
                      (slice(None),slice(None)),
                      (..., slice(1, 2, 3)),
                      (..., (1, 2, 3)),
                      (1, 2, ..., slice(1, 2, 3)),
                      (slice(None), ..., slice(1, 2, 3)),
                      (..., slice(None), slice(1, 2, 3)),
                      (0, 1, 2, 3, 4)]

        for subscript in subscripts:
            np.testing.assert_almost_equal(
                a[subscript].numpy(), 
                b[subscript],
                err_msg=f"error occured with {subscript = }"
            )

    def test_broadcast(self):

        print('test_broadcast')
        a = kp.arr('range', shape=[1, 4])
        np.testing.assert_almost_equal(
            a.broadcast([3, 4]).numpy(), 
            np.array([[0, 1, 2, 3],
                      [0, 1, 2, 3],
                      [0, 1, 2, 3]])
        )


    def runTest(self):
        self.test_init()
        self.test_subscript()
        self.test_reshape()
        self.test_shape_attr()
        self.test_print()

class TestKarrayMath(unittest.TestCase):

    pair_shapes = [((1,), (2,)),
                   ((1,), (4, 2)),
                   ((1,), (4, 2, 8)),
                   ((4, 2, 8), (1,)),
                   ((4, 1), (4, 4, 8)),
                   ((5, 4), (4, 5, 4)),
                   ((1, 1), (4, 2, 8)),
                   ((1, 5), (4, 3, 5)),
                   ((1, 5), (4, 1, 5)),
                   ((4, 3, 1, 5, 6, 1), (7, 1, 1, 6)),]
    pair_shapes_one_side = [((1,), (2,)),
                           ((1,), (4, 2)),
                           ((1,), (4, 2, 8)),
                           ((4, 1), (4, 4, 8)),
                           ((5, 4), (4, 5, 4)),
                           ((1, 1), (4, 2, 8)),
                           ((1, 5), (4, 3, 5)),
                           ((1, 5), (4, 1, 5)),]


    def test_add(self):

        print('test_add')

        with self.assertRaises(TypeError):
            kp.arr(1) + ''

        with self.assertRaises(TypeError):
            kp.arr(1) + '33'
            
        with self.assertRaises(TypeError):
            kp.arr(1) + 1
            
        with self.assertRaises(TypeError):
            1 + kp.arr(1)


        np.testing.assert_almost_equal(
            (kp.arr(1) + kp.arr(2)).numpy(), 
            [3]
        )

        for k in range(nb_random_checks):
            shape = rand_shape()

            a = np.random.rand(*shape)
            b = np.random.rand(*shape)
            
            np.testing.assert_almost_equal(
                (kp.arr(a) + kp.arr(b)).numpy(), 
                a + b,
                err_msg=f"{shape = }"
            )

        for sa, sb in self.pair_shapes:
            na = np.random.rand(*sa)
            nb = np.random.rand(*sb)

            a = kp.arr(na)
            b = kp.arr(nb)

            np.testing.assert_almost_equal(
                (a + b).numpy(), 
                na + nb,
                err_msg=f"{sa = }, {sb = }"
            )

            np.testing.assert_almost_equal(
                (b + a).numpy(), 
                na + nb,
                err_msg=f"{sb = }, {sa = }"
            )

    def test_inplace_add(self):

        print('test_inplace_add')

        ka = kp.arr(1)

        ka += kp.arr(2)
        np.testing.assert_almost_equal(
            ka.numpy(), 
            [3]
        )

        for k in range(nb_random_checks):
            shape = rand_shape()

            a = np.random.rand(*shape).astype(np.float32)
            b = np.random.rand(*shape).astype(np.float32)

            ka = kp.arr(a)
            ka += kp.arr(b)

            np.testing.assert_almost_equal(
                ka.numpy(), 
                a + b
            )

        for sa, sb in self.pair_shapes_one_side:
            na = np.random.rand(*sa)
            nb = np.random.rand(*sb)

            a = kp.arr(na)
            b = kp.arr(nb)

            b += a
            np.testing.assert_almost_equal(
                b.numpy(), 
                na + nb,
                err_msg=f"{sa = }, {sb = }"
            )

    def test_sub(self):

        print('test_sub')

        np.testing.assert_almost_equal(
            (kp.arr(2) - kp.arr(3)).numpy(), 
            [-1]
        )

    def test_inplace_sub(self):

        print('test_inplace_sub')

        a = kp.arr(2);
        a -= kp.arr(1)

        np.testing.assert_almost_equal(
            a.numpy(), 
            [1],
        )

    def test_mul(self):

        print('test_mul')

        np.testing.assert_almost_equal(
            (kp.arr(2) * kp.arr(3)).numpy(), 
            [6]
        )

    def test_inplace_mul(self):

        print('test_inplace_mul')

        a = kp.arr(2);
        a *= kp.arr(1.5)
        
        np.testing.assert_almost_equal(
            a.numpy(), 
            [3]
        )

    def test_div(self):

        print('test_div')

        np.testing.assert_almost_equal(
            (kp.arr(2) / kp.arr(3)).numpy(), 
            [0.6666666666666]
        )

    def test_inplace_div(self):

        print('test_inplace_div')

        a = kp.arr(2);
        a /= kp.arr(1.5)
        
        np.testing.assert_almost_equal(
            a.numpy(), 
            [1.33333333]
        )

    def test_matmul(self):

        print('test_matmul')
        a = kp.arr('range', shape=[3, 3]).reshape([1, 3, 3]).broadcast([2, 3, 3])
        b = kp.arr('range', shape=[3, 2])

        np.testing.assert_almost_equal(
            (a @ b).numpy(), 
            np.array([[[10., 13.],
                       [28., 40.],
                       [46., 67.]],
                      [[10., 13.],
                       [28., 40.],
                       [46., 67.]]])
        )

    def test_mean(self):

        print('test_mean')

        a = np.random.rand(2, 3, 4).astype(np.float32)
        ka = kp.arr(a)

        np.testing.assert_almost_equal(
            ka.mean().numpy(), 
            a.mean()
        )

        np.testing.assert_almost_equal(
            ka.mean(0).numpy(), 
            a.mean(0)
        )

        np.testing.assert_almost_equal(
            ka.mean(1).numpy(), 
            a.mean(1)
        )

        np.testing.assert_almost_equal(
            ka.mean(2).numpy(), 
            a.mean(2)
        )

        np.testing.assert_almost_equal(
            ka.mean(-1).numpy(), 
            a.mean(-1)
        )

    def runTest(self):
        self.test_add()
        self.test_inplace_add()
        self.test_sub()
        self.test_inplace_sub()
        self.test_mul()
        self.test_inplace_mul()
        self.test_div()
        self.test_inplace_div()



class TestGraph(unittest.TestCase):

    def test_init(self):
        global v, w, y, z, d

        v = kp.arr('randn', shape=(2,))
        w = kp.arr('randn', shape=(2,))
        y = kp.arr('randn', shape=(2,))
        z = kp.arr('randn', shape=(2,))
        d = kp.arr([[1, .3], [0.5, -1]])

        evaluator = np.random.randn(3, 2)

        def f(x):
            x = x @ ((d + v - w)* y / z)
            return -x

        g = kp.graph(f)
        result = g.compile(kp.arr(evaluator))

        v = v.numpy()
        w = w.numpy()
        y = y.numpy()
        z = z.numpy()
        d = d.numpy()

        np.testing.assert_almost_equal(
            result.numpy(), 
            f(evaluator),
            decimal=5,
            err_msg="behavior of graph and function differs"
        )




if __name__ == '__main__':
    unittest.main()
    # TestKarray().test_init()
    # TestKarray().test_subscript()
    # TestKarrayObject().test_init()
    # TestKarrayInternals().test_internals()

    # TestKarrayMath().test_inplace_add()
    # TestKarrayMath().run()
    # TestKarrayObject().run()


    




