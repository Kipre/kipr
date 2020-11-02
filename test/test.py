import unittest
import kipr as kp
import numpy as np

max_nd = kp.max_nd()
nb_random_checks = 5



class TestKarrayInternals(unittest.TestCase):

    def test_internals(self):
        self.assertTrue(kp.internal())




class TestKarrayObject(unittest.TestCase):

    def test_init(self):

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
        with self.assertRaises(TypeError):
            kp.arr(1, shape=[0, 1])
        

        # shape len must be > 1
        with self.assertRaises(TypeError):
            kp.arr(1, shape=[])
        
        # shape must be a sequence
        with self.assertRaises(TypeError):
            kp.arr(1, shape=2)
        
        # shape must be a sequence
        with self.assertRaises(TypeError):
            kp.arr(1, shape=np.array([]))
        
        self.assertTrue(kp.arr(1))
        
        np.testing.assert_almost_equal(
            kp.arr(1).numpy(), 
            np.array([1])
        )

        np.testing.assert_almost_equal(
            kp.arr(range(2), shape=[1, 1, 1, 2]).numpy(), 
            np.arange(2).reshape([1, 1, 1, 2])
        )


        for k in range(nb_random_checks):
            nd = np.random.randint(1, max_nd + 1)
            shape = np.random.randint(1,5, size=(nd))
            print(f'{shape = }')

            np.testing.assert_almost_equal(
                kp.arr(range(shape.prod()), shape=shape).numpy(), 
                np.array(range(shape.prod())).reshape(shape)
            )

        for k in range(nb_random_checks):
            nd = np.random.randint(1, max_nd + 1)
            shape = np.random.randint(1,5, size=(nd))
            a = np.random.rand(*shape)
            np.testing.assert_almost_equal(
                kp.arr(a).numpy(), 
                a
            )

    def test_reshape(self):

        with self.assertRaises(TypeError):
            kp.arr(1).reshape([2])

        with self.assertRaises(TypeError):
            kp.arr(1).reshape([])

        with self.assertRaises(TypeError):
            kp.arr(1).reshape(np.array([]))

        for k in range(nb_random_checks):
            nd = np.random.randint(1, max_nd + 1)
            shape = np.random.randint(2,5, size=(nd))
            print(f'reshape to {shape}')

            np.testing.assert_almost_equal(
                kp.arr(range(shape.prod())).reshape(shape).numpy(), 
                np.array(range(shape.prod())).reshape(shape)
            )

    def test_print(self):
        print(kp.arr(1, shape=[2, 3]))

    def test_shape_attr(self):

        shape = [3, 4, 5]
        print(kp.arr(1, shape=shape))

        
        with self.assertRaises(AttributeError):
            kp.arr(1).shape = [1, 2]

        self.assertEqual(
            kp.arr(1, shape=shape).shape,
            tuple(shape)
        )


    def test_subscript(self):

        a = kp.arr(range(5**5), shape=[5, 5, 5, 5, 5])
        b = a.numpy()

        
        with self.assertRaises(IndexError):
            a[..., ...]

        with self.assertRaises(IndexError):
            a[5]

        with self.assertRaises(IndexError):
            a[1.3]

        with self.assertRaises(IndexError):
            print(a[-7])

        with self.assertRaises(IndexError):
            a[..., 5]

        with self.assertRaises(IndexError):
            a[..., 5]

        a[1:1]

        print(kp.arr(1)[:])

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
            print(f"{subscript = }")
            np.testing.assert_almost_equal(
                a[subscript].numpy(), 
                b[subscript]
            )

    def runTest(self):
        self.test_init()
        self.test_subscript()
        self.test_reshape()
        self.test_shape_attr()
        self.test_print()

class TestKarrayMath(unittest.TestCase):


    def test_add(self):

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
            nd = np.random.randint(1, max_nd + 1)
            shape = np.random.randint(1, 5, size=(nd))
            a = np.random.rand(*shape)
            b = np.random.rand(*shape)

            print(a.shape)

            print(kp.arr(a) + kp.arr(b))
            
            np.testing.assert_almost_equal(
                (kp.arr(a) + kp.arr(b)).numpy(), 
                a + b
            )

    def test_inplace_add(self):

        ka = kp.arr(1)

        ka += kp.arr(2)
        np.testing.assert_almost_equal(
            ka.numpy(), 
            [3]
        )

        for k in range(nb_random_checks):
            nd = np.random.randint(1, max_nd + 1)
            shape = np.random.randint(1, 5, size=(nd))
            a = np.random.rand(*shape).astype(np.float32)
            b = np.random.rand(*shape).astype(np.float32)

            ka = kp.arr(a)
            print(a.shape)

            ka += kp.arr(b)

            np.testing.assert_almost_equal(
                ka.numpy(), 
                a + b
            )

    def test_sub(self):

        np.testing.assert_almost_equal(
            (kp.arr(2) - kp.arr(3)).numpy(), 
            [-1]
        )

    def test_inplace_sub(self):

        a = kp.arr(2);
        a -= kp.arr(1)

        np.testing.assert_almost_equal(
            a.numpy(), 
            [1]
        )

    def test_mul(self):

        np.testing.assert_almost_equal(
            (kp.arr(2) * kp.arr(3)).numpy(), 
            [6]
        )

    def test_inplace_mul(self):

        a = kp.arr(2);
        a *= kp.arr(1.5)
        
        np.testing.assert_almost_equal(
            a.numpy(), 
            [3]
        )

    def test_div(self):

        np.testing.assert_almost_equal(
            (kp.arr(2) / kp.arr(3)).numpy(), 
            [0.6666666666666]
        )

    def test_inplace_div(self):

        a = kp.arr(2);
        a /= kp.arr(1.5)
        
        np.testing.assert_almost_equal(
            a.numpy(), 
            [1.33333333]
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





if __name__ == '__main__':
    unittest.main()
    # TestKarray().test_init()
    # TestKarray().test_subscript()
    # TestKarrayObject().test_init()
    # TestKarrayInternals().test_internals()


    # TestKarrayMath().run()
    # TestKarrayObject().run()


    




