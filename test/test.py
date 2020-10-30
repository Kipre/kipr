import unittest
import kipr as kp
import numpy as np

max_nd = kp.max_nd()
nb_random_checks = 5

class TestKarray(unittest.TestCase):

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
        print("calling")
        print(kp.arr(1, shape=[2, 3]))

    def test_shape_attr(self):

        print("here")
        shape = [3, 4, 5]
        print(kp.arr(1, shape=shape))

        
        with self.assertRaises(AttributeError):
            kp.arr(1).shape = [1, 2]

        self.assertEqual(
            kp.arr(1, shape=shape).shape,
            tuple(shape)
        )

    def test_add_operation(self):

        np.testing.assert_almost_equal(
            (kp.arr(1) + kp.arr(2)).numpy(), 
            [3]
        )

        np.testing.assert_almost_equal(
            (kp.arr(1) + 2).numpy(), 
            [3]
        )

        np.testing.assert_almost_equal(
            (1 + kp.arr(2)).numpy(), 
            [3]
        )

        for k in range(nb_random_checks):
            nd = np.random.randint(1, max_nd + 1)
            shape = np.random.randint(1, 5, size=(nd))
            a = np.random.rand(*shape)
            b = np.random.rand(*shape)

            print(kp.arr(a) + kp.arr(b))
            
            np.testing.assert_almost_equal(
                (kp.arr(a) + kp.arr(b)).numpy(), 
                a + b
            )

            np.testing.assert_almost_equal(
                (kp.arr(a) + b).numpy(), 
                a + b
            )

            # BUG
            # print(a)
            # np.testing.assert_almost_equal(
            #     (a + kp.arr(b)).numpy(), 
            #     a + b
            # )


    def test_subscript(self):

        a = kp.arr(range(5**5), shape=[5, 5, 5, 5, 5])
        b = a.numpy()

        
        with self.assertRaises(IndexError):
            a[..., ...]

        with self.assertRaises(IndexError):
            a[5]

        with self.assertRaises(IndexError):
            print(a[-7])


        with self.assertRaises(IndexError):
            a[..., 5]

        with self.assertRaises(IndexError):
            a[..., 5]

        a[1:1]

        subscripts = [(1),
                      (-1),
                      (-3),
                      (-3),
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




if __name__ == '__main__':
    unittest.main()
    # TestKarray().test_init()
    # TestKarray().test_add_operation()
    # TestKarray().test_subscript()



    




