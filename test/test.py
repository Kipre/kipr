import unittest
import kipr as kp
import numpy as np

max_nd = 8

class TestKarray(unittest.TestCase):

    def test_init(self):

        with self.assertRaises(TypeError):
            kp.arr()
        
        with self.assertRaises(TypeError):
            kp.arr(1, shape=[])
        
        self.assertTrue(kp.arr(1))
        
        np.testing.assert_almost_equal(
            kp.arr(1).numpy(), 
            np.array([1])
        )
        np.testing.assert_almost_equal(
            kp.arr(range(2), shape=[1, 1, 1, 2]).numpy(), 
            np.arange(2).reshape([1, 1, 1, 2])
        )
        for k in range(5):
            nd = np.random.randint(max_nd)
            shape = np.random.randint(1,5, size=(nd))
            print(f'{shape = }')

            np.testing.assert_almost_equal(
                kp.arr(range(shape.prod()), shape=shape).numpy(), 
                np.array(range(shape.prod())).reshape(shape)
            )

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()

    




