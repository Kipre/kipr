import unittest
import kipr as k
import numpy as np

class TestKarray(unittest.TestCase):

    def test_init(self):
        np.testing.assert_equal(k.arr().numpy(), np.array([0], dtype=np.float32))

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
    




