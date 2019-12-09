import os
import sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from package_name import core

class TestCore(unittest.TestCase):
	def test_add(self):
		a = 1
		b = 2
		ans = 3
		sum = core.add(a,b)
		self.assertEqual(sum, ans)

if __name__ == '__main__':
	unittest.main()
