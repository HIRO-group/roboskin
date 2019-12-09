import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from robotic_skin import core

a = 1
b = 2

sum = core.add(a, b)
print('{}+{}={}'.format(a, b, sum))
