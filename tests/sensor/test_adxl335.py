import unittest
import argparse
# import numpy as np
from roboskin.sensor.adxl335 import ADXL335


class ADXL335Test(unittest.TestCase):
    """
    ADXL335 sensor test cases.
    """
    def test_read(self):
        """
        Test that a read from the adxl335 sensor works.
        """
        adxl335 = ADXL335(xpin=0, ypin=1, zpin=2)

        # Let the sensor read for 100 times before testing
        for _ in range(100):
            data = adxl335.read()

        self.assertTrue(data.size == 3)
        self.assertTrue(data[0] > 0.0)
        self.assertTrue(data[1] > 0.0)
        self.assertTrue(data[2] > 0.0)


def parse_arguments():
    """
    Argument parser with `argparse`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbosity', default=2, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments
    unittest.main()
    # unittest.main(verbosity=args.verbosity)
