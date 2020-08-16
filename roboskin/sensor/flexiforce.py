"""
FlexiForce module
"""
import time
from mcp3208 import MCP3208

import roboskin.const as C


class FlexiForce():
    """
    FlexiForce class
    """
    def __init__(self, pin):
        """
        Initialize the FlexiForce connected to MCP3208 AD Converter.

        Parameters
        --------------
        pin: int
            Pin number of the MCP3208 where FlexiForce is connected to.
        """
        self.pin = pin
        self.adc = MCP3208()

    def read(self):
        """
        Read a force value
        """
        return self.adc.read(self.pin)


if __name__ == '__main__':
    flexiforce = FlexiForce(C.FLEXIFORCE_PIN)

    while True:
        value = flexiforce.read()
        print('Force: %.2f' % (value))
        time.sleep(0.5)
