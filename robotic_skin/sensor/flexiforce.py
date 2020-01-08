import time
from mcp3208 import MCP3208

FLEXIFORCE_PIN = 3

class FlexiForce():
    def __init__(self, pin):
        self.pin = pin
        self.adc = MCP3208()

    def read(self):
        return self.adc.read(self.pin)

if __name__ == '__main__':
    flexiforce = FlexiForce(FLEXIFORCE_PIN)

    while True:
        value = flexiforce.read()
        print('Force: %.2f'%(value))
        time.sleep(0.5)
