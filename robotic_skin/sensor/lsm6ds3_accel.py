import smbus2
from robotic_skin.sensor import Sensor
import math

'''
This code is heavily inspired by this wonderful GitHub repo:
https://github.com/CRImier/python-lsm6ds3
Thanks Homie!
Datasheet Link: https://cdn.sparkfun.com/assets/learn_tutorials/4/1/6/DM00133076.pdf
'''


class LSM6DS3_acclerometer(Sensor):
    def __init__(self, bus: int = 1, addr: int =0x6b):
        """
        Initializes the LSM6DS3 accelerometer. Checks for the I2C connection and checks whether it's the correct
        accelerometer or not.
        :param bus: This is the bus number. Basically The I2C port number. For our circuit, I connected it I2C port 1,
        So by default it's value I kept as 1. Feel free to pass your own value if you need it.
        :type bus: int
        :param addr: The I2C address of the accelerometer. According to the datasheet of LSM6DS3, there can be only two
        addresses 0x6b or 0x6a. By default I am using Sparkfun breakout board and the address to that is 0x6b which I
        have kept as default
        :type addr: int (It would be easy for you to pass hexadecimal int of the form 0xNN, directly according to the
        datasheet)
        """
        super().__init__()
        # Below are Accelerometer Output registers
        self.OUTX_L_XL = 0x28
        self.OUTX_H_XL = 0x29
        self.OUTY_L_XL = 0x2A
        self.OUTY_H_XL = 0x2B
        self.OUTZ_L_XL = 0x2C
        self.OUTZ_H_XL = 0x2D
        # Below is a register used to find out if the device is LSM6DS3 or not
        # According to Page 51 this register will always output 0x69
        self.WHO_AM_I = 0x0F
        # Below are control registers used to set specific preferences
        # TODO: Explore more of this settings for our Optimal Use
        self.CTRL1_XL = 0x10
        self.CTRL2_G = 0x11
        self.CTRL3_C = 0x12
        self.CTRL4_C = 0x13
        self.CTRL5_C = 0x14
        self.CTRL6_C = 0x15
        self.CTRL7_G = 0x16
        self.CTRL8_XL = 0x17
        self.CTRL9_XL = 0x18
        self.CTRL10_C = 0x19
        # Below are initial register values along with their respective names in initial_registers list
        # these will be used to set values to registers
        self.initial_reg_values = [0x70, 0x4c, 0x44, 0x0, 0x0,
                                   0x0, 0x50, 0x0, 0x38, 0x38]
        self.initial_registers = ['CTRL1_XL', 'CTRL2_G', 'CTRL3_C', 'CTRL4_C', 'CTRL5_C',
                                  'CTRL6_C', 'CTRL7_G', 'CTRL8_XL', 'CTRL9_XL', 'CTRL10_C']
        # Setting the SMBus
        self.bus_num = bus
        self.bus = smbus2.SMBus(self.bus_num)
        # If int is not passed, then convert it to int
        if isinstance(addr, str):
            addr = int(addr, 16)
        # Address of the Acceleromter I2C device
        self.addr = addr
        self.setup()

    def calibrate(self):
        pass

    def write_reg(self, reg, val):
        """
        Write value to the register specified
        :param reg: Value of the register to which you want the write some value
        :type reg: int
        :param val: Value you want to write to the register
        :type val: int
        :return: None
        :rtype: None
        """
        return self.bus.write_byte_data(self.addr, reg, val)

    def read_reg(self, reg):
        """
        Read the Register Value in form of int
        :param reg: Register from which you want to read the value from
        :type reg: int
        :return: int value read from register
        :rtype: int
        """
        return self.bus.read_byte_data(self.addr, reg)

    def make_16bit_value(self, vh, vl):
        """
        The acceleration is usually from 2 Byte sized registers. We obtain acceleration value in 2's complement form
        So first we obatin both MSByte as well as LSByte, combine them both, and convert them into 2's complement form
        :param vh: The MSByte
        :type vh: int
        :param vl: The LSByte
        :type vl: int
        :return: Accleration value in G
        :rtype:float
        """
        v = (vh << 8) | vl
        # return v
        return (self.twos_comp(v, 16))/math.pow(2, 14)

    def twos_comp(self, val, bits):
        """
        compute the 2's complement of int value val
        :param val: The original value, which we have to convert to 2's complement
        :type val: int
        :param bits: # of bits, this is particularly important because if you don't know bit size, you dont what's the
        MS Bit, and entire thing can go wrong. Fortunately our both Accelerometer registers combined are of 16 bit
        length. So that's what we will pass
        :type bits: int
        :return: Two's complement value of passed value
        :rtype: int
        """
        if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << bits)  # compute negative value
        return val

    def detect(self):
        """
        This function will detect if the accelerometer is LSM6DS3 or not
        :return: True if the detected accelerometer is LSM6DS3, else False
        :rtype: Bool
        """
        assert (self.read_reg(self.WHO_AM_I) == 0x69), "Identification register value \
                                                       is wrong! Pass 'detect=False' \
                                                       to setup() to disable the check."

    def setup(self):
        """
        Setup the LSM6DS3 accelerometer with the preferences hexadecimal values from self.initial_reg_values. It also
        checks if the accelerometer is LSM6DS3 or not. Execution of this function without any problems is a litmus
        test that attached device is LSM6DS3.
        :return: Return True if all assert statements are executed and all code is executed without exceptions,
        else False
        :rtype: Bool
        """
        self.detect()
        # Safety check
        assert (len(self.initial_reg_values) == len(self.initial_registers)), \
            "Number of initial registers is not equal to number of initial \
                 register values. Set 'lsm.initial_registers' properly!"
        # Writing initial values into registers
        for i, reg_name in enumerate(self.initial_registers):
            self.write_reg(getattr(self, reg_name), self.initial_reg_values[i])
        return True

    def _read_raw(self):
        axh = self.read_reg(self.OUTX_H_XL)
        axl = self.read_reg(self.OUTX_L_XL)
        ax = self.make_16bit_value(axh, axl)
        ayh = self.read_reg(self.OUTY_H_XL)
        ayl = self.read_reg(self.OUTY_L_XL)
        ay = self.make_16bit_value(ayh, ayl)
        azh = self.read_reg(self.OUTZ_H_XL)
        azl = self.read_reg(self.OUTZ_L_XL)
        az = self.make_16bit_value(azh, azl)

        return [ax, ay, az]

    def _calibrate_value(self, input_value):
        """
        Output the calibrated value from the function you manually made.
        #TODO: Make this function an argument and pass it for future use
        :param input_value: Original value which you want to convert it to calibrated value
        :type input_value: float
        :return: Returns the calibrated value
        :rtype: float
        """
        return input_value

    def read(self):
        """
        Reads the sensor values and continuously streams them back to the function whoever called it. This is the
        function you need to put while(True) loop for continuous acquisition of accelerometer values.
        :return: List of floats with Acceleration values in G
        :rtype: list
        """
        return [self._calibrate_value(each_value) for each_value in self._read_raw()]


if __name__ == "__main__":
    from time import sleep
    lsm = LSM6DS3_acclerometer()
    while True:
        raw_accel_list = lsm.read()
        print(
            "Raw accel values: \t X {x:.4f} \t Y {y:.4f} \t Z {z:.4f}".format(
                x=raw_accel_list[0],
                y=raw_accel_list[1],
                z=raw_accel_list[2]))
        sleep(0.02)