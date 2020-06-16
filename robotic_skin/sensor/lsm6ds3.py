"""
This code is heavily inspired by this wonderful GitHub repo:
https://github.com/CRImier/python-lsm6ds3
Thanks Homie!
Datasheet Link: https://cdn.sparkfun.com/assets/learn_tutorials/4/1/6/DM00133076.pdf
"""
import math
import smbus2
from robotic_skin.sensor import Sensor
from robotic_skin.const import GRAVITATIONAL_CONSTANT
from time import sleep


class LSM6DS3_IMU(Sensor):
    """
    This is the Python Class for LSM6DS3. This includes all subroutines including calibration to handle everything
    related to the device.
    """

    def __init__(self, config_file):  # noqa: E999
        """
        Initializes the LSM6DS3 accelerometer. Checks for the I2C connection and checks whether it's the correct
        accelerometer or not. This class requires the below variables to be set in yaml configuration file:
        RPi_bus_num: The Raspberry Pi I2C bus number
        imu_i2c_address: The I2C address of the sensor
        gravity_constant: gravity constant. It's usually set in params.yaml.
        Additionally this class extends Sensor, so all sensor's configuration should also be passed to this class
        Parameters
        ----------
        config_file : str
            config_file is the full path to the config file which contains all parameters in yaml to execute
            successfully as explained above
        """
        super(LSM6DS3_IMU, self).__init__(config_file)
        # Below are Accelerometer Output registers
        self.OUTX_L_XL = 0x28
        self.OUTX_H_XL = 0x29
        self.OUTY_L_XL = 0x2A
        self.OUTY_H_XL = 0x2B
        self.OUTZ_L_XL = 0x2C
        self.OUTZ_H_XL = 0x2D

        # Below are Gyroscope Output registers
        self.OUTX_L_G = 0x22
        self.OUTX_H_G = 0x23
        self.OUTY_L_G = 0x24
        self.OUTY_H_G = 0x25
        self.OUTZ_L_G = 0x26
        self.OUTZ_H_G = 0x27
        # Below is a register used to find out if the device is LSM6DS3 or not
        # According to Page 51 this register will always output 0x69
        self.WHO_AM_I = 0x0F
        # Below are control registers used to set specific preferences

        # FOR LATER: Explore more of this settings for our Optimal Use
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
        self.LSM6DS3_RegisterIdentification_NUM = 0x69
        # Below are initial register values along with their respective names in initial_registers list
        # these will be used to set values to registers
        self.initial_reg_values = [0x70, 0x4c, 0x44, 0x0, 0x0,
                                   0x0, 0x50, 0x0, 0x38, 0x38]
        self.initial_registers = ['CTRL1_XL', 'CTRL2_G', 'CTRL3_C', 'CTRL4_C', 'CTRL5_C',
                                  'CTRL6_C', 'CTRL7_G', 'CTRL8_XL', 'CTRL9_XL', 'CTRL10_C']
        # Setting the SMBus
        self.bus_num = self.config_dict['RPi_bus_num']
        self.bus = smbus2.SMBus(self.bus_num)
        # If int is not passed, then convert it to int
        addr = self.config_dict['imu_i2c_address']
        if isinstance(addr, str):
            addr = int(addr, 16)
        # Address of the Acceleromter I2C device
        self.addr = addr
        self.setup()

    def calibrate(self):
        """
        # Need to implement this function

        Returns
        -------
        None

        """

    def write_reg(self, reg, val):
        """
        Write value to the register specified
        Parameters
        ----------
        reg : int
            Value of the register to which you want the write some value
        val : int
            Value you want to write to the register

        Returns
        -------
        None

        """
        return self.bus.write_byte_data(self.addr, reg, val)

    def read_reg(self, reg):
        """
        Read the Register Value in form of int
        Parameters
        ----------
        reg : int
            Register from which you want to read the value from

        Returns
        -------
        int
            int value read from register
        """

        return self.bus.read_byte_data(self.addr, reg)

    def make_16bit_value(self, vh, vl):
        """
        The acceleration is usually from 2 Byte sized registers. We obtain acceleration value in 2's complement form
        So first we obtain both MSByte as well as LSByte, combine them both, and convert them into 2's complement form
        Parameters
        ----------
        vh : int
            The MSByte
        vl : int
            The LSByte

        Returns
        -------
        float
            Acceleration Value in G
        """

        v = (vh << 8) | vl
        # return v
        return (self.twos_comp(v, 16)) / math.pow(2, 14)

    def twos_comp(self, val, num_of_bits):
        """
        compute the 2's complement of int value val. Reference:
        https://en.wikipedia.org/wiki/Two%27s_complement
        Parameters
        ----------
        val : int
            The original value, which we have to convert to 2's complement
        num_of_bits : int
            # of bits, this is particularly important because if you don't know bit size, you dont what's the
            MS Bit, and entire thing can go wrong. Fortunately our both Accelerometer registers combined are of 16 bit
            length. So that's what we will pass

        Returns
        -------
        int
            Two's complement value of passed value

        """
        if (val & (1 << (num_of_bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << num_of_bits)  # compute negative value
        return val

    def detect(self):
        """
        This function will detect if the accelerometer is LSM6DS3 or not

        Returns
        -------
        Bool
            True if the detected accelerometer is LSM6DS3, otherwise false
        """
        assert (self.read_reg(self.WHO_AM_I) == self.LSM6DS3_RegisterIdentification_NUM), "Identification register value \
                                                       is wrong! Pass 'detect=False' \
                                                       to setup() to disable the check."

    def setup(self):
        """
        Setup the LSM6DS3 accelerometer with the preferences hexadecimal values from self.initial_reg_values. It also
        checks if the accelerometer is LSM6DS3 or not. Execution of this function without any problems is a litmus
        test that attached device is LSM6DS3.

        Returns
        -------
        Bool
            Return True if all assert statements are executed and all code is executed without exceptions,
            else False

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
        ax = self.make_16bit_value(axh, axl) * GRAVITATIONAL_CONSTANT
        ayh = self.read_reg(self.OUTY_H_XL)
        ayl = self.read_reg(self.OUTY_L_XL)
        ay = self.make_16bit_value(ayh, ayl) * GRAVITATIONAL_CONSTANT
        azh = self.read_reg(self.OUTZ_H_XL)
        azl = self.read_reg(self.OUTZ_L_XL)
        az = self.make_16bit_value(azh, azl) * GRAVITATIONAL_CONSTANT
        gxh = self.read_reg(self.OUTX_H_G)
        gxl = self.read_reg(self.OUTX_L_G)
        gx = self.make_16bit_value(gxh, gxl)
        gyh = self.read_reg(self.OUTY_H_G)
        gyl = self.read_reg(self.OUTY_L_G)
        gy = self.make_16bit_value(gyh, gyl)
        gzh = self.read_reg(self.OUTZ_H_G)
        gzl = self.read_reg(self.OUTZ_L_G)
        gz = self.make_16bit_value(gzh, gzl)

        return [ax, ay, az, gx, gy, gz]

    def _calibrate_value(self, input_value):
        """
        Output the calibrated value from the function you manually made.
        #TODO: Make this function an argument and pass it for future use
        Parameters
        ----------
        input_value : float
            Original value which you want to convert it to calibrated value

        Returns
        -------
        float
            Returns the calibrated value

        """
        return input_value

    def read(self):
        """
        Reads the sensor values and continuously streams them back to the function whoever called it. This is the
        function you need to put while(True) loop for continuous acquisition of accelerometer values.
        Returns
        -------
        list
            List of floats with Acceleration values and angular velocity values in G in the form of
            [ax, ay, az, gx, gy, gz] respective directions

        """
        return [self._calibrate_value(each_value) for each_value in self._read_raw()]


if __name__ == "__main__":
    lsm6ds3 = LSM6DS3_IMU("/home/hiro/catkin_ws/src/ros_robotic_skin/config/accelerometer_config1.yaml")
    while 1:
        print(lsm6ds3.read()[0:3])
        sleep(0.5)
