"""
This code is heavily inspired by this wonderful GitHub repo:
https://github.com/CRImier/python-lsm6ds3
Thanks Homie!
Datasheet Link: https://cdn.sparkfun.com/assets/learn_tutorials/4/1/6/DM00133076.pdf
"""
import smbus2
from roboskin.sensor import Sensor
from roboskin.sensor import utils
from roboskin.const import GRAVITATIONAL_CONSTANT
from time import sleep


class LSM6DS3_IMU(Sensor):
    """
    This is the Python Class for LSM6DS3. This includes all subroutines including calibration to handle everything
    related to the device.
    """

    def __init__(self, raspi_bus_number=1, i2c_address='0x6b'):
        """
        Initializes the LSM6DS3 accelerometer.
        """
        super(LSM6DS3_IMU, self).__init__()

        # Setting the SMBus
        self._raspi_bus_number = raspi_bus_number
        self._bus = smbus2.SMBus(self._raspi_bus_number)

        # If int is not passed, then convert it to int
        if isinstance(i2c_address, str):
            i2c_address = int(i2c_address, 16)
        # Address of the Acceleromter I2C device
        self._i2c_address = i2c_address

        self._setup_register_values()
        self.is_lsm6ds3()
        self._initialize_registers()

    def _setup_register_values(self):
        """
        Setup register values that were in the data sheet
        """
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

    def is_lsm6ds3(self):
        """
        This function will detect if the accelerometer is LSM6DS3 or not

        Returns
        -------
        Bool
            True if the detected accelerometer is LSM6DS3, otherwise false
        """
        assert (self._read_reg(self.WHO_AM_I) == self.LSM6DS3_RegisterIdentification_NUM), "Identification register value \
                                                       is wrong! Pass 'detect=False' \
                                                       to setup() to disable the check."

    def _initialize_registers(self):
        """
        Setup the LSM6DS3 accelerometer with the preferences hexadecimal values from initial register values.
        It also checks if the accelerometer is LSM6DS3 or not.
        Execution of this function without any problems is a litmus
        test that attached device is LSM6DS3.

        Returns
        -------
        Bool
            Return True if all assert statements are executed and all code is executed without exceptions,
            else False

        """
        # Safety check
        assert (len(self.initial_reg_values) == len(self.initial_registers)), \
            "Number of initial registers is not equal to number of initial \
                 register values. Set 'lsm.initial_registers' properly!"

        # Writing initial values into registers
        for i, reg_name in enumerate(self.initial_registers):
            self._write_reg(getattr(self, reg_name), self.initial_reg_values[i])

        return True

    def get_bus_number(self):
        return self._raspi_bus_number

    def get_i2c_address(self):
        return self._i2c_address

    def calibrate(self):
        """
        # Need to implement this function

        Returns
        -------
        None

        """

    def _write_reg(self, reg, val):
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
        return self._bus.write_byte_data(self._i2c_address, reg, val)

    def _read_reg(self, reg):
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

        return self._bus.read_byte_data(self._i2c_address, reg)

    def read_raw(self):
        accels = self.read_raw_accels()
        gyros = self.read_raw_gyros()

        # Addition of lists
        # [a, b, c] + [d, e, f] -> [a, b, c, d, e, f]
        return accels + gyros

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
        return [self._calibrate_value(each_value) for each_value in self.read_raw()]

    def read_raw_accelX(self):
        """
        Read a raw acceleration value for x axis

        Returns
        -------
        accelX: float
        """
        axh = self._read_reg(self.OUTX_H_XL)
        axl = self._read_reg(self.OUTX_L_XL)
        return utils.make_16bit_value(axh, axl) * 0.061 * 0.001 * GRAVITATIONAL_CONSTANT

    def read_raw_accelY(self):
        """
        Read a raw acceleration value for y axis

        Returns
        -------
        accelY: float
        """
        ayh = self._read_reg(self.OUTY_H_XL)
        ayl = self._read_reg(self.OUTY_L_XL)
        return utils.make_16bit_value(ayh, ayl) * 0.061 * 0.001 * GRAVITATIONAL_CONSTANT

    def read_raw_accelZ(self):
        """
        Read a raw acceleration value for z axis

        Returns
        -------
        accelZ: float
        """
        azh = self._read_reg(self.OUTZ_H_XL)
        azl = self._read_reg(self.OUTZ_L_XL)
        return utils.make_16bit_value(azh, azl) * 0.061 * 0.001 * GRAVITATIONAL_CONSTANT

    def read_raw_accels(self):
        """
        Read raw acceleration values for all axis

        Returns
        -------
        accels: List[float]
        """
        return [
            self.read_raw_accelX(),
            self.read_raw_accelY(),
            self.read_raw_accelZ()
        ]

    def read_raw_gyroX(self):
        """
        Read a raw angular velocity value for x axis

        Returns
        -------
        gyroX: float
        """
        gxh = self._read_reg(self.OUTX_H_G)
        gxl = self._read_reg(self.OUTX_L_G)
        return utils.make_16bit_value(gxh, gxl)

    def read_raw_gyroY(self):
        """
        Read a raw angular velocity value for y axis

        Returns
        -------
        gyroY: float
        """
        gyh = self._read_reg(self.OUTY_H_G)
        gyl = self._read_reg(self.OUTY_L_G)
        return utils.make_16bit_value(gyh, gyl)

    def read_raw_gyroZ(self):
        """
        Read a raw angular velocity value for z axis

        Returns
        -------
        gyroZ: float
        """
        gzh = self._read_reg(self.OUTZ_H_G)
        gzl = self._read_reg(self.OUTZ_L_G)
        return utils.make_16bit_value(gzh, gzl)

    def read_raw_gyros(self):
        """
        Read raw angular velocity values for all axis

        Returns
        -------
        gyros: List[float]
        """
        return [
            self.read_raw_gyroX(),
            self.read_raw_gyroY(),
            self.read_raw_gyroZ()
        ]


# useful for debugging
if __name__ == "__main__":
    # A ROS Pkg can be used and the config file can be passed so that there is no need of
    # hard coding the path. But this main function is used to debug to check the functionality
    # of the driver.
    lsm6ds3 = LSM6DS3_IMU(raspi_bus_number=1)
    while 1:
        print(lsm6ds3.read()[0:3])
        sleep(0.5)
