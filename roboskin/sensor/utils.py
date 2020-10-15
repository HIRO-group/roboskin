def make_16bit_value(vh, vl):
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

    v = vl | (vh << 8)
    # return v
    return (twos_comp(v, 16))  # / math.pow(2, 14)


def twos_comp(val, num_of_bits):
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
