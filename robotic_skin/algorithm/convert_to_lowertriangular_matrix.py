"""
This is a python file containing a test case and a class to convert any matrix which is convertible Lower Triangular(LT)
Matrix to Lower Triangular Matrix
Algorithm:
*) As this is a binary matrix, and also lower triangular, the total sum of elements should be at max n(n+1)/2, else it
can't be a LT matrix
*) There can only be one reference segment, hence only one zero row array, else the algorithm will throw an exception
*) You are only allowed to switch rows and columns to get the LT matrix
Custom Sort Row:
    *) Arrange the rows in ascending order of number of ones present in the row
    *) If there is a tie in number of ones, then you assume the row as a binary number, with MSB at first and then the
    number you get, convert it to a floating point number and add it to the row sum. This mainly signifies as a weight
    value to the row sum which are same, so that argsort can give preference. Finally the row sum is arranged in
    ascending order

Custom Sort Column:
    *) Arrange the columns in the descending order of number of ones in the column
    *) *) Arrange the rows in ascending order of number of ones present in the row
    *) If there is a tie in number of ones, then you assume the row as a binary number, with MSB at first and then the
    number you get, convert it to a floating point number and add it to the row sum. This mainly signifies as a weight
    value to the row sum which are same, so that argsort can give preference. Finally the row sum is arranged in
    descending order

# TODO: Next PR, I will add more examples of why this approach is correct. I derived some in my IPad
"""
import math
import numpy as np


class ConvertToLT:
    """
    This is a class to convert LT convertible matrix to LT matrix
    """

    def __init__(self, input_matrix):
        """
        Initialize the class and start the main algorithm
        Parameters
        ----------
        input_matrix : np.ndarray
            the input matrix passed into the class, to convert it to LT matrix
        """
        self.input_matrix = input_matrix
        self.zero_array_index = []
        self.rows = []
        self.columns = []
        self.main_algorithm()

    def _is_there_only_one_zero_array(self):
        """
        This function will check if there is only one zero array in the passed matrix, if there are more, it will
        throw an exception
        """
        # There can be only one zero row in the activity matrix. More info in the exception string
        self.zero_array_index = []
        # Remove the nd arrays which are zero arrays for now, we can add them back later
        for index, each_array in enumerate(self.input_matrix):
            if self.find_zero_ndarray(each_array):
                self.input_matrix = np.delete(self.input_matrix, index, 0)
                self.zero_array_index.append(index)
        if len(self.zero_array_index) != 1:
            raise Exception("More than one row is a zero row. According to the paper there can be only one reference "
                            "zero row array, so Activity matrix might be wrong")

    def main_algorithm(self):
        """
        Main Algorithm implementation. Algorithm details at the top of the file.
        Returns
        -------
        None

        """
        self._is_there_only_one_zero_array()
        # After removing the zero arrays the residue matrix should be a square matrix, be mindful of that
        self.is_matrix_lt_convertible(self.input_matrix)
        self.input_matrix = self.sort_rows_in_ascending_order(self.input_matrix)
        self.input_matrix = self.sort_columns_in_descending_order(self.input_matrix)
        # Check if output matrix is an LT matrix
        self.is_lower_triangular = self.is_lower_triangular_fn(self.input_matrix)
        # The row numbers need to be changed as we removed zero array at the start
        self._arrange_row_column_numbers()
        # Finally we add zero array at the start of the array
        self._append_zero_array_at_start()
        # I know I can just return the input matrix, but for the sanity of people who will be using my algorithm in the
        # future I am writing this extra line
        self.final_matrix = self.input_matrix
        # Saving the reference segment number to a variable
        self.reference_segment_accelerometer = self.zero_array_index[0] + 1

    def _arrange_row_column_numbers(self):
        """
        When we remove the zero row array, and then do shuffling of rows and columns, we finally need where our original
        row and column were shuffled to, to get the LT matrix. Issue is that as the zero row array can be somewhere in
        between we have to change the row numbers of some rows. That algorithm is explained in more detailed in my blog
        https://krishnachaitanya9.github.io/posts/lower_triangular_algorithm_problem/
        This is just an implementation of that algorithm
        """
        self.rows += np.array((self.rows - self.zero_array_index[0] + 1) > 0).astype(int) + 1
        # Finally Increment the column number
        self.columns += 1

    def _append_zero_array_at_start(self):
        for _ in range(len(self.zero_array_index)):
            self.input_matrix = np.insert(self.input_matrix, 0, 0, axis=0)

    def numpyarray_toint(self, my_array: np.ndarray, reverse: bool):
        """
        This function will convert a numpy 1D array which is considered a binary number into a decimal number
        Parameters
        ----------
        my_array : np.ndarray
            Input array
        reverse : bool
            if reverse MSB is considered first, else last

        Returns
        -------
        int
            Decimal number

        """
        if reverse:
            my_array = my_array[::-1]
        return my_array.dot(2 ** np.arange(my_array.size)[::-1])

    def convert_to_floating_point(self, my_number):
        """
        This method id used to convert some number like 768 to 0.768. Basically used to add weight value
        Parameters
        ----------
        my_number : int
            Input Number

        Returns
        -------
        float
            floating point number from the input number

        """
        return my_number / math.pow(10, len(str(my_number)))

    def sort_rows_in_ascending_order(self, input_array: np.ndarray):
        """
        Sort rows of input matrix. Algorithm at the top.
        References:
        # Stackoverflow links: https://stackoverflow.com/questions/52216526/sort-array-columns-based-upon-sum/52216674
        # https://stackoverflow.com/questions/7235785/sorting-numpy-array-according-to-the-sum
        Parameters
        ----------
        input_array : np.ndarray
            the input array we pass into the function, for it to have it's rows sorted

        Returns
        -------
        np.ndarray
            The array with its rows sorted

        """
        # So each row has to be unique for us to convert to Lower triangular Matrix
        row_sum = np.sum(input_array, axis=1, dtype=np.float)
        row_sum_unique, counts = np.unique(row_sum, return_counts=True)
        duplicates = row_sum_unique[counts > 1]
        if len(duplicates) == 0:
            # All values are unique, no repetition
            self.rows = row_sum.argsort()
            return input_array[self.rows, :]
        else:
            # Values ain't unique, there is repetition
            # First find which indices are repeating
            for each_duplicate in duplicates:
                for index, value in enumerate(row_sum):
                    if each_duplicate == value:
                        row_sum[index] = value + \
                                         self.convert_to_floating_point(self.numpyarray_toint(input_array[index], True))
            self.rows = row_sum.argsort()
            return input_array[self.rows, :]

    def sort_columns_in_descending_order(self, input_array: np.ndarray):
        """
        Sort columns of input matrix. The algorithm is specified at the top.
        Parameters
        ----------
        input_array :

        Returns
        -------

        """
        input_array = input_array.transpose()
        # So each row has to be unique for us to convert to Lower triangular Matrix
        row_sum = np.sum(input_array, axis=1, dtype=np.float)
        row_sum_unique, counts = np.unique(row_sum, return_counts=True)
        duplicates = row_sum_unique[counts > 1]
        if len(duplicates) == 0:
            # All values are unique, no repetition
            self.columns = row_sum.argsort()[::-1]
            return input_array[self.columns, :].transpose()
        else:
            # Values ain't unique, there is repetition
            # First find which indices are repeating
            for each_duplicate in duplicates:
                for index, value in enumerate(row_sum):
                    if each_duplicate == value:
                        row_sum[index] = value + self.convert_to_floating_point(
                            self.numpyarray_toint(input_array[index], True))
            self.columns = row_sum.argsort()[::-1]
            return input_array[self.columns, :].transpose()

    def get_lt_matrix_infos(self):
        """
        This function is just used to return values in the order
        bool: Whether the matrix output generated is an LT matrix or not
        np.ndarray: The LT matrix, if not possible the near LT matrix. Basically whatever the output generated
            from the algorithm
        int: The reference segment Accelerometer ID
        np.ndarray: The accelerometer ID's in the order of their arrangement on the robotic arm
        np.ndarray: The Joint ID's in the order of their arrangement on the robotic arm
        Returns
        -------
            bool, np.ndarray, int, np.ndarray, np.ndarray

        """

        return self.is_lower_triangular, self.final_matrix, self.reference_segment_accelerometer, self.rows, self.columns

    def is_square_matrix(self, input_array):
        """
        This is a function which raises an exception, if the output matrix isn't square matrix
        Parameters
        ----------
        input_array : Input Matrix

        Returns
        -------
        None
            It doesn't return any value, but throws an exception if the passes matrix isn't a square matrix

        """
        if not len(input_array) == len(input_array[0]):
            raise Exception("The matrix isn't square")

    def find_zero_ndarray(self, input_array):
        """
        As this is a binary matrix, with values only 0 and 1, equating min value and max value to 0, we can say
        the array is a zero array
        Parameters
        ----------
        input_array : np.ndarray
            Input matrix is the matrix we want to find out whether it's a zero array or not.

        Returns
        -------
        Bool
            True if the array is a zero array, otherwise False

        """
        if input_array.min(axis=0) == input_array.max(axis=0) == 0:
            return True
        return False

    def is_lower_triangular_fn(self, input_array: np.ndarray):
        """
        Directly copied from: https://www.geeksforgeeks.org/program-check-matrix-lower-triangular/
        And also checked. Will output if input matrix input_array is a lower triangular matrix or not
        Parameters
        ----------
        input_array : np.ndarray
            The matrix in question, whether it's lower triangular or not

        Returns
        -------
        bool
            True if numpy array inputted is lower triangular, else False

        """
        rows, cols = input_array.shape
        for i in range(rows):
            for j in range(i+1, cols):
                if input_array[i][j] != 0:
                    return False
 
        return True

    def is_matrix_lt_convertible(self, input_array):
        """
        Assuming that passed triangle is binary matrix, this will only work in that case
        Parameters
        ----------
        input_array :

        Returns
        -------

        """

        self.is_square_matrix(input_array)
        length_of_input_array = len(input_array)
        if np.sum(input_array) <= (length_of_input_array * (length_of_input_array + 1)) / 2:
            pass
        else:
            raise Exception("Matrix is not convertible to lower traiangular matrix")


if __name__ == "__main__":
    # Original Matrix given in the paper
    TEST_ARRAY = np.array([
        [0, 1, 0, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0]
    ])
    print(ConvertToLT(TEST_ARRAY).get_lt_matrix_infos())
