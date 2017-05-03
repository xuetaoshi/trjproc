import re
import os
import numpy as np
import itertools
import warnings


class UtilityPackage(object):
    """Utility function package:
UtilityPackage.npz_unilen(npz, n_pts, masked=True, filled=0)
    Unifying the length of the input numpy npz package. For a package of multi-dimensional arrays, the "length" refers
    to the first dimension.
    Parameters: npz: numpy npz, numpy array, python list
                    first argument. Input array.
                n_pts: integer
                    second argument. Target length to unify to.
                masked: optional, bool, default=True
                    if true, makes masked arrays instead of regular ones.
                filled: optional, any type, default=0
                    when the original array has less elements than the array with target length, fill in the missing
                    spots with value passed in via this parameter.
    Returns: a numpy (masked/regular) array.

UtilityPackage.expand_multi_dims(a_shape, array, n_axis)
    Adds singleton dimensions to an array to have the same number of dimensions as a multi-dimensional numpy array for
    convenient broadcasting.
    Parameters: a_shape: tuple
                    first argument. Shape of the target multi-dimensional array.
                array: 1-D numpy array, python list
                    second argument. Input array.
                n_axis: integer.
                    third argument. The number of axis on which to put the input array. a_shape[n_axis] must be the
                    same as len(array), or an error will be raised.
    Returns: a numpy array.
    Examples:
        a[2,3,4,5] is to be multiplied by b[4]. Directly doing so with a*b would yield an error due to numpy not
        knowing which axis (axes) to broadcast along. UtilityPackage.expand_multi_dims((2,3,4,5), b, 3) returns
        b[1,1,4,1]. a[2,3,4,5]*b[1,1,4,1] can now be executed by numpy.

UtilityPackage.maxpts_from_npz(npz_file, axis=0)
    Obtain the maximum length of arrays among a numpy npz package.
    Parameters: npz_file: numpy npz package
                    first argument. Input array. Its arguments have to be the default ones, e.g. "arr_0", "arr_1", ...
                axis: integer
                    optional, default=0. Which axis to be looking at in the case of multi-dimensional arrays in the
                    numpy npz package.
    Returns: an integer.

UtilityPackage.rep_pinpoint(array, pinpoint, diss=None)
    This method selects representing points along a trajectory based on pinpoint array and optionally on diss
    array (dissociation detection results array). When diss is not passed in, pinpoint array should have same form
    as diss normally would, namely, a n_channel by n_trajectory matrix where -1 marks non-dissociation
    trajectories and any integer other than -1 signifies the number of point to be taken as representative of such
    trajectory. Note: the input array has to have the first two dimensions corresponding to trajectories and
    points along trajectory.
    Parameters: array: numpy array
                    first argument. Input array to have its elements selected based on pinpoint array and/or diss array.
                pinpoint: numpy array
                    second argument. Dimension of this array should be pinpoint[channel, trj], where channel is the
                    number of dissociation channels, and trj is the number of trajectories overall. The value in this
                    array is either -1, signifying that trajectroy should not be selected in that channel, or any
                    integer >= 0, signifying which point to take as the representing point along that trajectory.
                diss: numpy array
                    optional, default=None. Dimension of this array is the same as poinpoint array, diss[channel, trj].
                    The value in this array is similar to that of poinpoint array, that is either -1, signifying that
                    trajectroy did not dissociate in that channel, or any integer >= 0, signifying at which point the
                    trajectory dissociated. Passing in an array that is not None will make this method select trajectory
                    based on diss array, instead of pinpoint array. But the points along each of those trajectories will
                    still be selected based on pinpoint array.
    Returns: a list of numpy arrays.

UtilityPackage.cts2xyz(cts_npz, atom_list, convert_factor=1.889725989)
    Convert a Cartesian coordinate numpy npz package into a list of string that constitute a .xyz file.
    Parameters: cts_npz: numpy npz package
                    first argument. The input numpy npz package.
                atom_list: python list, numpy array of strings
                    second argument. This list contains the atomic symbols of the molecule.
                convert_factor: float
                    optional, default=1.889725989. This is the unit conversion factor. The default is to convert Bohr to
                    angstrom.
    Returns: a list of strings.

UtilityPackage.regulate_data(data, threshold_reg, axis_time=0)
    Regulate the data so that the absolute of the values above a certain threshold along a certain axis can be
    discarded.
    Parameters: data: numpy array
                    first argument. Input array.
                threshold_reg: float, integer, positive
                    second argument. The value of threshold.
                axis_time: integer
                    optional, default=0. The number of axis corresponding to time in the case of trajectory analysis.
                    If at any points along this axis the absolute of the value is larger than threshold_reg, the entire
                    trajectory (the whole axis) is skipped, i.e. discarded.
    Returns: a list of trajectory numbers that passed the test.
    """
    @staticmethod
    def npz_unilen(npz, n_pts, masked=True, filled=0):
        """Unifying the length of the input numpy npz package, with the ability to also handle regular list and regular
        numpy array"""
        def array_unilen(array_in, n_pts, masked=True, filled=0):
            # array has to be numpy array
            n_pts_current = array_in.shape[0]
            if n_pts_current >= n_pts:
                return array_in[:n_pts, ...]
            else:
                if masked:
                    temp = np.zeros((n_pts, ) + array_in.shape[1:])
                    temp[:n_pts_current, ...] = array_in
                    array_masked = np.ma.array(temp)
                    array_masked[n_pts_current:, ...] = np.ma.masked
                    return array_masked
                else:
                    temp = np.zeros((n_pts, ) + array_in.shape[1:])
                    temp[:n_pts_current, ...] = array_in
                    if filled != 0:
                        temp[n_pts_current:, ...] = filled
                    return temp

        if type(npz) is np.lib.npyio.NpzFile:
            n_trj = len(npz.files)
            if masked:
                return np.ma.array([array_unilen(npz['arr_' + str(i)], n_pts, masked=True, filled=filled)
                                    for i in list(range(n_trj))])
            else:
                return np.array([array_unilen(npz['arr_' + str(i)], n_pts, masked=False, filled=filled)
                                 for i in list(range(n_trj))])
        elif type(npz) is list:
            n_trj = len(npz)
            if masked:
                return np.ma.array([array_unilen(npz[i], n_pts, masked=True, filled=filled)
                                    for i in list(range(n_trj))])
            else:
                return np.array([array_unilen(npz[i], n_pts, masked=False, filled=filled)
                                 for i in list(range(n_trj))])
        elif type(npz) is np.ndarray:
            if masked:
                return np.ma.array(array_unilen(npz, n_pts, masked=True, filled=filled))
            else:
                return np.array(array_unilen(npz, n_pts, masked=False, filled=filled))
        else:
            raise Exception('Wrong type of npz file in npz_unilen!')

    @staticmethod
    def expand_multi_dims(a_shape, array, n_axis):
        # Expand the dimension of array to match a_shape along axis n_axis in order to properly broadcast
        # For example, an array of size 5, and a shape of (3,5,7) and axis=-2, output array will be (1,5,1).
        # Notice the chosen dimension of a_shape has to be the same size as array
        if a_shape[n_axis] != len(array):
            raise Exception(
                'Array length not equal to length of the chosen axis in a_shape! \n'
                'An error would occur when broadcasting!\n')
        re_shape = [1] * len(a_shape)
        re_shape[n_axis] = len(array)
        return np.reshape(array, tuple(re_shape))

    @staticmethod
    def maxpts_from_npz(npz_file, axis=0):
        if type(npz_file) is not np.lib.npyio.NpzFile:
            raise Exception("Expected npz file in maxpts_from_npz(npz_file)")
        n_file = len(npz_file.files)
        return np.max([npz_file['arr_'+str(i)].shape[axis] for i in list(range(n_file))])

    @staticmethod
    def rep_pinpoint(array, pinpoint, diss=None):
        # This method selects representing points along a trajectory based on pinpoint array and optionally on diss
        # array (dissociation detection results array). When diss is not passed in, pinpoint array should have same form
        # as diss normally would, namely, a n_channel by n_trajectory matrix where -1 marks non-dissociation
        # trajectories and any integer other than -1 signifies the number of point to be taken as representative of such
        # trajectory. Note: the input array has to have the first two dimensions corresponding to trajectories and
        # points along trajectory.
        if diss is None:  # This selects all the trajectories
            trj = [np.where(i != -1)[0] for i in pinpoint]
        else:
            trj = [np.where(i != -1)[0] for i in diss]
        pts = [p[t] for t, p in zip(trj, pinpoint)]  # This gives the corresponding pinpoint
        return [array[t, p, ...] for t, p in zip(trj, pts)]

    @staticmethod
    def cts2xyz(cts_npz, atom_list, convert_factor=1.889725989):
        atom_array = np.array(atom_list, dtype=str)
        xyz_str_list = list()
        for i in range(len(cts_npz.files)):
            xyz = cts_npz['arr_' + str(i)] * convert_factor
            xyz_aug = np.insert(np.array(xyz, dtype=str), 0, atom_array, axis=-1)
            one_file = list()
            for j in range(len(xyz_aug)):
                one_file.append(str(len(atom_list)))
                one_file.append("point " + str(j))
                one_file += [" ".join(line) for line in xyz_aug[j]]
            one_file_str = "\n".join(one_file)
            xyz_str_list.append(one_file_str)
        return xyz_str_list

    @staticmethod
    def regulate_data(data, threshold_reg, axis_time=0):
        if type(data) is not list and type(data) is not np.ndarray:
            raise Exception("Unsupported data type for regulate_data() method.")
        else:
            if type(data) is list:
                shape = data[0].shape
                dims = len(shape)
            else:
                shape = data.shape
                dims = len(shape)
        result = [i for i in range(len(data))
                  if (np.abs(data[i] - data[i][tuple(slice(shape[j]) if j != axis_time else slice(1)
                                                     for j in range(dims))]) > threshold_reg).any()]
        return result


class FileRead(object):
    """Read Gaussian trajectory log file based on regex patterns and then do proper processing.

Initialization: FileRead( n_atom=-1)
    The initialization of this object will attempt to determine the number of atoms in the molecule and set up a pattern
    dictionary, partially based on the number of atoms.
    Parameters: n_atom: integer
                    optional, default=-1, which is a flag to signify this value was not directly passed in. This gives
                    the number of atoms in the molecule for the Gaussian log files to read.

FileRead.pattern_dict = {
    "Electric Field": r"An electric field of +([\-+]*[.\d]+D*[\-+]*[\d]*) +([\-+]*[.\d]+D*[\-+]*[\d]*) +"
                      r"([\-+]*[.\d]+D*[\-+]*[\d]*)|Standard basis: .+(?:\n.+){1,2} basis functions,",
    "Electric Field Alt": r"Electric field = +([\-+]*[.\d]+D*[\-+]*[\d]*) +([\-+]*[.\d]+D*[\-+]*[\d]*) +"
                          r"([\-+]*[.\d]+D*[\-+]*[\d]*)|Standard basis: .+(?:\n.+){1,2} basis functions,",
    "Cartesian Coordinates #VarRow": r"Cartesian coordinates.+\n"
                                     + r".+ X= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                                       r" +Y= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                                       r" +Z= +([\-+]*[.\d]+D*[\-+]*[\d]*)\n" * self.n_atom,
    "Mass Weighted Velocity #VarRow": r"MW [cC]artesian velocity:.*\n"
                                      + r" I= +\d+ X= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                                        r" +Y= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                                        r" +Z= +([\-+]*[.\d]+D*[\-+]*[\d]*)\n" * self.n_atom,
    "Time": r"Time \(fs\) +([\-+]*[.\d]+D*[\-+]*[\d]*)",
    "Total Energy": r"ETot += +([\-+]*[.\d]+D*[\-+]*[\d]*)",
    "Kinetic Energy": r"EKin = +([\-+]*[.\d]+D*[\-+]*[\d]*);",
    "Potential Energy": r"EPot = +([\-+]*[.\d]+D*[\-+]*[\d]*);",
    "Total Angular Momentum": r"Jtot = +([\-+]*[.\d]+D*[\-+]*[\d]*) H-BAR",
    "Angular Momemtum Components": r"Angular momentum \(instantaneous\)\n +JX ="
                                   r" +([\-+]*[.\d]+D*[\-+]*[\d]*) +JY = +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                                   r" +JZ = +([\-+]*[.\d]+D*[\-+]*[\d]*)",
    "Mulliken Charges #VarRow": r"(?<=\n) Mulliken charges(?::| and spin densities:)(?! with)\n.+\n"
                                + r" +\d+ +\S+ +([\-+]*[.\d]+)(?:\n| +[\-+]*[.\d]+\n)" * self.n_atom,
    "Dipole Moment": r" Dipole moment .+:\n +X= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                     r" +Y= +([\-+]*[.\d]+D*[\-+]*[\d]*) +Z= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
}
    There are three types of patterns:
        1. Fixed dimension: for example, "Time": r"Time \(fs\) +([\-+]*[.\d]+D*[\-+]*[\d]*)"
            This means a 1-dimensional array will be read from every data block at each time step.
        2. Variable row number: for example, "Mulliken Charges #VarRow":
            r"(?<=\n) Mulliken charges(?::| and spin densities:)(?! with)\n.+\n"
            + r" +\d+ +\S+ +([\-+]*[.\d]+)(?:\n| +[\-+]*[.\d]+\n)" * self.n_atom
            This means a 2-dimensional array will be read from every data block at each time step. In order to preserve
            the shape of such array, the regex pattern is generated by a variable, self.n_atom, which is the number of
            atoms determined previously. "#VarRow" in the key of this dictionary entry signifies the follow-up method
            to treat such data type accordingly.
        3. Variable column number: similar to Variable row number type. Right now there is not any data type needed in
            this category, but the functionality is built in already.
        So far the only variable in the above mentioned variable type patterns is the number of atoms, and can only be
        the number of atoms.

FileRead.findall(file_str, data_type)
    The method to extract the data from a log file.
    Parameters: file_str: string, python File object
                    first argument. The raw string of a log file or the file handle of said file to be read from.
                data_type: string
                    second argument. The string of the key in the pattern_dict dictionary. An error will be raised if
                    such entry was not added before using this method.
    Returns: numpy array.

    Example:
        Suppose all the Cartesian coordinates are to be extracted from a log file. The following lines will do the job:
            fh = open("trajectory.log", "r")
            reader = FileRead(file_str=fh)
            cts = reader.findall(fh, "Cartesian Coordinates #VarRow")
            fh.close()
        Array cts is now a multi-dimensional array containing such data. cts[points, n_atoms, xyz]
    """

    def __init__(self, n_atom=-1):
        self.pattern_dict = {}
        if n_atom == -1:  # This is when FileRead object was created without n_atom specified.
            self.flag_init = False
        else:
            self.n_atom = n_atom
            self.init_pattern()
            self.flag_init = True

    def __add_pattern(self, key, pattern):
        self.pattern_dict[key] = pattern

    def get_pattern_keys(self):
        return [key for key, value in self.pattern_dict.items()]

    def init_natom(self, raw_str):
        if len(raw_str) > 300*80:
            init_str = raw_str[:300*80]
        else:
            init_str = raw_str
        pa = re.compile(r" Stoichiometry +([A-Za-z]+)", re.M)
        pr = pa.findall(init_str)
        if len(pr) == 0:
            raise Exception("Failed to find atom number in init_natom()!")
        self.n_atom = len([i for i in pr[0] if i.isupper()])

    def init_pattern(self):
        # This dictionary defines the regex strings for reading various physical quantities that are available in
        # a Gaussian trajectory log file. The key of each item in this dictionary has no significance except for the
        # part following "#": #VarRow signifies variable number of rows to read and #VarColumn variable number of
        # column to read. The default option is to set such number to the number of atoms.
        self.pattern_dict = {
            "Electric Field": r"An electric field of +([\-+]*[.\d]+D*[\-+]*[\d]*) +([\-+]*[.\d]+D*[\-+]*[\d]*) +"
                              r"([\-+]*[.\d]+D*[\-+]*[\d]*)|Standard basis: .+(?:\n.+){1,2} basis functions,",
            "Electric Field Alt": r"Electric field = +([\-+]*[.\d]+D*[\-+]*[\d]*) +([\-+]*[.\d]+D*[\-+]*[\d]*) +"
                                  r"([\-+]*[.\d]+D*[\-+]*[\d]*)|Standard basis: .+(?:\n.+){1,2} basis functions,",
            "Cartesian Coordinates #VarRow": r"Cartesian coordinates.+\n"
                                             + r".+ X= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                                               r" +Y= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                                               r" +Z= +([\-+]*[.\d]+D*[\-+]*[\d]*)\n" * self.n_atom,
            "Mass Weighted Velocity #VarRow": r"MW [cC]artesian velocity:.*\n"
                                              + r" I= +\d+ X= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                                                r" +Y= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                                                r" +Z= +([\-+]*[.\d]+D*[\-+]*[\d]*)\n" * self.n_atom,
            "Time": r"Time \(fs\) +([\-+]*[.\d]+D*[\-+]*[\d]*)",
            "Total Energy": r"ETot += +([\-+]*[.\d]+D*[\-+]*[\d]*)",
            "Kinetic Energy": r"EKin = +([\-+]*[.\d]+D*[\-+]*[\d]*);",
            "Potential Energy": r"EPot = +([\-+]*[.\d]+D*[\-+]*[\d]*);",
            "Total Angular Momentum": r"Jtot = +([\-+]*[.\d]+D*[\-+]*[\d]*) H-BAR",
            "Angular Momemtum Components": r"Angular momentum \(instantaneous\)\n +JX ="
                                           r" +([\-+]*[.\d]+D*[\-+]*[\d]*) +JY = +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                                           r" +JZ = +([\-+]*[.\d]+D*[\-+]*[\d]*)",
            "Mulliken Charges #VarRow": r"(?<=\n) Mulliken charges(?::| and spin densities:)(?! with)\n.+\n"
                                        + r" +\d+ +\S+ +([\-+]*[.\d]+)(?:\n| +[\-+]*[.\d]+\n)" * self.n_atom,
            "Dipole Moment": r" Dipole moment .+:\n +X= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
                             r" +Y= +([\-+]*[.\d]+D*[\-+]*[\d]*) +Z= +([\-+]*[.\d]+D*[\-+]*[\d]*)"
        }

    @staticmethod
    def str_process_fixed_dim(raw_string, pa):
        pc = re.compile(pa, re.M)
        arr1 = np.array(pc.findall(raw_string))  # Makes character numpy array
        if len(arr1) != 0:
            arr1[arr1 == ''] = '0.0'
            # Scientific notation is labeled by "D" in Gaussian output
            arr2 = np.core.defchararray.replace(arr1, 'D', 'e')
            return np.array(arr2, dtype=float)
        else:
            return np.array([])

    @staticmethod
    def str_process_var_row(raw_string, pa, n_row):
        pc = re.compile(pa, re.M)
        arr1 = np.array(pc.findall(raw_string))  # Makes character numpy array
        if len(arr1) == 0:
            return np.array([])
        arr2 = np.core.defchararray.replace(arr1, 'D', 'e')  # Scientific notation is labeled by "D" in Gaussian output
        arr3 = np.array(arr2, dtype=float)
        if arr3.shape[1] % n_row != 0:
            raise Exception("Invalid number of rows in str_process_var_row()!")
        else:
            n_column = arr3.shape[1] / n_row
        if n_column > 1:
            return np.reshape(arr3, (arr3.shape[0], n_row, n_column))
        elif n_column == 1:
            return arr3

    @staticmethod
    def str_process_var_column(raw_string, pa, n_column):
        pc = re.compile(pa, re.M)
        arr1 = np.array(pc.findall(raw_string))  # Makes character numpy array
        arr1[arr1 == ''] = '0.0'
        arr2 = np.core.defchararray.replace(arr1, 'D', 'e')  # Scientific notation is labeled by "D" in Gaussian output
        arr3 = np.array(arr2, dtype=float)
        if arr3.shape[1] % n_column != 0:
            raise Exception("Invalid number of rows in str_process_var_row()!")
        else:
            n_row = arr3.shape[1] / n_column
        return np.reshape(arr3, (arr3.shape[0], n_row, n_column))

    def findall(self, file_str, data_type):
        if hasattr(file_str, 'read'):
            file_actual = file_str.read()
        elif isinstance(file_str, str):
            file_actual = file_str
        else:
            raise Exception("Wrong type of file handle passed in file_read class!")
        if not self.flag_init:  # Pattern and n_atom were not initialized yet
            self.init_natom(file_actual)
            self.init_pattern()
        if data_type not in self.get_pattern_keys():
            raise Exception("Unable to find ", data_type, " in pattern dictionary key list!")
        else:
            key, pa = data_type, self.pattern_dict[data_type]
            if "#VarRow" in key:
                array = self.str_process_var_row(file_actual, pa, self.n_atom)
            elif "#VarColumn" in key:
                array = self.str_process_var_column(file_actual, pa, self.n_atom)
            else:
                array = self.str_process_fixed_dim(file_actual, pa)
            return array


class ArrayTrim(object):
    """Trim the arrays read by FileRead according to how each type of data was printed out in Gaussian log files. This
    is necessary due to Gaussian trajectory log file routinely print out additional information at the start and/or the
    end of the whole calculation in a similar fashion as it would be for the actual time steps.


Initialization: ArrayTrim(file_str="", max_pts=-1)
    The initialization of this object will attempt to determine the maximum number of points, aka time steps, in the
    log file and set up a trim parameter dictionary.
    Parameters: file_str: string, python File object
                    optional. However, if both this and the next optional argument were not passed in during
                    initialization, an error would be raised. This argument corresponds to either string of a log file,
                    or the file handle of such file from which the initialization attempts to determine the maximum
                    number of points if not directly passed in by the next argument.
                max_pts: integer
                    optional, default=-1, which is a flag to signify this value was not directly passed in. This gives
                    the maximum number of points for the Gaussian log files.

ArrayTrim.trim_par_dict = {
    "Electric Field": (1, 0),
    "Electric Field Alt": (1, 0),
    "Cartesian Coordinates #VarRow": (1, 0),
    "Mass Weighted Velocity #VarRow": (1, 0),
    "Time": (1, 0),
    "Total Energy": (1, 0),
    "Kinetic Energy": (1, 0),
    "Potential Energy": (1, 0),
    "Total Angular Momentum": (1, 0),
    "Angular Momemtum Components": (1, 0),
    "Mulliken Charges #VarRow": (1, 1),
    "Dipole Moment": (1, 1)
}
    Two variables would be needed in the tuples: one for indicating how many points to discard at the beginning and one
    for how many points to discard at the end. Since not all types of information would be printed out at the beginning,
    usually sampling step, each type of data may require different number of points to be trimmed. This is true for the
    end of the log file as well.

ArrayTrim.set_extra_trim(ex_par)
    Set additional trimming parameter which acts on top of the orginial ones.
    Parameters: ex_par: integer
                    first argument. Extra number of points to be trimmed from the array. If ex_par > 0, trim from the
                    beginning of the array; if ex_par < 0, trim from the end of the array.

    ArrayTrim.trim(array, data_type)
    Trim the input array by the number of points based on the data_type.
    Parameters: array: numpy array
                    first argument. The array to be trimmed.
                data_type: string
                    second argument. The string of the key in the trim_par_dict dictionary. An error will be raised if
                    such entry was not added before using this method.
    """

    def __init__(self, file_str="", max_pts=-1):
        if max_pts == -1:  # This is when ArrayTrim object was created without max_pts specified.
            if file_str == "":
                raise Exception("Expected to read maximum number of points from file, "
                                "but did not pass in file handle or file string itself!")
            else:
                if hasattr(file_str, 'read'):
                    temp_file = file_str.read()
                    file_str.seek(0, 0)  # Reset file handle so it could be used for actual reading data
                elif isinstance(file_str, str):
                    temp_file = file_str
                else:
                    raise Exception("Wrong type of file handle passed in file_read class!")
                pa_maxpts = r",42=(\d+)"  # 1/42 is the IOp where number of steps are defined.
                self.max_pts = int(re.compile(pa_maxpts, re.M).search(temp_file).group(1))
        else:
            self.max_pts = max_pts
        # This dictionary should items that have values being a tuple containing the two trim parameters.
        self.trim_par_dict = {
            "Electric Field": (1, 0),
            "Electric Field Alt": (1, 0),
            "Cartesian Coordinates #VarRow": (1, 0),
            "Mass Weighted Velocity #VarRow": (1, 0),
            "Time": (1, 0),
            "Total Energy": (1, 0),
            "Kinetic Energy": (1, 0),
            "Potential Energy": (1, 0),
            "Total Angular Momentum": (1, 0),
            "Angular Momemtum Components": (1, 0),
            "Mulliken Charges #VarRow": (1, 1),
            "Dipole Moment": (1, 1)
        }
        # This parameter signifies additional number of points to be trimmed. If it is set to >= 0, extra points will be
        # trimmed from the beginning; if it is set to < 0, extra points will be trimmed from the end.
        self.extra_trim_par = 0

    def add_trim_par(self, key, par_tuple):
        self.trim_par_dict[key] = par_tuple

    def get_trim_par_keys(self):
        return [key for key, value in self.trim_par_dict.items()]

    def set_extra_trim(self, ex_par):
        self.extra_trim_par = ex_par
        
    def trim(self, array, data_type):
        par1, par2 = self.trim_par_dict[data_type]
        if array.shape[0] < self.max_pts + par1 + par2:  # This happens for incomplete trajectory
            trim1 = par1
            trim2 = None
        elif par2 == 0:  # Not trimming points at the end means array[par1:None]
            trim1 = par1
            trim2 = None
        else:
            trim1 = par1
            trim2 = -par2
        if self.extra_trim_par > 0:
            trim1 += self.extra_trim_par
        elif self.extra_trim_par < 0:
            if trim2 is None:
                trim2 = self.extra_trim_par
            else:
                trim2 += self.extra_trim_par
        return array[trim1:trim2]


class MassRead(FileRead, ArrayTrim):
    """Read multiple types of data from multiple log files and store them into numpy npz files after being trimmed
    according to how such type of data is normally printed out in Gaussian log files.
Initialization: MassRead(n_log=-1, log_list=0, ext=".log", init_fh=None, input_dir="", output_dir="",
                     data_type=None)
    Parameters: data_type: python list
                    optional, default=None. A list containing the data type names to be read from the log files. If not
                    passed in, i.e. set to None, the default data type list will be used. The default data type list is:
                        ["Cartesian Coordinates #VarRow", "Mass Weighted Velocity #VarRow", "Time",
                        "Total Energy", "Kinetic Energy", "Potential Energy", "Total Angular Momentum",
                        "Angular Momemtum Components", "Mulliken Charges #VarRow", "Dipole Moment"]
                    Only the data type name that subclasses FileRead and ArrayTrim recognise should be passed in, which
                    can be tedious to do. Therefore, if the data type needed to be read is among the already supported
                    ones, the next object attribute, MassRead.data_name_short could be more convenient to use: any data
                    type shorthand in this dictionary will be translated into its actual data type name.
                    In the case of reading and trimming unsupported data type, these two processes have to be
                    initialized manually by assign the corresponding object to attributes MassRead.read_operator and
                    MassRead.trim_operator.
                n_log: integer
                    optional, default=-1, which is a flag signifying nothing passed in. This is the number of log files
                    to read.
                log_list: python list of strings, python File object
                    optional, default=0, which is a flag signifying nothing passed in. This is either a list of log
                    file names to read or the file handle of the text file containing such list. If the previous
                    argument, n_log, was passed in, the initialization will generate a list of log file names in the
                    following manner:
                    [0.log, 1.log, 2.log, 3.log, ...]; otherwise, the actual list passed in here or the one read from
                    the file handle passed in here will be used.
                init_fh: tuple, string, python File object
                    optional, default=None. This parameter is what the initialization attempts to initialize the
                    FileRead and ArrayTrim objects with. If passed in a tuple of integers,
                    (init_n_atom, init_max_pts)=init_fh; if passed in the raw string or file handle of a log file,
                    the n_atom and max_pts parameters needed to initialize the FileRead and ArrayTrim objects will be
                    read from such log file.
                input_dir: string
                    optional, default="". Input directory from where the log file would be read. If nothing passed in,
                    the current directory from where the script is running in will be taken.
                output_dir: string
                    optional, default="". Output directory to where the results will be saved. If nothing passed in,
                    the current directory from where the script is running in will be taken.
                ext: string
                    optional, default=".log". File extension of the log files. No need to change unless somehow the
                    file extension of the Gaussian log files are not ".log".

MassRead.data_name_short = {
    "ef": "Electric Field",
    "ef_alt": "Electric Field Alt",
    "xyz": "Cartesian Coordinates #VarRow",
    "MWVxyz": "Mass Weighted Velocity #VarRow",
    "time": "Time",
    "Etot": "Total Energy",
    "Ekin": "Kinetic Energy",
    "Epot": "Potential Energy",
    "Jtot": "Total Angular Momentum",
    "Jtot_xyz": "Angular Momemtum Components",
    "MlkC": "Mulliken Charges #VarRow",
    "dipole": "Dipole Moment"
}
    In the case that the data type needed to be read is among the already supported ones, any data type shorthand in
    this dictionary will be translated into its actual data type name in the MassRead.read() method.

MassRead.read_operator = None
    This should be assigned to a FileRead object. In the case of reading unsupported data type, do the following:
        fh = open(log_file_name, "r")
        mr = MassRead(n_log=100)
        mr.read_operator = FileRead(file_str=fh)
        mr.read_operator.pattern_dict[custom_type_name] = custom_type_pattern
        fh.close()

MassRead.trim_operator = None
    This should be assigned to a ArrayTrim object. In the case of trimming unsupported data type, do the following:
        fh = open(log_file_name, "r")
        mr = MassRead(n_log=100)
        mr.trim_operator = ArrayTrim(file_str=fh)
        mr.trim_operator.trim_par_dict[custom_type_name] = custom_type_pattern
        fh.close()

MassRead.data_read = list()
    The results from MassRead.read() is stored in this attribute.

MassRead.read(n_log=-1, log_list=None)
    Read the data types in MassRead.data_type_list from log files with names in log_list.
    Parameters: n_log: integer
                    optional, default=-1, which is a flag signifying nothing passed in. This is the number of log files
                    to read.
                log_list: python list of strings, python File object
                    optional, default=0, which is a flag signifying nothing passed in. This is either a list of log
                    file names to read or the file handle of the text file containing such list. If the previous
                    argument, n_log, was passed in, the initialization will generate a list of log file names in the
                    following manner:
                    [0.log, 1.log, 2.log, 3.log, ...]; otherwise, the actual list passed in here or the one read from
                    the file handle passed in here will be used.
                These two parameters are used identically as in initialization and are used to read files differently
                as set up during initialization if needed.
    Returns: a list of lists of numpy arrays. The dimension is this 2-D list is [data_type, log_files]

MassRead.save(list_of_data=None, list_of_names=None)
    Save the data from list_of_data as numpy npz package files with the names in list_of_names.
    Parameters: list_of_data: a list of lists of numpy arrays.
                optional, default=None, which signifies using MassRead.data_read instead. The dimension of this 2-D list
                is [data_type, log_files].
                list_of_names: a list of names of the data types
                optional, default=None, which signifies using MassRead.data_type_list instead. The
                MassRead.data_name_short will be used reversely to translate actual data type names into shorhands when
                saving into numpy npz files.

Example:
    Suppose there are 100 log files named numerically, starting from "0.log". The following script will read and save
    all supported data types:
        mr = MassRead(n_log=100, input_dir=INPUT_DIRECTORY, output_dir=OUTPUT_DIRECTORY)
        mr.read()
        mr.save()
"""

    def __init__(self, n_log=-1, log_list=0, ext=".log", init_fh=None, input_dir="", output_dir="", data_type=None):
        FileRead.__init__(self, n_atom=0)
        ArrayTrim.__init__(self, max_pts=0)
        # This part determines the directory delimiter. For Unix based system, this is simply forward slash, "/";
        # for Windows, this is backslash, "\".
        if '/' in os.getcwd():
            self.slash = '/'
        elif '\\' in os.getcwd():
            self.slash = '\\'
        self.ext = ext
        self.data_read = list()
        # data_type_list will determine the types of data to be read in for all the log files. The following is the
        # default list that could be used or overwritten.
        self.data_type_list_default = ["Cartesian Coordinates #VarRow", "Mass Weighted Velocity #VarRow", "Time",
                                       "Total Energy", "Kinetic Energy", "Potential Energy", "Total Angular Momentum",
                                       "Angular Momemtum Components", "Mulliken Charges #VarRow", "Dipole Moment"]
        if data_type is None:
            self.data_type_list = self.data_type_list_default
        else:
            if type(data_type) is list:
                self.data_type_list = data_type
            elif type(data_type) is str:
                self.data_type_list = [data_type]
            else:
                raise Exception("data_type for reading should be either a list or a string.")
        # In order to initialize FileRead and ArrayTrim objects, n_atom and max_pts are needed. Therefore init_fh
        # should be either a tuple like (n_atom, max_pts), a file handle to a sample log file, or such file itself.
        # If none of those were specified, a re-initialization is needed whenever such operator was called.
        if init_fh is None:
            self.read_operator = None
            self.trim_operator = None
        elif type(init_fh) is tuple:
            (init_n_atom, init_max_pts) = init_fh
            self.read_operator = FileRead(n_atom=init_n_atom)
            self.trim_operator = ArrayTrim(max_pts=init_max_pts)
        elif type(init_fh) is str or hasattr(init_fh, 'read'):
            self.read_operator = FileRead()
            self.trim_operator = ArrayTrim(file_str=init_fh)
        else:
            raise Exception("Wrong type of init_fh parameter was passed in MassRead!")
        # Three methods to specify log_list are allowed:
        # 1. Directly pass in a list of log file names. Obviously constructing such a list outside of this object
        # is very easy.
        # 2. A file handle linking to a text file in which each line corresponds to file name of a log file.
        # 3. A integer n. This corresponds to a list of file names in the format of "i.log",
        # where i takes the value of 1 to n.
        if n_log != -1 or log_list is not None:
            self.log_list = self.gen_log_list(n_log=n_log, log_list=log_list)
        else:
            self.log_list = list()
        # Now setting up input and output directory
        if input_dir == "":
            self.input_dir = self.check_dir(os.getcwd())
        else:
            self.input_dir = self.check_dir(input_dir)
        if output_dir == "":
            self.output_dir = self.check_dir(os.getcwd())
        else:
            self.output_dir = self.check_dir(output_dir)
        # Setting up short-hand names for each data type
        self.data_name_short = {
            "ef": "Electric Field",
            "ef_alt": "Electric Field Alt",
            "xyz": "Cartesian Coordinates #VarRow",
            "MWVxyz": "Mass Weighted Velocity #VarRow",
            "time": "Time",
            "Etot": "Total Energy",
            "Ekin": "Kinetic Energy",
            "Epot": "Potential Energy",
            "Jtot": "Total Angular Momentum",
            "Jtot_xyz": "Angular Momemtum Components",
            "MlkC": "Mulliken Charges #VarRow",
            "dipole": "Dipole Moment"
        }

    @staticmethod
    def check_log_list(log_list, ext, slash):
        """This method will check each item in log_list and add or change extension when necessary"""
        if type(log_list) is not list:
            raise Exception("Expected log_list to be a list, obtained "+type(log_list)+" instead!")
        else:
            if "." not in ext:
                ext = "." + ext
            log_list_checked = list()
            for item in log_list:
                if type(item) is not str:
                    raise Exception("Non-string item found in log list!")
                else:
                    temp = item.strip().strip(slash).split(slash)[-1]
                    if ext == "."+temp.split(".")[-1]:
                        log_list_checked.append(temp)
                    elif "." in temp:
                        log_list_checked.append(".".join(temp.split(".")[:-1])+ext)
                    elif temp != "":
                        log_list_checked.append(temp + ext)
        return log_list_checked

    def gen_log_list(self, n_log=-1, log_list=None):
        if n_log == -1:  # This means the number of log files were not specified
            if type(log_list) is list:  # This is when a list of log file names were directly passed in
                init_log_list = log_list
            elif hasattr(log_list, 'read'):
                # This is when a file handle linking to an assumed text file containing log file names were passed in
                init_log_list = log_list.readlines()
            else:
                raise Exception("Did not specify n_log then failed to obtain log list from log_list parameter!")
        elif n_log > 0:
            init_log_list = [str(i+1) + ext for i in list(range(n_log))]
        else:
            raise Exception("Invalid n_log parameter passed in gen_log_list()!")
        return self.check_log_list(init_log_list, self.ext, self.slash)

    def check_dir(self, dir):
        if type(dir) is not str:
            raise Exception("dir expected to be a string!")
        elif dir.endswith(self.slash):
            return dir
        else:
            return dir + self.slash

    def check_data_type_list(self):
        temp_data_type = list()
        for item in self.data_type_list:
            if item in self.data_type_list_default:
                temp_data_type.append(item)
            elif item in self.data_name_short:
                temp_data_type.append(self.data_name_short[item])
                if self.data_name_short[item] not in self.data_type_list_default:
                    warnings.warn("Type "+self.data_name_short[item]+" not supported by default!")
            else:
                warnings.warn("Type "+item+" not supported by default!")
                temp_data_type.append(item)
        self.data_type_list = temp_data_type

    def read(self, n_log=-1, log_list=None):
        """This method will mass read various data (types defined by data_type_list) from log files (defined by
        log_list)"""
        if n_log != -1 or log_list is not None:
            log_list = self.gen_log_list(n_log=n_log, log_list=log_list)
        else:
            log_list = self.log_list
        if len(log_list) == 0:
            raise Exception("Empty log_list in MassRead.read()! ")
        if self.read_operator is None:
            self.read_operator = FileRead()
        if self.trim_operator is None:
            with open(self.input_dir + log_list[0], "r") as fh:
                temp_str = fh.read()
            self.trim_operator = ArrayTrim(file_str=temp_str)
        if len(self.data_type_list) == 0:
            raise Exception("Data type list is empty!")
        self.check_data_type_list()
        raw_data = list()
        for log_name in log_list:
            with open(self.input_dir + log_name, "r") as fh:
                log_str = fh.read()
            raw_data.append([self.trim_operator.trim(self.read_operator.findall(log_str, data), data)
                             for data in self.data_type_list])
        # raw_data is an len(log_list) sized list of lists with size of len(data_type_list). We obviously would like to
        # transpose this 2D list so that each item in this list is a list of raw data from all log files for a certain
        # data type.
        results = list(map(list, zip(*raw_data)))
        self.data_read = results
        return results
        
    def save(self, list_of_data=None, list_of_names=None):
        """This method will save the list_of_data presumably created by method read(). List of lists created by other
        means is acceptable but the list_of_names should reflect that. A check will be performed on this list to make
        sure it corresponds to each item in list_of_data."""
        name_short_retro = {i: j for j, i in self.data_name_short.items()}
        if list_of_data is None:
            list_of_data = self.data_read
        if list_of_names is None:
            list_of_names = self.data_type_list
        if len(list_of_data) != len(list_of_names):
            raise Exception("Length of list_of_data not equal to length of list_of_names!")
        proc_list_names = list()
        for name in list_of_names:
            if name not in name_short_retro:
                proc_list_names.append("".join(x for x in name if x.isalnum()))
            else:
                proc_list_names.append(name_short_retro[name])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        for data, name in zip(list_of_data, proc_list_names):
            np.savez(self.output_dir + name + ".npz", *data)


class TrjScreen(UtilityPackage):
    """This is a bundle of tests: first test on Mulliken charges to detect very sudden fluctuations (mostly unphysical)
    during laser pulse; second test on total energy to detect very sudden jumps during and after laser pulse, such
    occurrence usually signifies an abrupt change of electronic state, which should be excluded from future
    calculations.

TrjScreen.mlk_test(mlk, mlk_trunc=True, n_cutoff=374, charge_threshold=0.9, n_threshold=94)
    This tests Mulliken charges to detect very sudden fluctuations (mostly unphysical) during laser pulse. If the
    Mulliken charge of a certain atom goes from -charge_threshold to +charge_threshold (or +charge_threshold to
    -charge_threshold) within n_threshold time steps, the entire trajectory fails the test and its file number will be
    recorded.
    Parameters: mlk: numpy array
                    first argument. The input Mulliken Charge array upon which this test is applied.
                mlk_trunc: bool
                    optional, default=True. Flag for turning on truncating mlk array. This is required for mlk array
                    that is either a python list or numpy npz package file.
                n_cutoff: positive integer
                    optional, default=374. The number of points, i.e. time steps, of the laser pulse duration.
                n_threshold: positive integer
                    optional, default=94. The number of points, i.e. time steps, of the threshold fluctuation window.
                    Usually this is set to the duration of one laser cycle.
                charge_threshold: positive float
                    optional, default=0.9. The charge threshold to determine if the charge value is "too extreme", i.e.
                    the window of charge fluctuation is defined as < -charge_threshold to > +charge_threshold.
    Returns: a numpy array with bool values. True value means the trajectory failed this test, False means it passed.
    CAUTION: This method involves two n*n matrices, where n = n_atoms * n_trj * n_time_steps. Therefore it is
        potentially extremely memory demanding. This is a drawback of implementing this function in a completely
        vectorized fashion.

TrjScreen.ejump_whitelist(etot, mlk_test_result=None, n_cutoff=374, threshold=0.01, pre_thresh=100)
    This detects unphysically large (defined by parameter threshold=0.01 in the same unit as input total energy array)
    energy jump AFTER the laser pulse duration, and DURING the laser pulse duration (threshold defined by
    threshold * pre_thresh). Any points, i.e. time steps, that passed such test will be recorded into a white list.
    Parameters: etot: numpy npz package, numpy array, python list
                    first argument. The array that contains the energy data, usually the total energy.
                mlk_test_result: numpy array.
                    optional, default=None, which signifies all trajectories are to be considered as being passed
                    such test. This is the result from the previous test. Any trajectory that did not pass this test is
                    entirely excluded from the white list, i.e. an array with all elements being False will be
                    representing this trajectory.
                n_cutoff: positive integer
                    optional, default=374. The number of points, i.e. time steps, of the laser pulse duration. The
                    distinction between during and after the laser pulse is important since the energy jump threshold
                    is set differently according to the next two parameters, threshold=0.01 and pre_thresh=100.
                threshold: positive float
                    optional, default=0.01. The threshold that determines an unphysically large energy jump occurs, in
                    the same unit as input energy array, after the laser pulse.
                pre_thresh: positive integer, positive float
                    optional, default=100. The ratio by which the threshold that determines an unphysically large energy
                    jump occurs during the laser pulse. The actual threshold is threshold * pre_thresh.
    Returns: a list of numpy arrays, each of which is a bool white list that signifies the points that behaved properly,
                 value=True, or improperly, value=False.

TrjScreen.screen_bundle(etot, mlk=None, mlk_test_results=None, mlk_trunc=True, n_cutoff=374, charge_threshold=0.9,
                        n_threshold=94, threshold=0.01, pre_thresh=100)
    This method combines the previous two tests. All the parameters are kept the same as in the previous two methods.
"""

    @staticmethod
    def mlk_test(mlk, mlk_trunc=True, n_cutoff=374, charge_threshold=0.9, n_threshold=94):
        """mlk_trunc signifies raw Mulliken charge data truncating to a unified length, required for uneven data.
        n_cutoff usually corresponds to the length of the whole laser (in terms of number of time steps).
        n_threshold determines "how sudden" the fluctuation is allowed to be: if charge changes, for example, from
        -charge_threshold to +charge_threshold within n_threshold number of steps, the test fails, i.e. the charges
        fluctuates too fast. This is obviously the same for the case of charges changing from +charge_threshold to
        -charge_threshold."""
        if mlk_trunc:
            mlk_prd = UtilityPackage.npz_unilen(mlk, n_cutoff)  # mlk_prd[trajectories,points,atoms]
            if type(mlk) is np.ndarray:
                mlk_prd = mlk_prd[np.newaxis, ...]  # This simply gives a dummy axis at the beginning.
        else:
            if type(mlk) is list or type(mlk) is np.lib.npyio.NpzFile:
                raise Exception("Data truncating is required for mlk that is of list or npz type.")
            mlk_prd = mlk
        map_charge = (2 * (mlk_prd > charge_threshold) + (-1) * (mlk_prd < -charge_threshold)).astype(np.int8)
        change_map = np.diff(map_charge, axis=1)
        screen_initial = np.sum((change_map == 3) + (change_map == -3), axis=(1, 2))
        # This corresponds to when chage change from +charge_threshold to -charge_threshold within one step
        idx_ex = (UtilityPackage.expand_multi_dims(change_map.shape, np.arange(1, n_cutoff), 1)).astype(
            np.int16)  # This is an index array with dimensions prepared for broadcasting
        plus2zero = (change_map == -2).filled(False)  # Filling masked element with the value False
        minus2zero = (change_map == 1).filled(False)
        zero2plus = (change_map == 2).filled(False)
        zero2minus = (change_map == -1).filled(False)
        plus2minus_dis = (-(plus2zero * idx_ex)[:, :, np.newaxis, :] + (zero2minus * idx_ex)[:, np.newaxis, :, :])
        # The np.newaxis is arranged so that the summation of these two matrices
        # would contain plus2minus[i,j]=zero2minus[i]-plus2zero[j] for the one dimension along all points.
        plus2minus_mask = np.logical_and((plus2minus_dis > 0), (np.logical_and(plus2zero[:, :, np.newaxis, :],
                                                                               zero2minus[:, np.newaxis, :, :])))
        # When element in plus2minus_dis == 0, it is either because "+ to 0" and "0 to -" happened at the same point,
        # which is impossible, or was a masked point during npz_unilen(mlk,n_cutoff). The second "logical and" here
        # makes sure the mask contains combinations of i,j where j is "+ to 0" AND i is "0 to -"
        plus2minus = np.sum((plus2minus_dis < n_threshold) * plus2minus_mask, axis=(1, 2, 3))
        # The three-fold summation is simply "logical or" operation along these 3 axes(pts,pts,atom)
        del plus2minus_dis
        del plus2minus_mask
        minus2plus_dis = (-(minus2zero * idx_ex)[:, :, np.newaxis, :] + (zero2plus * idx_ex)[:, np.newaxis, :, :])
        minus2plus_mask = np.logical_and((minus2plus_dis > 0), (np.logical_and(minus2zero[:, :, np.newaxis, :],
                                                                               zero2plus[:, np.newaxis, :, :])))
        minus2plus = np.sum((minus2plus_dis < n_threshold) * minus2plus_mask, axis=(1, 2, 3))
        # The summation here is effectively "logical or" operation
        return (screen_initial + plus2minus + minus2plus) > 0

    @staticmethod
    def ejump_whitelist(etot, mlk_test_result=None, n_cutoff=374, threshold=0.01, pre_thresh=100):
        """This tests unphysical total energy jump during and after laser pulse
        pre_thresh defines the ratio of threshold during laser to the one after laser, this is a safety procedure
        for the rare case that Etot jumps unphysically high during laser pulse"""
        if type(etot) is np.lib.npyio.NpzFile:
            n_trj = len(etot.files)
            flag = 1
        elif type(etot) is list:
            n_trj = len(etot)
            flag = 0
        elif type(etot) is np.ndarray:
            n_trj = 1
            flag = -1
        else:
            raise Exception("Wrong file type of etot array!")
        if mlk_test_result is None:
            mlk_test_result = np.zeros(n_trj, dtype=bool)
        whitelist = list()
        for i in list(range(n_trj)):
            if flag == 1:
                etot_1trj = etot['arr_' + str(i)]
            elif flag == 0:
                etot_1trj = etot[i]
            else:
                etot_1trj = etot
            n_pts = len(etot_1trj)
            if mlk_test_result[i]:
                whitelist.append(np.zeros(n_pts, dtype=bool))
            elif n_pts <= n_cutoff:
                whitelist.append((etot_1trj - etot_1trj[0]) < (pre_thresh * threshold))
            else:
                temp = np.ones(n_pts, dtype=bool)
                temp[:n_cutoff] = (etot_1trj[:n_cutoff] - etot_1trj[0]) < (pre_thresh * threshold)
                temp[n_cutoff:] = (etot_1trj[n_cutoff:] - etot_1trj[n_cutoff - 1]) < threshold
                whitelist.append(temp)
        return whitelist

    @staticmethod
    def screen_bundle(etot, mlk=None, mlk_test_results=None, mlk_trunc=True, n_cutoff=374, charge_threshold=0.9,
                      n_threshold=94, threshold=0.01, pre_thresh=100):
        # This is a simple bundle of the two tests above for convenience. If Mulliken charge test results are needed,
        # simply use the two corresponding methods individually. Or if Mulliken test results can be read in, simply
        # pass it in.
        if mlk_test_results is not None:
            mlk_result = mlk_test_results
        elif mlk is not None:
            mlk_result = TrjScreen.mlk_test(mlk, mlk_trunc=mlk_trunc, n_cutoff=n_cutoff,
                                            charge_threshold=charge_threshold, n_threshold=n_threshold)
        else:
            if type(etot) is np.lib.npyio.NpzFile:
                n_trj = len(etot.files)
            elif type(etot) is list:
                n_trj = len(etot)
            elif type(etot) is np.ndarray:
                n_trj = 1
            else:
                raise Exception("Wrong file type of etot array!")
            mlk_result = np.zeros(n_trj, dtype=bool)
        return TrjScreen.ejump_whitelist(etot, mlk_result, n_cutoff=n_cutoff, threshold=threshold,
                                         pre_thresh=pre_thresh)


class FragGen(object):
    """Generate rule string based on fragmentation for dissociation detection.
    The fragmentation definition consists of the following parts:
        1. Molecular connectivity. Similar to Gaussian connectiviy table, each line is in the following format:
            [atom label] [atom number 1] [bond length 1] [atom number 2] [bond length 2] ...
            Each line defines connectivity to the corresponding atom. The following example defines a ethane molecule:
                C
                H 1 1.10
                H 1 1.10
                H 1 1.10
                C 1 1.53
                H 5 1.10
                H 5 1.10
                H 5 1.10
        2. Fragmentation. This begins with a line in the following format:
            Channel 1:
            The signifier is "Channel" and can be changed by optional parameters.
            The next are two connectivity tables, separated by a blank line, that define two fragments. Note: at this
            point, only two fragments can be handled simultaneously. The following is an example for H + C2H5
            dissociation:
                Channel 1:
                H

                C
                C 1 1.8
                H 1 3.0 2 3.0
                H 1 3.0 2 3.0
                H 1 3.0 2 3.0
                H 1 3.0 2 3.0
                H 1 3.0 2 3.0
            Please be noted the atom number here refers to the atom within that fragment, and the bond definition here
            has higher priority than the same bond in molecular connectivity definition. There could also be
            multiple channels.

    The output rule string consists of lines of simple logic rules in the following format:
        [atom number 1]-[atom number 2] >(or <) [value 1]; [atom number 3]-[atom number 4] >(or <) [value 2]; ... or
        [atom number 5]-[atom number 6] >(or <) [value 3]; [atom number 7]-[atom number 8] >(or <) [value 4]; ... or ...
        The conditions separated by ";" (can be customized by parameter later on) will be joined by logical and, and
        the resulted conditions separated by "or" (cannot be customized) will then be joined by logical or.
        The following is an example:
            2-1 > 1.65;5-1 < 2.70 or 8-5 > 1.65;5-1 < 2.70
        This means the following condition: (bond between atom 1 and atom 2 is larger than 1.65 and bond between atom 5
        and 1 is smaller than 2.7) or (bond between atom 8 and atom 5 is larger than 1.65 and bond between atom 5
        and 1 is smaller than 2.7).
        Each line corresponds to a single dissociation channel.

    The logic on which the generation of such rule string is based is to break all the existing bonds (defined by
        molecular connectivity section) between the two fragments (defined by fragmentation section) and maintain or
        form all the bonds within the fragments that were either defined by molecular connectivity section or by
        fragmentation section.
    Another important feature is that generated rule string will contain all possible combinations of fragmentation,
        since the fragments can be defined implicitly using only the atom symbols of its constituent atoms. Therefore,
        in the previous examples, there are 6 H + C2H5 fragmentation possibilities, all of which are considered when
        generating the output rule string.

Initialization: FraGen()
    Put attribute FraGen.frag_array in name space.

FragGen.frag_array = None
    This attribute contains the raw fragment tuples, wrapped in a multi-dimensional python list.

FragGen.rule_gen(def_str, identifier="Channel", break_factor=1.5, remain_factor=1.5, delim=";", precision=2)
    This method generates the rule string from molecular and fragmentation connectivity definitions.
    Parameters: def_str: string
                    first argument. The input string that contains molecular and fragmentation connectivity definitions.
                identifier: string
                    optional, default="Channel". The string that signifies the fragmentation definition corresponding
                    to a new channel.
                break_factor: positive float, positive integer
                    optional, default=1.5. The multiplier for breaking bonds. If the bond length of such bond is l_bond,
                    the rule string will be atom1-atom2 > break_factor * l_bond.
                remain_factor: positive float, positive integer
                    optional, default=1.5. The multiplier for remaining or newly formed bonds. If the bond length of
                    such bond is l_bond, the rule string will be atom1-atom2 < remain_factor * l_bond.
                delim: string
                    optional, default=";". The delimiter that separates conditions intended to be joined by logical and.
                precision: positive integer
                    optional, default=2. The number of digits of the float values in the output rule string.
    Returns: rule string that could be used by the DissociationDetect(UtilityPackage, FragGen) class.
"""

    def __init__(self):
        self.frag_array = None

    @staticmethod
    def line_parser(line):
        def chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        tokens = line.strip().split(" ")
        label = tokens[0]
        com_list = list(chunks(tokens[1:], 2))
        return label, com_list

    @staticmethod
    def raw_com_gen(lines):
        label_list = list()
        com_list = list()
        i = 0
        for line in lines:
            label, raw_com = FragGen.line_parser(line)
            label_list.append(label)
            com_list.append([[i, int(j[0])-1, float(j[1])] for j in raw_com])
            i += 1
        return label_list, com_list

    @staticmethod
    def comb_gen(atom_dict, frag_list):
        """This method generate all possible combinations of atom numbers corresponding to frag_list, which defines the
        fragment. The main functionality is for implicit definition, such as [C, H, H] where the molecule has multiple
         C and/or H"""
        temp_dict = atom_dict
        seq_list = list()
        comb_count_dict = {}
        # First we separate the implicit definitions from explicit ones.
        for item in frag_list:
            if type(item) is not str:
                raise Exception("Expected strings in frag_list!")
            if item.isdigit():
                seq_list.append(int(item))
                del temp_dict[int(item)]  # This is to prevent double-counting on implicit and explicit definitions.
            else:
                if item in comb_count_dict:
                    comb_count_dict[item] += 1
                else:
                    comb_count_dict[item] = 1
                seq_list.append(item)
        if len(comb_count_dict) == 0:  # This happens when all atoms in the fragment are explicitly defined.
            return [tuple(seq_list), ]  # This would make a length 1 list which contains only a tuple
        frag_atom_list = list(comb_count_dict.keys())
        # The following list contains combination generators corresponding to their atom label in the same order
        # as frag_atom_list.
        comb_generators = [itertools.combinations([i for i in temp_dict if temp_dict[i] == atom], comb_count_dict[atom])
                           for atom in frag_atom_list]
        comb_product = itertools.product(*comb_generators)  # Combine all the separate combination possibilities
        # The following makes a list of  dictionaries mapping fragment index tuple to each type of atom for each
        # possbile combination.
        comb_dict_list = [{atom: frag for atom, frag in zip(frag_atom_list, combination)}
                          for combination in comb_product]
        product = list()
        for dict_1comb in comb_dict_list:
            counting_dict = {key: 0 for key in dict_1comb}  # Initializing counting dictionary for each combination
            # Re-assemble sequence for each combination possibility
            seq_product = list()
            for item in seq_list:
                if type(item) is int:
                    seq_product.append(item)
                else:
                    seq_product.append(dict_1comb[item][counting_dict[item]])
                    counting_dict[item] += 1
            product.append(tuple(seq_product))
        return product

    @staticmethod
    def conn_gen(n_atom, conn_list):
        """This method generate the connectivity matrix. Each item in conn_list should be a tuple in the form (atom_n1,
        atom_n2, distance). Since distance cannot be negative, therefore the initial value is set to be -1.0."""
        mat = np.ones((n_atom, n_atom), dtype=float) * -1.0
        for item in conn_list:
            if len(item) == 0:
                continue
            mat[item[0], item[1]] = item[2]
            mat[item[1], item[0]] = item[2]  # This matrix should be symmetric
        return mat

    @staticmethod
    def mol_gen(section):
        """This method convert the strings defining molecule by what is entirely similar Gaussian input connectivity
        table into necessary commands and atom label lists for other functions. Variable section should contain a
        list of lines."""
        label_list, command_list = FragGen.raw_com_gen(section)
        if any([i.isdigit() for i in label_list]):
            raise Exception("Atom label in molecule definition section cannot be integer.")
        atom_dict = {key: value for key, value in enumerate(label_list)}
        conn_mat = FragGen.conn_gen(len(label_list), [item for sublist in command_list for item in sublist])
        return atom_dict, conn_mat

    @staticmethod
    def bond_break(frag_tuple, conn_mat, break_factor, delim=";", precision=2):
        n_atom = len(conn_mat)
        remain_frag = tuple(item for item in range(n_atom) if item not in frag_tuple)
        possible_connections = itertools.product(frag_tuple, remain_frag)
        rule_list = list()
        for connection in possible_connections:
            if conn_mat[connection] > 0:
                rule_list.append(str(connection[0] + 1) + "-" + str(connection[1] + 1) + " > "
                                 + ("{:." + str(precision) + "f}").format(conn_mat[connection] * break_factor))
        return delim.join(rule_list)

    @staticmethod
    def bond_form(frag_tuple, frag_command_list, remain_factor, delim=";", precision=2):
        commands = [item for sublist in frag_command_list for item in sublist]
        rule_list = list()
        for command in commands:
            if len(command) == 0:
                continue
            rule_list.append(str(frag_tuple[command[0]] + 1) + "-" + str(frag_tuple[command[1]] + 1) + " < "
                             + ("{:." + str(precision) + "f}").format(command[2] * remain_factor))
        return delim.join(rule_list)

    def frag_gen(self, mol_section, frag_section, break_factor=1.5, remain_factor=1.5, delim=";", precision=2):
        """This method convert the strings defining fragments by what is entirely similar to Gaussian input connectivity
        table into necessary commands and atom label lists for other functions. As stated in comb_gen(), the
        definition can be implicit or explicit. At the moment, only two fragments, the leaving fragment and the
        remaining fragment, can be defined.
        Variables mol_section and frag_section should contain a list of lines, each corresponding to an atom. In the
        case of len(frag_section) > 1, frag_section should contain a list of lists of lines as bonding
        breaking/forming definition module will be recursively run for more than once."""
        def frag_recurse(atom_dict, list_frag_label, nfrag):
            all_combinations_1st = FragGen.comb_gen(atom_dict, list_frag_label[0])
            if nfrag == 1:
                return [[i, ] for i in all_combinations_1st]
            else:
                all_combinations_list = [[i, ] for i in all_combinations_1st]  # Initialize this list
                for i in range(nfrag-1):
                    new_combinations = list()
                    for comb_prev_frag in all_combinations_list:
                        used_atoms = [item for sublist in comb_prev_frag for item in sublist]
                        current_atom_dict = {key: atom_dict[key] for key in atom_dict if key not in used_atoms}
                        i_new_comb = FragGen.comb_gen(current_atom_dict, list_frag_label[i+1])
                        new_combinations += [comb_prev_frag + [item, ] for item in i_new_comb]
                    all_combinations_list = new_combinations
            return all_combinations_list

        atom_dict, mol_conn_mat = FragGen.mol_gen(mol_section)
        nfrag = len(frag_section)
        frag_label_list = list()
        frag_command_list = list()
        for frag in frag_section:
            frag_label, frag_command = FragGen.raw_com_gen(frag)
            frag_label_list.append(frag_label)
            frag_command_list.append(frag_command)
        all_combinations = frag_recurse(atom_dict, frag_label_list, nfrag)
        if type(self.frag_array) is list:
            self.frag_array.append(all_combinations)
        if len(all_combinations) == 1:  # This happens when fragments are entirely explicitly defined.
            frags = all_combinations[0]
            bond_break_list = [FragGen.bond_break(frag, mol_conn_mat, break_factor, delim=delim, precision=precision)
                               for frag in frags]
            bond_form_list = [FragGen.bond_form(frag, command, remain_factor, delim=delim, precision=precision)
                              for frag, command in zip(frags,frag_command_list)]
            bond_break_str = delim.join([i for i in bond_break_list if len(i) > 0])
            bond_form_str = delim.join([i for i in bond_form_list if len(i) > 0])
            rule_str = delim.join([i for i in (bond_break_str, bond_form_str) if len(i) > 0])
        else:
            rule_str_list = list()
            for frags in all_combinations:
                bond_break_list = [FragGen.bond_break(frag, mol_conn_mat, break_factor, delim=delim,
                                                      precision=precision) for frag in frags]
                bond_form_list = [FragGen.bond_form(frag, command, remain_factor, delim=delim, precision=precision)
                                  for frag, command in zip(frags, frag_command_list)]
                bond_break_str = delim.join([i for i in bond_break_list if len(i) > 0])
                bond_form_str = delim.join([i for i in bond_form_list if len(i) > 0])
                rule_str_list.append(delim.join([i for i in (bond_break_str, bond_form_str) if len(i) > 0]))
            rule_str = " or ".join(rule_str_list)
        return rule_str

    def rule_gen(self, def_str, identifier="Channel", break_factor=1.5, remain_factor=1.5, delim=";", precision=2):
        """This is the main method to call in order to process molecular connectivity table along with fragmentation
        definition into rule strings that can be used by class DissociationDetect(object)"""
        if type(def_str) is str:
            lines = def_str.splitlines()
        elif hasattr(def_str, 'read'):
            lines = def_str.readlines()
        else:
            lines = def_str
        channels = [list(g) for k, g in itertools.groupby(lines, lambda x: x.strip().startswith(identifier)) if not k]
        mol_def = [i for i in channels[0] if len(i) > 0]
        self.frag_array = list()
        if len(channels) < 2:
            raise Exception("Only molecule definition was given in rule_gen()!")
        rule_str_list = [self.frag_gen(mol_def, [list(g) for k, g in
                                                 itertools.groupby(channel, lambda x: len(x.strip()) == 0) if not k],
                                       break_factor=break_factor, remain_factor=remain_factor, delim=delim,
                                       precision=precision) for channel in channels[1:]]
        return "\n".join(rule_str_list)


class DissociationDetect(UtilityPackage, FragGen):
    """This is a dissociation detection module based on distance matrix.
Initialization: DissociationDetect(cts_raw, channel_def, frag_gen=False, frag_identifier="Channel",
                                   frag_break_factor=1.5, frag_remain_factor=1.5, frag_delim=";", frag_precision=2,
                                   cts_unit="Bohr", value_unit="Angstrom", atom_axis=1, dist_mat=None, whitelist=None,
                                   diss=None)
    Parameters: cts_raw: numpy npz package, numpy array, python list
                    first argument. The Cartesian coordinates.
                channel_def: string.
                    second argument. The definition of dissociation channels. This consists of lines of simple logic
                    rules in the following format:
                        [atom number 1]-[atom number 2] >(or <) [value 1]; [atom number 3]-[atom number 4] >(or <)
                        [value 2]; ... or [atom number 5]-[atom number 6] >(or <) [value 3];
                        [atom number 7]-[atom number 8] >(or <) [value 4]; ... or ...
                        The conditions separated by ";" (can be customized by parameter later on) will be joined by
                        logical and, and the resulted conditions separated by "or" (cannot be customized) will then be
                        joined by logical or.
                        The following is an example:
                            2-1 > 1.65;5-1 < 2.70 or 8-5 > 1.65;5-1 < 2.70
                        This means the following condition: (bond between atom 1 and atom 2 is larger than 1.65 and
                        bond between atom 5 and 1 is smaller than 2.7) or (bond between atom 8 and atom 5 is larger
                        than 1.65 and bond between atom 5 and 1 is smaller than 2.7).
                        Each line corresponds to a single dissociation channel.
                frag_gen: bool
                    optional, default=False. Flag to switch on treating channel_def as input string for FragGen() object
                    which takes in molecular and fragment connectivity definitions and generate dissociation channel
                    definitions to use. See documentation of FragGen class.
                frag_identifier="Channel", frag_break_factor=1.5, frag_remain_factor=1.5, frag_delim=";",
                    frag_precision=2: these optional parameters are used in FragGen() object. See documentation of
                    FragGen class for their usage
                cts_unit: string
                    optional, default="Bohr". Unit of input Cartesian coordinates.
                value_unit: string
                    optional, default="Angstrom". Unit of channel definition values are in. A conversion factor will be
                    generated based on these two optional parameters.
                atom_axis: integer
                    optional, default=1. The number of axis the atoms of the molecule is on. The expected Cartesian
                    coordinates array, cts_raw has the following dimensions: cts_raw[trj, points, atoms, xyz]. Since
                    cts_raw is treated as a list of arrays, each array has the dimensionality of
                    cts_1trj[points, atoms, xyz]. Its 1th axis corresponds to atoms of the molecule.
                dist_mat: python list of numpy arrays
                    optional, default=None, which signifies calculating distance matrix during initialization. Distance
                    matrix which the whole module is based on.
                whitelist: python list of numpy arrays
                    optional, default=None. Optional white list (see TrjScreen.ejump_whitelist() method for its
                    form and usage) for skipping any points along the trajectory that are not on this white list.
                diss: numpy array
                    optional, default=None. This is the result of the main function, dissociation detection, of this
                    class. But since there are some additional methods, e.g.  DissociationDetect.pinpoint_gen(),
                    DissociationDetect.frag_detect(), that need this as input. This dissociation detection array can be
                    passed in to skip the dissociation detection procedure and these additional methods can be used
                    directly.
    Note: besides the two positional, i.e. mandatory, arguments, cts_raw and channel_def, only frag_gen and its related
        optional arguments are needed in most cases.

DissociationDetect.dissociation_detect(whitelist=None, recombination=True)
    This method execute the main function of this class: dissociation detection based on the logic defined by rule
    strings.
    Parameters: whitelist: python list of numpy arrays
                    optional, default=None. Optional white list (see TrjScreen.ejump_whitelist() method for its
                    form and usage) for skipping any points along the trajectory that are not on this white list,
                    similar to its usage during initialization.
                recombination: bool
                    optional, default=True. Flag for considering recombination. If turned on, the trajectory whose
                    geometry satisfying dissociation definition at one point can be marked as "non-dissociating" if such
                    condition is broken later on. In other words, this implies the dissociation condition was met and
                    kept until the very last time step.
    Returns: numpy array[channel, trj]. Each element is either -1, marking "non-dissociating", or a positive integer,
                 which is the number of time step the dissociation first occured.

DissociationDetect.pinpoint_gen(trj_whitelist=None, length_laser=374, search_flag=1, diss=None, non_diss_filler=False,
                                default_filler=-1)
    This method pinpoint a single time step among "valid points" along a trajectory to represent the whole trajectory
        for each trajectory in each dissociation channel.
    "Valid points": a point, i.e. a time step, is called a "valid point" when it is on the white list,
        (more specifically, at which point the corresponding element in white list array equals to True), after laser
        pulse if possible (this condition will be ignored if trajectory finished before the end of laser pulse), and
        after the dissociation (determined by diss array) if the trajectory was marked as "dissociated".
    Parameters: trj_whitelist: python list of numpy arrays
                    optional, default=None. Optional white list (see TrjScreen.ejump_whitelist() method for its
                    form and usage) for skipping any points along the trajectory that are not on this white list,
                    similar to its usage during initialization.
                length_laser: positive integer
                    optional, default=374. The number of points, i.e. time steps, of the laser pulse duration.
                search_flag: integer
                    optional, default=1. When set to 1, the algorithm takes the first point among "valid points";
                    when set to anything else, it takes the last point among "valid points".
                non_diss_filler: bool
                    optional, default=False. When set to True, in the case of the trajectory being marked as
                    "non-dissociating", the algorithm takes first or last point among "valid points" (determined by
                    search_flag parameter) as a filler; otherwise, in the case of the trajectory being marked as
                    "non-dissociating", the algorithm takes whatever the default_filler parameter sets to as a filler.
                default_filler: integer
                    optional, default=-1. As mentioned previously, this is the default filler for the case of the
                    trajectory being marked as "non-dissociating". The default value, -1, makes the pinpoint array
                    behave very similar to diss array, since in both cases, -1 signifies "non-dissociating", any other
                    value is a representing time step along the trajectory.
                diss: numpy array
                    optional, default=None. If not passed in, the diss array stored in DissociationDetect.diss attribute
                    during previous usage of DissociationDetect.dissociation_detect() will be taken. This parameter is
                    to make it possible to read in a different diss array as what was generated by
                    DissociationDetect.dissociation_detect() or previous runs of the entire program.
    Returns: numpy array in a similar form as diss array, which is a result of DissociationDetect.dissociation_detect()
                 method. The dimensionality of it is pinpoint[channel, trj].

DissociationDetect.frag_detect(pinpoint=None)
    This method is built upon FragGen class to determine which one of the possible combinations were picked up by
        dissociation_detect() at a certain point (determined by either pinpoint array or diss array) along each
        trajectory and for each dissociation channel.
    Parameters: pinpoint: numpy array
                    optional, default=None. The result of DissociationDetect.pinpoint_gen() method, which pinpoint a
                    single time step among "valid points" along a trajectory to represent the whole trajectory for each
                    trajectory in each dissociation channel. See DissociationDetect.pinpoint_gen() for the detailed
                    documentation of this array. If nothing passed in, the diss array generated by previous method for
                    the same instance of DissociationDetect class will be taken when determine which point along a
                    trajectory to be looking at.
    Returns: python list of lists of tuples. The tuples contains fragmentation definition in terms of atomic number.

Example:
    Suppose a Cartesian coordinates array has been read and saved by class MassRead. Read this array into cts_raw.
    And suppose a text file, rule_string.txt, was written containing a few lines of geometry conditions as rule string,
    in similar form as the example mentioned above:
        2-1 > 1.65;5-1 < 2.70 or 8-5 > 1.65;5-1 < 2.70
    The following script would yield a dissociation array and a representing point array:
        with open("rule_string.txt", "r") as fh:
            rules = fh.read()
        detector = DissociationDetect(cts_raw, rules)
        diss = detector.dissociation_detect()
        pinpoint = detector.pinpoint_gen()
        # Now print out the channel number and all the trajectories dissociated in this channel as well as the
        representing points.
        for i, i_diss_pp in enumerate(zip(diss,pinpoint)):
            i_diss, i_pp = i_diss_pp
            print("Channel "+str(i)+": ")
            print(" ".join([np.where(i_diss!=-1)[0]]))
            print("at points: "+" ".join([i_pp[np.where(i_diss!=-1)[0]]]))
"""
    def __init__(self, cts_raw, channel_def, frag_gen=False, frag_identifier="Channel", frag_break_factor=1.5,
                 frag_remain_factor=1.5, frag_delim=";", frag_precision=2, cts_unit="Bohr", value_unit="Angstrom",
                 atom_axis=1, dist_mat=None, whitelist=None, diss=None):
        """cts_raw contains Cartesian coordinates, which could be either regular numpy array or .npz file. In the case
        of .npz file, it will be directly unpacked into a list."""
        FragGen.__init__(self)
        if type(cts_raw) is np.lib.npyio.NpzFile:
            cts = [cts_raw["arr_"+str(i)] for i in list(range(len(cts_raw.files)))]
        else:
            cts = cts_raw
        self.cts = cts
        if dist_mat is None:
            if type(cts) is list:
                vector = [np.expand_dims(item, axis=atom_axis + 1) - np.expand_dims(item, axis=atom_axis)
                          for item in cts]
                self.dist_mat = [np.sqrt(np.sum(np.square(item), axis=-1)) for item in vector]
            else:
                vector = np.expand_dims(cts, axis=atom_axis + 1) - np.expand_dims(cts, axis=atom_axis)
                # Making dist_mat a length 1 list
                self.dist_mat = [np.sqrt(np.sum(np.square(vector), axis=-1))]  # Assuming xyz is the last axis
        else:
            self.dist_mat = dist_mat
        if whitelist is not None:
            self.whitelist = whitelist
        else:
            self.whitelist = [np.ones(i.shape[0], dtype=bool) for i in cts]  # A whitelist of all points.
        if cts_unit == value_unit:
            self.convert = 1
        elif cts_unit == "Bohr" and value_unit == "Angstrom":
            self.convert = 1.889725989
        elif cts_unit == "Angstrom" and value_unit == "Bohr":
            self.convert = 0.529177249
        else:
            raise Exception("Wrong unit for either cts or value in channel definition.")
        self.atom_axis = atom_axis
        self.channel_raw_str = channel_def
        if frag_gen:
            self.frag_generator = FragGen()
            self.channel_def_str = self.frag_generator.rule_gen(channel_def, identifier=frag_identifier,
                                                                break_factor=frag_break_factor,
                                                                remain_factor=frag_remain_factor, delim=frag_delim,
                                                                precision=frag_precision)
            self.frag_array = self.frag_generator.frag_array
        else:
            self.channel_def_str = channel_def
            self.frag_generator = None
            self.frag_array = None
        self.rule_list = self.str2logic_multiple(self.channel_def_str)
        self.diss = diss

    def make_rule_list(self, rule_str=None, delim_and=";", delim_or="or"):
        if rule_str is None:
            self.rule_list = self.str2logic_multiple(self.channel_def_str, delim_and=delim_and, delim_or=delim_or)
        else:
            self.rule_list = self.str2logic_multiple(rule_str, delim_and=delim_and, delim_or=delim_or)

    @staticmethod
    def mat_comb(idx1, idx2, dist_mat, atom_axis=1):
        def slice_gen(idx1, idx2, axis1, dims):
            slice_list = list()
            for i in list(range(len(dims))):
                if i == axis1:
                    slice_list += [idx1, idx2]
                elif i == axis1 + 1:
                    continue
                else:
                    slice_list.append(slice(dims[i]))
            return slice_list

        if type(dist_mat) is list:
            return [item[slice_gen(idx1, idx2, atom_axis, item.shape)] for item in dist_mat]
        else:
            return dist_mat[slice_gen(idx1, idx2, atom_axis, dist_mat.shape)]

    @staticmethod
    def str2logic_single(in_str):
        """This method dissolves a string corresponding to a single logic definition, which could be an entire or
        partial dissociation channel definition. In most cases, a dissociation channel definition requires several
        such single logic definition combined (connected with logical). The string should be in the following form:
        1-2 <=12 (bond between atom1 and atom2 <= 12 angstrom/bohr)"""
        pattern = r" *(?P<idx1>[0-9]+)\-{1,2}(?P<idx2>[0-9]+) *(?P<logic>[><=]+) *(?P<value>[\d.]+)\S*"
        parser = re.compile(pattern, re.I)
        match = parser.match(in_str)
        if match is None:
            raise Exception("Failed to match dissociation channel definition string in str2logic_single(in_str).")
        # Atom number is expected to be counted from 1 when writing channel definition strings
        idx1 = int(match.group("idx1")) - 1
        idx2 = int(match.group("idx2")) - 1
        logic = DissociationDetect.apply_logic(match.group("logic").strip())  # This stores a function address
        value = float(match.group("value").strip())
        return idx1, idx2, logic, value

    @staticmethod
    def str2logic_multiple(file_str, delim_and=";", delim_or="or"):
        """This method convert the entire dissociation channel definitions into a multip-dimensional list in the
        following shape:
        (channels, combinations, conditions)
        channels correspond to one ore more dissociation channels, each of which is one line in file_str;
        combinations correspond to the possible atom number combinations for a certain channel, separated by logic or;
        conditions correspond to individual logic condition that will be passed into str2logic_single(in_str)"""
        if hasattr(file_str, 'read'):
            temp_file = file_str.readlines()
        elif isinstance(file_str, str):
            temp_file = file_str.splitlines()
        else:
            raise Exception("Wrong type of file handle passed in file_read class!")
        op_list = list()
        for line in temp_file:
            if len(line.strip()) == 0:
                continue
            comb_list = line.strip().split(delim_or)
            op_list_percomb = list()
            for comb in comb_list:
                logic_list = comb.strip().split(delim_and)
                op_list_percomb.append([DissociationDetect.str2logic_single(i)
                                        for i in logic_list if len(i.strip()) > 0])
            op_list.append(op_list_percomb)
        return op_list

    @staticmethod
    def apply_logic(rule_string):
        if '>' in rule_string and '=' not in rule_string:
            return lambda x, y: np.greater(x, y)
        elif '<' in rule_string and '=' not in rule_string:
            return lambda x, y: np.less(x, y)
        elif '>=' in rule_string:
            return lambda x, y: np.greater_equal(x, y)
        elif '<=' in rule_string:
            return lambda x, y: np.less_equal(x, y)
        else:
            raise Exception('Could not find logic operator!')

    @staticmethod
    def detector(dist_mat, op_list, whitelist, recombination, convert_factor, atom_axis=1):
        if type(dist_mat) is not list:
            raise Exception("Wrong detector used! The one for list type used, but received a"+str(type(dist_mat)))
        diss = list()
        for channel in op_list:
            diss_mat_per_channel = list()
            for trj in dist_mat:
                diss_mat_raw = [[logic(DissociationDetect.mat_comb(idx1, idx2, trj, atom_axis=atom_axis),
                                       value * convert_factor) for idx1, idx2, logic, value in comb]
                                for comb in channel]
                # diss_mat_raw has the dimension: (combinations, conditions); now use np.prod to combine "conditions"
                # and np.sum to combine "combinations".
                diss_mat_per_channel.append(np.array(np.sum(np.stack([np.prod(np.stack(combination), axis=0)
                                                                      for combination in diss_mat_raw]),
                                                            axis=0), dtype=bool))
            i = 0
            for diss_mat_i, whitelist_i in zip(diss_mat_per_channel, whitelist):
                if len(diss_mat_i) != len(whitelist_i):
                    if len(diss_mat_i) > len(whitelist_i):
                        diss_mat_per_channel[i] = diss_mat_i[:len(whitelist_i), ...]
                    else:
                        whitelist[i] = whitelist_i[:len(diss_mat_i), ...]
                i += 1
            if whitelist is not None:
                diss_mat_scr = [diss_mat_i * whitelist_i for diss_mat_i, whitelist_i
                                in zip(diss_mat_per_channel, whitelist)]
                diss_pos = [np.where(item)[0] for item in diss_mat_scr]  # np.where() returns a tuple
                wl_pos = [np.where(item)[0] for item in whitelist]
                if recombination:
                    diss_tf = np.array([diss_pos_i[-1] >= wl_pos_i[-1] if len(diss_pos_i) > 0 and len(wl_pos_i) > 0
                                        else False for diss_pos_i, wl_pos_i in zip(diss_pos, wl_pos)], dtype=bool)
                else:
                    diss_tf = np.array([np.array(np.sum(item), dtype=bool) for item in diss_mat_scr], dtype=bool)
            else:
                diss_mat_scr = diss_mat_per_channel
                diss_pos = [np.where(item)[0] for item in diss_mat_scr]
                if recombination:
                    diss_tf = np.array([diss_mat_scr_i[-1] if len(diss_mat_scr_i) >0 else False
                                        for diss_mat_scr_i in diss_mat_scr], dtype=bool)
                else:
                    diss_tf = np.array([np.array(np.sum(item), dtype=bool) for item in diss_mat_scr], dtype=bool)
            diss_pos_pick = np.array([item[0] if len(item) > 0 else -1 for item in diss_pos])
            diss.append(diss_pos_pick * diss_tf + -1 * (1 - diss_tf))
        return np.array(diss)

    def dissociation_detect(self, whitelist=None, recombination=True):
        """This is the main method detecting dissociation according to the channels definition string/file."""
        if type(self.dist_mat) is not list:
            dist_mat = [self.dist_mat]
        else:
            dist_mat = self.dist_mat
        if whitelist is None:
            if self.whitelist is None:
                raise Exception("dissociation_detect() requires whitelist to function.")
            whitelist = self.whitelist
        diss = self.detector(dist_mat, self.rule_list, whitelist, recombination, self.convert, self.atom_axis)
        self.diss = diss
        return diss

    def frag_detect(self, pinpoint=None):
        """This method is built upon FragGen class to determine which one of the possible combinations were picked up by
        dissociation_detect() at a certain point along each trajectory"""
        if pinpoint is None:
            pinpoint = self.diss
        if type(pinpoint) is list:
            if len(pinpoint) > 0:
                if isinstance(pinpoint[0], list):
                    n_dim = 2
                else:
                    n_dim = 1
                pinpoint = np.array(pinpoint)
            else:
                n_dim = 0
        elif type(pinpoint) is np.ndarray:
            n_dim = len(pinpoint.shape)
            if n_dim > 2:
                raise Exception("pinpoint array cannot have more than two dimensions.")
        else:
            raise Exception("pinpoint has to be at least a one-dimensional list or a numpy array.")
        # information will be obtained from that was passed in during initialization of the whole instance.
        rules = self.rule_list
        if self.frag_generator is None:
            raise Exception("frag_generator cannot be None type.")
        frag_array = self.frag_array
        diss = self.diss
        if n_dim == 0:  # This is simply a safety procedure, n_dim should not normally be zero.
            return np.array([])
        elif n_dim == 1:  # This is when pinpoint has no information about different dissociation channels,
            # such information will be obtained through frag_array. Additionally, pinpiont should have same length as
            # diss[i] for ith channel.
            if len(pinpoint) != len(self.dist_mat) or any((len(pinpoint) != len(channel) for channel in diss)):
                raise Exception("Mismatch occurs between pinpoint(dim=1), diss and dist_mat arrays.")
            dist_mat_red = [np.array([dist_trj[pinpoint_trj, ...] for dist_trj, pinpoint_trj, diss_trj
                                      in zip(self.dist_mat, pinpoint, channel) if diss_trj != -1]) for channel in diss]
        else:  # n_dim == 2
            if any((len(pinpoint_channel) != len(diss_channel) or len(diss_channel) != len(self.dist_mat)
                    for diss_channel, pinpoint_channel in zip(diss, pinpoint))):
                raise Exception("Mismatch occurs between pinpoint(dim=2), diss and dist_mat arrays: ")
            dist_mat_red = [np.array([dist_trj[pinpoint_trj, ...] for dist_trj, pinpoint_trj, diss_trj
                                      in zip(self.dist_mat, pinpoint_channel, diss_channel) if diss_trj != -1])
                            for diss_channel, pinpoint_channel in zip(diss, pinpoint)]
        frag_det = list()
        for rule_channel, dist_mat_channel, frag_channel in zip(rules, dist_mat_red, frag_array):
            diss_mat_raw = [[logic(self.mat_comb(idx1, idx2, dist_mat_channel), value * self.convert)
                             for idx1, idx2, logic, value in comb] for comb in rule_channel]
            # The dimensionality of diss_mat is diss_mat[combinations, trj].
            diss_mat = np.array(np.stack([np.prod(np.stack(combination), axis=0) for combination in diss_mat_raw]),
                                dtype=bool)  # Forcing dtype=bool is very important for the following code to work
            frag_det_channel = [[frag_comb if trj else [] for trj in diss_comb]
                                for diss_comb, frag_comb in zip(diss_mat, frag_channel)]
            frag_det.append([[frag_det_channel[j][i] for j in range(len(frag_det_channel))
                              if len(frag_det_channel[j][i]) > 0] for i in range(diss_mat.shape[1])])
        return frag_det

    def pinpoint_gen(self, trj_whitelist=None, length_laser=374, search_flag=1, diss=None, non_diss_filler=False,
                     default_filler=-1):
        """search_flag == 1, pinpoint would be the first point within valid points to take; search_flag == anything else,
         it would be the last point non_diss_filler=True, (default) a similar point would be taken even for
        non-dissociating trajectories; non_diss_filler=False, 0 will be filled."""
        if trj_whitelist is None:
            if self.whitelist is None:
                raise Exception("dissociation_detect() requires whitelist to function.")
            trj_whitelist = self.whitelist
        if diss is None:
            diss = self.diss
        if type(trj_whitelist) is list:
            eoc = [len(i)-1 for i in trj_whitelist]  # An array of the index of the last points on each trajectory
        elif type(trj_whitelist) is np.lib.npyio.NpzFile:
            eoc = [len(trj_whitelist['arr_'+str(i)])-1 for i in list(range(len(trj_whitelist.files)))]
        else:
            if len(trj_whitelist.shape) == 1:  # This happens when whitelist is created for a single trajectory.
                trj_whitelist = trj_whitelist[np.newaxis, :]
            eoc = trj_whitelist.shape[1] - 1
        mask_diss = (diss != -1)
        mask_non_diss = np.array(1 - mask_diss, dtype=bool)
        n_pts_max = np.max(eoc)+1
        trj_whitelist_unilen = np.array(UtilityPackage.npz_unilen(trj_whitelist, n_pts_max, masked=False, filled=False),
                                        dtype=bool)  # Extend whitelist array to max length for each trajectory
        gt_laser_diss = np.maximum(diss+1, length_laser)
        # print("shape of diss: ", diss.shape, "shape of gt_laser_diss", gt_laser_diss.shape)
        eoc_via_trj_whitelist = np.max(trj_whitelist_unilen * np.expand_dims(np.arange(n_pts_max, dtype=int), axis=0),
                                       axis=-1)  # This is safer than eoc
        valid_pts_start = np.minimum(eoc_via_trj_whitelist[np.newaxis, ...], gt_laser_diss)
        # The following is the dimensionality of these two arrays: eoc[trjs], gt_laser_diss[channels][trjs]
        valid_pts_all = trj_whitelist_unilen * np.expand_dims(np.arange(n_pts_max, dtype=int), axis=0)
        # index array[trjs,pts] where each element is the index of the point only when it is on whitelist
        mask_valid_pts_trimmed = (np.expand_dims(valid_pts_all, axis=0) >= np.expand_dims(valid_pts_start, axis=-1))
        # mask_valid_pts_trimmed[channels,trjs,pts]
        if search_flag == 1:
            if non_diss_filler:
                filler = valid_pts_start
            else:
                filler = default_filler
            not_mask_valid_pts_trimmed=np.array(1-mask_valid_pts_trimmed, dtype=bool)
            valid_pts = mask_valid_pts_trimmed \
                * UtilityPackage.expand_multi_dims(mask_valid_pts_trimmed.shape, np.arange(n_pts_max, dtype=int), -1) \
                + not_mask_valid_pts_trimmed * n_pts_max
            # Giving all the "bad" points a large value which is impossible to obtain otherwise makes sure np.min()
            # would not pick them up; otherwise, those points would have value of 0, which would always be picked up
            pinpoint = mask_diss * np.min(valid_pts, axis=-1) + mask_non_diss * filler
        else:
            if non_diss_filler:
                filler = eoc
            else:
                filler = default_filler
            valid_pts = mask_valid_pts_trimmed \
                * UtilityPackage.expand_multi_dims(mask_valid_pts_trimmed.shape, np.arange(n_pts_max, dtype=int), -1)
            pinpoint = mask_diss * np.max(valid_pts, axis=-1) + mask_non_diss * filler
        return pinpoint


class AnalysisPackage(UtilityPackage):
    """This is a package of numerous analysis methods.
Initialization: AnalysisPackage(dir_in, dir_out, ref_energy=None, energy_unit="Hartree")
    The purpose of this initialization is to set up some commonly used input/output directory and constants.
    Parameters: dir_in: string
                    optional, default=None. Input directory from where the input arrays will be read. If nothing passed
                    in, the current directory will be taken.
                dir_out: string
                    optional, default=None. Output directory from where the output arrays will be written. If nothing
                    passed in, the current directory will be taken.
                ref_energy: float
                    optional, default=None. Reference energy value for some of the methods in this package.
                energy_unit: string
                    optional, default="Hartree". String lable for energy unit used.

AnalysisPackage.diss2csv(diss=None, name=None, diss_time=True, time_step=0.25)
    This method generates a summary spreadsheet (csv format) for dissociation identification array.
    Parameters: diss: numpy array
                    optional, default=None. Dissociation identification array. If not passed in, the method assumes the
                    input data will be read from disk by specifying name parameter.
                name: string
                    optional, default=None. Dissociation identification array name.
                diss_time: bool
                    optional, default=True. Turn on generating the dissociation time stamp summary.
                time_step: float
                    optional, default=0.25. Time step size in whatever unit the original data was using.

AnalysisPackage.etot2csv(etot=None, name=None, pinpoint=None, avg=False, ref_energy=None, unit_out="kcal/mol")
    This method generates a summary spreadsheet (csv format) for total energy array.
    Parameters: etot: numpy array, python, list, numpy npz package
                    optional, default=None. Total energy array. If not passed in, the method assumes the input data will
                    be read from disk by specifying name parameter.
                name: string
                    optional, default=None. Total energy array name.
                pinpoint: numpy array
                    optional, default=None. Pinpoint array that contains the index of representing points along each
                    trajectory in each dissociation channel. This is usually generated by the method
                    DissociationDetect.pinpoint_gen() in DissociationDetect class.
                avg: bool
                    optional, default=False. Take the average along a trajectory instead of using pinpoint array.
                ref_energy: float
                    optional, default=None. Reference energy value so that the output energies will be relative energies
                    with respect to this value. If not passed in, the reference energy value specified during the
                    initialization of AnalysisPackage will be used. If neither was given, 0.0 will be taken.
                unit_out: string
                    optional, default="kcal/mol". Output unit.

AnalysisPackage.calc_bond(select, xyz_raw=None, xyz_name=None, xyz_unit="Bohr", bond_unit="Angstrom", save=False)
    This method calculates bond length for the atom pairs defined by select array, which is a python list of atomic
    number pairs.
    Parameters: select: numpy array, python list of 2-element tuples/lists
                    first argument. This is a python list of atomic number pairs that defines the set of bond lengths
                    needed to be calculated.
                xyz_raw: numpy npz package
                    optional, default=None. Pass in the numpy npz package of Cartesian coordinates directly.It is
                    assumed that this array has the following dimensionality: xyz_raw[..., atom, xyz].
                xyz_name: string
                    optional, default=None. Pass in the numpy npz package of Cartesian coordinates indirectly. The
                    package will be read from dir_in+xyz_name. If both of these two attempts at obtaining Cartesian
                    coordinates array failed, an error will be raised. It is also assumed that this array has the
                    following dimensionality: xyz_raw[..., atom, xyz].
                xyz_unit: string
                    optional, default="Bohr". Input Cartesian coordinates unit.
                bond_unit: string
                    optional, default="Angstrom". Output bond length unit. A conversion factor will be generated based
                    on the choice of these two optional parameters.
                save: bool
                    optional, default=False. Save the results into numpy npz package.
    Returns: a list of numpy arrays, each of which contains the bond lengths of selected atomic pairs for one trajectory.

AnalysisPackage.angle_between(v1, v2)
    This method calculates the angle between vector v1 and vector v2. This is a vectorized implementation, therefore
    v1, v2, and the result can be multi-dimensional arrays.
    Parameters: v1: numpy array
                    first argument. The first vector.
                v2: numpy array
                    second argument. The second vector.
    Returns: numpy array that contains the angle(s) between v1 and v2.

AnalysisPackage.angle_signed(v1, v2, n)
    This method calculates the signed angle from vector v1 and vector v2 referenced to normal vector n. The sign of such
    angle is determined by comparing the cross product (2-D or 3-D cross product) of v1 and v2 with normal vector n. If
    this cross product is co-linear with normal vector n, the angle is positive; if anti-linear, negative. This is also
    a vectorized implementation, therefore v1, v2, n, and the result can be multi-dimensional arrays.
    Parameters: v1: numpy array
                    first argument. The first vector.
                v2: numpy array
                    second argument. The second vector.
                n: python list, numpy array
                    third argument. The normal vector that the determination of signs of angles are referenced to.
    Returns: numpy array that contains the signed angle(s) from v1 to v2 with respect to normal vector n.

AnalysisPackage.calc_angle(select, xyz_raw=None, xyz_name=None, v_normal=None, signed=True, unit="degree", save=False)
    This This method calculates bond angles for the atom sets defined by select array, which is a python list of atomic
    number tuples containing three atom numbers, representing three points in space, A, B, C. And the angle will be
    defined as the angle from vector BA to vector BC.
    Parameters: select: numpy array, python list of 3-element tuples/lists
                    first argument. This is a python list of atomic number sets that defines the set of bond angles
                    needed to be calculated.
                xyz_raw: numpy npz package
                    optional, default=None. Pass in the numpy npz package of Cartesian coordinates directly.It is
                    assumed that this array has the following dimensionality: xyz_raw[..., atom, xyz].
                xyz_name: string
                    optional, default=None. Pass in the numpy npz package of Cartesian coordinates indirectly. The
                    package will be read from dir_in+xyz_name. If both of these two attempts at obtaining Cartesian
                    coordinates array failed, an error will be raised. It is also assumed that this array has the
                    following dimensionality: xyz_raw[..., atom, xyz].
                v_normal: python list, numpy array
                    optional, default=None. The normal vector to which the signed angle is calculated with respect. If
                    not passed in, the Z-axis, i.e. vector [0, 0, 1] will be taken. This is only meaningful if signed
                    angle is required to calculate instead of unsigned.
                signed: bool
                    optional, default=True. Turn on signed angle calculation.
                unit: string
                    optional, default="degree". The unit of output angles. The other option is "radian". A conversion
                    factor will be generated based on this option.
                save: bool
                    optional, default=False. Save the results into numpy npz package.
    Returns: a list of numpy arrays, each of which contains the bond angles of selected atomic sets for one trajectory.

AnalysisPackage.rotation_matrix_given_axis(axis_vector, angle2rotate):
    The formula for this method is taken from Wikipedia at:
        https://en.wikipedia.org/wiki/Rotation_matrix#cite_note-5
        Which was original referenced at https://dspace.lboro.ac.uk/dspace-jspui/handle/2134/18050
        The function of this method is to generate a 3-dimensional rotational matrix with a given rotation axis. The
        input axis_vector can be multi-dimensional, but the last axis needs to be (x,y,z). The number of dimension of
        angle2rotate has to be one less than that of axis_vector when not considering broadcasting. In other words,
        a multi-dimensional list of axis vectors should have the same dimensional list of rotation angle values. But
        since a vector inherently has one higher dimension than a scalar, hence the dimensionality requirement. The
        method automatically takes care of two types of broadcasting cases: 1. when axis_vector has the lowest possible
        dimension, i.e. 1-dimension; and 2. when angle2rotate has the lowest possible dimension, i.e. 0-dimension, a
        single scalar. Only for these two cases, the broadcasting is unambiguous without prior knowledge of the
        dimensionality of the two input arrays; in such cases, custom broadcasting could be done by feeding in this
        method with arrays that have singleton dimensions satisfactory for broadcasting. The implementation is
        vectorized.
    Parameters: axis_vector: python list, numpy array
                    first argument. The vector(s) representing the rotation axis(axes).
                angle2rotate: float, numpy array
                    second argument. The angle(s) of the rotation(s).
    Returns: numpy array. The 3 by 3 rotation matrices are appended to the shared dimensions between two input arrays.
                 That is, suppose we have axis_vector[a,b,...,n,3] and angle2rotate[a,b,...,n], the output array would
                 be rot_mat[a,b,...,n,3,3].

AnalysisPackage.rotation_matrix_target_vector(vector1, vector2):
    This method uses the above method to generate a rotation matrix. The axis of ration is calculated by
        vector1 cross vector2, and the angle to rotate is the signed angle from vector1 to vector2. The result of such
        rotation should make vector1 coincide with vector2.
    Parameters: vector1: numpy array
                    first argument. First vector.
                vector2: numpy array
                    second argument. Second vector.
    Returns: numpy array. The 3 by 3 rotation matrices are appended to the shared dimensions between two input arrays.

AnalysisPackage.cwt(data=None, data_name=None, diss=None, diss_name="diss.npy", axis_time=0, data_reg=False,
                    threshold_reg=1000, msg_print=True, save=False)
    This method calculates continuous wavelet transformation (CWT) for trajectory data based on dissociation
        identification array. For general usage of CWT, simply use the wavelets module directly.
    The wavelets module is credited to https://github.com/aaren/wavelets by Aaron O'Leary.
    Parameters: data: python list, numpy npz package
                    optional, default=None. Data array to have CWT performed on. If not passed in, the method
                    assumes the input data will be read from disk by specifying data_name parameter.
                data_name: string
                    optional, default=None. Data array name for directly reading from the disk.
                diss: python list, numpy npz package
                    optional, default=None. Dissociation identification array. If not passed in, the method
                    assumes the input data will be read from disk by specifying diss_name parameter.
                diss_name: string
                    optional, default="diss.npy". Dissociation identification array name for directly reading from
                    the disk.
                axis_time: integer
                    optional, default=0. The number of dimension time steps are located. If the first dimension of
                    the input data is not time steps, whichever specified will be rolled to the first dimension.
                data_reg: bool
                    optional, default=False. Turn on data regulation. In some cases, the resulted CWT power spectra
                    could have a few extremely large values resulting in significant distortion. This is especially
                    common when averaging over many spectra for a set of trajectories. Discarding the problematic
                    spectra is appropriate since such occurrence is usually a result of ill-behaving trajectory.
                threshold_reg: float, integer, positive
                    optional, default=1000. The value of threshold for data regulation.
                msg_print: bool
                    optional, default=True. Print out message during CWT calculations. This is useful when
                    calculating a large number of separate CWTs.
                save: bool
                    optional, default=False. Set to true if the various results need to be saved directly on disk.
    Returns: scales: numpy array
                 first return. Wavelet transformation scales.
             power_perch: python list
                 second return. Power spectra for each trajectory in each dissociation channel.
             power_sum: numpy array
                 third return. Average power spectra over all trajectories in each dissociation channel.


class AnalysisPackage.LinearRegression(object):
    This class is a simple implementation (gradient decent) of two basic machine learning algorithms, linear regression
        and logistic regression. Due to fundamental similarity between these two algorithms, logistic regression is
        treated as a variation of linear regression.
    The idea of machine learning, more specifically supervised machine learning, is to "train" a set of parameters to
        obtain a mapping from a set features to a certain answer. To obtain such parameter, the simplest algorithm is
        gradient decent, being applied to a training set with pre-determined answers. And then such parameters can be
        used to predict answers given different sets of features.
    The parameters are referred to as "theta", the features as "x", the answers as "y", and the hypothesis as "h".
        Then we would have: h(theta * x) = y. (Equation 1)
        For linear regression, h(theta * x) is simply the matrix multiplication, theta * x; for logistic regression,
        it has one more layer: suppose z = theta * x, h(theta * x) = 1/(1+exp(-z)).
        If x and y are from training set, theta is optimized to reproduce Equation 1; if x is from actual problems with
        no answers, i.e. no y available, optimized theta is used to predict y by using Equation 1.

    Initialization: AnalysisPackage.LinearRegression(x=None, y=None, random_init=True, feature_norm=True, cost_f=None,
                                                     hyp_f=None, grad_f=None)
        Parameters: x: numpy array
                        optional, default=None. Unprocessed "raw" (usually feature scaling or other processes may be
                        needed) training set feature data.
                    y: numpy array
                        optional, default=None. Training set answer data.
                    random_init: bool
                        optional, default=True. Turn on random initialization of parameters.
                    feature_norm: bool
                        optional, default=True. Turn on feature normalization. This is recommended.
                    cost_f: callable, string
                        optional, default=None. The cost function. This can be a external function or the cost functions
                        predefined for the two types of regression algorithms mentioned. In the latter case, a string
                        can be passed in to signify the choice: "linear" for linear regression; "logistic" for logistic
                        regression.
                    hyp_f: callable, string
                        optional, default=None. The hypothesis function. This can be a external function or the
                        hypothesis functions predefined for the two types of regression algorithms mentioned. In the
                        latter case, a string can be passed in to signify the choice: "linear" for linear regression;
                        "logistic" for logistic regression.
                    grad_f: callable
                        optional, default=None. The gradient function. This can be a external function or the
                        hypothesis functions predefined for the two types of regression algorithms mentioned. In the
                        latter case, nothing need to be specified since these two types of regression have the same
                        symbolic gradient function.

    AnalysisPackage.LinearRegression.flag_ts_init = False
        The flag signifying if the training set is initialized yet.

    AnalysisPackage.LinearRegression.feature_norm = True
        The flag signifying if feature normalized was done during initialization.

    AnalysisPackage.LinearRegression.x = None
        Processed feature array.

    AnalysisPackage.LinearRegression.y = None
        Answer array.

    AnalysisPackage.LinearRegression.theta = None
        Parameter array.

    AnalysisPackage.LinearRegression.x_mean = None
        Feature mean array for feature scaling.

    AnalysisPackage.LinearRegression.x_std = None
        Feature standard deviation for feature scaling.

    AnalysisPackage.LinearRegression.cost_f = None
        Cost function.

    AnalysisPackage.LinearRegression.hyp_f = None
        Hypothesis function.

    AnalysisPackage.LinearRegression.grad_f = None
        Gradient function.

    AnalysisPackage.LinearRegression.init_training_set(x, y, random_init=True)
        This method is used to initialize training set. In the case that feature and answer arrays were passed in during
            the initialization of the whole LinearRegression class, this method is not needed.
        Parameters: x: numpy array
                        optional, default=None. Unprocessed "raw" (usually feature scaling or other processes may be
                        needed) training set feature data. If not passed in, whatever stored in
                        AnalysisPackage.LinearRegression.x will be taken.
                    y: numpy array
                        optional, default=None. Training set answer data. If not passed in, whatever stored in
                        AnalysisPackage.LinearRegression.y will be taken.
                    random_init: bool
                        optional, default=True. Turn on random initialization of parameters.

    AnalysisPackage.LinearRegression.normlx(x, array_mean=None, xstd=None, gen_new=False)
        This method does "feature scaling", which is advised to do in most cases. The feature array x will undergo the
            following process:
                x_std = standard_deviation(x)
                norm = (x - mean(x)) / x_std
        The purpose of this process is to avoid certain features being numerically too large or small than other
            features, and therefore greatly speed up gradient descent in most cases.
        Note: it is assumed that x takes the following dimensionality: x[redundant, examples, features].
            For simple problems, x is treated as 2-D matrix of a collection of features for individual examples. There
            could be additional, i.e. redundant, dimension on top of it, such as different classifiers in multi-class
            classification problems.
        The calculated mean and standard deviation are needed to repeat such process on data sets other than the
            training set. More specifically, the mean and standard deviation arrays produced when normalizing training
            set features are needed to do the same exact normalization on feature set without answers during predication
            using trained parameters.
        Parameters: x: numpy array
                        first argument. Feature array that needs to be normalized.
                    array_mean: numpy array
                        optional, default=None. Mean of the array. If not passed in, and attribute
                        AnalysisPackage.LinearRegression.x_mean is None, i.e. never generated, the method generates
                        array_mean; otherwise, either array_mean or the attribute
                        AnalysisPackage.LinearRegression.x_mean will be taken (array_mean has higher priority).
                    xstd: numpy array
                        optional, default=None. Standard deviation of the array. If not passed in, and attribute
                        AnalysisPackage.LinearRegression.x_std is None, i.e. never generated, the method generates xstd;
                        otherwise, either xstd or the attribute AnalysisPackage.LinearRegression.x_std will be taken
                        (xstd has higher priority).
                    gen_new: bool
                        optional, default=False. Setting this to True will force the method to generate new set of array
                        mean and standard deviation.
        Returns: norm: numpy array
                     first return. The normalized, i.e. feature scaled, feature array
                 array_mean: numpy array
                     second return. Array mean. This will also be stored in attribute
                     AnalysisPackage.LinearRegression.x_mean.
                 xstd: numpy array
                     third return. Array standard deviation. This will also be stored in attribute
                     AnalysisPackage.LinearRegression.x_std.

    AnalysisPackage.LinearRegression.train(max_cycle=int(1e5), alpha=None, convergence=1e-5)
        This method trains the parameters, theta, on a training set, using gradient descent algorithm.
        Parameters: max_cycle:
                        optional, default=int(1e5). Maximum number of iterations before the method is aborted. This is
                        simply a safety measure due to the iterative nature of this method.
                    alpha:
                        optional, default=None. Learning rate, which is a factor determining how far each iteration
                        takes in the direction of descending gradients. If not passed in, a learning rate is estimated.
                        Manually setting up this parameter is highly recommended.
                    convergence:
                        optional, default=1e-5. Convergence criteria. If the change of cost function between current
                        and previous iteration is smaller than this value, the parameter set is deemed "converged".
        Returns: numpy array
                     parameters (theta) array.

    AnalysisPackage.LinearRegression.predict(hyp_f=None, x=None, theta=None, feature_norm=None, array_mean=None,
                                             xstd=None)
        This method uses optimized theta array to predict y values by applying hypothesis function, h.
        Parameters: hyp_f: callable
                        optional, default=None. Hypothesis function. If not passed in, whatever function that was
                        initialized previously will be used.
                    x: numpy array
                        optional, default=None. Feature array. Unless this method is used to check if the training set
                        can be reproduced by the optimized parameter array, the feature array for the actual problem
                        should be passed in.
                    theta: numpy array
                        optional, default=None. Parameters array. If not passed in, whatever was obtained during the
                        previous run of AnalysisPackage.LinearRegression.train() method will be used.
                    feature_norm: bool
                        optional, default=None. Turn on feature normalization. If feature array passed in is already
                        normalized or feature normalization was not done when training the parameters, this needs to be
                        turned off. When this parameter was not specified, the method assumes the same procedure should
                        be done as was done during initialization.
        Returns: numpy array
                     answer array, y. The answers predicted by optimized parameters array based on input feature array.

    Example:
        Here is a "real life" example where we attempt to decide which one out of a few pre-determined structures a
            certain molecular structure is the "most similar" to. The model molecule in this case is methane. There are
            five pre-determined structures with different symmetry: Td, D2d, C2v, C3v, and Cs.
        Step 1: recognize the type of machine learning problem. Since we have pre-determined structures, these can serve
            as training set, i.e. the data set in which we do have correct answers. This is called "supervised machine
            learning". Since we would like to classify a certain structure into 5 different groups. This is multi-class
            classification problem. The easiest algorithm is "one-vs-all" algorithm, in which case for each class we use
            1, i.e. True, to represent that class, and 0, i.e. False, to represent all other classes, resulting in 5
            different classifiers for our 5-class classification problem. The predication would be a value between
            0 and 1, i.e. the probability of one class instead of all other classes, for each class. The class that has
            the highest probability would then be taken as the final classification.
        Step 2: decide the features. Intuitively, we noticed that there are patterns unique to each symmetry among the
            6 H-C-H bond angles and 4 C-H bond lengths. These are the initial features. We have an additional function,
            AnalysisPackage.LinearRegression.gen_feature_TdD2dC2vC3vCs() to map these 10 features to 25 features
            in order to represent those recognized patterns. These are basically standard deviations and/or average
            values among different subgroups of the 6 H-C-H bond angles and 4 C-H bond lengths.
        Step 3: decide the algorithm to optimize the parameters. Here we simply use the easiest to implement: gradient
            descent.
        Step 4: train the parameters, i.e. obtain 5 different sets of parameters corresponding to 5 classifiers.
        Step 5: predict the answers, i.e. predict the probabilities of the structure being one class vs. all other
            classes; then choose the class that has the highest probability.
        Following lines are the code to implement these steps:
            import trjproc
            import numpy as np
            # Training set unprocessed features:
            geom_Td = [109.47]*6+[1.09188]*4
            geom_D2d = [96.4]*4+[141.0]*2+[1.1139]*4
            geom_C2v = [54.6]+[114.2]*4+[125.0]+[1.0782]*2+[1.1745]*2
            geom_C3v = [95.4]*3+[119.1]*3+[1.0829]*3+[1.3758]
            geom_Cs = [91.6]*2+[101.3]*2+[141.0]*2+[1.1012, 1.1123, 1.1123, 1.1250]
            data_raw = np.stack([geom_Td, geom_D2d, geom_C2v, geom_C3v, geom_Cs]) # Stack them to vectorize the run
            # The following is just a 5 by 5 identity matrix, corresponding to five classifiers and five classes.
            y = np.eye(5, dtype=float)
            # Generate the features from bond angles and bond lengths.
            x_rawfeature = trjproc.AnalysisPackage.LinearRegression.gen_feature_TdD2dC2vC3vCs(data_raw)
            x = np.array([x_rawfeature, ]*5)  # Replicate x_rawfeature five times for the five classifiers
            # Initialize LinearRegression instance. Feature scaling and random parameter initialization are done by
            # default and the results are automatically stored in respective attributes.
            lr = trjproc.AnalysisPackage.LinearRegression(x=x, y=y, cost_f="logistic",
                                                          hyp_f="logistic")
            # Now train the parameters. The choice of alpha (learning rate) may need some tweaking before a satisfying
            # convergence is achieved.
            theta = lr.train(alpha=0.01)
            # Load some test data (this example is the bond angles and bond lengths along a BOMD simulation trajectory).
            test_data = np.loadtxt("time_angle_bonds_allpt_trj0.txt", skiprows=1)
            # Generate the features from bond angles and bond lengths similarly.
            test_rawfeature = lr.gen_feature_TdD2dC2vC3vCs(test_data)
            # Predict the answers
            prediction_test = lr.predict(x=np.array([test_rawfeature, ] * 5), theta=theta)
            # Take the class with the highest probability among the five classifiers as the correct answer: 1 for Td,
            # 2 for D2d, 3 for C2v, 4 for C3v, 5 for Cs, in the same order as was set up initially.
            answers = np.argmax(prediction_test, axis=0)+1


"""
    def __init__(self, dir_in=None, dir_out=None, ref_energy=None, energy_unit="Hartree"):
        import os
        if dir_in is None:
            self.dir_in = os.getcwd()
        else:
            self.dir_in = dir_in
        if dir_out is None:
            self.dir_out = os.getcwd()
        else:
            self.dir_out = dir_out
        if '/' in os.getcwd():
            slash = '/'
        elif '\\' in os.getcwd():
            slash = '\\'
        else:
            raise Exception("Error in finding directory separator.")
        if ref_energy is None:
            self.ref_energy = None
        else:
            self.ref_energy = ref_energy
        self.e_unit = energy_unit
        self.slash = slash
        self.dir_in = self.dir_in.rstrip(slash)
        self.dir_out = self.dir_out.rstrip(slash)

    def unpack_data(self, data, data_name):
        if data is None:
            if data_name is None:
                raise Exception("Either data or data_name has to be passed in when unpacking.")
            else:
                data_raw = np.load(self.dir_in + self.slash + data_name)
        else:
            data_raw = data
        if type(data_raw) is np.lib.npyio.NpzFile:
            data_proc = [data_raw["arr_" + str(i)] for i in range(len(data_raw.files))]
        elif type(data_raw) is list:
            data_proc = data_raw
        else:
            raise Exception("Wrong type of data when unpacking!")
        return data_proc

    def calc_bond(self, select, xyz_raw=None, xyz_name=None, xyz_unit="Bohr", bond_unit="Angstrom", save=False):
        xyz_uni = self.unpack_data(xyz_raw, xyz_name)
        select_np = np.array(select)
        if select_np.shape[-1] != 2:
            raise Exception("select array in calc_bond must have length 2 for the last dimension.")
        bond_list = list()
        for xyz in xyz_uni:
            if len(select_np.shape) > 1:
                v = np.stack([xyz[..., i[1], :] - xyz[..., i[0], :] for i in select_np], axis=-2)
            else:
                v = xyz[..., select_np[1], :] - xyz[..., select_np[0], :]
            bond = np.sqrt(np.sum(np.square(v), axis=-1))
            if xyz_unit == bond_unit:
                bond_list.append(bond)
            elif xyz_unit == "Bohr" and bond_unit == "Angstrom":
                bond_list.append(bond*0.52918)
            elif xyz_unit == "Angstrom" and bond_unit == "Bohr":
                bond_list.append(bond * 1.8897161646320724)
            else:
                raise Exception("Unrecognised xyz or bond unit!")
        if save:
            np.save(self.dir_out + self.slash + "bond.npz", *bond_list)
        return bond_list

    @staticmethod
    def unit_vector(vector):
        return vector / np.expand_dims(np.sqrt(np.sum(np.square(vector), axis=-1)), axis=-1)

    @staticmethod
    def angle_between(v1, v2):
        v1_u = AnalysisPackage.unit_vector(v1)
        v2_u = AnalysisPackage.unit_vector(v2)
        angle = np.arccos(np.sum(v1_u * v2_u, axis=-1))
        mask_isnan = np.isnan(angle)
        mask_v1eqv2 = np.prod(v1_u == v2_u, axis=-1)
        mask_else = np.ones_like(mask_v1eqv2, dtype=int) - mask_v1eqv2
        angle_corr = np.nan_to_num(angle) + mask_isnan * mask_else * np.pi
        return angle_corr

    @staticmethod
    def angle_signed(v1, v2, n):  # n needs to have either number of dimension of 1 or same as v1 and v2
        if type(n) is list:
            n = np.array(n)
        if type(v1) is list:
            v1 = np.array(v1)
        if type(v2) is list:
            v2 = np.array(v2)
        angle = AnalysisPackage.angle_between(v1, v2)
        if v1.shape[-1] == 2 and v2.shape[-1] == 2:
            mask_sign = np.cross(v1, v2) < 0
        elif v1.shape[-1] == 3 and v2.shape[-1] == 3:
            if len(n.shape) < len(v1.shape) and len(n.shape) == 1:
                n_expand = UtilityPackage.expand_multi_dims(v1.shape, n, -1)
            else:
                n_expand = n
            mask_sign = np.sum(n_expand * np.cross(v1, v2), axis=-1) < 0
        else:
            print('Size of last dimension of v1 or v2 not equal to 2 or 3')
            return None
        return angle * (np.ones_like(mask_sign) - mask_sign + mask_sign * -1)

    def calc_angle(self, select, xyz_raw=None, xyz_name=None, v_normal=None, signed=True, unit="degree", save=False):
        xyz_uni = self.unpack_data(xyz_raw, xyz_name)
        select_np = np.array(select)
        if select_np.shape[-1] != 3:
            raise Exception("select array in calc_angle must have length 3 for the last dimension.")
        # Calculating the two vector arrays based on select array
        angle_list = list()
        for xyz in xyz_uni:
            if len(select_np.shape) > 1:
                v1 = np.stack([xyz[..., i[0], :]-xyz[..., i[1], :] for i in select_np], axis=-2)
                v2 = np.stack([xyz[..., i[2], :] - xyz[..., i[1], :] for i in select_np], axis=-2)
            else:
                v1 = xyz[..., select_np[0], :]-xyz[..., select_np[1], :]
                v2 = xyz[..., select_np[2], :]-xyz[..., select_np[1], :]
            if signed:
                if v_normal is None:
                    n = np.array([0, 0, 1])
                else:
                    n = v_normal
                angle = self.angle_signed(v1, v2, n)
            else:
                angle = self.angle_between(v1, v2)
            if unit == "degree":
                angle_list.append(angle * 180 / np.pi)
            elif unit == "radian":
                angle_list.append(angle)
            else:
                raise Exception("Unrecognised angle unit!")
        if save:
            np.save(self.dir_out+self.slash+"angle.npz", *angle_list)
        return angle_list

    @staticmethod
    def rotation_matrix_given_axis(axis_vector, angle2rotate):
        """The formula for this method is taken from Wikipedia at:
        https://en.wikipedia.org/wiki/Rotation_matrix#cite_note-5
        Which was original referenced at https://dspace.lboro.ac.uk/dspace-jspui/handle/2134/18050
        The function of this method is to generate a 3-dimensional rotationa matrix with a given rotation axis.
        The input can be multi-dimensional, but the last axis needs to be (x,y,z)."""
        if type(axis_vector) is list:
            axis_vector = np.array(axis_vector)
        if len(np.shape(axis_vector)) - len(np.shape(angle2rotate)) == 1:  # No need to broadcast
            angle = np.array(angle2rotate)
            axisv = axis_vector
        elif len(np.shape(axis_vector)) > 1:  # Case of multi-dimensional input axis_vector
            axisv = axis_vector
            if len(np.shape(angle2rotate)) == 0:  # Need to prepare for broadcasting
                angle = UtilityPackage.expand_multi_dims(np.shape(axis_vector)[:-1], angle2rotate, -1)
            else:
                raise Exception("Cannot match the shape of axis_vector with angle2rotate due to ambiguity!")
        elif len(np.shape(angle2rotate)) > 0:  # Case of multi-dimensional input angle2rotate
            angle = np.array(angle2rotate)
            if len(np.shape(axis_vector)) == 1:  # Need to prepare for broadcasting
                axisv = UtilityPackage.expand_multi_dims(np.shape(angle2rotate)+(3, ), axis_vector, -1)
            else:
                raise Exception("Cannot match the shape of axis_vector with angle2rotate due to ambiguity!")
        else:
            raise Exception("Cannot match the shape of axis_vector with angle2rotate!")
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x = axisv[..., 0]
        y = axisv[..., 1]
        z = axisv[..., 2]
        # The matrix elements
        r1 = cos_angle+np.square(x)*(1-cos_angle)
        r2 = x*y*(1-cos_angle)-z*sin_angle
        r3 = x*z*(1-cos_angle)+y*sin_angle
        r4 = y*x*(1-cos_angle)+z*sin_angle
        r5 = cos_angle+np.square(y)*(1-cos_angle)
        r6 = y*z*(1-cos_angle)-x*sin_angle
        r7 = z*x*(1-cos_angle)-y*sin_angle
        r8 = z*y*(1-cos_angle)+x*sin_angle
        r9 = cos_angle+np.square(z)*(1-cos_angle)
        r_list = [r1, r2, r3, r4, r5, r6, r7, r8, r9]
        # Stack then reshape the matrix
        r = np.reshape(np.stack(r_list, axis=-1), tuple(np.maximum(np.shape(axisv)[:-1], np.shape(angle)))+(3, 3))
        return r

    @staticmethod
    def rotation_matrix_target_vector(vector1, vector2):
        """This method uses the above method to generate a rotation matrix. The axis of ration is calculated by
        vector1 cross vector2, and the angle to rotate is the signed angle from vector1 to vector2. The result of such
        rotation should make vector1 coincide with vector2."""
        v1xv2 = np.cross(vector1, vector2)
        u = v1xv2 * (1 / np.linalg.norm(v1xv2))
        angle = AnalysisPackage.angle_signed(vector1, vector2, u)
        return AnalysisPackage.rotation_matrix_given_axis(u, angle)

    class LinearRegression(object):
        def __init__(self, x=None, y=None, random_init=True, feature_norm=True, cost_f=None, hyp_f=None,
                     grad_f=None):
            self.flag_ts_init = False
            self.x = x
            self.x_mean = None
            self.x_std = None
            self.feature_norm = feature_norm
            if feature_norm:
                self.x, self.x_mean, self.x_std = self.normlx(x)
            self.y = y
            self.m = None
            self.n = None
            self.theta = None
            if x is not None and y is not None:
                self.init_training_set(random_init=random_init)
            self.cost_f = None
            self.hyp_f = None
            self.grad_f = None
            self.init_cost_f(cost_f)
            self.init_hyp_f(hyp_f)
            self.init_grad_f(grad_f)

        def init_training_set(self, x=None, y=None, random_init=True):
            def theta_init(target_shape, random=True):
                if random:
                    return np.random.rand(*target_shape)
                else:
                    return np.zeros(target_shape)

            # Insert zeroth order terms
            if x is not None:
                self.x = x
            if y is not None:
                self.y = y
            x0 = np.ones(self.x.shape[:-1])
            self.x = np.concatenate((x0[..., np.newaxis], self.x), axis=-1)
            self.m = self.y.shape[-1]
            self.n = self.x.shape[-1]
            self.theta = theta_init(self.y.shape[:-1]+(self.n, ), random=random_init)
            if not self.self_chk():
                raise Exception("Failed self check!")
            self.flag_ts_init = True

        def init_cost_f(self, cost_f):
            if cost_f is None:  # Assign default cost function to linear type
                self.cost_f = self.costf_linear
            elif type(cost_f) is str:
                if cost_f.lower() == "linear":
                    self.cost_f = self.costf_linear
                elif cost_f.lower() == "logistic":
                    self.cost_f = self.costf_logistic
            else:
                self.cost_f = cost_f

        def init_hyp_f(self, hyp_f):
            if hyp_f is None:  # Assign default hypothesis function to linear type
                self.hyp_f = self.hypf_linear
            elif type(hyp_f) is str:
                if hyp_f.lower() == "linear":
                    self.hyp_f = self.hypf_linear
                elif hyp_f.lower() == "logistic":
                    self.hyp_f = self.hypf_logistic
            else:
                self.hyp_f = hyp_f

        def init_grad_f(self, grad_f):
            if grad_f is None:
                self.grad_f = self.gradf_default
            else:
                self.grad_f = grad_f

        @staticmethod
        def costf_linear(h, y):
            # h: hypothesis(linear); y: target results, k*m array, where m is the number of training examples,
            # k is any additional dimensions
            m = y.shape[-1]
            return (0.5/m)*np.sum(np.square(h-y), axis=-1)

        @staticmethod
        def costf_logistic(h, y):
            # h: hypothesis(logistic); y: target results, k*m array, where m is the number of training examples,
            # k is any additional dimensions
            m = y.shape[-1]
            cost_true = -np.log(h)
            cost_false = -np.log(1-h)
            return (1/m)*np.sum(cost_true*(y == 1)+cost_false*(y == 0), axis=-1)

        @staticmethod
        def hypf_linear(x, theta):
            if len(x.shape) > len(theta.shape):
                return np.sum(theta[..., np.newaxis, :]*x, axis=-1)
            elif len(x.shape) == len(theta.shape):
                return np.sum(theta*x, axis=-1)
            else:
                raise Exception("x has less number of dimension than theta in hyp_linear!")

        @staticmethod
        def hypf_logistic(x, theta):
            if len(x.shape) > len(theta.shape):
                z = np.sum(theta[..., np.newaxis, :]*x, axis=-1)
            elif len(x.shape) == len(theta.shape):
                z = np.sum(theta*x, axis=-1)
            else:
                raise Exception("x has less number of dimension than theta in hyp_linear!")
            return 1/(1+np.exp(-z))

        @staticmethod
        def gradf_default(h, y, x):
            return (h-y)[..., np.newaxis]*x

        def normlx(self, x, array_mean=None, xstd=None, gen_new=False):
            """Method to normlize data[..., examples, feature]"""
            if (array_mean is None and self.x_mean is None) or gen_new:
                array_mean = np.mean(x, axis=-2)[..., np.newaxis, :]
            elif array_mean is None:
                array_mean = self.x_mean
            if (xstd is None and self.x_std is None) or gen_new:
                xstd = np.std(x, axis=-2)[..., np.newaxis, :]
            elif xstd is None:
                xstd = self.x_std
            x1 = x - array_mean
            norm = x1 / xstd
            self.x_mean = array_mean
            self.x_std = xstd
            return norm, array_mean, xstd

        def self_chk(self):
            """m is the number of training examples, n is the number of features (including constant term),
             k is any additional dimensions"""
            shape_x = self.x.shape  # x[k,m,n]
            shape_y = self.y.shape  # y[k,m]
            shape_theta = self.theta.shape  # theta[k,n]
            flag = True
            if len(shape_x) < 2:
                print("Dimension of x is smaller than 2")
                flag = False
            if len(shape_y) < 1:
                print("Dimension of y is smaller than 1")
                flag = False
            if shape_x[:-1] != shape_y:
                print("shape mis-match between x and y")
                flag = False
            if shape_x[:-2]+(shape_x[-1], ) != shape_theta:
                print("shape mis-match between x and theta")
                flag = False
            return flag

        def grad_update(self, alpha, x=None, y=None, h=None, theta=None):
            """alpha is the step size for gradient descent update"""
            if h is None:
                h = self.hyp_f(self.x, self.theta)
            if theta is None:
                theta = self.theta
            if x is None:
                x = self.x
            if y is None:
                y = self.y
            m = y.shape[-1]
            return theta-(alpha/m)*np.sum(self.grad_f(h, y, x), axis=-2)

        def train_singular(self, max_cycle, alpha, convergence, x, y, theta):
            j_prev = self.cost_f(self.hyp_f(x, theta), y)
            flag = False
            for i in range(max_cycle):
                h = self.hyp_f(x, theta)
                theta = self.grad_update(alpha, x=x, y=y, h=h, theta=theta)
                j = self.cost_f(h, y)
                if np.abs(j-j_prev) < convergence:
                    flag = True
                    break
                j_prev = j
            if flag:
                print("Convergence ("+str(convergence)+") achieved.")
            return theta

        def train(self, max_cycle=int(1e5), alpha=None, convergence=1e-5):
            if not self.flag_ts_init:
                raise Exception("Training set has not been properly initialized yet!")
            if alpha is None:
                # Attempting to estimate learning rate alpha
                alpha = (np.amax(self.x)-np.amin(self.x))*convergence
            if len(self.x.shape) > 2:
                x = np.reshape(self.x, (np.prod(self.x.shape[:-2]), )+self.x.shape[-2:])
                y = np.reshape(self.y, (np.prod(self.y.shape[:-1]), )+(self.y.shape[-1], ))
                theta = np.reshape(self.theta, (np.prod(self.theta.shape[:-1]), )+(self.theta.shape[-1], ))
                theta_new = np.array([self.train_singular(max_cycle, alpha, convergence,
                                                          x[i, ...], y[i, :], theta[i, :])
                                      for i in range(y.shape[0])])
                self.theta = np.reshape(theta_new, self.theta.shape)
            else:
                self.theta = self.train_singular(max_cycle, alpha, convergence, self.x, self.y, self.theta)
            return self.theta

        def predict(self, hyp_f=None, x=None, theta=None, feature_norm=None, array_mean=None, xstd=None):
            if x is None:
                x = self.x
            else:
                if feature_norm is None:
                    feature_norm = self.feature_norm
                if feature_norm:
                    x, array_mean, xstd = self.normlx(x, array_mean=array_mean, xstd=xstd)
                x0 = np.ones(x.shape[:-1])
                x = np.concatenate((x0[..., np.newaxis], x), axis=-1)
            if theta is None:
                theta = self.theta
            if self.hyp_f is None:
                self.init_hyp_f(hyp_f)
            return self.hyp_f(x, theta)

        @staticmethod
        def gen_feature_TdD2dC2vC3vCs(raw_x):
            if raw_x.shape[-1] != 10:
                raise Exception("Last axis of x is not len=10")
            x = np.concatenate((np.sort(raw_x[..., :6]), np.sort(raw_x[..., 6:])), axis=-1)
            # Td
            a1 = np.std(x[..., :6], axis=-1)
            b1 = np.std(x[..., 6:], axis=-1)
            # D2d
            a2 = np.std(x[..., :4], axis=-1)
            a3 = np.std(x[..., 4:6], axis=-1)
            a4 = np.abs(np.mean(x[..., :4], axis=-1)-np.mean(x[..., 4:6], axis=-1))
            # C2v
            a5 = np.std(x[..., 1:5], axis=-1)
            a6 = np.abs(np.mean(x[..., 1:5], axis=-1)-x[..., 0])
            a7 = np.abs(np.mean(x[..., 1:5], axis=-1) - x[..., 5])
            b2 = np.std(x[..., 6:8], axis=-1)
            b3 = np.std(x[..., 8:], axis=-1)
            b4 = np.abs(np.mean(x[..., 6:8], axis=-1)-np.mean(x[..., 8:], axis=-1))
            # C3v
            a8 = np.std(x[..., :3], axis=-1)
            a9 = np.std(x[..., 3:6], axis=-1)
            a10 = np.abs(np.mean(x[..., :3], axis=-1) - np.mean(x[..., 3:6], axis=-1))
            b5 = np.std(x[..., 6:9], axis=-1)
            b6 = np.abs(np.mean(x[..., 6:9], axis=-1)-x[..., -1])
            # Cs
            a11 = np.std(x[..., :2], axis=-1)
            a12 = np.std(x[..., 2:4], axis=-1)
            a13 = np.std(x[..., 4:6], axis=-1)
            a14 = np.abs(np.mean(x[..., :2], axis=-1) - np.mean(x[..., 2:4], axis=-1))
            a15 = np.abs(np.mean(x[..., 2:4], axis=-1) - np.mean(x[..., 4:6], axis=-1))
            a16 = np.abs(np.mean(x[..., :2], axis=-1) - np.mean(x[..., 4:6], axis=-1))
            b7 = np.std(x[..., 7:9], axis=-1)
            b8 = np.abs(np.mean(x[..., 7:9], axis=-1)-x[..., 6])
            b9 = np.abs(np.mean(x[..., 7:9], axis=-1)-x[..., -1])
            return np.stack([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, b1, b2, b3, b4, b5,
                             b6, b7, b8, b9], axis=-1)

        @staticmethod
        def str2class_TdD2dC2vC3vCs(str_in):
            output = np.zeros((5, len(str_in)))
            for i, i_str in enumerate(str_in):
                if i_str == "Td":
                    output[0, i] = 1
                if i_str == "D2d":
                    output[1, i] = 1
                if i_str == "C2v":
                    output[2, i] = 1
                if i_str == "C3v":
                    output[3, i] = 1
                if i_str == "Cs":
                    output[4, i] = 1
            return output

        @staticmethod
        def class2str_TdD2dC2vC3vCs(class_in):
            def v2str(value):
                if value == 0:
                    return "Td"
                elif value == 1:
                    return "D2d"
                elif value == 2:
                    return "C2v"
                elif value == 3:
                    return "C3v"
                elif value == 4:
                    return "Cs"
                else:
                    raise Exception("Invalid value in v2str()")
            max_prob = np.argmax(class_in, axis=0)
            return list(map(v2str, max_prob)), class_in[max_prob, range(class_in.shape[-1])]

    def cwt(self, data=None, data_name=None, diss=None, diss_name="diss.npy", axis_time=0, data_reg=False,
            threshold_reg=1000, msg_print=True, save=False):
        """This method calculates continuous wavelet transformation (CWT) for trajectory data based on dissociation
            identification array. For general usage of CWT, simply use the wavelets module directly.
        The wavelets module is credited to https://github.com/aaren/wavelets by Aaron O'Leary."""
        import datetime
        import gc
        try:
            import wavelets
        except:
            raise Exception("Unable to import wavelet analysis library!")
        dir_in = self.dir_in
        slash = self.slash
        if data is None:
            if data_name is None:
                raise Exception("Either data or data_name has to be passed in for cwt() method.")
            else:
                data_raw = np.load(dir_in + slash + data_name)
        else:
            if data_name is None:
                data_name = "default_name"
            data_raw = data
        if type(data_raw) is np.lib.npyio.NpzFile:
            data_proc = [data_raw["arr_" + str(i)] for i in range(len(data_raw.files))]
            n_pt_max = np.max([data_raw["arr_" + str(i)].shape[axis_time] for i in range(len(data_raw.files))])
        elif type(data_raw) is list:
            data_proc = data_raw
            n_pt_max = np.max([data_raw[i].shape[axis_time] for i in range(len(data_raw))])
        else:
            raise Exception("Wrong type of data!")
        if msg_print:
            print("Max #points: ", n_pt_max)
        if axis_time != 0:
            data = self.npz_unilen([np.rollaxis(data_proc[i], axis_time)
                                    for i in range(len(data_proc))], n_pt_max, masked=False)
        else:
            data = self.npz_unilen(data_proc, n_pt_max, masked=False)
        if diss is None:
            if diss_name is None:
                raise Exception("Either diss_array or diss_name has to be passed in for cwt() method.")
            else:
                diss = np.load(dir_in + slash + diss_name)
        if data_reg:
            print("before", diss)
            abnormal_list = UtilityPackage.regulate_data(data_proc, threshold_reg, axis_time=axis_time)
            if len(abnormal_list) > 0:
                if msg_print:
                    print("The following trajectories has abnormal data: ", abnormal_list)
                    idx_aray=np.array([np.arange(diss.shape[1])]*diss.shape[0])
                    diss[np.where(np.sum(np.stack([idx_aray == i for i in abnormal_list]), axis=0))] = -1
            print("after", diss)
        n_trans = np.prod(data.shape[:1] + data.shape[2:])
        scales = None
        if n_trans > 100:
            power_allch_list = list()
            i = 0
            for data_piece in np.array_split(data, np.ceil(n_trans / 100)):
                if msg_print:
                    print('Current time at loop' + str(i + 1) + ': ', datetime.datetime.time(datetime.datetime.now()))
                wa_piece = wavelets.WaveletAnalysis(data_piece - np.mean(data_piece, axis=1)[:, np.newaxis, ...],
                                                    wavelet=wavelets.Morlet(w0=12), axis=1, unbias=True)
                # The scale dimension from CWT will be inserted at the first dimension
                power_allch_list.append(wa_piece.wavelet_power)
                if i == 0:
                    scales = wa_piece.scales
                gc.collect()
                i += 1
            power_allch = np.concatenate(power_allch_list, axis=1)
            power_perch = [power_allch[:, diss[i] != -1, ...] for i in list(range(len(diss)))]
            print([tuple(range(len(item.shape)))[3:] for item in power_perch])
            power_sum = np.array([np.mean(np.sum(item, axis=tuple(range(len(item.shape)))[3:]), axis=1)
                                  for item in power_perch])
        else:
            wa_allch = wavelets.WaveletAnalysis(data - np.mean(data, axis=1)[:, np.newaxis, ...],
                                                wavelet=wavelets.Morlet(w0=12), axis=1, unbias=True)
            power_allch = wa_allch.wavelet_power
            scales = wa_allch.scales
            power_perch = [power_allch[:, diss[i] != -1, ...]
                           for i in list(range(len(diss)))]
            power_sum = np.array([np.mean(np.sum(item, axis=tuple(range(len(item.shape)))[3:]), axis=1)
                                  for item in power_perch])
        if save:
            data_name_out = data_name.split(".")[0]
            np.save(self.dir_out + self.slash + "wavelet_scales.npy", scales)
            np.savez(self.dir_out + self.slash + "wavelet_power_each_channel_" + data_name_out + ".npz", *power_perch)
            np.save(self.dir_out + self.slash + "wavelet_power_sum_" + data_name_out + ".npy", power_sum)
            np.savetxt(self.dir_out + self.slash + "wavelet_scales.txt", scales, fmt="%.10e")
            for i in list(range(len(power_sum))):
                np.savetxt(self.dir_out + self.slash + "wavelet_power_sum_" + data_name_out + "_ch" + str(i + 1)
                           + ".txt", power_sum[i, ...], fmt="%.10e")
        return scales, power_perch, power_sum

    def save_cwt(self, data_name, power_perch, power_sum):
        dir_out = self.dir_out
        slash = self.slash
        data_name = data_name.split(".")[0]
        np.savez(dir_out + slash + "wavelet_power_each_channel_" + data_name + ".npz", *power_perch)
        np.save(dir_out + slash + "wavelet_power_sum_" + data_name + ".npy", power_sum)
        for i in list(range(len(power_sum))):
            np.savetxt(dir_out + slash + "wavelet_power_sum_" + data_name + "_ch" + str(i + 1) + ".txt",
                       power_sum[i, ...], fmt="%.10e")

    def bisector_angle_HCH(self, angle, xyz_raw=None, xyz_name=None):
        """This method calculates the angle between two vectors which are bisectors of H-C-H angles, assuming one of
        these two angles is the minimum angle at that point."""
        xyz = self.unpack_data(xyz_raw, xyz_name)
        imin_angle = [np.argmin(iangle, axis=-1) for iangle in angle]
        icomp_angle = [-1 - i for i in imin_angle]
        comb_array = np.array(list(itertools.combinations([1, 2, 3, 4], 2)))
        idx_array = [(comb_array[i], comb_array[j]) for i, j in zip(imin_angle, icomp_angle)]
        v_angle = list()
        for idx, ixyz in zip(idx_array, xyz):
            i, j = idx
            npts = len(i)
            pt1 = ixyz[np.arange(npts), i[:, 0]]
            pt2 = ixyz[np.arange(npts), i[:, 1]]
            pt3 = ixyz[np.arange(npts), j[:, 0]]
            pt4 = ixyz[np.arange(npts), j[:, 1]]
            pt_c = ixyz[:, 0, :]  # XYZ of Carbon atoms
            v1 = 0.5 * (pt1 + pt2) - pt_c
            v2 = 0.5 * (pt3 + pt4) - pt_c
            v_angle.append(self.angle_between(v1, v2) * 180 / np.pi)
        return v_angle

    def diss2csv(self, diss=None, name=None, diss_time=True, time_step=0.25):
        import csv
        if diss is None:
            if name is None:
                raise Exception("Either diss or name needs to be passed in diss2csv()")
            diss = np.load(self.dir_in + self.slash + name + ".npy")
        diss_mat = [np.where(i != -1)[0] for i in diss]
        overlap_comb = list(itertools.combinations(range(len(diss_mat)), 2))
        overlap_mat = [np.intersect1d(diss_mat[comb[0]], diss_mat[comb[1]], assume_unique=True)
                       for comb in overlap_comb]
        combined_mat = diss_mat + overlap_mat
        n_diss = np.array([len(i) for i in combined_mat], dtype=int)
        combined_mat_filled = np.zeros((len(combined_mat), max(n_diss)), dtype=np.dtype('U8'))
        for i in range(len(combined_mat)):
            combined_mat_filled[i, :n_diss[i]] = np.array(combined_mat[i], dtype=np.dtype('U8'))
        with open(self.dir_out + self.slash + name + ".csv", "w") as fh:
            writer = csv.writer(fh)
            writer.writerow([""]+["Channel " + str(i+1) for i in range(len(diss_mat))] +
                            ["Channel " + str(i+1) + " and " + str(j+1) for i, j in overlap_comb])
            writer.writerow(np.insert(np.array(n_diss, dtype=np.dtype('U8')), 0, "Total"))
            writer.writerow(np.insert(np.array(n_diss[:len(diss_mat)]/len(np.unique([j for i in diss_mat for j in i])),
                                               dtype=np.dtype('U10')), 0, "Percentage"))
            combined_mat_str = np.vstack((np.array(["Entry No."+str(i)
                                                    for i in range(combined_mat_filled.shape[1])],
                                                   dtype=np.dtype('U16')), combined_mat_filled))
            writer.writerows(np.transpose(combined_mat_str))
        if diss_time:
            time_diss = np.array([np.mean(j) for j in [diss[i, diss_mat[i]] for i in range(len(diss_mat))]],
                                 dtype=float) * time_step
            time_diss_filled = np.zeros((len(diss_mat), max(n_diss)), dtype=np.dtype('U8'))
            for i in range(len(diss_mat)):
                time_diss_filled[i, :n_diss[i]] = np.array(diss[i, diss_mat[i]] * time_step, dtype=np.dtype('U8'))
            with open(self.dir_out + self.slash + name + "_time.csv", "w") as fh:
                writer = csv.writer(fh)
                writer.writerow([""] + ["Channel " + str(i + 1) for i in range(len(diss_mat))] + ["All Trj."])
                writer.writerow(np.append(np.insert(np.array(time_diss, dtype=np.dtype('U8')), 0, "Total"),
                                          np.mean([j for i in diss_mat for j in i]) * time_step))
                time_diss_str = np.vstack((np.array(["Entry No." + str(i)
                                                     for i in range(time_diss_filled.shape[1])],
                                                    dtype=np.dtype('U16')), time_diss_filled))
                writer.writerows(np.transpose(time_diss_str))

    def etot2csv(self, etot=None, name=None, pinpoint=None, avg=False, ref_energy=None, unit_out="kcal/mol"):
        import csv
        if etot is None:
            if name is None:
                raise Exception("Either etot or name needs to be passed in etot2csv()")
            etot = np.load(self.dir_in + self.slash + name + ".npz")
        if type(etot) is np.lib.npyio.NpzFile:
            etot_pkg = [etot["arr_"+str(i)] for i in range(len(etot.files))]
        elif type(etot) is list:
            etot_pkg = etot
        else:
            raise Exception("Wrong type of etot!")
        if avg:
            if pinpoint is None:
                etot_proc = [[np.mean(i) for i in etot_pkg]]
            else:
                etot_proc = [[np.mean(etot_trj[pinpoint_trj:]) for pinpoint_trj, etot_trj in zip(channel, etot_pkg)
                              if pinpoint_trj != -1] for channel in pinpoint]
        else:
            if pinpoint is None:
                raise Exception("pinpoint array required when not picking up etot data by averaging over all points.")
            else:
                etot_proc = [np.array([e_trj[pin] for e_trj, pin in zip(etot_pkg, channel) if pin != -1]) for channel in pinpoint]
        if unit_out != self.e_unit:
            if unit_out == "kcal/mol":
                if self.e_unit == "Hartree":
                    convert = 627.503
                else:
                    raise Exception("Unrecognized energy unit passed in.")
            else:
                if self.e_unit == "kcal/mol":
                    convert = 0.00159362
                else:
                    raise Exception("Unrecognized energy unit passed in.")
        else:
            convert = 1.0
        if ref_energy is None:
            if self.ref_energy is None:
                ref_energy = 0.0
            else:
                ref_energy = self.ref_energy
        etot_avg = [np.mean(np.array(etot_trj) - ref_energy) * convert for etot_trj in etot_proc]
        etot_std = [np.std(np.array(etot_trj) - ref_energy) * convert for etot_trj in etot_proc]
        etot_all = (np.array([j for i in etot_proc for j in i]) - ref_energy) * convert
        n_trj_max = np.max([len(i) for i in etot_proc])
        etot_char = np.zeros((len(etot_proc), n_trj_max), dtype="U16")
        for channel_i, channel_j in zip(etot_char, etot_proc):
            channel_i[:len(channel_j)] = [u"{:.2f}".format((i - ref_energy) * convert) for i in channel_j]
        with open(self.dir_out + self.slash + name + ".csv", "w") as fh:
            writer = csv.writer(fh)
            writer.writerow([""] + ["Channel " + str(i + 1) for i in range(len(etot_proc))] + ["All Trj."])
            writer.writerow(["Total"] + [u"{:.2f}\u00B1{:.2f}".format(avg, std) for avg, std
                                         in zip(etot_avg, etot_std)] + [u"{:.2f}\u00B1{:.2f}".format(np.mean(etot_all),
                                                                                                     np.std(etot_all))])
            etot_str = np.vstack((np.array(["Entry No." + str(i) for i in range(n_trj_max)], dtype=np.dtype('U16')),
                                  etot_char))
            writer.writerows(np.transpose(etot_str))


class WorkFlow(FragGen):
    """This work flow class is designed for working in a single set of data, from reading various types of data to
    select trajectories based on Mulliken charge test, total energy test and dissociation test, to numerous analysis
    procedures."""
    def __init__(self, dir_in, dir_out):
        import os
        FragGen.__init__(self)
        self.dir_in = dir_in
        self.dir_out = dir_out
        if '/' in os.getcwd():
            slash = '/'
        elif '\\' in os.getcwd():
            slash = '\\'
        else:
            raise Exception("Error in finding directory separator.")
        self.slash = slash
        self.dir_in = self.dir_in.rstrip(slash)
        self.dir_out = self.dir_out.rstrip(slash)

    def read_continuous(self, n_log, data_type=None, ext=".log", save=True):
        # Reading from trajectory log files that has file names in a continuous form such as: 1.log, 2.log, 3.log, ...,
        #  n_log.log
        reader = MassRead(n_log=n_log, ext=ext, input_dir=self.dir_in, output_dir=self.dir_out)
        if data_type is not None:
            shorhand_dict = {value: key for key, value in reader.data_name_short}
            if type(data_type) is str:
                if data_type in reader.data_type_list:
                    reader.data_type_list = [data_type]
                elif data_type in shorhand_dict:
                    reader.data_type_list = [shorhand_dict[data_type]]
                else:
                    raise Exception("data_type is str, but could not find available type to read in MassRead.")
            elif type(data_type) is list:
                type_list = list()
                for idata in data_type:
                    if idata in reader.data_type_list:
                        type_list.append(idata)
                    elif idata in shorhand_dict:
                        type_list.append(shorhand_dict[idata])
                if len(type_list) == 0:
                    raise Exception("Could not find any available type to read in MassRead.")
                reader.data_type_list = type_list
            else:
                raise Exception("Wrong type for data_type was passed in.")
        data = reader.read()
        if save:
            reader.save(data)
        return data

    def select_no_charge_test(self, read_then_select=True, etot_name="Etot.npz", cts_name="xyz.npz", rule_str=None,
                              frag_gen=False, rule_identifier="Channel", break_factor=1.5, remain_factor=1.5,
                              rule_delim=";", rule_precision=2, cts_unit="Bohr", value_unit="Angstrom",
                              recombination=True, save=False):
        # Select trajectories without the Mulliken charge test, which is to test unphysically fast charge fluctuation
        # during laser pulse. Obviously this method is for trajectories without laser field.
        if read_then_select:
            dir_in = self.dir_out
        else:
            dir_in = self.dir_in
        etot = np.load(dir_in + self.slash + etot_name)
        if rule_str is None:
            cts = None
        else:
            cts = np.load(dir_in + self.slash + cts_name)
        whitelist = TrjScreen.screen_bundle(etot, n_cutoff=1)
        fh_flag = False
        if type(rule_str) is str:
            # The following happens when rule_str is rule definition itself.
            if ">" in rule_str or "<" in rule_str or "=" in rule_str:
                pass
            elif self.slash in rule_str:  # This happens when rule_str is a full file directory
                rule_str = open(rule_str, "r")
                fh_flag = True
            else:
                rule_str = open(self.dir_in + self.slash + rule_str, "r")
                fh_flag = True
        if frag_gen:
            if rule_str is None:
                rules = None
            else:
                rules = rule_str
        else:
            rules = rule_str
        if rules is not None:
            if frag_gen:
                detector = DissociationDetect(cts, rules, frag_gen=True, frag_identifier=rule_identifier,
                                              frag_break_factor=break_factor, frag_remain_factor=remain_factor,
                                              frag_delim=rule_delim, frag_precision=rule_precision, cts_unit=cts_unit,
                                              value_unit=value_unit, whitelist=whitelist)
            else:
                detector = DissociationDetect(cts, rules, cts_unit=cts_unit, value_unit=value_unit, whitelist=whitelist)
            diss = detector.dissociation_detect(recombination=recombination)
            pinpoint = detector.pinpoint_gen(whitelist, length_laser=0, diss=diss)
        else:
            diss = None
            pinpoint = None
        if save:
            np.save(self.dir_out + self.slash + "whitelist.npy", whitelist)
            if rules is not None:
                np.save(self.dir_out + self.slash + "diss.npy", diss)
                np.save(self.dir_out + self.slash + "pinpoint.npy", pinpoint)
        if fh_flag:
            rule_str.close()
        return whitelist, diss, pinpoint

    def select_with_charge_test(self, read_then_select=True, etot_name="Etot.npz", mlk_name="MlkC.npz",
                                mlk_test_name="MlkCtest.npy", whitelist_name="whitelist.npy",
                                pinpoint_name="pinpoint.npy", diss_name="diss.npy", reload=True, n_cutoff=374,
                                charge_threshold=0.9, n_threshold=94,threshold=0.01, pre_thresh=100, cts_name="xyz.npz",
                                rule_str=None, frag_gen=False,rule_identifier="Channel", break_factor=1.5,
                                remain_factor=1.5, rule_delim=";",rule_precision=2, cts_unit="Bohr",
                                value_unit="Angstrom", recombination=True,save=False):
        # Select trajectories with the Mulliken charge test, which is to test unphysically fast charge fluctuation
        # during laser pulse. Obviously this method is for trajectories with laser field. The following parameters are
        # specifically for this charge test:
        # n_cutoff: the number of time steps the entire laser spans;
        # n_threshold: the number of time steps one laser cycle spans;
        # charge_threshold: charge scanning from -charge_threshold to +charge_threshold.
        if read_then_select:
            dir_in = self.dir_out
        else:
            dir_in = self.dir_in
        etot = np.load(dir_in + self.slash + etot_name)
        if os.path.isfile(dir_in + self.slash + mlk_test_name) and reload:
            mlk_test = np.load(dir_in + self.slash + mlk_test_name)
        else:
            mlk = np.load(dir_in + self.slash + mlk_name)
            mlk_test = TrjScreen.mlk_test(mlk, n_cutoff=n_cutoff, charge_threshold=charge_threshold,
                                          n_threshold=n_threshold)
            if type(mlk_test) is np.ma.core.MaskedArray:
                mlk_test = np.array(mlk_test, dtype=bool)
        if rule_str is None:
            cts = None
        else:
            cts = np.load(dir_in + self.slash + cts_name)
        if os.path.isfile(dir_in + self.slash + whitelist_name) and reload:
            whitelist = np.load(dir_in + self.slash + whitelist_name)
        else:
            whitelist = TrjScreen.screen_bundle(etot, mlk_test_results=mlk_test, n_cutoff=n_cutoff, threshold=threshold,
                                            pre_thresh=pre_thresh)
        fh_flag = False
        if type(rule_str) is str:
            # The following happens when rule_str is rule definition itself.
            if ">" in rule_str or "<" in rule_str or "=" in rule_str:
                pass
            elif self.slash in rule_str:  # This happens when rule_str is a full file directory
                rule_str = open(rule_str, "r")
                fh_flag = True
            else:
                rule_str = open(self.dir_in + self.slash + rule_str, "r")
                fh_flag = True
        if frag_gen:
            if rule_str is None:
                rules = None
            else:
                rules = rule_str
        else:
            rules = rule_str
        if os.path.isfile(dir_in + self.slash + diss_name) and reload:
            diss = np.load(dir_in + self.slash + diss_name)
            if os.path.isfile(dir_in + self.slash + pinpoint_name) and reload:
                pinpoint = np.load(dir_in + self.slash + pinpoint_name)
            elif rules is not None:
                if frag_gen:
                    detector = DissociationDetect(cts, rules, frag_gen=True, frag_identifier=rule_identifier,
                                                  frag_break_factor=break_factor, frag_remain_factor=remain_factor,
                                                  frag_delim=rule_delim, frag_precision=rule_precision,
                                                  cts_unit=cts_unit,
                                                  value_unit=value_unit, whitelist=whitelist)
                else:
                    detector = DissociationDetect(cts, rules, cts_unit=cts_unit, value_unit=value_unit,
                                                  whitelist=whitelist)
                pinpoint = detector.pinpoint_gen(whitelist, length_laser=n_cutoff, diss=diss)
            else:
                pinpoint = None
        elif rules is not None:
            if frag_gen:
                detector = DissociationDetect(cts, rules, frag_gen=True, frag_identifier=rule_identifier,
                                              frag_break_factor=break_factor, frag_remain_factor=remain_factor,
                                              frag_delim=rule_delim, frag_precision=rule_precision, cts_unit=cts_unit,
                                              value_unit=value_unit, whitelist=whitelist)
            else:
                detector = DissociationDetect(cts, rules, cts_unit=cts_unit, value_unit=value_unit, whitelist=whitelist)
            diss = detector.dissociation_detect(recombination=recombination)
            if os.path.isfile(dir_in + self.slash + pinpoint_name) and reload:
                pinpoint = np.load(dir_in + self.slash + pinpoint_name)
            else:
                pinpoint = detector.pinpoint_gen(whitelist, length_laser=n_cutoff, diss=diss)
        else:
            diss = None
            pinpoint = None
        if save:
            np.save(self.dir_out + self.slash + mlk_test_name, mlk_test)
            np.save(self.dir_out + self.slash + "whitelist.npy", whitelist)
            if rules is not None:
                np.save(self.dir_out + self.slash + "diss.npy", diss)
                np.save(self.dir_out + self.slash + "pinpoint.npy", pinpoint)
        if fh_flag:
            rule_str.close()
        return mlk_test, whitelist, diss, pinpoint


class GetDir(object):
    """Get directory from routine_list"""
    def __init__(self, routine_list):
        with open(routine_list,"r") as fh:
            self.str = fh.readlines()

    def rout_dir(self):
        return [line.split(";")[2] for line in self.str]
#
# input_par = "/Volumes/XSHI/BOMD/OUTPUT/CH4+_FldFree_with0ptE"
# output_par2 = "/Users/xuetaoshi/Documents/CH4+_1/CH4+_FldFree_with0ptE"
# output_par = "/Users/xuetaoshi/Documents/test_read"
# output_par3 = "/Users/xuetaoshi/Documents/ClCHO+_singleL1/ClCHO+_Lin0xFiStr0p03WL10p5Cos16C"
# wf = WorkFlow(output_par3, output_par)
# diss_csv = wf.diss2csv(name="diss")
# fg = FragGen()
# fg.comb_gen(["H","H","C"])
# line = "C 1 1.3 3 2.3 4 2.1"
# lines_mol = """C
# C
# H
# H
# H
# H
# H 1 1.2 3 3.4
# H 4 2.5
# C 1 2.2"""
# lines_frag1 = """H
# C 2 1.0
# H"""
# lines_frag2 = """H
# H"""
# lines_frag3 = """C"""
# lines_frag = """H 2 1.0
# # H """

# lines_mol = """C
# H 1 1.0
# H 1 1.0
# C 1 1.8
# H 1 1.0
# H 4 1.0
# H 4 1.0
# H 4 1.0"""
# lines_frag1 = """H
# H 1 1.1 3 1.1
# H 1 1.1"""
# # rule_str=fg.frag_gen(lines_mol.split("\n"),[lines_frag1.split("\n"),lines_frag2.split("\n"),lines_frag3.split("\n")])
# lines = """C
# H 1 1.0
# H 1 1.0
# C 1 1.8
# H 1 1.0
# H 4 1.0
# H 4 1.0
# H 4 1.0
#
# Channel 1:
# H
# H 1 1.1 3 1.1
# H 1 1.1
#
# Channel 2:
# C
# H 1 1.5
# H 1 1.5
# H 1 1.5
#
# """
# # rule_str=fg.frag_gen(lines_mol.split("\n"),[lines_frag1.split("\n")])
# rule_str=fg.rule_gen(lines)
# rule_list = DissociationDetect.str2logic_multiple(rule_str)
# print("rule_list", rule_list)
# print(rule_str)


# print([len(line.split("or")) for line in rule_str.splitlines()])
# print(len([print(i) for i in rule_str.split("or")]))
# print(fg.com_gen(lines_raw.split("\n")))
# print(fg.line_parser(line))
# mlk = np.load(output_par3+"/"+"MlkC.npz")
# # mlk_test = TrjScreen.mlk_test(mlk,n_cutoff=2242,n_threshold=141)
# mlk_test = np.load(output_par3+"/"+"mlk_test_results.npy")
# # np.save(output_par3+"/"+"mlk_test_results.npy",mlk_test.filled(False))
# etot = np.load(output_par3+"/"+"etot.npz")
# whitelist = TrjScreen.screen_bundle(etot, mlk_test_results=mlk_test, n_cutoff=2242)
# # [print(i) for i in whitelist]
# cts = np.load(output_par3+'/'+"xyz.npz")
# rule="""1-2 > 2.2 ; 1-4 < 3.36 or 1-2 < 2.2 ; 1-4 > 3.36
# 1-2 > 2.2 ; 1-4 > 3.36 ; 2-4 < 2.66
# 1-2 > 2.2 ; 1-4 > 3.36 ; 2-4 > 2.66
# 1-2 < 2.2 ; 1-4 < 3.36"""
# # rule="1-2 < 2.2 ; 1-4 < 3.36"
# dd = DissociationDetect(cts,rule,whitelist=whitelist)
# # dd = DissociationDetect(cts, rule, )
# diss = dd.dissociation_detect()
# print(diss)
# pp = dd.pinpoint_gen(whitelist, length_laser=2242)
# print(pp)
# np.save("pinpoint.npy",pp)
# diss=[DissociationDetect(cts['arr_'+str(i)], rule,
#                          atom_list=['C', 'H', 'O', 'Cl']).dissociation_detect(whitelist=whitelist[i])
#       for i in list(range(100))]
# print(diss)
# dir_list = GetDir("/Users/xuetaoshi/Documents/routine_list_CH4+_3copy.txt").rout_dir()
# diss_avg_list = list()
# dissl1=list()
# dissl2=list()
# for dir in dir_list:
#     diss = np.load("/Users/xuetaoshi/Documents/CH4+_3/" + dir + "/diss.npy")
#     diss_mask = np.ma.masked_where(diss == -1, diss)
#     diss_avg=np.mean(diss_mask,axis=1)
#     diss_nmb = np.sum(diss_mask.filled(0)>0, axis=1)
#     diss1 = np.sum((diss_avg * diss_nmb)[:4])/np.sum(diss_nmb[:4])
#     diss2 = np.sum((diss_avg * diss_nmb)[4:10]) / np.sum(diss_nmb[4:10])
#     diss_avg_list.append(np.sum((diss_avg * diss_nmb)[:10])/np.sum(diss_nmb[:10]))
#     print(diss_nmb,diss1,diss2,np.sum((diss_avg * diss_nmb)[:10])/np.sum(diss_nmb[:10]))
#     dissl1.append(diss1)
#     dissl2.append(diss2)
# np.savetxt("test_diss1.csv",np.ma.masked_array(dissl1),fmt="%.1f",delimiter=",")
# np.savetxt("test_diss2.csv",np.ma.masked_array(dissl2),fmt="%.1f",delimiter=",")
# np.savetxt("test_diss.csv",np.ma.masked_array(list(zip(dissl1,dissl2,diss_avg_list))),fmt="%.1f",delimiter=",")
#
#np.savetxt("test_avg.csv",np.ma.masked_array(diss_avg_list),fmt="%.1f",delimiter=",")
# mlk = np.load(output_par3+"/"+"MlkC.npz")
# mlktest = TrjScreen(n_cutoff=2242,n_threshold=141)
# result = mlktest.test(mlk)
# print(np.where(result))
# mr = MassRead(n_log=100, input_dir=input_par, output_dir=output_par)
# for data in mr.data_type_list:
#     re1=np.load(output_par+"/"+mr.data_name_short[data]+".npz")
#     re2=np.load(output_par2+"/"+mr.data_name_short[data]+".npz")
#     diff=np.sum(np.array([np.sum(re1["arr_"+str(i)]-re2["arr_"+str(i)]) for i in list(range(100))]))
#     print(data,diff)
# print('Before read Current time: ',datetime.datetime.time(datetime.datetime.now()))
# data = mr.read()
# mr.save(data)
# print('After save Current time: ',datetime.datetime.time(datetime.datetime.now()))
# fh_test=open("test.txt","r")
# mass1=MassRead(log_list=fh_test,ext=".txt")
# fh_test.close()
# print(mass1.log_list)
# trim = ArrayTrim(file1)
# trim.set_extra_trim(-10)
# print("max points: ",trim.max_pts)
# arr1=np.arange(100)
# print("trim1: ",trim.trim(arr1,"Dipole Moment").shape)
# print("before: ",arr1[:10],trim.trim(arr1,"Dipole Moment")[:10])
# print("trim2: ",trim.trim(arr1,"Time").shape)
# print("before: ",arr1[:10],trim.trim(arr1,"Time")[:10])
#
# arr2=np.arange(trim.max_pts+2)
# print("trim1: ",trim.trim(arr2,"Dipole Moment").shape)
# print("before: ",arr2[:10],trim.trim(arr2,"Dipole Moment")[:10],arr2[-10:],trim.trim(arr2,"Dipole Moment")[-10:])
# print("trim2: ",trim.trim(arr2,"Time").shape)
# print("before: ",arr2[:10],trim.trim(arr2,"Time")[:10],arr2[-10:],trim.trim(arr2,"Dipole Moment")[-10:])
# # di1=read.findall(file1,"Dipole Moment")
# di2=read.findall(file2,"Dipole Moment")
# print(di1.shape)
# print("di1",di1[:10])
# print(di2.shape)
# print("di2",di2[:10])
# #
# # Command line options defined here
# arg_parser = argparse.ArgumentParser(description='i/o options')
# arg_parser.add_argument("-n", "--name_list", type=str, help="name of the name list files to extract",
#                         default='name_list.txt')
# arg_parser.add_argument("-i", "--input", type=str, help="Input file path", default="/Users/xuetaoshi/Documents")
# arg_parser.add_argument("-o", "--output", type=str, help="Output directory", default=os.getcwd())
#
# args = arg_parser.parse_args()
# # End of command line options
# with open(args.input + '/' + args.name_list, "r") as file_str_in:
#     for line in file_str_in:
#         if '.inp' in line:
#             fname = line.strip().replace('.inp', '.log')
#         elif '.' in line:
#             fname = line.strip()
#         else:
#             fname = line.strip() + '.log'
#         with open(args.input + '/' + fname) as file_str_log:
#             file_read = FileRead(file_str_log, ["Electric Field Alt"])
#             arr = file_read.findall_list()[0]
#             print(arr.shape)
#             np.savetxt(args.output + '/' + fname.replace('.log', '.txt').replace('_', ''), arr, fmt="%f")
#
