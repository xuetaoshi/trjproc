# trjproc
Extract and analyze data from Gaussian trajectory calculations.
The following classes are included:
  class UtilityPackage(object): 
    Utility function package.
  class FileRead(object):
    Read Gaussian trajectory log file based on regex patterns and then do proper processing.
  class ArrayTrim(object):
    Trim the arrays read by FileRead according to how each type of data was printed out in Gaussian log files. This
    is necessary due to Gaussian trajectory log file routinely print out additional information at the start and/or the
    end of the whole calculation in a similar fashion as it would be for the actual time steps.
  class MassRead(FileRead, ArrayTrim):
    Read multiple types of data from multiple log files and store them into numpy npz files after being trimmed
    according to how such type of data is normally printed out in Gaussian log files.
  class TrjScreen(UtilityPackage):
    This is a bundle of tests: first test on Mulliken charges to detect very sudden fluctuations (mostly unphysical)
    during laser pulse; second test on total energy to detect very sudden jumps during and after laser pulse, such
    occurrence usually signifies an abrupt change of electronic state, which should be excluded from future
    calculations.
  class FragGen(object):
    Generate rule string based on fragmentation for dissociation detection.
  class DissociationDetect(UtilityPackage, FragGen):
    This is a dissociation detection module based on distance matrix.
  class AnalysisPackage(UtilityPackage):
    This is a package of numerous analysis methods.
