#!/usr/bin/python
""""
Convert the "old" HB2B data, now used to test peak fitting, to new HidraProject format

*How to run*
1. Add PyRS path to python path (refer to pyrsdev.sh)
1. Run this script

"""
import numpy
from pyrs.utilities import rs_scan_io
from pyrs.projectfile import HidraProjectFile


def main():
    """
    Main method to convert diffraction data in the old HDF5 format
    :return:
    """
    # Load source data in 'old' format
    source_h5 = 'tests/testdata/16-1_TD.cor_Log.h5'
    reader = rs_scan_io.DiffractionDataFile()
    diff_data_dict, sample_logs = reader.load_rs_file(source_h5)

    # Create a Hidra project
    target_project_file_name = 'tests/testdata/Hydra_16-1_cor_log.h5'
    target_file = HidraProjectFile(target_project_file_name, 'w')

    # Create sub runs
    target_file.write_sub_runs(sorted(diff_data_dict.keys()))

    # Add (reduced) diffraction data
    two_theta_vector = None
    diff_data_matrix = None

    # construct the matrix of intensities
    for sub_run_index, sub_run_number in enumerate(sorted(diff_data_dict.keys())):
        two_theta_vector_i, intensity_vector_i = diff_data_dict[sub_run_index]

        # create data set
        if two_theta_vector is None:
            two_theta_vector = two_theta_vector_i
            diff_data_matrix = numpy.ndarray(shape=(len(diff_data_dict.keys()), intensity_vector_i.shape[0]),
                                             dtype='float')
        # END-IF

        # set vector
        diff_data_matrix[sub_run_index] = intensity_vector_i
    # END-FOR

    # Add data
    target_file.write_reduced_diffraction_data_set(two_theta_vector, {None: diff_data_matrix})

    # Save
    target_file.save(verbose=True)

    return


if __name__ == '__main__':
    main()
