from pyrs.core import workspaces
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode
from pyrs.core.instrument_geometry import HidraSetup
from pyrs.dataobjects.constants import HidraConstants
from pyrs.peaks.peak_collection import PeakCollection
import os
import numpy as np
import h5py
import pytest


def test_rw_raw():
    """Test read a project to workspace and write in the scope of raw data

    Returns
    -------

    """
    raw_project_name = os.path.join(os.getcwd(), 'data/HZB_Raw_Project.h5')

    # Read to workspace
    source_project = HidraProjectFile(raw_project_name, 'r')

    # To the workspace
    source_workspace = workspaces.HidraWorkspace('Source HZB')
    source_workspace.load_hidra_project(source_project, load_raw_counts=True,
                                        load_reduced_diffraction=False)

    # Export
    target_project = HidraProjectFile('HZB_HiDra_Test.h5', 'w')
    # Experiment data
    source_workspace.save_experimental_data(target_project, sub_runs=range(1, 41))

    # Instrument
    detector_setup = source_workspace.get_instrument_setup()
    instrument_setup = HidraSetup(detector_setup=detector_setup)
    target_project.write_instrument_geometry(instrument_setup)

    # Save
    target_project.save(True)

    return


@pytest.mark.parametrize('source_file, hidra_file, peak_tag',
                         [('tests/data/BD_Data_Log.hdf5', 'HB2B_Test_BD.h5', 'BD'),
                          ('tests/data/LD_Data_Log.hdf5', 'HB2B_Test_LD.h5', 'LD'),
                          ('tests/data/ND_Data_Log.hdf5', 'HB2B_Test_ND.h5', 'ND')],
                         ids=['TestBD', 'TestLD', 'TestND'])
def test_peaks_read_write(source_file, hidra_file, peak_tag):
    """Test to create a Hidra project file containing peaks parameters

    Files:
        data/BD_Data_Log.hdf5   data/LD_Data_Log.hdf5   data/ND_Data_Log.hdf5

    Parameters
    ----------
    source_file
    hidra_file
    peak_tag : str
        peak tag

    Returns
    -------

    """
    # write sample logs
    hidra_ws = import_nrsf2_data(source_file)

    # Create Hidra project file
    hidra_peak_file = HidraProjectFile(hidra_file, HidraProjectFileMode.OVERWRITE)

    hidra_ws.save_experimental_data(hidra_file, None, True)

    # write peaks
    peaks = PeakCollection(peak_tag=peak_tag, peak_profile='PseudoVoigt', background_type='Linear')

    hidra_peak_file.write_peak_fit_result(peaks)

    # close
    hidra_peak_file.close()

    # Verify
    verify_peak_file = HidraProjectFile(hidra_peak_file, HidraProjectFileMode.READONLY)

    # check peaks
    peak_tags = verify_peak_file.read_peak_tags()
    assert peak_tags[0] == '', 'Peak tag shall be {} but not {}'.format(peak_tag, peak_tags[0])

    saved_peaks = verify_peak_file.read_peak_parameters(peak_tag='')

    return


def import_nrsf2_data(data_h5_name):
    """Import an NRSF2 data and create an HidraWorkspace

    Parameters
    ----------
    data_h5_name : str
        h5 name

    Returns
    -------
    pyrs.core.workspaces.HidraWorkspace

    """
    # Read h5
    data_h5 = h5py.File(data_h5_name, 'r')
    main_h5_entry = data_h5['Diffraction Data']

    # Create HidraWorkspace
    hydra_ws = workspaces.HidraWorkspace(os.path.basename(data_h5).split('.')[0])

    # Go over for sub runs
    sub_run_list = list()
    for key_name in list(main_h5_entry.keys()):
        if key_name.startswith('Log'):
            sub_run = int(key_name.split()[-1])
            sub_run_list.append(sub_run)
    sub_runs = np.array(sorted(sub_run_list))

    # log values in interest
    log_names = ['time', 'Wavelength', 'sx', 'sy', 'sz', 'vx', 'vy', 'vz']
    log_dict = dict()
    for log_name in log_names:
        log_dict[log_name] = np.ndarray(shape=sub_runs.shape, dtype='float')

    # read sample logs
    for i_sb in range(len(sub_runs)):
        log_i_entry = main_h5_entry['Log {}'.format(sub_runs[i_sb])]
        for log_name in log_names:
            log_value_i = log_i_entry[log_name].value
            log_dict[log_name][i_sb] = log_value_i
    # END-FOR

    # add sample logs
    for log_name in log_names:
        hydra_ws.set_sample_log(log_name, sub_runs, log_dict[log_name])

    # Close
    data_h5.close()

    return hydra_ws


if __name__ == '__main__':
    pytest.main([__file__])
