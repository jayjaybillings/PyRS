import numpy as np

from pyrs.interface.peak_fitting.config import LIST_AXIS_TO_PLOT
from pyrs.interface.peak_fitting.config import fit_dict as FIT_DICT


class DataRetriever:

    def __init__(self, parent=None):
        self.parent = parent
        self.hidra_workspace = self.parent.hidra_workspace

    def get_data(self, name='Sub-runs', peak_index=0):

        if name == 'Sub-runs':
            return [np.array(self.hidra_workspace.get_sub_runs()), None]

        if name in LIST_AXIS_TO_PLOT['raw'].keys():
            return [self.hidra_workspace._sample_logs[name], None]

        if name == 'd-spacing':
            peak_collection = self.parent.fit_result.peakcollections[peak_index]
            _d_reference = np.float(str(self.parent.ui.peak_range_table.item(peak_index, 3).text()))
            peak_collection.set_d_reference(values=_d_reference)
            values, error = peak_collection.get_dspacing_center()
            return [values, error]

        if name == 'microstrain':
            peak_collection = self.parent.fit_result.peakcollections[peak_index]
            values, error = peak_collection.get_strain()
            return [values*1e6, error*1e6]

        if name in LIST_AXIS_TO_PLOT['fit'].keys():
            return self.get_fitted_value(peak=self.parent.fit_result.peakcollections[peak_index],
                                         value_to_display=name)

    def get_fitted_value(self, peak=None, value_to_display='Center'):
        """
        return the values and errors of the fitted parameters of the given peak
        :param peak:
        :param value_to_display:
        :return:
        """
        value, error = peak.get_effective_params()

        mantid_value_to_display = FIT_DICT[value_to_display]
        value_selected = value[mantid_value_to_display]
        error_selected = error[mantid_value_to_display]
        return value_selected, error_selected
