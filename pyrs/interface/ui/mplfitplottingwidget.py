# This is a special widget for plotting fit result.
# It consists of 2 plots (top one for experimental data and model, bottom one for residual, aka difference)
# plus a tool bar

from pyrs.interface.ui.mplgraphicsview1d import MyNavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout
from mantidqt.MPLwidgets import FigureCanvasQTAgg as FigureCanvas
from pyrs.interface.ui.mplconstants import MplBasicColors


class MplFitPlottingWidget(QWidget):
    """ Specially implemented widget for showing data against fitted model along with residual (difference)
    """

    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(MplFitPlottingWidget, self).__init__(parent)
        self.parent = parent

        # set up UI and widgets
        self._myCanvas = QtMplFitCanvas(self)
        self._myToolBar = MyNavigationToolbar(self, self._myCanvas)

        # set up layout
        self._vBox = QVBoxLayout(self)
        self._vBox.addWidget(self._myCanvas)
        self._vBox.addWidget(self._myToolBar)

        # class variables
        self._data_line_list = list()   # allow multiple
        self._model_line = None
        self._residual_line = None

        # class variables
        self._curr_color_index = 0
        self._curr_x_label = None

        self.list_peak_ranges = []
        self.list_peak_ranges_matplotlib_id = []
        self.list_fit_peak_labels = []
        self.list_peak_labels_matplotlib_id = []
        self.list_fit_peak_d0 = []

        self._working_with_range_index = -1
        self._peak_label_index = 0
        self._button_pressed = False
        self._left_line = None
        self._right_line = None
        self._myCanvas.mpl_connect('button_press_event', self.button_clicked)
        self._myCanvas.mpl_connect('button_release_event', self.button_released)
        self._myCanvas.mpl_connect('motion_notify_event', self.mouse_moved)

    def any_toolbar_button_clicked(self):
        if self._myToolBar.NAVIGATION_MODE_ZOOM == self._myToolBar._myMode:
            return True

        if self._myToolBar.NAVIGATION_MODE_PAN == self._myToolBar._myMode:
            return True

        return False

    def button_clicked(self, event):
        if (self.any_toolbar_button_clicked()):
            return

        self._button_pressed = True
        self._add_initial_point(x=event.xdata)
        self.parent.update_peak_ranges_table(click=True,
                                             list_fit_peak_labels=self.list_fit_peak_labels,
                                             list_fit_peak_ranges=self.list_peak_ranges,
                                             list_fit_peak_d0=self.list_fit_peak_d0,
                                             list_fit_peak_ranges_matplotlib_id=self.list_peak_ranges_matplotlib_id,
                                             list_fit_peak_labels_matplotlib_id=self.list_peak_labels_matplotlib_id)

    def button_released(self, event):
        if not self._button_pressed:
            return

        self._button_pressed = False
        self._validate_second_point(x=event.xdata)
        self.parent.update_peak_ranges_table(release=True,
                                             list_fit_peak_labels=self.list_fit_peak_labels,
                                             list_fit_peak_ranges=self.list_peak_ranges,
                                             list_fit_peak_d0=self.list_fit_peak_d0,
                                             list_fit_peak_ranges_matplotlib_id=self.list_peak_ranges_matplotlib_id,
                                             list_fit_peak_labels_matplotlib_id=self.list_peak_labels_matplotlib_id)
        self.parent.update_save_peak_range_widget()

    def mouse_moved(self, event):
        if self._button_pressed:
            self._change_second_point(x=event.xdata)
        self.parent.update_peak_ranges_table(move=True,
                                             list_fit_peak_labels=self.list_fit_peak_labels,
                                             list_fit_peak_ranges=self.list_peak_ranges,
                                             list_fit_peak_d0=self.list_fit_peak_d0,
                                             list_fit_peak_ranges_matplotlib_id=self.list_peak_ranges_matplotlib_id,
                                             list_fit_peak_labels_matplotlib_id=self.list_peak_labels_matplotlib_id)

    def _add_initial_point(self, x=np.NaN):

        if not self.list_peak_ranges:
            self.list_peak_ranges = [[x, np.NaN]]
            self.list_fit_peak_labels = ['Peak0']
            self.list_fit_peak_d0 = [1]
            self._peak_label_index += 1
        else:
            _was_part_of_one_range = False
            for _index, _range in enumerate(self.list_peak_ranges):
                if (x >= np.min(_range)) and (x <= np.max(_range)):
                    self.list_peak_ranges[_index] = [x, np.NaN]
                    _was_part_of_one_range = True
                    self._working_with_range_index = _index
                    break

            if _was_part_of_one_range is False:
                self.list_peak_ranges.append([x, np.NaN])
                self.list_fit_peak_labels.append("Peak{}".format(self._peak_label_index))
                self.list_fit_peak_d0.append(1)
                self._working_with_range_index = -1
                self._peak_label_index += 1

        self.plot_data_with_fitting_ranges()

    def _validate_second_point(self, x=np.NaN):

        _working_range = self.list_peak_ranges[self._working_with_range_index]
        if _working_range[0] == x:  # remove this range
            self.list_peak_ranges.remove([_working_range[0], np.NaN])
            [left_peak, right_peak] = self.list_peak_ranges_matplotlib_id[self._working_with_range_index]
            left_peak.remove()
            right_peak.remove()
            self.list_peak_ranges_matplotlib_id.remove([left_peak, right_peak])
            _peak_label = self.list_fit_peak_labels[self._working_with_range_index]
            self.list_fit_peak_labels.remove(_peak_label)
            del self.list_fit_peak_d0[self._working_with_range_index]
        else:
            _working_range = [_working_range[0], x]
            self.list_peak_ranges[self._working_with_range_index] = _working_range
        self.plot_data_with_fitting_ranges()

    def _change_second_point(self, x=np.NaN):
        _working_range = self.list_peak_ranges[self._working_with_range_index]
        self.list_peak_ranges[self._working_with_range_index] = [_working_range[0], x]
        self.plot_data_with_fitting_ranges()

    def _get_next_color(self):
        """
        get the next available color
        :return:
        """
        # color = MplBasicColors[0]

        self._curr_color_index += 1
        if self._curr_color_index >= len(MplBasicColors):
            self._curr_color_index = 0

    def clear_canvas(self):
        """
        clear (aka reset) the canvas
        :return:
        """
        self._myCanvas.reset_plot()

        # reset the class variables managing the plots
        self._data_line_list = list()
        self._model_line = None
        self._residual_line = None

    def evt_toolbar_home(self):
        """
        Handle event triggered by 'home' button in tool bar is pressed
        :return:
        """
        # TODO - 20181125 - Implement!
        # No NavigrationToolbar2.evt_toolbar_home()

        # self._myCanvas.set_x_y_home()
        return

    def evt_view_updated(self):
        """ Handling the event when a 'draw()' is called from tool bar
        :return:
        """
        return

    def evt_zoom_released(self, event):
        """
        Handling the event that is triggered by zoom-button in tool bar is released and a customized signal
        is thus emit.
        Note: 1. axis_pos = event.inaxes.get_position()
                 (axis_pos.x0, axis_pos.x1, axis_pos.y0, axis_pos.y1)
                 (axis_pos.xmin, axis_pos.xmax, axis_pos.ymin, axis_pos.ymax) are the same!
              2. zooming on y won't be synchronized!

        :param event:
        :return:
        """
        event_triggered_axis = event.inaxes
        if event_triggered_axis is None:
            # do it in a nasty way
            upper_x_range = self._myCanvas.get_curr_x_range(False)
            lower_x_range = self._myCanvas.get_curr_x_range(True)
            if lower_x_range[1] - lower_x_range[0] > upper_x_range[1] - upper_x_range[0]:
                # upper is in a smaller range, set lower's xrange
                # upper_y_range = self._myCanvas.get_curr_y_range(False)
                self._myCanvas.set_x_range(upper_x_range[0], upper_x_range[1], is_residual=True)
                # self._myCanvas.set_y_range(upper_y_range[0], upper_y_range[1], is_residual=True)
            else:
                # lower is in a smaller range, set upper's xrange
                # lower_y_range = self._myCanvas.get_curr_y_range(True)
                self._myCanvas.set_x_range(lower_x_range[0], lower_x_range[1], is_residual=False)
                # self._myCanvas.set_y_range(lower_y_range[0], lower_y_range[1], is_residual=False)
        else:
            # a more elegant way to determine
            axis_pos = event.inaxes.get_position()
            if axis_pos.y0 < 0.348:
                # make sure to avoid round error and the event is from lower subplot: set to upper subplots
                lower_x_range = self._myCanvas.get_curr_x_range(True)
                # lower_y_range = self._myCanvas.get_curr_y_range(True)
                self._myCanvas.set_x_range(lower_x_range[0], lower_x_range[1], is_residual=False)
                # self._myCanvas.set_y_range(lower_y_range[0], lower_y_range[1], is_residual=False)
            else:
                # signal from upper subplot: set lower subplot
                upper_x_range = self._myCanvas.get_curr_x_range(False)
                # upper_y_range = self._myCanvas.get_curr_y_range(False)
                self._myCanvas.set_x_range(upper_x_range[0], upper_x_range[1], is_residual=True)
                # self._myCanvas.set_y_range(upper_y_range[0], upper_y_range[1], is_residual=True)

    def get_x_limit(self):
        """
        get X limit
        :return:
        """
        return self._myCanvas.get_curr_x_range()

    def plot_data(self, data_set, color=None, line_label=''):
        """ plot experimental data
        :param data_set:
        :param color:
        :param line_label:
        :return:
        """
        if color is None:
            color = self._get_next_color()

        self._color = color
        self._data_set = data_set
        self._line_label = line_label
        self._diff_data_set = data_set

        data_line_id = self._myCanvas.add_plot_upper_axis(data_set, line_color=color, label=line_label)
        self._data_line_list.append(data_line_id)

    def plot_diff_data(self, data_set, color=None, line_label=''):
        if color is None:
            color = self._get_next_color()

        self._color = color
        self._data_set = data_set
        self._line_label = line_label

        data_line_id = self._myCanvas.add_plot_upper_axis(data_set, line_color=color, label=line_label)
        self._data_line_list.append(data_line_id)

    def plot_data_with_fitting_ranges(self):
        # self.clear_canvas()

        for _peak_label in self.list_peak_labels_matplotlib_id:
            _peak_label.remove()

        for [_left_line, _right_line] in self.list_peak_ranges_matplotlib_id:
            _left_line.remove()
            _right_line.remove()

        color = self._color

        data_set = self._data_set
        line_label = self._line_label

        self._myCanvas.add_plot_upper_axis(data_set, line_color=color, label=line_label)
        self.list_peak_ranges_matplotlib_id = []
        self.list_peak_labels_matplotlib_id = []
        list_peak_labels = self.list_fit_peak_labels

        for _index, _range in enumerate(self.list_peak_ranges):
            x_right = np.nanmax(_range)
            x_left = np.nanmin(_range)
            self._left_line = self._myCanvas._data_subplot.axvline(x_left, color='r', linestyle='--')
            self._right_line = self._myCanvas._data_subplot.axvline(x_right, color='r', linestyle='--')
            self.list_peak_ranges_matplotlib_id.append([self._left_line, self._right_line])
            txt_id = self._myCanvas._data_subplot.text(x_left, 0, list_peak_labels[_index],
                                                       fontsize=16,
                                                       rotation=90,
                                                       rotation_mode='anchor')
            self.list_peak_labels_matplotlib_id.append(txt_id)

        self._myCanvas.draw()

    def plot_data_model(self, data_set, data_label, model_set, model_label, residual_set):
        """
        plot data along with model
        :param data_set:
        :param data_label:
        :param model_set:
        :param model_label:
        :param residual_set:
        :return:
        """
        # clear plot
        self.clear_canvas()

        # plot and set
        data_line_id = self._myCanvas.add_plot_upper_axis(data_set, label=data_label,
                                                          line_color='black', line_marker='+',
                                                          marker_size=4, line_style='--', line_width=1)
        self._data_line_list.append(data_line_id)

        model_line_id = self._myCanvas.add_plot_upper_axis(model_set, label=model_label,
                                                           line_color='red', line_style='-', line_width=1)
        self._model_line = model_line_id

        diff_line_id = self._myCanvas.add_plot_lower_axis(residual_set)
        self._residual_line = diff_line_id

    def reset_color(self):
        """
        reset the auto color index
        :return:
        """
        self._curr_color_index = 0

    def set_x_label(self, new_x_label):
        """
        set unit
        :param new_x_label:
        :return:
        """
        if new_x_label == self._curr_x_label:
            # do nothing
            pass
        else:
            # set
            self._myCanvas.set_x_label(new_x_label)
            self._curr_x_label = new_x_label


# TEST - 20181124 - Make split diffraction view work!
# TEST            - It can be an option to create new MplGraphics1D class as a special widget!
class QtMplFitCanvas(FigureCanvas):
    """ Canvas containing 2 vertical plots and 1 tool bar

    """

    def __init__(self, parent):
        """ initialization for a canvas with 2 sub plots
        :param parent:
        """
        # Instantiating matplotlib Figure. It is a requirement to initialize a figure canvas
        self.fig = Figure()
        self.fig.patch.set_facecolor('white')

        # Initialize parent class and set parent
        super(QtMplFitCanvas, self).__init__(self.fig)
        self.setParent(parent)

        # set 2 axes with fixed ratio
        self._data_subplot = self.fig.add_axes([0.15, 0.35, 0.8, 0.55])  # top
        self._residual_subplot = self.fig.add_axes([0.15, 0.1, 0.8, 0.20])  # bottom

        self._data_subplot.get_shared_x_axes().join(self._data_subplot, self._residual_subplot)

        self._line_index = 0
        self._data_plot_dict = dict()
        self._residual_dict = dict()

        return

    def _setup_legend(self, location='best', font_size=10):
        """Set up legend
        self.axes.legend(): Handler is a Line2D object. Lable maps to the line object
        :param location:
        :param font_size:
        :return:
        """
        allowed_location_list = [
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "right",
            "center left",
            "center right",
            "lower center",
            "upper center",
            "center"]

        # Check legend location valid or not
        if location not in allowed_location_list:
            location = 'best'

        # main axes on subplot
        handles, labels = self._data_subplot.get_legend_handles_labels()
        self._data_subplot.legend(handles, labels, loc=location, fontsize=font_size)

    def add_plot_lower_axis(self, data_set):
        """
        add a plot to the lower axis as residual
        :param data_set:
        :return:
        """
        vec_x = data_set[0]
        vec_y = data_set[1]

        # print ('[DB...BAT] Plot residual:\n{}\n{}'.format(vec_x, vec_y))

        plot_info = self._residual_subplot.plot(vec_x, vec_y, label=None, color='green',
                                                linestyle='-', linewidth=2)
        self._residual_subplot.axhline(2, linestyle='--', color='blue')
        self._residual_subplot.axhline(-2, linestyle='--', color='blue')

        self._data_subplot.set_aspect('auto')

        # Register
        line_id = self._line_index  # share the line ID counter with main axis
        if len(plot_info) == 1:
            self._residual_dict[line_id] = plot_info[0]
            self._line_index += 1
        else:
            # returned tuple has more than 1 element, i.e., API changed
            msg = 'Matplotlib API changed: the return from plot is a %d-tuple: %s.. \n' % (len(plot_info), plot_info)
            for i_r in range(len(plot_info)):
                msg += 'r[%d] = %s\n' % (i_r, str(plot_info[i_r]))
            raise NotImplementedError(msg)

        # Flush/commit
        self.draw()

        return line_id

    def add_plot_upper_axis(self, data_set, label, line_color, line_marker='.', marker_size=4,
                            line_style='-', line_width=1):
        """ add a plot to the upper axis
        :param data_set:
        :param label:
        :param line_color:
        :param line_marker:
        :param marker_size:
        :param line_style:
        :param line_width:
        :return:
        """
        vec_x = data_set[0]
        vec_y = data_set[1]

        plot_info = self._data_subplot.plot(vec_x, vec_y, label=label, color=line_color,
                                            marker=line_marker, markersize=marker_size,
                                            linestyle=line_style, linewidth=line_width)

        self._data_subplot.set_aspect('auto')

        real_max = np.nanmax(vec_y)
        max_value_with_offset = real_max + 0.1*real_max
        self._data_subplot.set_ylim([0, max_value_with_offset])

        # set/update legend
        # self._setup_legend()

        # Register
        line_id = self._line_index  # share the line ID counter with main axis
        if len(plot_info) == 1:
            self._data_plot_dict[line_id] = plot_info[0]
            self._line_index += 1
        else:
            # returned tuple has more than 1 element, i.e., API changed
            msg = 'Matplotlib API changed: the return from plot is a %d-tuple: %s.. \n' % (len(plot_info), plot_info)
            for i_r in range(len(plot_info)):
                msg += 'r[%d] = %s\n' % (i_r, str(plot_info[i_r]))
            raise NotImplementedError(msg)

        # Flush/commit
        self.draw()

        return line_id

    def get_curr_x_range(self, is_residual=False):
        """
        get X range
        :param is_residual: flag whether the returned value is for residual/lower plot
        :return:
        """
        if is_residual:
            x_lim = self._residual_subplot.get_xlim()
        else:
            x_lim = self._data_subplot.get_xlim()

        return x_lim

    def get_curr_y_range(self, is_residual=False):
        """
        get Y range
        :param is_residual: flag whether the returned value is for residual/lower plot
        :return:
        """
        if is_residual:
            y_lim = self._residual_subplot.get_ylim()
        else:
            y_lim = self._data_subplot.get_ylim()

        return y_lim

    def remove_data_lines(self, line_index_list=None):
        """
        remove data line by line index
        :param line_index_list:
        :return:
        """
        # default for all lines
        if line_index_list is None:
            line_index_list = self._data_plot_dict.keys()

        for line_index in line_index_list:
            line_handler = self._data_plot_dict[line_index]
            if line_handler:
                self._data_subplot.lines.remove(line_handler)
            self._data_plot_dict[line_index] = None

    def remove_residual_line(self):
        """
        remove the residual line
        :return:
        """
        # skip if there is no residual line on subplot currently
        if len(self._residual_dict) == 0 or not self._residual_dict:
            return

        if 0 not in self._residual_dict:
            return

        line_handler = self._residual_dict[0]  # this is the broken line
        self._residual_subplot.lines.remove(line_handler)
        self._residual_dict = dict()
        # TODO something is wrong with this code
        '''
        line_handler = self._residual_dict.values()[0]  # this is the broken line
        self._residual_subplot.lines.remove(line_handler)
        self._residual_dict = dict()
        '''

    def reset_plot(self):
        """
        remove all the lines plot on subplots currently
        :return:
        """
        self.remove_data_lines()
        self.remove_residual_line()

    def set_x_range(self, x_min, x_max, is_residual):
        """
        set X range .. x limit to either upper subplot or lower subplot
        :param x_min:
        :param x_max:
        :param is_residual:
        :return:
        """
        if x_min >= x_max:
            raise RuntimeError('Set wrong range to X... min = {} >= max = {}'.format(x_min, x_max))

        if is_residual:
            self._residual_subplot.set_xlim([x_min, x_max])
        else:
            self._data_subplot.set_xlim([x_min, x_max])
        # END-IF

        return

    def set_y_range(self, y_min, y_max, is_residual):
        """
        set Y range .. x limit to either upper subplot or lower subplot
        :param y_min:
        :param y_max:
        :param is_residual:
        :return:
        """
        if y_min >= y_max:
            raise RuntimeError('Set wrong range to Y... min = {} >= max = {}'.format(y_min, y_max))

        if is_residual:
            self._residual_subplot.set_ylim([y_min, y_max])
        else:
            self._data_subplot.set_ylim([y_min, y_max])
        # END-IF

        return
