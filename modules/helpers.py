import os
import json
import math
import time
import functools
import asyncio

import plotly.graph_objects as go
import plotly.express as px
from matplotlib import cm
import matplotlib.pyplot as matplt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter

import sys
from glob import glob

import pandas as pd
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool

from modules import segments as segments_module
import definitions

###########
# Classes #
###########


class Plotter():
    """
    Args:
    """

    def __init__(self, debug=False):
        self.debug = debug

        # If os is windows use matlibplot
        # Plotly does not work well on windows
        if os.name == "nt":
            self.os = "windows"
        else:
            self.os = "linux"

        self.plot_dictionaries = []

    def create_figure_2d(self, figure_name, lock_aspect_ratio=False):
        if self.os == "windows":
            return self.create_figure_2d_windows(figure_name)
        else:
            return self.create_figure_2d_linux(figure_name, lock_aspect_ratio=lock_aspect_ratio)

    def add_pd_series(self, figure, series, series_name):
        if self.os == "windows":
            self.add_pd_series_windows(figure, series, series_name)
        else:
            self.add_pd_series_linux(figure, series, series_name)

    def add_raw_series(self, figure, x_values, y_values, series_name):
        if self.os == "windows":
            self.add_raw_series_windows(
                figure, x_values, y_values, series_name)
        else:
            self.add_raw_series_linux(figure, x_values, y_values, series_name)

    def add_segments(self, figure, segments, series_name, number_of_points=100):
        for segment_index, segment in enumerate(segments):
            self.add_segment(figure, segment, series_name +
                             f"_{segment_index}", number_of_points)

    def add_segment(self, figure, segment, series_name, number_of_points=100):
        x_values = np.linspace(segment.start, segment.end, number_of_points)
        self.add_2d_shape(figure, segment, x_values, series_name)

    def add_2d_shape(self, figure, shape, x_values, series_name):
        y_values = []
        for x_value in x_values:
            y_values.append(shape.y(x_value))

        self.add_raw_series(figure, x_values, y_values, series_name)

    def show(self, figure):
        if self.os == "windows":

            plots = [plot_dictionary["plot"]
                     for plot_dictionary in self.plot_dictionaries]
            labels = [plot_dictionary["label"]
                      for plot_dictionary in self.plot_dictionaries]

            figure.legend(plots, labels)
            figure.show()
        else:
            figure.show()

    #########
    # Linux #
    #########

    def create_figure_2d_linux(self, figure_name, lock_aspect_ratio=False):
        fig = go.Figure()
        fig.update_layout(title=figure_name)

        if lock_aspect_ratio:
            fig.update_layout(
                yaxis=dict(
                    scaleanchor="x",
                    scaleratio=1,
                )
            )

        return fig

    def add_pd_series_linux(self, figure, series, series_name):
        figure.add_trace(go.Scatter(x=series.index.values,
                                    y=series, mode='lines', name=series_name))

    def add_raw_series_linux(self, figure, x_values, y_values, series_name):
        figure.add_trace(go.Scatter(x=x_values, y=y_values,
                                    mode='lines', name=series_name))

    ###########
    # Windows #
    ###########

    def create_figure_2d_windows(self, figure_name):
        self.plot_dictionaries = []
        matplt.title(figure_name)
        return matplt

    def add_pd_series_windows(self, figure, series, series_name):
        plot = figure.plot(series.index.values,
                           series.values, label=series_name)
        self.plot_dictionaries.append({
            "plot": plot,
            "label": series_name
        })

    def add_raw_series_windows(self, figure, x_values, y_values, series_name):
        plot = figure.plot(x_values, y_values, label=series_name)
        self.plot_dictionaries.append({
            "plot": plot,
            "label": series_name
        })

    ###############
    # Quick Plots #
    ###############

    # TODO: Needs to follow multi OS implementation
    def quick_plot_3d(self, dataframe, graph_title="None"):
        fig = go.Figure(data=[go.Surface(z=dataframe.values)])
        fig.update_layout(title=graph_title, autosize=True,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.show()
        # self.figs.append(fig)

    # TODO: Needs to follow multi OS implementation
    def quick_plot_3d_windows(self, dataframe):

        ny, nx = dataframe.shape
        # Create the x-axis
        # Create an array of length nx with values starting from 0 to nx with an even delta between all values
        x = np.linspace(0, nx, nx)
        # Create the y-axis
        y = np.linspace(0, ny, ny)
        # Create a 2D mesh using our x and y axis
        xv, yv = np.meshgrid(x, y)

        fig = matplt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xv, yv, dataframe, cmap=cm.coolwarm)
        ax.set_xlabel('Scanner view window')
        ax.set_ylabel('Scan count')
        ax.set_zlabel('Distance from scanner [mm]')
        matplt.show()

    def quick_plot_2d_pd(self, series, series_name="None", lock_aspect_ratio=False):
        figure = self.create_figure_2d(
            "Quick Figure", lock_aspect_ratio=lock_aspect_ratio)
        self.add_pd_series(figure, series, series_name)
        figure.show()

    def quick_plot_2d_raw(self, x_values, y_values, series_name="None", lock_aspect_ratio=False):
        figure = self.create_figure_2d(
            "Quick Figure", lock_aspect_ratio=lock_aspect_ratio)
        self.add_raw_series(figure, x_values, y_values, series_name)
        figure.show()

    def quick_plot_2d_shape(self, shape, x_values, series_name="None", lock_aspect_ratio=False):
        y_values = []
        for x_value in x_values:
            y_values.append(shape.y(x_value))

        self.quick_plot_2d_raw(
            x_values, y_values, series_name, lock_aspect_ratio=lock_aspect_ratio)

    def quick_plot_2d_segment(self, segment, series_name="None", number_of_points=100, lock_aspect_ratio=False):
        x_values = np.linspace(segment.start, segment.end, number_of_points)
        self.quick_plot_2d_shape(
            segment, x_values, series_name, lock_aspect_ratio=lock_aspect_ratio)

    def quick_plot_2d_segments(self, segments, lock_aspect_ratio=False):
        figure = self.create_figure_2d(
            "Quick Figure", lock_aspect_ratio=lock_aspect_ratio)
        for index, segment in enumerate(segments):

            x_values = np.linspace(segment.start, segment.end, 100)
            y_values = []
            for x_value in x_values:
                y_values.append(segment.y(x_value))

            self.add_raw_series(figure, x_values, y_values, index)

        figure.show()

    ###########
    # Helpers #
    ###########

    def plot_series_set(self, series_set, lock_aspect_ratio=False):

        figure = self.create_figure_2d(
            "Series Set", lock_aspect_ratio=lock_aspect_ratio)

        for series in series_set:
            self.add_pd_series(figure, series["data"], series["name"])

        figure.show()
    
    def plot_segment_end_derivative_differences(self, segments):

        figure = self.create_figure_2d(
        "Segments Derivative Differences")

        number_of_segments = len(segments)
        segment_indices = []
        segment_derivative_differences = []
        for segment_index, segment in enumerate(segments):

            next_segment_index = segment_index + 1
            if next_segment_index < number_of_segments:
                segment_indices.append(f"{segment_index}:{next_segment_index}")

                current_segment = segment
                next_segment = segments[next_segment_index]

                segment_derivative_difference = calculate_segment_end_derivatives_difference(
                    current_segment, next_segment)
                segment_derivative_differences.append(
                    segment_derivative_difference)

        self.add_raw_series(
            figure, segment_indices, segment_derivative_differences, "Segment Differences")
        self.show(figure)

    def plot_segment_generated_circles(self, segment, circle_segments, lock_aspect_ratio=False):

        figure = self.create_figure_2d(
            "Segment Conversion", lock_aspect_ratio=lock_aspect_ratio)

        segment_x_series, segment_y_series, segment_derivative_series = create_segment_debug_series(
            segment)
        plotter_instance.add_raw_series(
            figure, segment_x_series, segment_y_series, "Original Segment")

        for circle_segment_index, circle_segment in enumerate(circle_segments):
            circle_segment_x_series, circle_segment_y_series, circle_segment_derivative_series = create_segment_debug_series(
                circle_segment)
            plotter_instance.add_raw_series(
                figure, circle_segment_x_series, circle_segment_y_series, f"Circle Segment {circle_segment_index}")

        figure.show()

    def plot_segments(self, segments, lock_aspect_ratio=False):
        figure = self.create_figure_2d(
            "Segments", lock_aspect_ratio=lock_aspect_ratio)

        for segment_index, segment in enumerate(segments):
            segment_x_series, segment_y_series, derivative_series = create_segment_debug_series(
                segment)
            plotter_instance.add_raw_series(
                figure, segment_x_series, segment_y_series, f"Segment {segment_index}")

        figure.show()

    def plot_blades_toolpath_parameters(self, blades_toolpath_parameters, initial_blade_instances, lock_aspect_ratio=False):

        # Separate our blades
        blades = {}
        for blade_toolpath_parameters in blades_toolpath_parameters:
            for toolpath_parameter in blade_toolpath_parameters:

                blade_key = toolpath_parameter.blade_instance.z_shape.identifier_string()

                if blade_key not in blades.keys():
                    blades[blade_key] = []

                blades[blade_key].append(toolpath_parameter.blade_instance)

        # Create a plot per blade
        for blade_index, blade_key in enumerate(blades):
            blade_instances = blades[blade_key]
            figure_name = "Top Blade" if blade_index == 1 else "Bottom Blade"
            figure = self.create_figure_2d(
                figure_name, lock_aspect_ratio=lock_aspect_ratio)

            self.add_pd_series(
                figure, initial_blade_instances[blade_index].profile, "Scan")

            for blade_instance_index, blade_instance in enumerate(blade_instances):
                self.add_segments(
                    figure, blade_instance.segments, f"toolpath_{blade_instance_index}")

            self.show(figure)

    def plot_series_generated_segments(self, series, segment_sets, lock_aspect_ratio=False):
        figure = self.create_figure_2d(
            "Series Conversion", lock_aspect_ratio=lock_aspect_ratio)
        self.add_pd_series(figure, series, "Original Segment")

        for segment_set in segment_sets:
            for segment_index, segment in enumerate(segment_set):
                segment_x_series, segment_y_series, segment_derivative_series = create_segment_debug_series(
                    segment)
                self.add_raw_series(
                    figure, segment_x_series, segment_y_series, f"Segment {segment_index}")

        figure.show()


class Timer():

    def __init__(self):

        self.timestamps = [] 
        self.benchmarks = []

    def add_benchmark(self, timestamp):

        benchmark = next((benchmark for benchmark in self.benchmarks if benchmark.function_name == timestamp.function_name), None)
        if benchmark is None:
            benchmark = TimerBenchmark(timestamp.function_name)
            self.benchmarks.append(benchmark)

        benchmark.append_function_call(timestamp.runtime)

        # Sort benchmark keys by runtime
        self.benchmarks = sorted(self.benchmarks, key=lambda x: (x.total_runtime))

    # Add timestamp and add benchmark
    def add_timestamp(self, timestamp):
        self.timestamps.append(timestamp)
        self.add_benchmark(timestamp)

        # Sort timestamps by runtime
        self.timestamps = sorted(self.timestamps, key=lambda x: (x.runtime))[::-1]

    def add_manual_timestamp(self, name):
        timestamp_instance = TimerTimestamp(name, time.time(), 0)
        self.timestamps.append(timestamp_instance)

    def update_manual_timestamp(self, name):
        # find timestamp instance
        for timestamp_instance in self.timestamps:
            # Update runtime
            if timestamp_instance.function_name == name:
                true_finish_time = time.time()
                timestamp_instance.runtime = true_finish_time - timestamp_instance.timestamp 
                timestamp_instance.timestamp = true_finish_time
                break

        # Sort timestamps by runtime
        self.timestamps = sorted(self.timestamps, key=lambda x: (x.runtime))[::-1]

    # Output

    def print_benchmarks(self):
        for benchmarks in self.benchmarks:
            print(str(benchmarks))

    def print_timestamps(self):
        for timestamp in self.timestamps:
            print(str(timestamp))

class TimerTimestamp():

    def __init__(self, function_name, timestamp, runtime):
        self.function_name = function_name
        self.timestamp = timestamp
        self.runtime = runtime

    def __str__(self):
        return f"{self.function_name}: finished at {round(self.timestamp, 10)} and took {round(self.runtime, 10)}s to complete"

class TimerBenchmark():

    def __init__(self, function_name):

        self.function_name = function_name
        self.total_runtime = 0
        self.count = 0

    def append_function_call(self, runtime):
        self.count += 1
        self.total_runtime += runtime

    def __str__(self):
        print(f"{self.function_name}: was called {self.count} times and had a total runtime of {round(self.total_runtime, 10)}")

#############
# Constants #
#############

plotter_instance = Plotter()
timer_instance = Timer()

#############
# Functions #
#############

# Can be used as a decorator to benchmark function
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        runtime = (te - ts) #milliseconds
        timestamp = TimerTimestamp(method.__name__, te, runtime)
        timer_instance.add_timestamp(timestamp)

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f'{method.__name__} ran in {(te - ts)} seconds')
        return result
    return timed

def handle_in_parrallel(args, function, threads):
    pool = ThreadPool(threads)
    results = pool.map(function, args)
    pool.close()
    pool.join()
    return results

# Read config.js file and return json


def read_config(config_directory_path):
    with open(os.path.join(config_directory_path, 'config.json')) as json_file:
        return json.load(json_file)

# Segment off leading and trailing NaNs in a dataslice


def omit_edge_nan_data(dataslice):
    """
    This function trims x axis based-off whether or not there is a NaN in the y-edge domain
    The existence of NaNs in the dataset are intepretated as scan setup non-conformance IRL
    """
    middle_index = int(dataslice.size / 2)
    # find first nonnan and last nonnan indices
    index_start, index_end = find_nan_range_in_slice(
        dataslice, middle_index, middle_index)

    # segment dataframe from the first nonnan to the last nonnan
    modified_dataframe = dataslice.iloc[int(index_start):int(index_end)]

    return modified_dataframe

# TODO: Add a second check to confirm noise via second derivative
# Method can be; "2_diff", "1_diff", or "1_diff_simple"
def omit_series_noise(series, threshold, method):
    local_series = series.copy()

    if method == "1_diff" or method == "1_diff_simple":
        
        local_series_diff = local_series.diff()
        noise_series = local_series_diff.where(abs(local_series_diff) > threshold)
        
        if method == "1_diff":
            for index, value in enumerate(noise_series):
                if np.isnan(value) == False:
                    local_series[index + noise_series.index.start] = np.NaN
        else:
            last_value = None
            for index in local_series.index.values:
                value = local_series[index]
                if last_value is None:
                    last_value = value
                else:
                    value_delta = abs(last_value - value)
                    if value_delta > threshold:
                        local_series[index] = np.NaN
                        last_value = None
                    else:
                        last_value = value

    elif method == "2_diff":
        
        local_series_second_diff = local_series.diff().diff()
        noise_series = local_series_second_diff.where(abs(local_series_second_diff) > threshold)

        for index, value in enumerate(noise_series):
                if np.isnan(value) == False:
                    local_series[index + noise_series.index.start] = np.NaN

    else:
        raise Exception(f"omit_series_noise bad method \"{method}\". Valid options are: \"2_diff\", \"1_diff\", or \"1_diff_simple\"")

    return local_series


def loop_rolling_series_omit_series_noise(series, initial_threshold, cycles, threshold_mutation_factor=0.75, window_width=3, upward_jump_limit=5.0):

    local_series = series.copy()
    current_threshold = initial_threshold
    for cycle in range(cycles):
        local_series = rolling_series_average_omit_series_noise(local_series, current_threshold, window_width=window_width, upward_jump_limit=upward_jump_limit)
        local_series = local_series.interpolate(method="akima")
        current_threshold *= threshold_mutation_factor
    
    return local_series

# Tolerance of 0.2 shift over 0.06666 is sufficient for a 60 degree angle
# Rework this using first derivative
def rolling_series_average_omit_series_noise(series, threshold, window_width=3, upward_jump_limit=5.0, max_miss_count=3):

    local_series = series.copy()

    # Keep track of how many values were outside of tolerance in a row
    miss_count = 0
    current_truth = None
    value_queue = []

    for index in local_series.index.values:

        value = local_series[index]

        # Ignore nan values
        if np.isnan(value):
            continue

        # If no values in value_queue
        # Add current point
        if len(value_queue) < 1:

            value_queue.append(value)
            current_truth = value

        # Check if significant upward shift is detected
        elif (value - current_truth) > upward_jump_limit:

            miss_count = 0
            value_queue = [value]
            current_truth = value

        else:                

            # Check if this value is within tolerance
            if abs(value - current_truth) > threshold:

                local_series[index] = np.NaN

                if miss_count > max_miss_count:
                    miss_count = 0
                    value_queue = [value]
                    current_truth = value
                else:
                    miss_count += 1

            else:
                
                miss_count = 0
                value_queue.append(value)

            if len(value_queue) > window_width:
                value_queue = value_queue[1:]

            current_truth = list_mean(value_queue)

    return local_series

def list_mean(input_list):
    return sum(input_list) / len(input_list)

def trim_to_toe_and_heel_requirements(series, heel_angle=25.0, toe_angle=55.0, window_size=6):

    local_series = series.copy()

    # Split series in half
    split_series = split_series_in_half(local_series)

    for index, series_section in enumerate(split_series):

        target_angle = heel_angle if index == 0 else toe_angle
        target_angle *= -1
        target_angle_radians = target_angle * math.pi/180

        direction = -1 if index == 0 else 1

        smoothed_split_series = apply_savgol_filter(series_section, 81, 2)
        smoothed_split_series = omit_edge_nan_data(smoothed_split_series)

        current_angle_set = []
        current_angle_mean = None

        found_index = None
        for series_index in range(smoothed_split_series.size-1):

            if found_index is not None:
                continue

            true_series_index = smoothed_split_series.index.values[0]
            # Do the search from the center of the series
            # This is done to avoid noise at ends of scan
            if index == 0:
                true_series_index += smoothed_split_series.size - 1 - series_index
            else:
                true_series_index += series_index

            series_index_y_value = smoothed_split_series[true_series_index]

            next_series_index = true_series_index + direction
            next_series_index_y_value = smoothed_split_series[next_series_index]

            # get angle
            x_diff = definitions.SPACING_PER_X_INDEX
            y_diff = next_series_index_y_value - series_index_y_value

            current_angle_radians = math.atan(y_diff/x_diff)

            # Keep track of last window_size angle values
            current_angle_set.append(current_angle_radians)
            if len(current_angle_set) > window_size:
                # Remove first element
                current_angle_set = current_angle_set[1:]

            current_angle_mean = list_mean(current_angle_set)

            # Precision of 0.01 gets us an accuracy around 0.8 degrees
            if current_angle_mean < target_angle_radians:
                # index found
                found_index = true_series_index

        if found_index is None:

            missed_angle = "Heel" if index == 0 else "Toe"
            print(f"{missed_angle} series angle of {target_angle} not found")

        else:
            # trim series
            if index == 0:
                local_series = local_series.loc[found_index:]
            else:
                local_series = local_series.loc[:found_index]

    return local_series
            
def set_nan_sections(series, set_value=np.NaN):
    # Iterate through values in series
    # Look for columns with multiple nans in a row

    local_series = series.copy()

    # Find boundaries to set

    # Cases
    # 1) single nan point
    # 2) multiple consecutive nan points
    # 3) nan section with some single points within
    first_nan_index = None
    nan_boundaries = []
    for index in local_series.index.values:

        value = local_series[index]
        next_index = index + 1

        if next_index in local_series.index.values:

            if np.isnan(value):
                if first_nan_index is None:
                    first_nan_index = index
            else:
                if np.isnan(local_series[next_index]) == False and first_nan_index is not None:
                    nan_boundaries.append([first_nan_index, index - 1])
                    first_nan_index = None

    for boundary in nan_boundaries:
        start_index = boundary[0]
        end_index = boundary[1]

        iterator_range = range(start_index, end_index, 1)
        # for iteration_index, index in enumerate(iterator_range):
        for index in iterator_range:
            local_series[index] = set_value

    return local_series

################
#Find Functions#
################

def find_nan_range_in_slice(data_slice, initial_min, initial_max, index_offset=0.0):
    """ 
    Scans a dataframe slice (1D) and finds the range of indices where 
    we start to have non NaN values. If values found exceed the range of 
    our initial min and max return our new found range instead.

    Arguments:
        data_slice {DataFrame} -- 1D subset of scan data.
        initial_min {int} -- The initial min index. If the min index in this
         slice is less return the newly found index, else the initial index.
        initial_max {int} -- The initial max index. If the max index in this
         slice is greater return the newly found index, else the initial index.
    """

    current_min = initial_min
    current_max = initial_max

    # If no values in row other than NaN then ignore it
    if math.isnan(data_slice.max()):
        return current_min, current_max

    # Find first not NaN value (min)
    first_non_nan = find_first_non_nan(data_slice, index_offset)
    # Find last not NaN value (max)
    last_non_nan = find_last_non_nan(data_slice, index_offset)

    # Compare new and initial indices
    if initial_min > first_non_nan:
        current_min = first_non_nan

    if initial_max < last_non_nan:
        current_max = last_non_nan

    return current_min, current_max


def find_first_non_nan(values, index_offset=0.0):
    for index, value in enumerate(values, start=0):
        if not math.isnan(value):
            return index + index_offset


def find_last_non_nan(values, index_offset=0.0):
    # reverse order of values
    # get value of first_non_nan in reversed list
    first_non_nan = find_first_non_nan(values[::-1], -index_offset)
    # return length of values - first_non_nan
    return len(values) - first_non_nan


def series_derivative(series, start_relative_index, end_relative_index):
    series_start_x = series.index.values[start_relative_index]
    series_start_y = series[series_start_x]
    series_end_x = series.index.values[end_relative_index]
    series_end_y = series[series_end_x]

    delta_x = series_end_x - series_start_x
    delta_y = series_end_y - series_start_y

    derivative = delta_y / delta_x

    return derivative


def calculate_segment_to_series_difference(reference_series, segment):
    # Find absolute difference between generated circles and segment over some numpy range
    difference_accumulator = 0

    # Create evaluation range
    for x_value in reference_series.index.values:
        difference_accumulator += abs(
            reference_series[x_value] - segment.y(x_value))

    return difference_accumulator


def find_best_series_value_at_index(series, index):
    # If the value exists in series return it directly
    # Else interpolate over closest points
    if index in series.index.values:
        value = series[index]
    else:
        # Find closest indices (Own method)
        closest_indices = find_closest_indices(index, series)
        # Apply linear interpolation
        value = linear_interpolate_series(series, closest_indices, index)

    return value


def find_closest_indices(index, series):
    for series_index_index, series_index_value in enumerate(series.index.values):

        assert series_index_index != series.size, f"Could not find index of {index} in series with range of {series.index}!"

        if index >= series_index_value and index <= series.index.values[series_index_index + 1]:
            return [series_index_value, series.index.values[series_index_index + 1]]


def find_closest_index(index, series):

    # Check edge conditions
    if index < series.index.values[0]:
        return series.index.values[0]
    elif index > series.index.values[-1]:
        return series.index.values[-1]

    closest_indices = find_closest_indices(index, series)
    return closest_indices[0] if abs(closest_indices[0] - index) < abs(closest_indices[1] - index) else closest_indices[1]


def find_series_index_at_percentage(percentage, series):
    start_index = series.index.values[0]
    end_index = series.index.values[-1]

    indices_per_percentage = (end_index - start_index)/100

    raw_index = percentage * indices_per_percentage + start_index
    closest_index = find_closest_index(raw_index, series)
    return closest_index


def linear_interpolate_series(series, indices, target_index):
    # Apply linear interpolation
    # Find slope between indices
    # Find delta x from smaller index
    # Find delta y at x
    slope_delta_x = indices[1] - indices[0]
    slope_delta_y = series[indices[1]] - series[indices[0]]
    slope = slope_delta_y/slope_delta_x

    index_delta_x = target_index - indices[0]
    index_delta_y = slope * index_delta_x
    return series[indices[0]] + index_delta_y


def create_segment_debug_series(segment, number_of_points=1000, y_offset=0.0):
    # Return an x and y series to use for debugging
    start_x = segment.get_start()
    end_x = segment.get_end()

    # Generate a range of x values of which our segment will get evaluated over
    x_series = np.linspace(start_x, end_x, number_of_points)
    y_series = []
    derivative_series = []

    for x_value in x_series:
        y_value = segment.y(x_value) + y_offset
        y_series.append(y_value)

        derivative_value = segment.derivative(x_value)
        derivative_series.append(derivative_value)

    return x_series, y_series, derivative_series


def create_profile_debug_figure(median_blade_series, interpolated_blade_series, cleaned_scan_slice, smoothed_scan_slice_opt_3):
    profile_figure = plotter_instance.create_figure_2d(
        "Blade Profile")
    plotter_instance.add_pd_series(
        profile_figure, median_blade_series, "Raw Median Profile")
    plotter_instance.add_pd_series(
        profile_figure, interpolated_blade_series, "Raw Interpolated Profile")
    plotter_instance.add_pd_series(
        profile_figure, cleaned_scan_slice, "Cleaned Profile")
    plotter_instance.add_pd_series(
        profile_figure, smoothed_scan_slice_opt_3, "Smooth Profile")
    profile_figure.show()


##############
#Segment Func#
##############

#Segments module for all helpers that take segments as a parameter


def join_segments_to_circle_segment(segments, center_segment_index):

    local_segments = list(map(lambda segment: segment.copy(), segments))
    # Take given segments and make sure they overlap without any jumps
    # Starts from center segment and work outwards
    right_segments = join_segments(local_segments, center_segment_index, 1)
    joint_segments = join_segments(right_segments, center_segment_index, -1)
    return joint_segments


def join_segments(segments, center_segment_index, direction):

    local_segments = list(map(lambda segment: segment.copy(), segments))
    current_index = center_segment_index

    if direction == 1:
        end_index = len(local_segments) - 1

        while current_index < end_index:

            left_x = local_segments[current_index].get_end()
            left_y = local_segments[current_index].y(left_x)
            right_y = local_segments[current_index + 1].y(left_x)

            y_delta = left_y - right_y

            local_segments = y_shift_segments(local_segments, y_delta, start_index=current_index + 1, end_index=end_index)

            current_index += direction

    else:
        end_index = 0

        while current_index > end_index:

            right_x = local_segments[current_index].get_start()
            right_y = local_segments[current_index].y(right_x)
            left_y = local_segments[current_index - 1].y(right_x)

            y_delta = right_y - left_y

            local_segments = y_shift_segments(local_segments, y_delta, start_index=end_index, end_index=current_index - 1)

            current_index += direction

    return local_segments


def y_shift_segments(segments, shift_value, start_index=0, end_index=None):

    local_segments = list(map(lambda segment: segment.copy(), segments))

    if end_index is None:
        end_index = len(local_segments) - 1

    current_index = start_index
    while current_index <= end_index:
        local_segments[current_index].y_shift(shift_value)
        current_index += 1

    return local_segments    


def calculate_segment_series_difference(series, segment, series_test_range):
    # Find average difference between generated circles and segment over series index range
    difference_accumulator = 0

    # Create test range
    sub_series = series.loc[series_test_range[0]:series_test_range[1]]
    for x_value in sub_series.index.values:
        difference_accumulator += abs(series[x_value] - segment.y(x_value))

    return difference_accumulator / len(sub_series.values)


def extend_line_from_segment(segment, x):
    if x > segment.get_end():
        segment_derivative_x = segment.get_end()
    elif x < segment.get_start():
        segment_derivative_x = segment.get_start()
    else:
        return segment.y(x)

    derivative = segment.derivative(segment_derivative_x)
    target_x_delta = x - segment_derivative_x
    target_y_delta = derivative * target_x_delta
    extended_y = segment.y(segment_derivative_x) + target_y_delta
    return extended_y


def rename_series_indices(series, index_mutator):
    local_series = series.copy()
    rename_dict = {}
    for index in local_series.index.values:
        rename_dict[index] = index * index_mutator
    local_series.rename(index=rename_dict, inplace=True)
    return local_series


def pad_text_until_length(text, character, target_length):
    """ 
    Take a string and pad it with character until target length is reached or surpassed.

    Arguments:
        text {string} -- Some text to pad onto.
        character {string} -- Character used to do the padding.
        target_length {int} -- Our goal length for the text.
    """

    current_length = len(text)
    if current_length < target_length:
        text = f"{character}{text}"
        text = pad_text_until_length(text, character, target_length)

    return text


def find_center_profiling_specification(profiling_specifications):

    start_percentage = None
    end_percentage = None
    for profiling_specification in profiling_specifications:
        if start_percentage is None or profiling_specification.start_percentage < start_percentage:
            start_percentage = profiling_specification.start_percentage

        if end_percentage is None or profiling_specification.end_percentage > end_percentage:
            end_percentage = profiling_specification.end_percentage

    center_percentage = (end_percentage + start_percentage) / 2

    for profiling_specification_index, profiling_specification in enumerate(profiling_specifications):
        if profiling_specification.start_percentage <= center_percentage and profiling_specification.end_percentage >= center_percentage:
            return profiling_specification_index, profiling_specification

    assert("Center profiling specification not found!")


def find_segment_with_x_value(segments, x_value):
    # Return segment that falls on the x value
    for index, segment in enumerate(segments):
        if x_value >= segment.get_start() and x_value <= segment.get_end():
            return segment, index

    assert(f"No segment found for x value {x_value}")


def find_segments_max_y_delta(segments, scan_series):
    # Test segments and find our max cut depth
    max_y_delta = 0
    for x_value in scan_series.index.values:

        find_segment_response = find_segment_with_x_value(segments, x_value)

        if find_segment_response is not None:
            segment = find_segment_response[0]

            # If within the range of our segments
            if segment is not None:

                y_delta = abs(scan_series[x_value] - segment.y(x_value))

                if y_delta > max_y_delta:
                    max_y_delta = y_delta

    return max_y_delta

def find_segments_max_perpendicular_delta(segments, profile, test_points=1000):

    largest_delta = 0
    # for debugging
    largest_delta_x = None
    largest_delta_intersection_point = None

    failed_intersection_point_index = []

    # Iterate through all profile points
    for profile_x_index, profile_x in enumerate(profile.index.values):

        # Ignore first and last point
        if profile_x_index == 0 or profile_x_index == (profile.size - 1):
            continue
        
        # Find average derivative at point
        profile_average_derivative = find_average_derivative_at_index_in_series(profile, profile_x_index)
        profile_y = profile[profile_x]

        # Create line using point and average derivative
        profile_point_line = segments_module.Line(profile_x, profile_y, profile_average_derivative)

        # Create perpendicular line
        profile_point_perpendicular_line = profile_point_line.inverse()

        # Iterate through segments
        # Find segment with intersection with perpendicular line
        # Intersection point must be within segment x domain
        found_intersection_point = None
        for segment in segments:

            if found_intersection_point is not None:
                continue

            if segment.get_shape() == "Line":
                intersection_point = profile_point_perpendicular_line.intersection_point(segment.raw_shape)

                if intersection_point["x"] > segment.get_start() and intersection_point["x"] <= segment.get_end():
                    found_intersection_point = intersection_point
                    
                    
            # For circles convert them into a list of line segments
            else:
                
                line_segments = segment.to_line_segments()

                for line_segment in line_segments:

                    if found_intersection_point is not None:
                        continue

                    # TODO: Clean up code duplication
                    intersection_point = profile_point_perpendicular_line.intersection_point(line_segment.raw_shape)

                    if intersection_point["x"] > line_segment.get_start() and intersection_point["x"] <= line_segment.get_end():
                        found_intersection_point = intersection_point

        if found_intersection_point is not None:
            profile_point = {
                "x":profile_x,
                "y":profile_y
            }
            point_distance = find_distance_between_two_points(intersection_point, profile_point)

            if point_distance > largest_delta:
                largest_delta = point_distance

                # For debugging
                largest_delta_x = profile_x
                largest_delta_intersection_point = found_intersection_point
                # plotter_instance.plot_series_generated_segments(profile, [segments], lock_aspect_ratio=True)
        
        else:

            # Check for gap in failed intersection points
            # This implies that there was a failed point in the middle of the blade
            if len(failed_intersection_point_index) > 0:
                last_failed_point =  failed_intersection_point_index[-1]

                if (profile_x_index - last_failed_point) > 1:
                    raise Exception(f"None consecutive failed intersection point detected. Jump from index {last_failed_point} to {profile_x_index}.") 

            failed_intersection_point_index.append(profile_x_index)

    return largest_delta


def find_distance_between_two_points(point_1, point_2):

    x_delta = point_1["x"] - point_2["x"]
    y_delta = point_1["y"] - point_2["y"]
    hypotenuse = (x_delta ** 2 + y_delta ** 2) ** 0.5

    return hypotenuse
        

def find_average_derivative_at_index_in_series(series, index):

    x_1 = series.index.values[index - 1]
    x_2 = series.index.values[index]
    x_3 = series.index.values[index + 1]

    y_1 = series[x_1]
    y_2 = series[x_2]
    y_3 = series[x_3]

    derivative_1_2 = (y_2 - y_1) / (x_2 - x_1)
    derivative_2_3 = (y_3 - y_2) / (x_3 - x_2)

    average_derivative = (derivative_1_2 + derivative_2_3) / 2
    return average_derivative

def find_segments_max_perpendicular_deltas(blade_segments, scan_series, test_points=1000):

    # Convert series to a set of segments
    scan_line_segments = convert_series_to_line_segments(scan_series)

    # Create a test domain
    first_segment = blade_segments[0]
    last_segment = blade_segments[-1]

    domain_length = last_segment.get_end() - first_segment.get_start()
    test_x_delta = domain_length / test_points

    test_counter = 0
    max_positive_perpendicular_delta = 0
    max_negative_perpendicular_delta = 0

    while test_counter < test_points:

        segment_x = test_counter * test_x_delta
        test_counter += 1

        find_segment_response = find_segment_with_x_value(
            blade_segments, segment_x)

        if find_segment_response is not None:
            segment = find_segment_response[0]

            segment_y = segment.y(segment_x)
            segment_derivative = segment.derivative(segment_x)
            segment_perpendicular_derivative = -1/segment_derivative
            perpendicular_line = segments_module.Line(
                segment_x, segment_y, segment_perpendicular_derivative)

            intersection_point = find_line_intersection_with_line_segments(
                perpendicular_line, scan_line_segments)

            if intersection_point is not None:

                # use pythagoras to solve for perpendicular line length
                intersection_point_to_segment_y_delta = intersection_point["y"] - segment_y
                intersection_point_to_segment_x_delta = intersection_point["x"] - segment_x
                perpendicular_delta = (intersection_point_to_segment_y_delta **
                                       2 + intersection_point_to_segment_x_delta ** 2)**(1/2)

                if intersection_point_to_segment_y_delta < 0:
                    perpendicular_delta *= -1

                if perpendicular_delta > max_positive_perpendicular_delta:
                    max_positive_perpendicular_delta = perpendicular_delta

                if perpendicular_delta < max_negative_perpendicular_delta:
                    max_negative_perpendicular_delta = perpendicular_delta

    return max_negative_perpendicular_delta, max_positive_perpendicular_delta


def convert_series_to_line_segments(series, skip_rate=0):

    line_x_points = []

    sample_spacing = 1 + skip_rate
    number_of_points = math.floor(series.size / sample_spacing)
    point_counter = 0

    while point_counter < number_of_points:

        line_x_points.append(
            series.index.values[point_counter * sample_spacing])
        point_counter += 1

    line_segments = []
    for x_value_index, x_value in enumerate(line_x_points):

        next_x_value_index = x_value_index + 1
        if next_x_value_index < number_of_points:

            first_point_x = x_value
            first_point_y = series[first_point_x]

            second_point_x = line_x_points[next_x_value_index]
            second_point_y = series[second_point_x]

            line_segments.append(segments_module.LineBladeSegment.from_two_points(
                first_point_x, first_point_y, second_point_x, second_point_y))

    return line_segments


def find_line_intersection_with_line_segments(line, line_segments):

    for line_segment in line_segments:

        intersection_point = line.intersection_point(line_segment.raw_shape)
        if intersection_point["x"] >= line_segment.get_start() and intersection_point["x"] <= line_segment.get_end():
            return intersection_point


def apply_savgol_filter(series, rolling_window_size, polyorder):

    local_series = series.copy()

    filtered_series_y = savgol_filter(
        local_series, rolling_window_size, polyorder)
    local_series.update(pd.Series(data=filtered_series_y,
                                  index=local_series.index.values))

    return local_series


def apply_savgol_filter_section_percentage(series, start_percentage, end_percentage, rolling_window_size, polyorder=2):
    
    start_index = find_series_index_at_percentage(start_percentage, series)
    end_index = find_series_index_at_percentage(end_percentage, series)

    return apply_savgol_filter_section(series, start_index, end_index, rolling_window_size, polyorder=polyorder)

def apply_savgol_filter_section(series, start_index, end_index, rolling_window_size, polyorder=2):

    local_series = series.copy()

    partial_smoothed_series = apply_savgol_filter(series.loc[start_index:end_index], rolling_window_size, polyorder)

    for series_index in partial_smoothed_series.index.values:
        local_series[series_index] = partial_smoothed_series[series_index]

    return local_series

def interpolate_series_at_index(series, index, interpolation_window=4, gap_points=3):

    local_series = series.copy()

    # Check if we have enough indices to remove
    # If not trim up to first point
    start_trim_index = index - math.ceil(interpolation_window/2)
    if start_trim_index < 0:
        start_trim_index = 1
    
    # Check if we have enough indices to remove
    # If not trim up to last point
    end_trim_index = index + math.ceil(interpolation_window/2)
    if end_trim_index > (local_series.size - 1):
        end_trim_index = local_series.size - 1

    start_trim_index_value = series.index.values[start_trim_index]
    end_trim_index_value = series.index.values[end_trim_index]

    # Create a step size to evenly space points between trim start and end
    step_size = (end_trim_index_value - start_trim_index_value) / (gap_points + 1)
    gap_series_x = []
    gap_series_y = []
    point_counter = 0
    while point_counter < gap_points:
        point_counter += 1
        gap_series_x_value = start_trim_index_value + point_counter * step_size 
        gap_series_x.append(gap_series_x_value)
        gap_series_y.append(np.NaN)

    gap_series = pd.Series(data=gap_series_y, index=gap_series_x)

    # trim series and add gap series
    trimmed_series = pd.concat([local_series.iloc[:start_trim_index], gap_series, local_series.iloc[end_trim_index:]])
    # interpolate over nan values in gap series
    trimmed_interpolated_series = trimmed_series.interpolate(method="slinear")

    return trimmed_interpolated_series


def split_series_in_half(series):

    start_index = series.index[0]
    split_index = start_index + int(series.size / 2)

    return [
        series.loc[:split_index],
        series.loc[split_index:start_index + series.size]
    ]


def split_series_into_sections(series, min_section_gap_mm=1.0):

    min_section_gap_indices = math.ceil(
        min_section_gap_mm / definitions.SPACING_PER_X_INDEX)

    key_index_pairs = []
    start_key_index = None
    in_nan_section = True

    for series_index in series.index.values:
        last_index = series_index == series.index.values[-1]
        series_value = series[series_index]
        series_value_is_nan = np.isnan(series_value)

        # Check if we have started to see non-nan values
        # This will be the start of our section
        if in_nan_section and series_value_is_nan == False:
            # Look for a real value
            # This will be the start of our section
            start_key_index = series_index
            in_nan_section = False

        # Check if we have stopped to see non-nan values
        # If we reach the end of the series without concluding this section
        elif not in_nan_section and (series_value_is_nan == True or last_index):

            # If we are at the last index and are still getting non-nan values we can use the 
            # last index instead of the previous index.
            end_series_index = series_index
            if not last_index:
                end_series_index -= 1

            key_index_pairs.append([start_key_index, end_series_index])
            in_nan_section = True

    # Test section lengths
    joint_pairs = []
    for pair_index, pair in enumerate(key_index_pairs):

        if len(joint_pairs) == 0:
            joint_pairs.append(pair)
        else:
            last_pair = joint_pairs[-1]
            if (pair[0] - last_pair[1]) < min_section_gap_indices:
                # Mutation here affects key_index_pairs items as well
                last_pair[1] = pair[1]
            else:
                joint_pairs.append(pair)

    return_series = []
    for joint_pair in joint_pairs:
        return_series.append(series.loc[joint_pair[0]:joint_pair[1]])

    return return_series


# Given a list of series find the series with the largest size
def find_largest_series_section(sections):

    largest_series = None

    for series in sections:

        if largest_series is None or largest_series.size < series.size:
            largest_series = series.copy()

    return largest_series


def trim_series(series, start_trim_length, end_trim_length):

    series_start_index = series.index.values[0]
    series_end_index = series.index.values[-1]

    new_start_index = find_closest_index(
        series_start_index + start_trim_length/definitions.SPACING_PER_X_INDEX, series)
    new_end_index = find_closest_index(
        series_end_index - end_trim_length/definitions.SPACING_PER_X_INDEX, series)

    return series.loc[new_start_index:new_end_index]

# Terrible name...
# Function is used to find true center of blades by eliminating data on the fixture
# Resize is a good keyword here


def focus_blade_in_series(series):

    # Remove values under 0.5
    local_series = series.copy().where(series > 0.5)

    # Find first and last non nan index
    first_non_nan_index = int(find_first_non_nan(local_series))
    last_non_nan_index = int(find_last_non_nan(local_series))

    return local_series.iloc[first_non_nan_index:last_non_nan_index]


def shift_series_indices(series, index_shift):
    local_series = series.copy()
    rename_dict = {}
    for index in local_series.index.values:
        rename_dict[index] = index + index_shift
    local_series.rename(index=rename_dict, inplace=True)
    return local_series


def create_subset_series_from_segments(segments, start, end, number_of_points=100):

    x_series = np.linspace(start, end, number_of_points)
    y_series = []

    for x_value in x_series:
        segment, segment_index = find_segment_with_x_value(segments, x_value)
        y_value = segment.y(x_value)
        y_series.append(y_value)

    return pd.Series(y_series, index=x_series)


def replace_series_values(series, start, end, value=np.NaN):

    local_series = series.copy()

    indices_to_replace = local_series.loc[start:end]

    for index_to_replace in indices_to_replace.index.values:
        local_series[index_to_replace] = np.NaN

    return local_series


def pd_series_to_json(series, number_of_points=None):

    total_number_of_profile_points = len(series.index.values)

    if number_of_points is None:
        number_of_points = total_number_of_profile_points

    index_increment = total_number_of_profile_points/number_of_points

    point_counter = 0
    points_x = []
    points_y = []

    while(point_counter < number_of_points):
        point_index = math.floor(point_counter * index_increment)
        point_x = float(series.index.values[point_index])
        point_y = float(series[point_x])

        points_x.append(point_x)
        points_y.append(point_y)
        point_counter += 1

    return {
        "x": points_x,
        "y": points_y
    }


def find_segment_index_in_segments(segments, target_segment):

    for segment_index, segment in enumerate(segments):
        if target_segment.get_start() == segment.get_start() and target_segment.get_end() == segment.get_end():
            return segment_index


def derivative_to_vector_components(derivative, magnitude=1.0):

    # Find theta
    # tan( Theta ) = y/x = derivative
    # Theta =  tan^-1 ( derivative )
    theta = math.atan(derivative)

    # Find our x and y components
    # x component
    # cos ( Theta ) = x / magnitude
    # x = cos ( Theta ) * magnitude

    # y component
    # sin ( Theta ) = y / magnitude
    # y = sin ( Theta ) * magnitude

    # Could also use a like triangles method instead
    return {
        "x": math.cos(theta) * magnitude,
        "y": math.sin(theta) * magnitude
    }


def calculate_segment_end_derivatives_difference(left_segment, right_segment):
    return left_segment.derivative(left_segment.get_end()) - right_segment.derivative(right_segment.get_start())


def float_equivalence(float_1, float_2, precision=0.0001):

    difference = abs(float_1 - float_2)

    if difference < precision:
        return True

    return False


async def retry_function(function_reference, *args, retry_count=5, delay=5, allowed_exceptions=(), **kwargs):

    result = None
    last_exception = None
    for _ in range(retry_count):
        try:
            result = function_reference(*args, **kwargs)
            if result:
                return result
        except allowed_exceptions as e:
            last_exception = e

        await asyncio.sleep(delay)

    if last_exception is not None:
        raise type(last_exception) from last_exception

    return result


def retry_decorator(retry_count=5, delay=5, allowed_exceptions=()):
    def decorator(function_reference):
        @functools.wraps(function_reference)
        async def wrapper(*args, **kwargs):
            return retry_function(function_reference, *args, retry_count=5, delay=5, allowed_exceptions=(), **kwargs)

        return wrapper
    return decorator


def remove_none_elements_from_list(input_list):

    output_list = []

    for element in input_list:
        if element is not None:
            output_list.append(element)

    return output_list


def denoise_loop(series, interpolation_method, cycles, threshold=0.8, threshold_divisor=2, minimum_threshold=0.05, method="1_diff"):

    denoise_threshold = threshold
    local_denoised_series = series.copy()
    cycles_left = cycles
    while cycles_left > 0:
        # Denoise
        local_denoised_series = omit_series_noise(
            local_denoised_series, denoise_threshold, method=method)

        # Interpolate
        local_denoised_series = local_denoised_series.interpolate(method="akima")

        cycles_left = cycles_left - 1

        denoise_threshold *= 1/threshold_divisor
        if denoise_threshold < minimum_threshold:
            denoise_threshold = minimum_threshold

    return local_denoised_series


def denoise_loop_max(series, interpolation_method, cycles, threshold=0.8, threshold_multiplier=2, maximum_threshold=2.5, method="1_diff"):

    denoise_threshold = threshold
    local_denoised_series = series.copy()
    cycles_left = cycles
    while cycles_left > 0:
        # Denoise
        local_denoised_series = omit_series_noise(
            local_denoised_series, denoise_threshold, method=method)

        # Interpolate
        local_denoised_series = local_denoised_series.interpolate(method="akima")

        cycles_left = cycles_left - 1

        denoise_threshold *= threshold_multiplier
        if denoise_threshold > maximum_threshold:
            denoise_threshold = maximum_threshold

    return local_denoised_series


def apply_smoothing_rounds(series, rounds):

    local_series = series.copy()

    for i in range(0, rounds):
        local_series = apply_savgol_filter(local_series, 31, 2)

    return local_series

def apply_scanner_offsets(series, x_offset, y_offset=None):

    if y_offset == None:
        y_offset = definitions.APP_CONFIG["toolPathGenerator"]["scanner"]["offsets"]["y"]

    local_series = series.copy()

    shifted_series = shift_series_indices(local_series, x_offset)
    shifted_series += y_offset

    return shifted_series

# Freelancer work


def trim_blade_edges(y_org, th_left=0.2, th_right=2.0):
    # Trim for extreme points
    L1, R1 = trim_index_v1(y_org)
    y_trim_v1 = y_org.loc[L1:R1]
    # y_trim_v1=TrimSeries(y_org,L1,R1)

    # Interpolate to fill the NaN values and small perturbations
    y_trim_v11 = interpolation_denoise_filter(y_trim_v1)
    L2, R2 = trim_index_v2(y_trim_v11, th_left, th_right)

    # Trim again to remove side profile perturbations
    y_trim_v12 = y_trim_v11.loc[L2:R2]

    return y_trim_v12


def trim_index_v1(y):
    L1 = 0
    R1 = len(y)-1
    dyL = pd.Series(np.abs(pd.Series.diff(y, 1)))
    dyR = pd.Series(np.abs(pd.Series.diff(y, -1)))

    for i in range(y.index.values[0], y.index.values[-1]):
        if dyL[i] > 0.1 and y[i+1] > -30:
            L1 = i
            break
    for i in range(y.index.values[-1], y.index.values[0], -1):
        if dyR[i] > 0.1 and y[i-1] > -30:
            R1 = i
            break
    return L1, R1


def trim_index_v2(y, th_left, th_right):
    #     th_left=.2,th_right=2
    L1 = y.index.values[0]
    R1 = y.index.values[-1]
    dyL = pd.Series(np.abs(pd.Series.diff(y, 1)))
    dyR = pd.Series(np.abs(pd.Series.diff(y, -1)))

    for i in range(y.index.values[0], y.index.values[0]+1000):
        if dyL[i] > th_left and y[i+1] > -30:
            L1 = i

    for i in range(y.index.values[-1], y.index.values[-1]-1000, -1):
        if dyR[i] > th_right and y[i-1] > -30:
            R1 = i
    return L1, R1


def interpolation_denoise_filter(y):
    ys = y.copy()
    idx = np.int32(len(ys)/2)
    for i in range(ys.index.values[0], ys.index.values[0] + idx):
        if ys[i] < -50:
            ys[i] = ys[i-1]

    for i in range(ys.index.values[-1], ys.index.values[-1] - idx, -1):
        if ys[i] < -50:
            ys[i] = ys[i+1]

    st = 15 + ys.index.values[0]
    for j in range(ys.index.values[0], st):
        for i in range(ys.index.values[0], ys.index.values[-1]-2):
            df1 = np.abs(y[i]-ys[i])
            if df1 > .02:
                if i == 0:
                    ys[i] = (ys[i]+ys[i+1])/2
                elif i == (len(ys)-1):
                    ys[i] = (ys[i-1]+ys[i])/2
                else:
                    ys[i] = (ys[i-1]+ys[i+1])/2

    for k in range(1, 2):
        for j in range(1, 2+2*k):
            for i in range(ys.index.values[0]+k, ys.index.values[-1]-2-k):
                df1 = np.abs(ys[i-k]-ys[i+k])
                if df1 < j*.05:
                    ys[i] = (ys[i-k]+ys[i+k])/2

    return ys
