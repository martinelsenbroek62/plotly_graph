import os, re, math, statistics
from glob import glob

import qprompt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from modules import helpers, blade, segments
import definitions

###########
# Modules #
###########
plotter_instance = helpers.Plotter()

class Scan():
    """ 
    Represents a scan of a blade.

    Arguments:
        scan_dataframe {DataFrame} -- Contains our scan data.

    Keyword Arguments:        
        y_offset {float} -- Contains our scan data. (default: {0.0})
        scanner_accuracy_range {list} -- Two element list defining a range of 
         values we accept to be accurate in our scan data. (default: {[-6,6]})
        debug {boolean} -- Enable extra output for troubleshooting.
         (default: {False})
    """

    def __init__(self, \
            scan_dataframe, \
            y_offset=0.0, \
            scanner_accuracy_range=definitions.DEFAULT_SCANNER_ACCURACY_RANGE, \
            debug_mode=definitions.APP_CONFIG["toolPathGenerator"]["debug"]):

        self.raw_scan_dataframe = scan_dataframe

        self.spacing_per_x_index = definitions.SPACING_PER_X_INDEX
        self.spacing_per_z_index = definitions.SPACING_PER_Z_INDEX

        # Once our new cartridges are created we can no longer assume this radius
        x_axis_reference_feature_length = definitions.APP_CONFIG["toolPathGenerator"]["fixture"]["cartridge"]["xAxisReference"]["length"]
        self.x_axis_reference_feature_length_in_indices = int(x_axis_reference_feature_length / self.spacing_per_x_index)
        self.max_circle_segment_difference = definitions.APP_CONFIG["toolPathGenerator"]["maxCircleSegmentDifference"]

        # Test scanner accuracy range validity
        Scan.validate_scanner_accuracy_range(scanner_accuracy_range)
        self.scanner_accuracy_range = scanner_accuracy_range

        self.y_offset = y_offset
        self.debug_mode = debug_mode

        self.accurate_scan_data, self.trimmed_scan_dataframe = Scan.clean_scan_data(self.raw_scan_dataframe, self.scanner_accuracy_range)

    def detect_blade_z_center(self, blade_series):
        
        # simplier method
        # blades are known to be at center of series
        # focus on this area first
        local_blade_series = blade_series.copy()

        center_index = (local_blade_series.index[0] + local_blade_series.index[-1]) / 2

        blade_thickness = 3.0
        section_thickness_indices = int((blade_thickness / 2.5) / self.spacing_per_x_index)

        local_blade_series = local_blade_series.loc[center_index - section_thickness_indices:center_index + section_thickness_indices]

        focused_blade_series = helpers.focus_blade_in_series(local_blade_series)
        interpolated_blade_series = focused_blade_series.interpolate(method='akima')

        smoothed_blade_series = helpers.apply_savgol_filter(interpolated_blade_series, 31, 5)

        return smoothed_blade_series.idxmin()

        # Old method using blade start and end
        # local_blade_series = local_blade_series.interpolate(method='akima')

        # # Reduce size of series to only include blades
        # first_derivative_blade_series = local_blade_series.diff()
        # second_derivative_blade_series = first_derivative_blade_series.diff()

        # smoothed_second_derivative_blade_series = helpers.apply_savgol_filter(second_derivative_blade_series, 31, 5)

        # # split in half
        # split_smoothed_second_derivative_blade_series = helpers.split_series_in_half(smoothed_second_derivative_blade_series)

        # # find min of left side
        # blade_start = split_smoothed_second_derivative_blade_series[0].idxmin()

        # # find min of right side
        # blade_end = split_smoothed_second_derivative_blade_series[1].idxmin()

        # return center index
        # return int((blade_start + blade_end) / 2)

    ##########################
    # Dataframe Manipulation #
    ##########################

    @staticmethod
    def clean_scan_data(dataframe, scanner_accuracy_range):
        """ 
        Remove useless data from scan DataFrame.
        """

        print("Scan session clean scan data")

        local_accurate_scan_dataframe = Scan.remove_out_of_scanner_range_values(dataframe, scanner_accuracy_range)
        return local_accurate_scan_dataframe, Scan.trim_dataframe(local_accurate_scan_dataframe)

    @staticmethod
    def remove_out_of_scanner_range_values(dataframe, scanner_accuracy_range):
        """ 
        Replace values outside the range of our scanner with None. 
        Uses our class property scanner_accuracy_range.
        """
        mins_removed = dataframe.where(dataframe > scanner_accuracy_range[0])
        return mins_removed.where(mins_removed < scanner_accuracy_range[1])

    @staticmethod
    def trim_dataframe(dataframe):
        """ 
        Resize our DataFrame to remove any dead space we have on the x and z axis.
        Dead space will be represented via NaN values.
        """
        print("Scan session trim x axis values")
        local_trimmed_scandata = dataframe.copy()
        local_trimmed_scandata = Scan.trim_dataframe_x_axis(local_trimmed_scandata)
        # local_trimmed_scandata = Scan.trim_dataframe_z_axis(local_trimmed_scandata)
        return local_trimmed_scandata

    @staticmethod
    def trim_dataframe_x_axis(dataframe):
        """
        We want to resize our x axis which is the number of rows in our dataframe.
        To improve performance we should reduce our dataframe's size.
        """
        index_offset = dataframe.index.start
        dataset_mid_point = (int)((dataframe.index.start + dataframe.index.stop) / 2)
        min_y_axis_index = dataset_mid_point
        max_y_axis_index = dataset_mid_point
        for x_axis_column_index in dataframe.columns.values:

            # Slice x axis column values
            x_axis_slice = dataframe[x_axis_column_index]

            """
            The first and last non nan indices make up the range of indices we
            should to keep. We should trim indices outside of this range.
            """
            min_y_axis_index, max_y_axis_index = helpers.find_nan_range_in_slice(x_axis_slice, min_y_axis_index, max_y_axis_index, index_offset=index_offset)

        # Trim away rows that do not hold any data
        return dataframe.iloc[min_y_axis_index:max_y_axis_index,:]

    @staticmethod
    def trim_dataframe_z_axis(dataframe):
        """
        We want to resize our z axis which is the number of columns in our dataframe.
        To improve performance we should reduce our dataframe's size.
        """
        dataset_mid_index = (int)((dataframe.columns.max() + dataframe.columns.min()) / 2)
        min_z_axis_index = dataset_mid_index
        max_z_axis_index = dataset_mid_index
        for z_axis_row_index in dataframe.index.values:

            # Slice z axis row values
            z_axis_slice = dataframe.loc[z_axis_row_index,:]

            """
            The first and last non nan indices make up the range of indices we
            should to keep. We should trim indices outside of this range.
            """
            min_z_axis_index, max_z_axis_index = helpers.find_nan_range_in_slice(z_axis_slice, min_z_axis_index, max_z_axis_index)

        # Trim away columns that do not hold any data
        return dataframe.loc[:,min_z_axis_index:max_z_axis_index]

    @staticmethod
    def normalize_z_axis_values(dataframe):
         # Flipping the dataframe values
        flipped_dataframe = dataframe.iloc[:,::-1]
        # Rename columns to match new positions
        rename_dict = {}
        max_z_index = flipped_dataframe.shape[1] - 1 # Minus one to convert to index values
        for index, value in enumerate(flipped_dataframe.columns):
            rename_dict[value] = max_z_index - value
        flipped_dataframe.rename(columns=rename_dict, inplace=True)

        return  flipped_dataframe

    @staticmethod
    def flip_range(max_value, range):
        range_copy = range.copy()
        new_end = max_value - range_copy[0]
        new_start = max_value - range_copy[1]
        range_copy[0] = new_start
        range_copy[1] = new_end
        return range_copy

    # Depreciated
    # scan_instance.detect_blades functionality has changed
    @staticmethod
    def from_multiple_scans(scan_file_set):    
        
        debug_mode = definitions.APP_CONFIG["toolPathGenerator"]["debug"]

        # Create a set of scan instances
        # While finding our largest scan instance
        scan_instances = []
        largest_scan_instance = None

        if debug_mode:
            scan_instance_2d_figure_0 =  plotter_instance.create_figure_2d("Raw Scanner Instances Pre-Merge Blade 0")
            scan_instance_2d_figure_1 =  plotter_instance.create_figure_2d("Raw Scanner Instances Pre-Merge Blade 1")

        for scan_index, scan_file_instance in enumerate(scan_file_set):
            scan_instance = Scan(\
                scan_file_instance.to_dataframe(), \
                debug_mode=debug_mode \
            )

            # 
            scan_instance.detected_blades = scan_instance.detect_blades(scan_instance.trimmed_scan_dataframe)
            scan_instances.append(scan_instance)

            if debug_mode:
                plotter_instance.add_pd_series(scan_instance_2d_figure_0, scan_instance.detected_blades[0], f"Scan {scan_index} Offset@{scan_instance.y_offset}")
                plotter_instance.add_pd_series(scan_instance_2d_figure_1, scan_instance.detected_blades[1], f"Scan {scan_index} Offset@{scan_instance.y_offset}")

            # Find largest scan instance
            # No advantage at the moment in not mutating the largest scan instance
            if largest_scan_instance is None:
                largest_scan_instance = scan_instance
            elif largest_scan_instance.trimmed_scan_dataframe.shape[0] < scan_instance.trimmed_scan_dataframe.shape[0]:
                largest_scan_instance = scan_instance

        if debug_mode:
            scan_instance_2d_figure_0.show()
            scan_instance_2d_figure_1.show()

        # Shift all scan instances in x and y
        # We will use the our largest instance as our reference
        if debug_mode:
            scan_instance_2d_figure_0 =  plotter_instance.create_figure_2d("Shifted Scanner Instances Pre-Merge Blade 0")
            scan_instance_2d_figure_1 =  plotter_instance.create_figure_2d("Shifted Scanner Instances Pre-Merge Blade 1")

        for scan_index, scan_instance in enumerate(scan_instances):
            x_delta = largest_scan_instance.cartridge_reference_features["x"][0] - scan_instance.cartridge_reference_features["x"][0]

            # Shift all x values by x_delta
            for blade_index, detected_blade in enumerate(scan_instance.detected_blades):
                if x_delta != 0:
                    rename_dict = {}
                    for x_index, value in enumerate(detected_blade.index.values):
                        rename_dict[value] = value + x_delta
                    detected_blade.rename(index=rename_dict, inplace=True)

                # Normalize all y values
                detected_blade += scan_instance.y_offset

            if debug_mode:
                plotter_instance.add_pd_series(scan_instance_2d_figure_0, scan_instance.detected_blades[0], f"Scan {scan_index} Offset@{scan_instance.y_offset}")
                plotter_instance.add_pd_series(scan_instance_2d_figure_1, scan_instance.detected_blades[1], f"Scan {scan_index} Offset@{scan_instance.y_offset}")

        # Iterate through all rows of largest_scan_instance and apply merge
        rows_to_merge = range(largest_scan_instance.trimmed_scan_dataframe.index.start, largest_scan_instance.trimmed_scan_dataframe.index.stop, 1)
        for row_index, row in enumerate(rows_to_merge):
            for blade_index, detected_blade in enumerate(largest_scan_instance.detected_blades):
                blades = []
                for scan_index, scan_instance in enumerate(scan_instances):
                    scan_blade = scan_instance.detected_blades[blade_index]
                    # Make sure row index exists
                    if row in scan_blade.index:
                        blades.append({
                            "value": scan_blade[row],
                            "y_offset": scan_instance.y_offset
                        })

                # Merging one blade does nothing
                if len(blades) > 1:
                    # Merge values of the blade row
                    largest_scan_instance.detected_blades[blade_index][row] = Scan.merge_values(blades, scanner_accuracy_range[1])

                    # Merge the rows and mutate our largest scan instance dataframe
                    # largest_scan_instance.trimmed_scan_dataframe.update(Scan.merge_row(rows, scanner_accuracy_range[1]))
                    # largest_scan_instance.trimmed_scan_dataframe.loc[[row]]
                    # rows[0]['dataframe']
                    # largest_scan_instance.trimmed_scan_dataframe.loc[[row]][100]
                    # largest_scan_instance.trimmed_scan_dataframe.loc[[row]].max(axis=1)
                    # largest_scan_instance.trimmed_scan_dataframe.loc[[row]].idxmax(axis=1)
                    print(f"row {row} merged for blade {blade_index}")

        if debug_mode:
            plotter_instance.add_pd_series(scan_instance_2d_figure_0, largest_scan_instance.detected_blades[0], f"Merged Scan")
            plotter_instance.add_pd_series(scan_instance_2d_figure_1, largest_scan_instance.detected_blades[1], f"Merged Scan")
            scan_instance_2d_figure_0.show()
            scan_instance_2d_figure_1.show()

        return largest_scan_instance

    @staticmethod 
    def merge_row(rows, max_delta_from_scanner_focal_point):
        # Should return a single row dataframe back

        # TODO: This method needs to be optimized to use dataframes more efficiently
        # We can create new dataframes for each operation instead of evaluating each value
        # Current run time is 10 minutes for joining three scans

        # Merging one row does nothing
        # Just return it instead
        if len(rows) == 1:
            return rows[0]["dataframe"]

        # Get largest row
        largest_row_dataframe = None
        for row_index, row in enumerate(rows):
            if largest_row_dataframe is None:
                largest_row_dataframe = row["dataframe"].copy()
            elif largest_row_dataframe.shape[1] < row["dataframe"].shape[1]:
                largest_row_dataframe = row["dataframe"].copy()

        for column in range(largest_row_dataframe.columns[0], largest_row_dataframe.columns[-1] + 1):

            # This value flag is used if we find a point that is exactly on the scanner focal point
            best_column_value = None
            column_total_weight = 0
            column_values = []

            for row_index, row in enumerate(rows):
                # Check if exists
                # Check if not nan
                if column in row["dataframe"] and np.isnan(row["dataframe"][column]).bool() is False:
                    
                    # Calculate distance from best case scanner accuracy
                    column_value = row["dataframe"][column].values[0]
                    abs_delta_scanner_focal_point =  abs(column_value - row["y_offset"])

                    # TODO: Consider using a tolerance instead
                    # Like if any delta is less than 0.001 then we consider it perfect
                    # Really the current use case just avoids us having an infinite weight
                    if abs_delta_scanner_focal_point == 0:
                        best_column_value = column_value
                    else:
                        # Calculate weight
                        weight = (abs_delta_scanner_focal_point / max_delta_from_scanner_focal_point) ** -1
                        column_values.append({
                            "weight": weight,
                            "value": column_value
                        })
                        # Add to total weight
                        column_total_weight += weight

            # Update value of column for largest_row
            # If a point is found exactly on the focal point of the scanner just use it
            if best_column_value is not None:
                largest_row_dataframe[column] = best_column_value
            elif len(column_values) > 0: # To avoid nan only column slices
                # Calculate best value given column values and total weight
                best_column_value_accumulator = 0
                for index, column_value in enumerate(column_values):
                    best_column_value_accumulator += column_value["value"] * column_value["weight"] / column_total_weight
                largest_row_dataframe[column] = best_column_value_accumulator

        return largest_row_dataframe
 
    @staticmethod
    def merge_values(blades, max_delta_from_scanner_focal_point):
        # Merging one blade does nothing
        # Just return it's value
        if len(blades) == 1:
            return blades[0]["value"]

        # This value flag is used if we find a point that is exactly on the scanner focal point
        total_weight = 0
        values = []

        for blade_index, blade in enumerate(blades):
            # Check if not nan
            if np.isnan(blade["value"]) == False:
                
                # Calculate distance from best case scanner accuracy
                value = blade["value"]
                abs_delta_scanner_focal_point =  abs(value - blade["y_offset"])

                # If a point is found exactly on the focal point of the scanner just use it
                # The scanner should be the most accurate here
                # TODO: Consider using a tolerance instead
                # Like if any delta is less than 0.001 then we consider it perfect
                # Really the current use case just avoids us having an infinite weight
                if abs_delta_scanner_focal_point == 0:
                    return value
                else:
                    # Calculate weight
                    weight_factor = 3 # Increasing this value will put more weight on points with lower scanner deltas
                    weight = (abs_delta_scanner_focal_point / max_delta_from_scanner_focal_point) ** (-1 * weight_factor)
                    values.append({
                        "weight": weight,
                        "value": value
                    })
                    # Add to total weight
                    total_weight += weight

        # Update value of column for largest_row
        if len(values) > 0: # To avoid nan values
            # Calculate best value given values and total weight
            best_value_accumulator = 0
            for index, value in enumerate(values):
                best_value_accumulator += value["value"] * value["weight"] / total_weight
            return best_value_accumulator

        return values[0]

    ##############
    # Validation #
    ##############

    @staticmethod
    def validate_scanner_accuracy_range(scanner_accuracy_range):
        if len(scanner_accuracy_range) != 2:
            raise Exception("The scanner_accuracy_range must be a list with two elements!")
        if scanner_accuracy_range[0] > scanner_accuracy_range[1]:
            raise Exception("The first element in scanner_accuracy_range must greater than second element!")

    ###########
    # Helpers #
    ###########

    def plot_segments_with_raw_scan_series(self, blade_scan_series, segments, blade_index, depth_of_cut):

        bottom_blade = (blade_index == 1)
        blade_segment_figure = plotter_instance.create_figure_2d(f"Segments of {'bottom' if bottom_blade else 'top'} blade")

        renamed_blade_scan_series = helpers.rename_series_indices(blade_scan_series, self.spacing_per_x_index)
        plotter_instance.add_pd_series(blade_segment_figure, renamed_blade_scan_series, "Scan")

        # for each segment
        for segment_index, segment in enumerate(segments):
            # Segment data
            segment_x_series, segment_y_series, segment_derivative_series = helpers.create_segment_debug_series(segment)
            plotter_instance.add_raw_series(blade_segment_figure, segment_x_series, segment_y_series, f"Segment {segment_index}")
            # Tool path
            segment_x_series, segment_y_series, segment_derivative_series = helpers.create_segment_debug_series(segment, y_offset=-depth_of_cut)
            plotter_instance.add_raw_series(blade_segment_figure, segment_x_series, segment_y_series, f"Cut Segment {segment_index}")

        blade_segment_figure.show()

    @staticmethod
    def find_cubic_beizer_search_range(initial_cubic_bezier_blade_segment, handle_start_control_point):
        # Select the appropriate control point and derivative line
        if handle_start_control_point:
            upper_bound_x = initial_cubic_bezier_blade_segment.raw_shape.P2["x"]
            lower_bound_x = initial_cubic_bezier_blade_segment.raw_shape.P0["x"]
        else:
            upper_bound_x = initial_cubic_bezier_blade_segment.raw_shape.P3["x"]
            lower_bound_x = initial_cubic_bezier_blade_segment.raw_shape.P1["x"]

        return [
            lower_bound_x, 
            upper_bound_x
        ]

    def convert_series_to_circle_segments(self, series, start_segment_flag, key_derivative):
        # TODO: A better approach might be to define an acceptable amount of error and then
        # when generating circles keep segmenting the circles until we meet our error requirements

        # Move one index in to avoid nan derivative at the start of the series
        if start_segment_flag:
            start = series.index.values[1]
            end = series.index.values[-1]
        else:
            start = series.index.values[0]
            end = series.index.values[-2]

        # Break up the series into multiple circles
        circle_segments = self.series_to_circle_segments(series, start_segment_flag, key_derivative, self.max_circle_segment_difference, start=start, end=end)
        # reduced_circle_segments = self.reduce_circle_segments(circle_segments, number_of_segments=5)

        if self.debug_mode:
            helpers.plot_series_generated_segments(series, [circle_segments])
            # helpers.plot_series_generated_segments(series, [circle_segments, reduced_circle_segments])

        return circle_segments
    
    def series_to_circle_segments(self, series, start_segment_flag, key_derivative, max_circle_segment_difference, start=None, end=None):

        if start is None:
            start = series.index.values[0]

        if end is None:
            end = series.index.values[-1]

        # Break up the segment into multiple circles    
        segment_length = end - start
        segment_minimum_length = self.spacing_per_x_index
        
        series_test_range = [start, end]

        # TODO: Investigate cases where very small circles act as bumps
        # This seems to be caused by negative radicand circles and the derivatives they provide
        if segment_length < segment_minimum_length:
            print(f"Segment length is less than our segment minimum length!")

        circle_segment = self.series_to_circle(series, start, end, key_derivative, start_segment_flag)
        circle_segment_difference = self.calculate_segment_series_difference(series, circle_segment, series_test_range)

        circle_segments = []
        if circle_segment_difference < max_circle_segment_difference or segment_length < segment_minimum_length:
            circle_segments.append(circle_segment)        
        else:
            
            left_start = start
            left_end = helpers.find_closest_index(left_start + segment_length / 2, series)

            if start_segment_flag:
            
                right_start = left_end
                right_end = end
                right_circle_segments = self.series_to_circle_segments(series, start_segment_flag, key_derivative, max_circle_segment_difference, start=right_start, end=right_end)

                joining_circle_segment = right_circle_segments[0]
                joining_circle_segment_start = joining_circle_segment.get_start()
                joining_circle_segment_start_derivative = joining_circle_segment.derivative(joining_circle_segment_start)

                left_circle_segments = self.series_to_circle_segments(series, start_segment_flag, joining_circle_segment_start_derivative, max_circle_segment_difference, start=left_start, end=left_end)

            else:                

                left_circle_segments = self.series_to_circle_segments(series, start_segment_flag, key_derivative, max_circle_segment_difference, start=left_start, end=left_end)

                joining_circle_segment = left_circle_segments[-1]
                joining_circle_segment_end = joining_circle_segment.get_end()
                joining_circle_segment_end_derivative = joining_circle_segment.derivative(joining_circle_segment_end)

                right_start = left_end
                right_end = end

                right_circle_segments = self.series_to_circle_segments(series, start_segment_flag, joining_circle_segment_end_derivative, max_circle_segment_difference, start=right_start, end=right_end)

            circle_segments = left_circle_segments + right_circle_segments
        
        return circle_segments

    def reduce_circle_segments(self, circle_segments, number_of_segments=3):

        if len(circle_segments) < number_of_segments:
            print(f"Can't reduce circle segments lower than {len(circle_segments)}")
            return circle_segments
        
        # For each circle we need three points with the last point overlaping with the next circle
        # The number of points we need to isolate = number of segments * 2 + 1
        
        number_of_points = number_of_segments * 2 + 1

        # Create our linespace
        x_domain = np.linspace(circle_segments[0].get_start(), circle_segments[-1].get_end(), number_of_points)

        # Create new circles
        reduced_circle_segments = []
        segment_counter = 0
        while segment_counter < number_of_segments:

            start_x_index = segment_counter * 2

            p1_x = x_domain[start_x_index]
            p1_y = helpers.find_segment_with_x_value(circle_segments, p1_x)[0].y(p1_x)

            p2_x = x_domain[start_x_index + 1]
            p2_y = helpers.find_segment_with_x_value(circle_segments, p2_x)[0].y(p2_x)

            p3_x = x_domain[start_x_index + 2]
            p3_y = helpers.find_segment_with_x_value(circle_segments, p3_x)[0].y(p3_x)

            new_circle_segment = segments.CircleBladeSegment.from_3_points(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, raw_series = circle_segments[0].raw_series)
            reduced_circle_segments.append(new_circle_segment)

            segment_counter += 1

        return reduced_circle_segments

    def create_main_radius_segment(self, circle_segment_series, profile, working_radius):
        main_radius_model_segment = segments.CircleBladeSegment.from_series(circle_segment_series)

        # If profiling enabled
        if profile:
            # Would be better to just clone the segment instance
            profiled_main_radius_model_segment = segments.CircleBladeSegment.from_series(circle_segment_series)
            print(f"Current profile radius is {profiled_main_radius_model_segment.raw_shape.radius}mm")
            # Change circle radius
            profiled_main_radius_model_segment.raw_shape.radius = working_radius
            main_radius_model_segment = profiled_main_radius_model_segment

        # Confirm segment is making contact with full blade
        x_domain = circle_segment_series.index.values
        max_y_delta = None
        for x_value in x_domain:
            circle_segment_y_value = circle_segment_series[x_value]
            main_radius_y_value = main_radius_model_segment.y(x_value)

            y_delta = circle_segment_y_value - main_radius_y_value

            if max_y_delta is None or y_delta < max_y_delta:
                max_y_delta = y_delta

        main_radius_model_segment.raw_shape.y0 += max_y_delta

        # # Confirm segment is making contact with full blade
        # # Get y value of scan at center of circle
        # old_circle_apex_y_value = helpers.find_best_series_value_at_index(circle_segment_series, main_radius_model_segment.raw_shape.x0)
        # # Find y delta between new y value and old circle y value
        # new_circle_apex_y_value = main_radius_model_segment.y(main_radius_model_segment.raw_shape.x0)
        # new_circle_apex_delta_y = new_circle_apex_y_value - old_circle_apex_y_value
        # # Move new circle by y delta
        # main_radius_model_segment.raw_shape.y0 -= new_circle_apex_delta_y

        # # Confirm we are making contact with blade at the ends of the circle segment
        # # Get y delta at two send of the circle segment
        # # If y delta is negative we are cutting the blade, no mutation change required
        # # If y delta is positive we are above the blade, move the circle down by the delta

        # # Test start
        # start_x = main_radius_model_segment.start
        # new_start_y = main_radius_model_segment.y(start_x)
        # old_start_y = helpers.find_best_series_value_at_index(circle_segment_series, start_x)
        # start_delta_y = new_start_y - old_start_y

        # if start_delta_y > 0:
        #     main_radius_model_segment.raw_shape.y0 -= start_delta_y

        # end_x = main_radius_model_segment.end
        # new_end_y = main_radius_model_segment.y(end_x)
        # old_end_y = helpers.find_best_series_value_at_index(circle_segment_series, end_x)
        # end_delta_y = new_end_y - old_end_y

        # if end_delta_y > 0:
        #     main_radius_model_segment.raw_shape.y0 -= end_delta_y

        return main_radius_model_segment

class ScanFile():
    """ 
    Class used to manage a scan file.

    Arguments:
        file_path {string} -- Path to file.
        y_offset {float} -- Y axis offset for the scan file.

    Keyword Arguments:
        debug {boolean} -- Enable extra output for troubleshooting.
         (default: {false})
    """

    def __init__(self, file_path, y_offset=0.0, debug_mode=definitions.APP_CONFIG["toolPathGenerator"]["debug"]):
        self.file_path = file_path
        self.y_offset = y_offset
        self.debug_mode = debug_mode

    def to_dataframe(self):
        return pd.read_csv(self.file_path, header=None)

    @staticmethod
    def from_directory(directory_path):
        """
        Display all scan files in given directory for user to select, and return a ScanFile instance.

        Arguments:
            directory_path {string} -- Path to directory of files.

        Returns:
            [ScanFile] -- A ScanFile instance.
        """

        # Find all scan files in directory
        scan_files = glob(f"{directory_path}/*.csv*") # Allow csv and any compressed version of a csv
        # Reverse their order so latest files show up first in the list
        # This only occurs due to our naming convention
        reversed_scan_files = sorted(scan_files, reverse=True)
        # Create and display the menu
        menu = qprompt.enum_menu(reversed_scan_files)
        selected_index = int(menu.show(dft='1', header='Select a scan file'))-1
        selected_scan_file = reversed_scan_files[selected_index]

        return ScanFile(selected_scan_file, 0.0)

    @staticmethod
    def from_directory_multi(directory_path):
        """
        Reviews all files in scans directory, finds related scan files, and
        displays them for the user to select. 
        
        For scan files to be related their filenames will need to have the same
        prefix. An '(Offset@X)' section in the filename dictates at what offset
        this scan is at compared to the base scan file. The base scan file is
        the scan file without an offset section in the filename. Each scan dataset 
        requires one base scan file.

        Arguments:
            directory_path {string} -- Path to directory of files.

        Returns:
            [Array:ScanFile] -- An array of ScanFile instances.
        """

        # Find all scan files in directory
        scan_files = glob(f"{directory_path}/*.csv*")# Allow csv and any compressed version of a csv
        # Reverse their order so latest files show up first in the list
        # This only occurs due to our naming convention
        reversed_scan_files = sorted(scan_files, reverse=True)

        # Keep track of all mapped files
        paired_scan_files = {}
            
        # Look for files with an (base_file_nameoffset:X) appened to the end of their name
        offset_scan_file_regex = re.compile("(.+)\(Offset@(-?\d+(\.\d+)?)\)(\.csv)?.*")
        for scan_file in reversed_scan_files:
            if offset_scan_file_regex.match(scan_file):
                """
                Get the base file name by selecting the first regex group.
                We add the file type again afterwards to complete the file name.
                """
                base_file_name = f'{re.search(offset_scan_file_regex, scan_file).group(1)}.csv.bz2'
                y_offset_value = float(re.search(offset_scan_file_regex, scan_file).group(2))

                # Create the base file element if it doesn't exist in dictionary yet
                if base_file_name not in paired_scan_files:
                    
                    # Make sure the base file exists
                    if os.path.exists(base_file_name) and os.path.isfile(base_file_name):
                        paired_scan_files[base_file_name] = []
                        paired_scan_files[base_file_name].append(ScanFile(file_path=base_file_name,y_offset=0.0))
                    else:
                        raise Exception(f"Base file for scan {scan_file} missing!")

                # Add to dictionary
                paired_scan_files[base_file_name].append(ScanFile(file_path=scan_file,y_offset=y_offset_value))
        
        # Create menu options to display
        options_to_display = []
        name_and_children_count_delimiter = ' : '
        for base_file_name in paired_scan_files:
                # Add file name and number of offset files
                options_to_display.append(f'{base_file_name}{name_and_children_count_delimiter}{str(len(paired_scan_files[base_file_name]))} offset files')

        # Display a menu with a list of mapped files
        menu = qprompt.enum_menu(options_to_display)
        selected_option = int(menu.show(dft='1', header='Select a file set'))-1
        # Get the option's base file name and trip any white space
        selected_base_file_name = options_to_display[selected_option].split(name_and_children_count_delimiter)[0].rstrip()

        return paired_scan_files[selected_base_file_name]