from modules import helpers, segments
from .base import Fixture
import definitions
import pandas as pd
import numpy as np
import math

# TODO: Over time we should improve these approximations
# Current approximation:
# https://docs.google.com/spreadsheets/d/1h8_uY1ovGZvX1A9rwJB_LeZROpR_pQifpS6uedweamE/edit?usp=sharing
blade_center_offsets = [
    {  # linear approximation
        "grip_gap_length": 5.348,
        "size": "212",
        "static_grip_offset": 10.436
    },
    {  # linear approximation
        "grip_gap_length": 11.009,
        "size": "221",
        "static_grip_offset": 7.538
    },
    {  # linear approximation
        "grip_gap_length": 16.67,
        "size": "230",
        "static_grip_offset": 4.64
    },
    {  # linear approximation
        "grip_gap_length": 21.702,
        "size": "238",
        "static_grip_offset": 2.064
    },
    {  # linear approximation
        "grip_gap_length": 26.734,
        "size": "246",
        "static_grip_offset": -0.512
    },
    {  # linear approximation
        "grip_gap_length": 31.766,
        "size": "254",
        "static_grip_offset": -3.088
    },
    {  # measured
        "grip_gap_length": 36.68522,
        "size": "263",
        "static_grip_offset": -5.42036
    },
    {  # measured
        "grip_gap_length": 42.99458,
        "size": "272",
        "static_grip_offset": -8.19404
    },
    {  # measured
        "grip_gap_length": 48.33874,
        "size": "280",
        "static_grip_offset": -13.33246
    },
    {  # measured
        "grip_gap_length": 53.77434,
        "size": "288",
        "static_grip_offset": -14.05636
    },
    {  # linear approximation
        "grip_gap_length": 58.184,
        "size": "296",
        "static_grip_offset": -16.612
    },
    {  # measured
        "grip_gap_length": 63.47968,
        "size": "306",
        "static_grip_offset": -19.13636
    },
]

# Fixture v0.2 by Sefortek


class FixtureM0P2(Fixture):

    def __init__(self, scan_instance):
        Fixture.__init__(self, scan_instance)

    # TODO: When a function is called store it's value in this class in case we need to re-reference it
    def reference_features(self):
        """
        Find the reference features for the provided scan
        """
        # TODO: We need a more reliable method
        # There are too many hardcoded assumptions for this algorithm to work

        # TODO: store search_dataframe a class property so we don't need to keep making it
        # Use z center of dataframe +/- 25
        scan_z_center_index = int(self.scan_instance.accurate_scan_data.shape[1]/2)
        x_reference_search_width = 40
        x_reference_search_lower_limit = int(scan_z_center_index - x_reference_search_width/2)
        x_reference_search_upper_limit = int(scan_z_center_index + x_reference_search_width/2)
        blade_dataframe = self.scan_instance.trimmed_scan_dataframe.copy()
        search_dataframe = blade_dataframe.loc[:,x_reference_search_lower_limit:x_reference_search_upper_limit]

        # Convert 3D dataframe to 2D median series to help reduce noise
        raw_median_series = search_dataframe.median(axis=1)

        # Split into main sections
        # 0th index is heel blade grip
        # 1st index is toe blade grip
        # If there are 3 sections the 1st index is an artifact we need to ignore
        main_sections = helpers.split_series_into_sections(raw_median_series, min_section_gap_mm=5.0)
        heel_grip_series = main_sections[0]
        toe_grip_series = main_sections[-1]

        dynamic_reference_feature_index = heel_grip_series.index.values[-1]
        print(f"Dynamic x-axis reference feature index: {dynamic_reference_feature_index}")

        static_end_reference_feature_index = toe_grip_series.index.values[0]
        print(f"Static end x-axis reference feature index: {static_end_reference_feature_index}")

        above_negative_one_series = toe_grip_series.where(toe_grip_series > -1)
        toe_grip_series_sections = helpers.split_series_into_sections(above_negative_one_series)
        start_edge_of_circle = toe_grip_series_sections[1].index.values[-1]
        end_edge_of_circle = toe_grip_series_sections[2].index.values[0]
        static_circle_reference_feature_index = int((start_edge_of_circle + end_edge_of_circle) / 2)
        print(f"Static circle x-axis reference feature index: {static_circle_reference_feature_index}")

        return {
            "x": {
                "static-end": static_end_reference_feature_index,
                "static-circle": static_circle_reference_feature_index,
                "dynamic": dynamic_reference_feature_index
            }
        }

    def blade_profile(self, blade_center_z_index, debug_mode=False):
        """
        Finds a blade profile series at a given z height
        """
        blade_dataframe = self.scan_instance.trimmed_scan_dataframe.copy()
        raw_blade_dataframe = blade_dataframe.loc[:,blade_center_z_index - 10:blade_center_z_index + 10]
        # plot raw 3D
        # plotter_instance.quick_plot_3d(raw_blade_dataframe, "Raw Blade")
        # median
        # interpolate
        median_blade_series = raw_blade_dataframe.median(axis=1)
        interpolated_blade_series = median_blade_series.interpolate(method='akima')

        # Clean up noise
        noise_omitted_scan_slice = helpers.omit_series_noise(interpolated_blade_series, 0.1, method="1_diff")
        cleaned_scan_slice = helpers.set_nan_sections(noise_omitted_scan_slice, np.NaN).interpolate(method='akima')
        
        # Remove trailing and leading NaNs
        trimmed_scan_slice = helpers.omit_edge_nan_data(cleaned_scan_slice)

        smoothed_scan_slice_opt_1 = helpers.apply_savgol_filter(trimmed_scan_slice, 51, 2)
        smoothed_scan_slice_opt_2 = helpers.apply_savgol_filter(smoothed_scan_slice_opt_1, 103, 2)
        smoothed_scan_slice_opt_3 = helpers.apply_savgol_filter(smoothed_scan_slice_opt_2, 153, 2)

        # TODO: Improve accuracy
        # find length of blade
        blade_length = (smoothed_scan_slice_opt_3.index.stop - smoothed_scan_slice_opt_3.index.start) * definitions.SPACING_PER_X_INDEX
        print(f"Blade found of length of {round(blade_length,3)}mm")

        # find blade ends angles
        blade_start_y_diff = abs(smoothed_scan_slice_opt_3.iloc[1] - smoothed_scan_slice_opt_3.iloc[0])
        blade_start_x_diff = definitions.SPACING_PER_X_INDEX
        blade_start_angle = math.atan(blade_start_y_diff/blade_start_x_diff) * 180 / math.pi

        blade_end_y_diff = abs(smoothed_scan_slice_opt_3.iloc[-1] - smoothed_scan_slice_opt_3.iloc[-2])
        blade_end_X_diff = definitions.SPACING_PER_X_INDEX
        blade_end_angle = math.atan(blade_end_y_diff/blade_end_X_diff) * 180 / math.pi

        print(f"Blade has a start angle of {round(blade_start_angle,3)}° and an end angle of {round(blade_end_angle,3)}°")

        if debug_mode:
            profile_figure = helpers.plotter_instance.create_figure_2d("Blade Profile")
            helpers.plotter_instance.add_pd_series(profile_figure, median_blade_series, "Raw Median Profile")
            helpers.plotter_instance.add_pd_series(profile_figure, interpolated_blade_series, "Raw Interpolated Profile")
            helpers.plotter_instance.add_pd_series(profile_figure, cleaned_scan_slice, "Cleaned Profile")
            helpers.plotter_instance.add_pd_series(profile_figure, smoothed_scan_slice_opt_3, "Smooth Profile")
            profile_figure.show()

        return smoothed_scan_slice_opt_3

    def blade_center_x(self):
        """
        Finds the x-axis center of blades
        """

        reference_features = self.reference_features()

        # The dynamic_grip_end is the dynamic blade grips edge closest to the center of the fixture
        # The static_grip_end is the static blade grips edge closest to the center of the fixture
        static_grip_end = reference_features["x"]["static-end"] * \
            definitions.SPACING_PER_X_INDEX
        dynamic_grip_end = reference_features["x"]["dynamic"] * \
            definitions.SPACING_PER_X_INDEX
        blade_grips_gap_length = static_grip_end - dynamic_grip_end

        # Find closest blade center length object
        best_blade_center_offset = blade_center_offsets[0]
        for blade_center_offset in blade_center_offsets:

            # See if the next blade_center_offset object is a better match than our current best match
            if abs(best_blade_center_offset["grip_gap_length"] - blade_grips_gap_length) > abs(blade_center_offset["grip_gap_length"] - blade_grips_gap_length):
                best_blade_center_offset = blade_center_offset

        print(f"Blade of size {best_blade_center_offset['size']} detected")
        print(f"Blade grip length of {blade_grips_gap_length}mm")

        return static_grip_end + best_blade_center_offset["static_grip_offset"]

    def middle_plate_center_groove(self):
        # TODO: store search_dataframe a class property so we don't need to keep making it
        # Use z center of dataframe +/- 25
        scan_z_center_index = int(self.scan_instance.accurate_scan_data.shape[1]/2)
        x_reference_search_width = 40
        x_reference_search_lower_limit = int(scan_z_center_index - x_reference_search_width/2)
        x_reference_search_upper_limit = int(scan_z_center_index + x_reference_search_width/2)
        blade_dataframe = self.scan_instance.trimmed_scan_dataframe.copy()
        search_dataframe = blade_dataframe.loc[:,x_reference_search_lower_limit:x_reference_search_upper_limit]

        # Convert 3D dataframe to 2D median series to help reduce noise
        raw_median_series = search_dataframe.median(axis=1)

        # Split into main sections
        # 0th index is heel blade grip
        # 1st index is toe blade grip
        # If there are 3 sections the 1st index is an artifact we need to ignore
        main_sections = helpers.split_series_into_sections(raw_median_series, min_section_gap_mm=5.0)
        heel_grip_series = main_sections[0]

        # Remove all values less than -0.75
        z_trimmed_heel_grip_series = heel_grip_series.where(heel_grip_series > -0.75)

        # Trim edge nan indices
        x_trimmed_heel_grip_series = helpers.omit_edge_nan_data(z_trimmed_heel_grip_series)

        # From end index sample with offsets [-20, [-70, -170]]
        end_x_sample_offsets = []
        end_x_sample_offsets += [-20] # Closest safe point from end index
        end_x_sample_offsets += list(range(-70,-170, -25)) + [-170] # Up till right screw
        end_x_sample_offsets += list(range(-425,-600, -25)) + [-620] # From right screw till right blade marking
        end_x_sample_offsets += list(range(-650,-950, -25)) + [-950] # From right blade marking till left blade marking
        end_x_sample_offsets += [-1030] # Add last safe point past the left blade marking
        end_x_index = x_trimmed_heel_grip_series.index.values[-1]

        sample_points = []

        for offset in end_x_sample_offsets:
            sample_points.append(end_x_index + offset)

        x_values = []
        z_values = []

        for sample_point in sample_points:
            x_values.append(sample_point * definitions.SPACING_PER_X_INDEX)
            z_values.append((find_circle_groove_center(search_dataframe, sample_point) - 400) * definitions.SPACING_PER_Z_INDEX )

        z_series = pd.Series(data=z_values, index=x_values)
        smoothed_z_series = helpers.apply_savgol_filter(z_series, 15, 2)

        # TODO: Convert this to a list of segments
        return segments.Polynomial.find_best_polynomial(smoothed_z_series)


def find_circle_groove_center(search_dataframe, series_index, median_window_width=7):
    
    median_window_upper_index = series_index + median_window_width
    median_window_lower_index = series_index - median_window_width

    scan_slice = search_dataframe.loc[median_window_lower_index:median_window_upper_index,:].median()
    
    smoothed_scan_slice = helpers.apply_savgol_filter(scan_slice, 11, 2)

    center_index = smoothed_scan_slice.idxmin()

    return center_index