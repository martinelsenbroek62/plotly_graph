from .base import Fixture

from modules import helpers

import pandas as pd
import numpy as np
import math

import definitions

# Fixture v0.3 by LAC

wedge_sizes = [
    {
        "wedge_size": 272,
        "wedge_length": 290.0
    },
    {
        "wedge_size": 296,
        "wedge_length": 316.0
    }
]

blade_sizes = [
    {
        "blade_size": 212,
        "blade_length": 220, #NOT TESTED
        "main_working_radius_length": 125,
        "offset": 120.65, #NOT TESTED
        "supported_wedge_size": 272 #NOT TESTED
    },
    {
        "blade_size": 221,
        "blade_length": 230, #NOT TESTED
        "main_working_radius_length": 130,
        "offset": 120.65, #NOT TESTED
        "supported_wedge_size": 272 #NOT TESTED
    },
    {
        "blade_size": 230,
        "blade_length": 240, #NOT TESTED
        "main_working_radius_length": 141,
        "offset": 120.65, #NOT TESTED
        "supported_wedge_size": 272 #NOT TESTED
    },
    {
        "blade_size": 238,
        "blade_length": 246, #NOT TESTED
        "main_working_radius_length": 144,
        "offset": 120.65, #NOT TESTED
        "supported_wedge_size": 272 #NOT TESTED
    },
    {
        "blade_size": 246,
        "blade_length": 252, #NOT TESTED
        "main_working_radius_length": 151,
        "offset": 120.65, #NOT TESTED
        "supported_wedge_size": 272 #NOT TESTED
    },
    {
        "blade_size": 254,
        "blade_length": 268,
        "main_working_radius_length": 155,
        "offset": 120.65,
        "supported_wedge_size": 272
    },
    {
        "blade_size": 263,
        "blade_length": 272,
        "main_working_radius_length": 159,
        "offset": 119.38,
        "supported_wedge_size": 272
    },
    {
        "blade_size": 272,
        "blade_length": 280,
        "main_working_radius_length": 165,
        "offset": 114.3,
        "supported_wedge_size": 272
    },
    {
        "blade_size": 280,
        "blade_length": 292,
        "main_working_radius_length": 169,
        "offset": 109.22,
        "supported_wedge_size": 272
    },
    {
        "blade_size": 288,
        "blade_length": 306,
        "main_working_radius_length": 174,
        "offset": 107.95, #116.84
        "supported_wedge_size":272 #296
    },
    {
        "blade_size": 296,
        "blade_length": 311,
        "main_working_radius_length": 185,
        "offset": 114.3,
        "supported_wedge_size":296
    },
    {
        "blade_size": 306,
        "blade_length": 330,
        "offset": 111.76,
        "main_working_radius_length": 194,
        "supported_wedge_size":296
    }
]

class FixtureM0P3(Fixture):

    def __init__(self, scan_instance):
        Fixture.__init__(self, scan_instance)
        
        self.stored_reference_feature = None
        self.wedge_profiles = {}
        self.wedge_size_dictionaries = {}
        self.blade_rough_profiles = {}
        self.blade_smooth_profiles = {}
        self.blade_size_dictionaries = {}

    def get_wedge_size_dictionary(self, blade_center_z_index):

        blade_z_index_string = str(blade_center_z_index)
        if blade_z_index_string in self.wedge_size_dictionaries:
            return self.wedge_size_dictionaries[blade_z_index_string]

        wedge_z_index = blade_z_index_to_wedge_index(blade_center_z_index)

        wedge_dataframe = self.scan_instance.trimmed_scan_dataframe.copy()
        raw_wedge_dataframe = wedge_dataframe.loc[:,wedge_z_index - 10:wedge_z_index + 10]
        median_wedge_series = raw_wedge_dataframe.median(axis=1)
        edge_omitted_series = helpers.omit_edge_nan_data(median_wedge_series)
        wedge_length = edge_omitted_series.size * definitions.SPACING_PER_X_INDEX
        
        # helpers.plotter_instance.plot_series_set([
        #     {
        #         "name": "median_wedge_series",
        #         "data": median_wedge_series
        #     },
        #     {
        #         "name": "edge_omitted_series",
        #         "data": edge_omitted_series
        #     },
        #     {
        #         "name": "smoothed_wedge_scan_slice",
        #         "data": smoothed_wedge_scan_slice
        #     }
        # ], lock_aspect_ratio=False)

        # Find closest wedge object
        best_wedge = wedge_sizes[0]
        for wedge in wedge_sizes:

            # See if the next blade_center_offset object is a better match than our current best match
            if abs(best_wedge["wedge_length"] - wedge_length) > abs(wedge["wedge_length"] - wedge_length):
                best_wedge = wedge
        
        print(f"Wedge of size {best_wedge['wedge_size']} detected at z {wedge_z_index}")

        self.wedge_size_dictionaries[blade_z_index_string] = best_wedge
        return self.wedge_size_dictionaries[blade_z_index_string]

    def wedge_profile(self, blade_center_z_index):
        
        blade_z_index_string = str(blade_center_z_index)
        if blade_z_index_string in self.wedge_profiles:
            return self.wedge_profiles[blade_z_index_string]

        wedge_z_index = blade_z_index_to_wedge_index(blade_center_z_index)

        wedge_dataframe = self.scan_instance.trimmed_scan_dataframe.copy()
        raw_wedge_dataframe = wedge_dataframe.loc[:,wedge_z_index - 10:wedge_z_index + 10]
        median_wedge_series = raw_wedge_dataframe.median(axis=1)
        edge_omitted_series = helpers.omit_edge_nan_data(median_wedge_series)
        interpolated_series = edge_omitted_series.interpolate(method="akima")
        smoothed_wedge_scan_slice = helpers.apply_savgol_filter(interpolated_series, 51, 2)
        
        # helpers.plotter_instance.plot_series_set([
        #     {
        #         "name": "median_wedge_series",
        #         "data": median_wedge_series
        #     },
        #     {
        #         "name": "edge_omitted_series",
        #         "data": edge_omitted_series
        #     },
        #     {
        #         "name": "smoothed_wedge_scan_slice",
        #         "data": smoothed_wedge_scan_slice
        #     }
        # ], lock_aspect_ratio=False)

        self.wedge_profiles[blade_z_index_string] = smoothed_wedge_scan_slice
        return self.wedge_profiles[blade_z_index_string]

    def get_blade_size_dictionary(self, blade_center_z_index):

        blade_center_z_index_string = str(blade_center_z_index)
        if blade_center_z_index_string in self.blade_size_dictionaries:
            return self.blade_size_dictionaries[blade_center_z_index_string]

        blade_profile = self.blade_rough_profile(blade_center_z_index)

        # find length of blade
        blade_length = blade_profile.size * definitions.SPACING_PER_X_INDEX
        print(f"Blade found of length of {round(blade_length,3)}mm")

        # find blade ends angles
        blade_start_y_diff = abs(
            blade_profile.iloc[1] - blade_profile.iloc[0])
        blade_start_x_diff = definitions.SPACING_PER_X_INDEX
        blade_start_angle = math.atan(
            blade_start_y_diff/blade_start_x_diff) * 180 / math.pi

        blade_end_y_diff = abs(
            blade_profile.iloc[-1] - blade_profile.iloc[-2])
        blade_end_X_diff = definitions.SPACING_PER_X_INDEX
        blade_end_angle = math.atan(
            blade_end_y_diff/blade_end_X_diff) * 180 / math.pi

        print(f"Blade has a start angle of {round(blade_start_angle,3)}° and an end angle of {round(blade_end_angle,3)}°")

        # Find closest blade size object
        best_blade_size = blade_sizes[0]
        for blade_size in blade_sizes:

            # See if the next blade_center_offset object is a better match than our current best match
            if abs(best_blade_size["blade_length"] - blade_length) > abs(blade_size["blade_length"] - blade_length):
                best_blade_size = blade_size
        
        print(f"Blade of size {best_blade_size['blade_size']} detected")

        self.blade_size_dictionaries[blade_center_z_index_string] = best_blade_size
        return self.blade_size_dictionaries[blade_center_z_index_string]


    # TODO: When a function is called store it's value in this class in case we need to re-reference it
    def reference_features(self):
        """
        Find the reference features for the provided scan
        """

        if self.stored_reference_feature is not None:
            return self.stored_reference_feature

        # Only has one x axis static feature
        # We will use the right edge of the receivers
        # https://imgur.com/r45iiNb.png

        upper_limit = 395
        lower_limit = 385
        search_dataframe = self.scan_instance.trimmed_scan_dataframe.loc[:,lower_limit:upper_limit]
        search_series = search_dataframe.median(axis=1)
        
        # When using larger wedges the screws used to add extra clamping force at the edges show up in the scan
        # We need to ignore them so we split the series into sections with a large gap tolerance so the largest
        # section will be our fixture.
        screws_split_series = helpers.split_series_into_sections(search_series, min_section_gap_mm=5.0)
        fixture_series = helpers.find_largest_series_section(screws_split_series)

        fixture_series_median = fixture_series.median()
        median_delta = 0.2
        
        upper_trimmed_series = fixture_series.where(fixture_series < fixture_series_median + median_delta)
        lower_trimmed_series = upper_trimmed_series.where(upper_trimmed_series > fixture_series_median - median_delta)

        edge_omitted_series = helpers.omit_edge_nan_data(lower_trimmed_series)

        fixture_left_edge = edge_omitted_series.index[0]
        fixture_right_edge = edge_omitted_series.index[-1]

        print(f"Fixture length of {abs(fixture_left_edge - fixture_right_edge) * definitions.SPACING_PER_X_INDEX}")

        self.stored_reference_feature = {
            "x": fixture_left_edge
        }
        return self.stored_reference_feature

    
    def blade_rough_profile(self, blade_center_z_index, debug_mode=False):
        blade_center_z_index_string = str(blade_center_z_index)
        if blade_center_z_index_string in self.blade_rough_profiles:
            return self.blade_rough_profiles[blade_center_z_index_string]

        blade_dataframe = self.scan_instance.trimmed_scan_dataframe.copy()
        raw_blade_dataframe = blade_dataframe.loc[:,blade_center_z_index - 20:blade_center_z_index + 20]
        median_blade_series = raw_blade_dataframe.median(axis=1)

        rollowing_omitted_series = helpers.loop_rolling_series_omit_series_noise(median_blade_series, 0.15, 5, threshold_mutation_factor=1.05, upward_jump_limit=3.0)
        rollowing_omitted_series = helpers.omit_edge_nan_data(rollowing_omitted_series)

        edge_trimmed_series = helpers.trim_to_toe_and_heel_requirements(rollowing_omitted_series, heel_angle=25.0, toe_angle=50.0)

        if debug_mode:

            helpers.plotter_instance.plot_series_set([
                {
                    "name": "median_blade_series",
                    "data": median_blade_series
                },
                {
                    "name": "edge_trimmed_series",
                    "data": edge_trimmed_series
                }
            ], lock_aspect_ratio=False)

        self.blade_rough_profiles[blade_center_z_index_string] = edge_trimmed_series
        return self.blade_rough_profiles[blade_center_z_index_string]

    def blade_smooth_profile(self, blade_center_z_index, debug_mode=False):
        """
        Finds a blade profile series at a given z height
        """
        
        blade_center_z_index_string = str(blade_center_z_index)
        if blade_center_z_index_string in self.blade_smooth_profiles:
            return self.blade_smooth_profiles[blade_center_z_index_string]

        edge_trimmed_series = self.blade_rough_profile(blade_center_z_index)

        # Smoothing
        # Find heel and toe points
        blade_size_dict = self.get_blade_size_dictionary(blade_center_z_index)
        x_axis_reference_feature = self.reference_features()["x"]
        blade_center_x_raw_index = x_axis_reference_feature + (blade_size_dict["offset"] / definitions.SPACING_PER_X_INDEX)
        
        main_working_radius_length = blade_size_dict["main_working_radius_length"]
        main_working_radius_indices = main_working_radius_length / definitions.SPACING_PER_X_INDEX

        blade_toe_x_raw_index = blade_center_x_raw_index + (main_working_radius_indices/2)
        blade_heel_x_raw_index = blade_center_x_raw_index - (main_working_radius_indices/2)

        blade_toe_x_index = helpers.find_closest_index(blade_toe_x_raw_index, edge_trimmed_series)
        blade_heel_x_index = helpers.find_closest_index(blade_heel_x_raw_index, edge_trimmed_series)

        smoothed_heel = helpers.apply_savgol_filter_section(edge_trimmed_series, edge_trimmed_series.index.values[0], blade_heel_x_index, 41)
        smoothed_toe = helpers.apply_savgol_filter_section(smoothed_heel, blade_toe_x_index, edge_trimmed_series.index.values[-1], 81)
        smoothed_scan_slice = helpers.apply_savgol_filter_section(smoothed_toe, blade_heel_x_index, blade_toe_x_index, 151)

        # When doing this section by secion smoothing there will be jumps between the segments
        # This should be interpolated out

        # interpolate heel joining sections
        heel_interpolated_series = helpers.interpolate_series_at_index(smoothed_scan_slice, blade_heel_x_index)

        # interpolate toe joining sections
        smoothed_scan_slice = helpers.interpolate_series_at_index(heel_interpolated_series, blade_toe_x_index)

        if debug_mode:

            helpers.plotter_instance.plot_series_set([
                {
                    "name": "edge_trimmed_series",
                    "data": edge_trimmed_series
                },
                {
                    "name": "smoothed_scan_slice",
                    "data": smoothed_scan_slice
                }
            ], lock_aspect_ratio=False)

        self.blade_smooth_profiles[blade_center_z_index_string] = smoothed_scan_slice
        return self.blade_smooth_profiles[blade_center_z_index_string]


    def blade_center_x(self, blade_center_z_index):

        # Use the left edge of the receiver tower as our origin
        x_axis_reference_feature = self.reference_features()["x"]
        blade_size = self.get_blade_size_dictionary(blade_center_z_index)
        blade_x_center = x_axis_reference_feature * definitions.SPACING_PER_X_INDEX + blade_size["offset"]
        return blade_x_center


    def blade_size(self, blade_center_z_index):

        wedge_size = self.get_wedge_size_dictionary(blade_center_z_index)
        blade_size = self.get_blade_size_dictionary(blade_center_z_index)

        # wedge and blade compatibility test
        assert blade_size["supported_wedge_size"] == wedge_size["wedge_size"], f"Blade size {blade_size['blade_size']} should not be loaded in {wedge_size['wedge_size']} wedge"

        return blade_size["blade_size"]


    def blade_z_level(self, blade_center_z_index, top = True):

        blade_size = self.get_blade_size_dictionary(blade_center_z_index)

        z_shape_dictionary = definitions.APP_CONFIG["toolPathGenerator"]["fixture"]["cartridge"]["zShapes"]

        blade_size_string = str(blade_size["blade_size"])
        if blade_size_string in z_shape_dictionary.keys():
            if top:
                return z_shape_dictionary[blade_size_string]["top"]
            else:
                return z_shape_dictionary[blade_size_string]["bottom"]
        
        else:

            raise Exception(f"Blade size {blade_size_string} has no z shape value!")
            

def blade_z_index_to_wedge_index(blade_z_index):
    if (blade_z_index > 400):
        return blade_z_index - 60
    else:
        return blade_z_index + 60