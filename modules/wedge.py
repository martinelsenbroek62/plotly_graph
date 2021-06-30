import math

import numpy as np
import pandas as pd

from modules import helpers
import definitions

class Wedge():
    """
    Args:
    """

    def __init__(self, top, size, profile):

        self.top = top
        self.size = size
        self.profile = profile.copy()

    def create_web_profile_json(self, number_of_points=1000):
        return helpers.pd_series_to_json(self.profile, number_of_points)

    def to_json(self):
        return {
            "top": self.top,
            "size": self.size,
            "profile": helpers.pd_series_to_json(self.profile)
        }

    @staticmethod
    def from_stored_wedge(stored_json):
        profile_series = pd.Series(data=stored_json["profile"]["y"], index=stored_json["profile"]["x"])
        top = stored_json["top"]
        size = stored_json["size"]

        return Wedge(top, size, profile_series)

    @staticmethod
    def parse_from_fixture_instance(fixture_instance, blade_center_z_index, x_axis_offset, top):
        
        wedge_size = fixture_instance.get_wedge_size_dictionary(blade_center_z_index)["wedge_size"]
        wedge_profile = fixture_instance.wedge_profile(blade_center_z_index)
        wedge_profile_mm = helpers.rename_series_indices(wedge_profile.copy(), definitions.SPACING_PER_X_INDEX)
        adjusted_wedge_profile_mm = helpers.apply_scanner_offsets(wedge_profile_mm, x_axis_offset)        
        return Wedge(top, wedge_size, adjusted_wedge_profile_mm)