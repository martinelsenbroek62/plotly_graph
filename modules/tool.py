import math

import definitions

from modules import helpers

class Tool():
    def __init__(self, number, form, diameter, number_of_flutes=1, side_profile_radius=None):
        self.number = number
        self.form = form
        self.diameter = diameter
        self.radius = diameter/2

        self.number_of_flutes = number_of_flutes
        self.side_profile_radius = side_profile_radius

    def generate_feed_and_speed(self, SFM, IPT, RPMMultiplier, CLF):

        tool_diameter_inch = self.diameter / definitions.MM_PER_INCH

        spindle_rpm = (SFM / (tool_diameter_inch * math.pi)) * definitions.INCH_PER_FT
        feed_rate_inch_min = spindle_rpm * CLF * self.number_of_flutes * IPT
        feed_rate_mm_min = feed_rate_inch_min * definitions.MM_PER_INCH
        final_spindle_rpm = spindle_rpm * RPMMultiplier

        return {
            "spindle_rpm": round(final_spindle_rpm, 2),
            "feed_rate_inch_min": round(feed_rate_inch_min, 2),
            "feed_rate_mm_min": round(feed_rate_mm_min, 2)
        }

    @staticmethod
    def from_dictionary(dictionary_object):

        return Tool(
            dictionary_object.get("number"),
            dictionary_object.get("form"),
            dictionary_object.get("diameter"),
            number_of_flutes=dictionary_object.get("numberOfFlutes"),
            side_profile_radius = dictionary_object.get("sideProfileRadius"),
        )

    @staticmethod
    def from_hollow_radius(hollow_radius):
        available_tools = definitions.APP_CONFIG["mill"]["tools"]

        for tool_dict in available_tools:

            tool_instance = Tool.from_dictionary(tool_dict)

            # If tool has no side profile radius ignore it
            if tool_instance.side_profile_radius is None:
                continue

            if helpers.float_equivalence(hollow_radius, tool_instance.side_profile_radius):
                return tool_instance

        # Return None if nothing found
        return None

    @staticmethod
    def from_tool_number(tool_number):
        available_tools = definitions.APP_CONFIG["mill"]["tools"]

        for tool_dict in available_tools:

            if tool_dict["number"] == tool_number:

                tool_instance = Tool.from_dictionary(tool_dict)
                return tool_instance

        # Return None if nothing found
        return None