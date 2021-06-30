import math

from modules import scan, blade, tool_path, helpers
import definitions

class MillSession():
    def __init__(self, blade_instances, tool_instance, debug_mode=definitions.APP_CONFIG["toolPathGenerator"]["debug"]):

        self.blade_instances = blade_instances
        self.debug_mode = debug_mode

        # TODO: Add scanner level to config. Ideally we could convert an angle to two perpendicular shift values
        # Since our scanner is not perfectly level we need to make some further adjustments here
        blade_instances[0].perpendicular_shift(-0.05, recreate_segments=True)

        # Remove blades that are not needed for output
        if (definitions.APP_CONFIG["toolPathGenerator"]["output"]["cut"]["top"] == False):
            blade_instances.pop(0)

        if (definitions.APP_CONFIG["toolPathGenerator"]["output"]["cut"]["bottom"] == False):
            blade_instances.pop(-1)

        self.tool_instance = tool_instance
        # Append gcode initialization lines
        self.tool_path_instance = tool_path.ToolPath()
        self.tool_path_instance.append_initialization_gcode()
        self.tool_path_instance.append_initial_cut_gcode(self.tool_instance)


    def to_json(self):

        return {
            "hollow_radius": self.tool_instance.side_profile_radius
        }
    
    # move to helper function
    def clears_wedge(self, final_cut_profile, wedge_profile, sample_points = 100):
        cut_series = final_cut_profile
        wedge_series = wedge_profile

        raw_step_size = (wedge_series.size - 1) / (sample_points - 1)

        test_x_indices_values = []
        while len(test_x_indices_values) < sample_points:
            raw_index = len(test_x_indices_values) * raw_step_size
            test_x_indices_values.append(math.floor(raw_index))

        for test_index in test_x_indices_values:
            wedge_x = wedge_series.index.values[test_index]
            cut_x = helpers.find_closest_index(wedge_x, cut_series)

            # calculate the y axis diff between cut and wedge
            diff_y = cut_series[cut_x] - wedge_series[wedge_x]
            if(diff_y < 5.0 ):
                return False
        
        return True

class SharpeningSession(MillSession):

    def __init__(self, blade_instances, tool_instance):
        MillSession.__init__(self, blade_instances, tool_instance)
        
        self.cut_blade_instances = list(map(lambda blade_instance: blade_instance.copy(), self.blade_instances))

        # Apply our depth of cut
        list(map(lambda blade_instance: blade_instance.perpendicular_shift(-1 * definitions.APP_CONFIG["toolPathGenerator"]["cut"]["depth"]), self.cut_blade_instances))

        # Plot generated segments 
        if self.debug_mode:
            for blade_index, blade_instance in enumerate(self.blade_instances):
                helpers.plotter_instance.plot_series_generated_segments(blade_instance.profile, [self.cut_blade_instances[blade_index].segments])

        # Create our cut parameters
        cut_SFM = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["SFM"]
        cut_IPT = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["IPT"]
        cut_RPM_multiplier = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["RPMMultiplier"]
        cut_CLF = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["CLF"]
        speeds_and_feeds =  tool_instance.generate_feed_and_speed(cut_SFM, cut_IPT, cut_RPM_multiplier, cut_CLF)
        feedrate = speeds_and_feeds["feed_rate_mm_min"]
        spindle_rpm = speeds_and_feeds["spindle_rpm"]

        # Append gcode lines for cutting
        for blade_index, cut_blade_instance in enumerate(self.cut_blade_instances):
            first_pass = blade_index == 0
            last_pass = blade_index == len(self.cut_blade_instances) - 1
            self.tool_path_instance.sharpen_blade(cut_blade_instance, self.tool_instance.number, first_pass, last_pass, feedrate, spindle_rpm)

        # Append deburr lines
        if (definitions.APP_CONFIG["toolPathGenerator"]["output"]["deburr"]):
            self.tool_path_instance.deburr_blades(self.cut_blade_instances)

        self.tool_path_instance.finalize()
        self.tool_path_instance.export()

    def get_cut_blade_instances(self, blade_index):
        return [self.cut_blade_instances[blade_index].copy()]

    def to_json(self):
        return_json = MillSession.to_json(self)
        return_json["sharpening"] = True
        return return_json

class ToolLifeSession(MillSession):

    def __init__(self, blade_instances, tool_instance, DoC, SFM, IPT, RPM, CLF, NoC):
        MillSession.__init__(self, blade_instances, tool_instance)

        # Create our cut parameters
        cut_SFM = SFM
        cut_IPT = IPT
        cut_RPM_multiplier = RPM
        cut_CLF =CLF
        speeds_and_feeds =  tool_instance.generate_feed_and_speed(cut_SFM, cut_IPT, cut_RPM_multiplier, cut_CLF)
        feedrate = speeds_and_feeds["feed_rate_mm_min"]
        spindle_rpm = speeds_and_feeds["spindle_rpm"]

        # This will hold all our cuts organized by blade
        # 0th index will be top blade
        # 1st index will be bottom blade
        self.blades_cut_instances = [[], []]

        for i in range(NoC):

            cut_blade_instances = list(map(lambda blade_instance: blade_instance.copy(), self.blade_instances))
            # Apply our depth of cut
            list(map(lambda blade_instance: blade_instance.perpendicular_shift(-1 * DoC * (i + 1)), cut_blade_instances))

            self.blades_cut_instances[0].append(cut_blade_instances[0])
            self.blades_cut_instances[1].append(cut_blade_instances[1])

        flat_blade_instances = []
        deburr_blade_instances = []
        for blade_cut_instances in self.blades_cut_instances:
            # Concat all our blades so we cut everything in order
            flat_blade_instances += blade_cut_instances

            # Only add the last blade instance for deburring
            deburr_blade_instances.append(blade_cut_instances[-1])

        # Append gcode lines for cutting
        first_pass = True
        for cut_index, blade_instance in enumerate(flat_blade_instances):

            if cut_index != 0:
                first_pass = False

            last_pass = cut_index == len(flat_blade_instances) - 1

            self.tool_path_instance.sharpen_blade(
                    blade_instance, \
                    self.tool_instance.number, \
                    first_pass, last_pass, \
                    feedrate=feedrate, \
                    spindle_rpm=spindle_rpm \
                )

        # Append deburr lines
        if (definitions.APP_CONFIG["toolPathGenerator"]["output"]["deburr"]):
            self.tool_path_instance.deburr_blades(deburr_blade_instances)

        self.tool_path_instance.finalize()
        self.tool_path_instance.export()

    def get_cut_blade_instances(self, blade_index):
        return list(map(lambda blade_instance: blade_instance.copy(), self.blades_cut_instances[blade_index]))

    def to_json(self):
        return_json = MillSession.to_json(self)
        return_json["sharpening"] = False
        return return_json


class ProfilingSession(MillSession):

    def __init__(self, blade_instances, tool_instance, profiling_specifications):
        MillSession.__init__(self, blade_instances, tool_instance)
        # Needed for to_json method
        self.profiling_specifications = profiling_specifications

        # Convert blade instances to meet our profiling requirements
        local_blade_instances = list(map(lambda blade_instance: blade_instance.copy(), self.blade_instances))

        # Add our depth of cut so we are at least removing this much material everywhere
        list(map(lambda blade_instance: blade_instance.perpendicular_shift(-1 * definitions.APP_CONFIG["toolPathGenerator"]["cut"]["depth"]), local_blade_instances))
        
        # Apply our profiling specifications
        profiled_blade_instances = list(map(lambda blade_instance: blade_instance.add_profiling_specifications(self.profiling_specifications), local_blade_instances))

        self.blades_toolpath_parameters = list(map(lambda blade_instance: blade_instance.generate_toolpath_parameters(tool_instance),profiled_blade_instances))

        # plot all profiling segments
        if definitions.APP_CONFIG["toolPathGenerator"]["debug"]:
            helpers.plotter_instance.plot_blades_toolpath_parameters(self.blades_toolpath_parameters, self.blade_instances)

        flat_toolpath_parameters = []
        deburr_blade_instances = []
        for blade_toolpath_parameters in self.blades_toolpath_parameters:
            # Concat all our blades so we cut everything in order
            flat_toolpath_parameters += blade_toolpath_parameters

            # Only add the last blade instance for deburring
            deburr_blade_instances.append(blade_toolpath_parameters[-1].blade_instance)

        # Append gcode lines for cutting
        last_feedrate = None
        last_spindle_rpm = None
        for cut_index, toolpath_parameters in enumerate(flat_toolpath_parameters):

            if cut_index == 0 or last_feedrate != toolpath_parameters.feedrate or last_spindle_rpm != toolpath_parameters.spindle_rpm:
                first_pass = True
                last_feedrate = toolpath_parameters.feedrate
                last_spindle_rpm = toolpath_parameters.spindle_rpm
            else:
                first_pass = False

            last_pass = cut_index == len(flat_toolpath_parameters) - 1
            self.tool_path_instance.sharpen_blade(
                    toolpath_parameters.blade_instance, \
                    self.tool_instance.number, \
                    first_pass, last_pass, \
                    feedrate=toolpath_parameters.feedrate, \
                    spindle_rpm=toolpath_parameters.spindle_rpm \
                )

        # Append deburr lines
        if (definitions.APP_CONFIG["toolPathGenerator"]["output"]["deburr"]):
            self.tool_path_instance.deburr_blades(deburr_blade_instances)
        
        self.tool_path_instance.finalize()
        self.tool_path_instance.export()

    def get_cut_blade_instances(self, blade_index):
        return list(map(lambda blade_toolpath_parameters: blade_toolpath_parameters.blade_instance, self.blades_toolpath_parameters[blade_index]))


    def to_json(self):
        return_json = MillSession.to_json(self)
        return_json["sharpening"] = False
        return_json["profile_parameters"] = blade.ProfilingSpecification.multiple_to_json(self.profiling_specifications)
        return return_json
