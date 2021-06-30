import numpy as np
import pandas as pd
import math
import copy

from modules import helpers, gcode, segments, tool
import definitions

class ToolPath():
    """ 
    Stores scan data and contains all scanning methods.

    Arguments:
        scan_instance {Scan} -- A scan class instance to convert into gcode.
        cutter_tool_number {int} -- What tool index in the ATC our desired cutting 
         head is on.
        spindle_rpm {int} -- RPM of spindle to use for cut.

    Keyword Arguments:
        debug_mode {Boolean} -- Enable extra output for troubleshooting.
         (default: {False})
    """

    def __init__(
                self, \
                debug_mode=definitions.APP_CONFIG["toolPathGenerator"]["debug"]
            ):

        self.debug_mode = debug_mode

        self.gcode_instance = gcode.Gcode()
    
    #############
    # Interface #
    #############

    def sharpen_blade(self, blade, tool_number, first_pass, last_pass, feedrate, spindle_rpm):

        
        local_blade = blade.copy()
        # Add cutter compensation
        tool_instance = tool.Tool.from_tool_number(tool_number)
        local_blade.perpendicular_shift(tool_instance.radius, recreate_segments=True)

        # Create tool path segments
        blade_tool_path_segments, gcode_boundaries = ToolPath.generate_blade_tool_path_segments(local_blade)
        
        # Add cut lines
        self.append_tool_path_segments(blade_tool_path_segments, first_pass, last_pass, spindle_rpm, tool_number, feedrate)
    
    @staticmethod
    def profile_blade(blades, gcode_instance, tool_number, tool_radius_x_y):
        # Add setup for cut lines

        # For each blade
        # Create gcode segments
        # Add cut lines

        print("not ready")

    def deburr_blades(self, blades):

        self.gcode_instance.append_line("(Deburring)")

        deburr_tool_number = definitions.APP_CONFIG["toolPathGenerator"]["deburr"]["toolNumber"]
        deburr_tool_instance = tool.Tool.from_tool_number(deburr_tool_number)
        assert deburr_tool_instance is not None, f"No deburr tool available with a number of {deburr_tool_number}"
        assert deburr_tool_instance.form == "deburr_brush", f"Can not use a {deburr_tool_instance.form} to deburr. Update the deburr tool number to a deburr brush."

        spindle_rpm = definitions.APP_CONFIG["toolPathGenerator"]["deburr"]["spindleRpm"]
        feedrate = definitions.APP_CONFIG["toolPathGenerator"]["deburr"]["feedrate"]
        
        # Append setup for deburr lines
        self.append_initial_cut_gcode(deburr_tool_instance)
        
        # For each blade
        for blade_index, blade in enumerate(blades):

            first_pass = blade_index == 0
            last_pass = blade_index == len(blades) - 1

            # Add our cutter compensation manually
            local_blade = blade.copy()
            local_blade.perpendicular_shift(deburr_tool_instance.radius, recreate_segments=True)

            # Create tool path segments
            blade_tool_path_segments, gcode_boundaries = ToolPath.generate_blade_tool_path_segments(local_blade)
            
            # Append deburr lines
            self.append_tool_path_segments(blade_tool_path_segments, first_pass, last_pass, spindle_rpm, deburr_tool_number, feedrate, clockwise_rotation = False)

    def finalize(self):
        self.gcode_instance.append_line("M2")

    def export(self):

        self.gcode_instance.export()

    ######################
    # Segment Processing #
    ######################

    @staticmethod
    def generate_blade_tool_path_segments(blade):
        # Break up sections into segments
        blade_segments = blade.segments

        tool_path_segments = []
        gcode_boundaries = []

        # Add lead in and lead out
        leadin_points, leadout_points = ToolPath.generate_leadinout(blade_segments, blade.z_shape)
        # leadin_points, leadout_points = ToolPath.generate_horizontal_leadinout(blade_segments, blade.z_shape)

        # Add initial leadin points
        tool_path_segments.append(ToolPathSegment(gcode_command="G01", points=leadin_points))

        for segment_index, blade_segment in enumerate(blade_segments):

            boundary = {
                "segment": blade_segment,
                "start_index": len(tool_path_segments),
            }

            tool_path_segments += ToolPath.create_tool_path_segments(segment_index, blade_segments, blade.z_shape)

            boundary["end_index"] = len(tool_path_segments) - 1
            gcode_boundaries.append(boundary)
            

        # Add final leadout points
        tool_path_segments.append(ToolPathSegment(gcode_command="G01", points=leadout_points))

        return tool_path_segments, gcode_boundaries

    @staticmethod
    def create_blade_segment_points(segment, blade_z_shape):
        # Create a range of x values
        segment_length = segment.get_end() - segment.get_start()
        number_of_points = math.floor(segment_length / 0.1)
        
        segment_start_x = segment.get_start()
        segment_end_x = segment.get_end()

        # Generate a range of x values of which our segment will get evaluated over
        x_domain = np.linspace(segment_start_x, segment_end_x, number_of_points)

        # Convert segments into a set of points
        segment_points = []

        for x_value in x_domain:

            y_value = segment.y(x_value)
            z_value = blade_z_shape.y(x_value)
            segment_points.append(Point(x_value, y_value, z_value))

        # TODO: Having the main radius segment index hardcoded here is terrible as it will not scale with additional segments
        # If segment to the right of the circle remove the first point
        # if segment_index == 2:
        #     segment_points = segment_points[1:]

        return segment_points

    @staticmethod
    def create_blade_circle_segment(segment, blade_z_shape, previous_segment, previous_gcode_point=None):

        if previous_gcode_point is None:
            if previous_segment.get_shape() == "Circle" or previous_segment.get_shape() == "CubicBezier":

                # Since our circle segments have x and y end values we can just use them
                previous_gcode_point = previous_segment

            else:

                # Get last point from series of points for a G01
                previous_segment_points = ToolPath.create_blade_segment_points(previous_segment, blade_z_shape)
                previous_gcode_point = previous_segment_points[-1]

        return ToolPath.create_blade_circle_tool_path_segment(segment, previous_gcode_point, blade_z_shape)

    @staticmethod
    def create_blade_circle_tool_path_segment(segment, previous_tool_segment, blade_z_shape):

        # Find last point of circle
        last_point_x = segment.get_end()
        last_point_y = segment.y(last_point_x)
        last_point_z = blade_z_shape.y(last_point_x)

        # Find delta from start point of circle and circle center
        first_point_to_circle_center_delta_x = segment.raw_shape.x0 - previous_tool_segment.x
        first_point_to_circle_center_delta_y = segment.raw_shape.y0 - previous_tool_segment.y

        circle_segment_type = "G02" if segment.raw_shape.radicand == 1 else "G03"

        return ToolPathSegment(gcode_command=circle_segment_type, x=last_point_x, y=last_point_y, z=last_point_z, i=first_point_to_circle_center_delta_x, j=first_point_to_circle_center_delta_y)

    @staticmethod
    def generate_leadinout(segments, blade_z_shape):

        # return a start and end point to append and prepend to our cutting steps
        first_segment = segments[0]
        last_segment = segments[len(segments) - 1]
        blade_first_x = first_segment.get_start()
        blade_first_y = first_segment.y(blade_first_x)
        blade_first_z = blade_z_shape.y(blade_first_x)

        blade_last_x = last_segment.get_end()
        blade_last_y = last_segment.y(blade_last_x)
        blade_last_z = blade_z_shape.y(blade_last_x)

        blade_first_point = Point(blade_first_x, blade_first_y, blade_first_z)
        blade_last_point = Point(blade_last_x, blade_last_y, blade_last_z)

        # Find derivative of start and end of blade segments
        # Get derivative of first segments and use the first value for the leadin
        # Get derivative of last segments and use the last value for the loadout
        leadin_derivative = first_segment.derivative(blade_first_point.x)
        leadout_derivative = last_segment.derivative(blade_last_point.x)
        
        # Use half of the leadinout length so we can generate two points instead
        # For the leadin the second point will apply the cutter compensation
        # For the leadout the first point will remove the cutter compensation
        half_leadinout_x_length = definitions.APP_CONFIG["toolPathGenerator"]["leadInOutXLength"] / 2

        leadinout_offsets = []
        # Lead in offsets
        leadinout_offsets.append(Point(-half_leadinout_x_length, leadin_derivative * -half_leadinout_x_length, 0))

        # Lead out offsets
        leadinout_offsets.append(Point(half_leadinout_x_length, leadout_derivative * half_leadinout_x_length, 0))

        leadin_point_1_x = blade_first_point.x + 2 * leadinout_offsets[0].x
        leadin_point_1_y = blade_first_point.y + 2 * leadinout_offsets[0].y
        leadin_point_1_z = blade_z_shape.y(leadin_point_1_x)

        leadin_point_2_x = blade_first_point.x + leadinout_offsets[0].x
        leadin_point_2_y = blade_first_point.y + leadinout_offsets[0].y
        leadin_point_2_z = blade_z_shape.y(leadin_point_2_x)

        leadout_point_1_x = blade_last_point.x + leadinout_offsets[1].x
        leadout_point_1_y = blade_last_point.y + leadinout_offsets[1].y
        leadout_point_1_z = blade_z_shape.y(leadout_point_1_x)

        leadout_point_2_x = blade_last_point.x + 2 * leadinout_offsets[1].x
        leadout_point_2_y = blade_last_point.y + 2 * leadinout_offsets[1].y
        leadout_point_2_z = blade_z_shape.y(leadout_point_2_x)

        leadin_points = [
            Point(leadin_point_1_x, leadin_point_1_y, leadin_point_1_z),
            Point(leadin_point_2_x, leadin_point_2_y, leadin_point_2_z),
        ]
        leadout_points = [
            Point(leadout_point_1_x, leadout_point_1_y, leadout_point_1_z),
            Point(leadout_point_2_x, leadout_point_2_y, leadout_point_2_z),
        ]

        return leadin_points, leadout_points

    @staticmethod
    def generate_horizontal_leadinout(segments, blade_z_shape):

        first_segment = segments[0]
        last_segment = segments[len(segments) - 1]
        start_x = first_segment.get_start()
        start_y = first_segment.y(start_x)
        start_z = blade_z_shape.y(start_x)
        last_x = last_segment.get_end()
        last_y = last_segment.y(last_x)
        last_z = blade_z_shape.y(last_x)

        leadinout_x_length = definitions.APP_CONFIG["toolPathGenerator"]["leadInOutXLength"]

        leadin_x = start_x - leadinout_x_length
        leadin_z = blade_z_shape.y(leadin_x)
        leadin_points = [
            Point(leadin_x, start_y, leadin_z),
        ]

        leadout_x = last_x + leadinout_x_length
        leadout_z = blade_z_shape.y(leadout_x)
        leadout_points = [
            Point(leadout_x, last_y, leadout_z),
        ]

        return leadin_points, leadout_points

    @staticmethod
    def create_tool_path_segments(segment_index, blade_segments, blade_z_shape):
        # Create a list of gcode points from this blade segment and return it

        tool_path_segments = []
        blade_segment = blade_segments[segment_index]
        segment_shape = blade_segment.get_shape()

        # TODO: Seems like multiple sections use a segment_index == 0 condition
        # We should move this out of the segment_shape switch

        if (segment_shape == "CubicBezier"):
            
            # Add a G01 for the first start point
            if segment_index == 0:
                start_point_x = blade_segment.raw_shape.P0["x"]
                start_point_y = blade_segment.raw_shape.P0["y"]
                start_point_z = blade_z_shape.y(start_point_x)
                previous_gcode_point = Point(start_point_x, start_point_y, start_point_z)
                tool_path_segments.append(ToolPathSegment(gcode_command="G01", points=[previous_gcode_point]))
            
            tool_path_segments.append(ToolPathSegment.from_cubic_bezier_segment(blade_segment, blade_z_shape))

        elif (segment_shape == "Circle"):

            previous_gcode_x = blade_segment.get_start()
            previous_gcode_y = blade_segment.y(previous_gcode_x)
            previous_gcode_z = blade_z_shape.y(previous_gcode_x)
            previous_gcode_point = Point(previous_gcode_x, previous_gcode_y, previous_gcode_z)                 

            # First circle segment needs an extra G01
            if segment_index == 0:
                tool_path_segments.append(ToolPathSegment(gcode_command="G01", points=[previous_gcode_point]))

            tool_path_segments.append(ToolPath.create_blade_circle_segment(blade_segment, blade_z_shape, None, previous_gcode_point))

        elif (segment_shape == "Line"):

            # Add a G01 for the first start point
            if segment_index == 0:
                start_point_x = blade_segment.get_start()
                start_point_y = blade_segment.y(start_point_x)
                start_point_z = blade_z_shape.y(start_point_x)
                previous_gcode_point = Point(start_point_x, start_point_y, start_point_z)
                tool_path_segments.append(ToolPathSegment(gcode_command="G01", points=[previous_gcode_point]))
            
            end_point_x = blade_segment.get_end()
            end_point_y = blade_segment.y(end_point_x)
            end_point_z = blade_z_shape.y(end_point_x)
            gcode_point = Point(end_point_x, end_point_y, end_point_z)
            tool_path_segments.append(ToolPathSegment(gcode_command="G01", points=[gcode_point]))

        else:
            # For polynomials
            # NOT UPDATED
            raise Exception("Bad segment name detected")
            # print("NOT UPDATED!")
            # previous_gcode_x = blade_segment.get_start()
            # previous_gcode_y = blade_segment.y(previous_gcode_x)
            # previous_gcode_point = {
            #     "x": previous_gcode_x,
            #     "y": previous_gcode_y,
            # }

            # if segment_index == 0:
            #     tool_path_segments.append(ToolPathSegment(gcode_command="G01", points=[previous_gcode_point]))

            # # Convert shapes that are not circles into a set of circles
            # circle_segments = ToolPath.convert_segment_to_circles(blade_segment, previous_gcode_point)
            # tool_path_segments += circle_segments

        return tool_path_segments

    ##################
    # G-Code Writing #
    ##################
        
    def append_tool_path_segments(self, tool_path_segments, first_pass, last_pass, spindle_rpm, tool_number, feedrate, clockwise_rotation = True):

        # Add setup commands
        self.append_start_cut_gcode(tool_path_segments, first_pass, spindle_rpm, feedrate, clockwise_rotation)

        # Add segments along our blade and our lead in/out
        for segment_index, tool_path_segment in enumerate(tool_path_segments):

            if tool_path_segment.gcode_command == "G01":
                self.append_g01_points(tool_path_segment, segment_index, tool_number)

            elif tool_path_segment.gcode_command == "G05":
                self.append_cubic_spline_segment(tool_path_segment)

            else:
                self.append_circle_segment(tool_path_segment)

        # Add finishing commands
        self.append_end_cut_gcode(tool_path_segments, last_pass)

    def append_initialization_gcode(self):
        """
        Add initial gcode command lines for setting up coordinate, and referencing system.
        """
        self.gcode_instance.append_line("G49 (Turn off tool length compensation)") # Turn off tool length compensation
        self.gcode_instance.append_line("G40 (Turn off cutter compensation)") # Turn off cutter compensation
        self.gcode_instance.append_line("G21 (Metric)") # Use metric system
        self.gcode_instance.append_line("G90 (Absolute coordinates)") # Switch to absolute coordinates
        self.gcode_instance.append_line("G64 P0.02 Q0.002 (Add path blending)") # Add path blending with naive cam detector
        self.gcode_instance.append_line(definitions.APP_CONFIG['toolPathGenerator']['workOffsets']['cut']['number']) # Add work offset

    def append_initial_cut_gcode(self, tool_instance):

        # Convert to acceptable gcode tool number
        padded_tool_number = helpers.pad_text_until_length(str(tool_instance.number), "0", 2)

        # Add default lines for each blade
        self.gcode_instance.append_line(f"M06 T{padded_tool_number}") # Change tool
        # TODO: Make this value a config variable... or some how safer using the work_offset
        self.gcode_instance.append_line(f"#<safe_z_position> = -260.0")

        tool_radius_x_y = tool_instance.diameter / 2
        self.gcode_instance.append_line(f"#<tool_radius_x_y> = {tool_radius_x_y}")

        self.gcode_instance.append_line(f"G00 G43 Z[#<safe_z_position>] H{padded_tool_number}") # Move spindle up out of the way

    def append_start_cut_gcode(self, tool_path_segments, first_pass, spindle_rpm, feedrate, clockwise_rotation):
        # Move to start of leadin. Offset by our tool_radius_x_y to make space for the cutter compenstaion move coming up
        # First point is on the lead in so G01 with points is expected
        first_point =  tool_path_segments[0].points[0]
        # self.gcode_instance.append_line(f"X[{first_point.x} - [#<tool_radius_x_y>] * 2] Y[{first_point.y} + [#<tool_radius_x_y>] * 2]")
        self.gcode_instance.append_line(f"X[{first_point.x} - #<tool_radius_x_y>] Y{first_point.y}")

        # Move down to the center of the blade at first point
        self.gcode_instance.append_line(f"Z{first_point.z}")

        if first_pass:

            rotation_modal_code = "M03" if clockwise_rotation else "M04"

            self.gcode_instance.append_line(f"{rotation_modal_code} S{spindle_rpm}") # Start spindle
            self.gcode_instance.append_line(f"F{feedrate}") # Set feedrate


    def append_end_cut_gcode(self, tool_path_segments, last_pass):

        ## Last point is on the lead out so we know it's a line
        # last_point =  tool_path_segments[-1].points[-1]
        # self.gcode_instance.append_line(f"X[{last_point.x} + #<tool_radius_x_y>] Y[{last_point.y} + #<tool_radius_x_y>]") # Move to default start cut location with tool radius
        # self.gcode_instance.append_line(f"X[{last_point.x} + #<tool_radius_x_y>] Y{last_point.y}") # Move to default start cut location with tool radius
        # self.gcode_instance.append(" G40") # Remove cutter compensation
        # Only stop the spindle if this is our second blade
        if last_pass:
            self.gcode_instance.append_line("M05") # Stop spindle

        self.gcode_instance.append_line(f"G00 Z[#<safe_z_position>]") # Move spindle up out of the way


    def append_g01_points(self, tool_path_segment, segment_index, tool_number):
        
        g01_points = tool_path_segment.points

        # Skipping is no longer required since we only produce one point for our leadin / out
        # if segment_index == 0:
        #     # Skip first point since it's already included in our leadin
        #     g01_points = g01_points[1:]

        for point_index, gcode_point in enumerate(g01_points):
            if point_index == 0 and segment_index == 0:
                
                # Cutter compenstaion no longer needed since we perpendicular shift our the tool ourselves
                # First line needs cutter compensation
                # Convert to acceptable gcode tool number
                padded_tool_number = helpers.pad_text_until_length(str(tool_number), "0", 2)

                # Add our tool radius to help transition to the cut
                # self.gcode_instance.append_line(f"G01 G41 X[{gcode_point.x} - #<tool_radius_x_y>] Y[{gcode_point.y} - #<tool_radius_x_y>] D{padded_tool_number}")
                self.gcode_instance.append_line(f"G01 X[{gcode_point.x} - #<tool_radius_x_y>] Y{gcode_point.y} Z{gcode_point.z}")
            
            elif point_index == 0:
                # No cutter compensation needed if not first segment
                self.gcode_instance.append_line(f"G01 X{gcode_point.x} Y{gcode_point.y} Z{gcode_point.z}")


            else:
                # Rest of blade gcode lines
                self.gcode_instance.append_line(f"X{gcode_point.x} Y{gcode_point.y} Z{gcode_point.z}")


    def append_cubic_spline_segment(self, tool_path_segment):
        self.gcode_instance.append_line(f"{tool_path_segment.gcode_command} I{tool_path_segment.i} J{tool_path_segment.j} P{tool_path_segment.p} Q{tool_path_segment.q} X{tool_path_segment.x} Y{tool_path_segment.y}")
        self.gcode_instance.append_line(f"G01 Z{tool_path_segment.z}")

    def append_circle_segment(self, tool_path_segment):
        self.gcode_instance.append_line(f"{tool_path_segment.gcode_command} I{tool_path_segment.i} J{tool_path_segment.j} X{tool_path_segment.x} Y{tool_path_segment.y} Z{tool_path_segment.z}")

    ###########
    # Helpers #
    ###########

    def get_top_z_position(self):

        # Find correct numbered parameter for work offset z value
        # http://linuxcnc.org/docs/html/gcode/overview.html#sub:numbered-parameters
        # Using correct_z_offset = 20 * work_offset + 4143
        
        correct_z_offset = 20 * self.work_offset + 4143
        current_tool_length = "#5403"
        distance_from_max_z = 5.0
        return f"[[-#{correct_z_offset} - {current_tool_length}] * 25.4 - {str(distance_from_max_z)}]"

    # TODO: rework this with new class structure
    def calculate_material_removed(self, blade_index, gcode_boundaries, gcode_points):

        bottom_blade = (blade_index == 1)

        # Create y values array
        # Create x values array
        # For each gcode point
        #   find correct segment using gcode boundaries
        #   calculate difference between gcode point y and segment y and append to y values array
        #   append x value to x values array
        # Use np.trapz given y array and x array

        x_values = []
        delta_y_values = []
        gcode_y_values = []
        segment_y_values = []

        for point_index, point in enumerate(gcode_points):
            
            # If not segment is found then we are in the lead in or out
            # We do not need to consider these points
            point_segment = None
            # TODO: create seperate method
            for boundary in gcode_boundaries:
                if point_index >= boundary["start_index"] and point_index <= boundary["end_index"]:
                    point_segment = boundary["segment"]

            if point_segment is not None:
                x_values.append(point["x"])
                gcode_y_values.append(point["y"])
                segment_y_value = point_segment.raw_y(point["x"])
                segment_y_values.append(segment_y_value)

                y_delta = point["y"] - segment_y_value
                # TODO: Catch cases where y_delta is positive
                # This implies that we are not cutting at this point

                delta_y_values.append(y_delta)
            
        material_to_be_removed = np.trapz(delta_y_values, x=x_values)

        if self.debug_mode:
            figure = helpers.plotter_instance.create_figure_2d("Material to be removed points")
            helpers.plotter_instance.add_raw_series(figure, x_values, gcode_y_values, "Gcode")
            helpers.plotter_instance.add_raw_series(figure, x_values, segment_y_values, "Blade Segments")
            figure.show()

        print(f"The estimated material to be removed for the {'bottom' if bottom_blade else 'top'} blade: {abs(material_to_be_removed)}mm^2")

    @staticmethod
    def new_segment_to_circles(segment, start, end, max_circle_segment_difference=definitions.APP_CONFIG["toolPathGenerator"]["maxCircleSegmentDifference"]):
        
        # Break up the segment into multiple circles    
        segment_length = end - start
        segment_test_length = 0.05
        if segment_length < segment_test_length:
            print("Segment length is less than our segment test length! Only start point evaluated.")
        
        segment_test_range = np.arange(start, end, 0.05)

        circle_segment = ToolPath.segment_to_circle(segment, start, end)
        circle_segment_difference = ToolPath.calculate_circle_segment_difference(segment, circle_segment, segment_test_range)

        circle_segments = []
        if circle_segment_difference < max_circle_segment_difference:
            circle_segments.append(circle_segment)
        else:
            left_start = start
            left_end = left_start + segment_length / 2

            left_circle_segments = ToolPath.new_segment_to_circles(segment, left_start, left_end, max_circle_segment_difference)

            right_start = left_end
            right_end = end

            right_circle_segments = ToolPath.new_segment_to_circles(segment, right_start, right_end, max_circle_segment_difference)

            circle_segments = left_circle_segments + right_circle_segments
        
        return circle_segments

    @staticmethod
    def segment_to_circles(segment, number_of_circles):
        # Take a segment and return a set of circle segments
        segment_start = segment.get_start()
        segment_end = segment.get_end()
        segment_length = segment_end - segment_start
        
        circle_segment_length = segment_length / number_of_circles

        circles_counter = 0
        circle_segments = []
        while circles_counter < number_of_circles:
            circle_start_x = circles_counter * circle_segment_length + segment_start
            circle_end_x = (circles_counter + 1) * circle_segment_length + segment_start

            circle_segments.append(ToolPath.segment_to_circle(segment, circle_start_x, circle_end_x))
            circles_counter += 1
        
        return circle_segments

    @staticmethod
    def calculate_circle_segment_difference(segment, circle_segment, segment_test_range):
        # Find absolute difference between generated circles and segment over some numpy range
        difference_accumulator = 0
        
        # Create evaluation range
        for x_value in segment_test_range:
            difference_accumulator += abs(segment.y(x_value) - circle_segment.y(x_value))

        return difference_accumulator

    # DEPRICATED
    @staticmethod
    def update_segments_y(gcode_segments, y_offset):

        # TODO: Confirm deep copy
        updated_gcode_segments = copy.deepcopy(gcode_segments)

        for updated_gcode_segment in updated_gcode_segments:
            if updated_gcode_segment["type"] == "G01":
                for segment_point in updated_gcode_segment["points"]:
                    segment_point["y"] = round(segment_point["y"] + y_offset, 3)
            else:
                updated_gcode_segment["y"] = round(updated_gcode_segment["y"] + y_offset, 3)

        return updated_gcode_segments    
    
class Point():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class ToolPathSegment():

    def __init__(self, gcode_command, points=None, x=None, y=None, z=None, i=None, j=None, p=None, q=None):

        self.gcode_command = gcode_command

        if self.gcode_command == "G01":
        
            self.points =  list(map(lambda point: Point(round(point.x, 3), round(point.y, 3), round(point.z, 3)), points))
        
        elif self.gcode_command == "G05":

            self.x = round(x, 3)
            self.y = round(y, 3)
            self.z = round(z, 3)
            self.i = round(i, 3)
            self.j = round(j, 3)
            self.p = round(p, 3)
            self.q = round(q, 3)

        else:

            self.x = round(x, 3)
            self.y = round(y, 3)
            self.z = round(z, 3)
            self.i = round(i, 3)
            self.j = round(j, 3)
        
    @staticmethod
    def from_cubic_bezier_segment(segment, blade_z_shape):

        end_point_x = segment.raw_shape.P3["x"]
        end_point_y = segment.raw_shape.P3["y"]
        end_point_z = blade_z_shape.y(end_point_x)

        start_control_point_x_delta = segment.raw_shape.P1["x"] - segment.raw_shape.P0["x"]
        start_control_point_y_delta = segment.raw_shape.P1["y"] - segment.raw_shape.P0["y"]

        end_control_point_x_delta = segment.raw_shape.P2["x"] - segment.raw_shape.P3["x"]
        end_control_point_y_delta = segment.raw_shape.P2["y"] - segment.raw_shape.P3["y"]

        return ToolPathSegment(gcode_command="G05", \
            x=end_point_x, y=end_point_y, z=end_point_z, \
            i=start_control_point_x_delta, j=start_control_point_y_delta,\
            p=end_control_point_x_delta, q=end_control_point_y_delta)
