import math

import numpy as np
import pandas as pd

from modules import helpers, segments as segments_module, scan, tool, wedge as wedge_module
from modules.fixture import FixtureM0P2, FixtureM0P3
import definitions

class Blade():
    """
    Args:
    """

    def __init__(self, size, profile, z_shape, center_of_blade_x, wedge_instance, segments=None, profiling_specifications=None, debug_mode=definitions.APP_CONFIG["toolPathGenerator"]["debug"]):
        
        self.debug_mode = debug_mode
        self.size = size
        self.raw_profile = profile
        self.profile = profile
        self.z_shape = z_shape
        self.center_of_blade_x = center_of_blade_x
        self.wedge_instance = wedge_instance
        self.profiling_specifications = profiling_specifications

        self.segments = segments
        if segments is None:
            self.segments = self.create_segments(self.profile)
        

    def create_segments(self, profile):

        # Use the profile here to get our true start index
        start_index = profile.index[1]
        end_index = profile.index[-1]
        blade_series = profile.copy()

        segment_type = definitions.APP_CONFIG["toolPathGenerator"]["segmentType"]

        produced_segments = None

        if (segment_type == "line"):

            min_segment_size = definitions.APP_CONFIG["toolPathGenerator"]["minLineSegmentLength"]
            produced_segments = segments_module.LineBladeSegment.convert_series_to_line_segments(blade_series, start_index=start_index, end_index=end_index, min_segment_size=min_segment_size)
        
        else:

            # We use the profile here to get the diff because the trimmed profile will not be able to
            # interpolate a diff for it's first point
            key_derivative = profile.diff()[start_index] / definitions.SPACING_PER_X_INDEX

            if (segment_type == "circle"):

                generated_segments = segments_module.CircleBladeSegment.convert_series_to_circle_segments(blade_series, key_derivative_index=0, key_derivative=key_derivative, start_index=start_index, end_index=end_index)
        
            else:

                generated_segments = segments_module.CubicBezierBladeSegment.convert_series_to_segments(blade_series, start_index, end_index)
            
            #TODO: update to handle change made to create_segments
            joint_segments = self.update_disjoint_segments(generated_segments)
            #TODO: update to handle changes made to create_segments
            produced_segments = self.update_segment_derivatives(joint_segments)

        if self.profiling_specifications is not None:
            produced_segments = self.apply_profiling(produced_segments)
        
        return produced_segments

    def copy(self):
        local_segments = list(map(lambda segment: segment.copy(), self.segments))
        profiling_specifications = None if self.profiling_specifications is None else self.profiling_specifications.copy()
        return Blade(self.size, self.profile.copy(), self.z_shape.copy(), self.center_of_blade_x, self.wedge_instance, segments = local_segments, profiling_specifications = profiling_specifications, debug_mode = self.debug_mode)

    def add_profiling_specifications(self, profiling_specifications):

        # Our produced radii are a little smaller than expected
        # Adding a factor of 1.1 here to help
        adjusted_profiling_specifications = profiling_specifications.copy()
        for adjusted_profiling_specification in adjusted_profiling_specifications:
            adjusted_profiling_specification.radius *= 1.02

        # Do not copy over segments so that new segments are generated using our profiling specifications
        return Blade(self.size, self.profile.copy(), self.z_shape.copy(), self.center_of_blade_x, self.wedge_instance, profiling_specifications = adjusted_profiling_specifications, debug_mode = self.debug_mode)

    ############
    # Mutators #
    ############

    def y_shift(self, shift_value):
        self.profile += shift_value
        self.y_shift_segments(shift_value)

    def y_shift_segments(self, shift_value):
        self.segments = helpers.y_shift_segments(self.segments, shift_value)

    def z_shift(self, shift_value):

        # Since our z_shape is a function in the xz plane a y shift will achieve what we want
        self.z_shape.y_shift(shift_value)

    def x_shift(self, shift_value):
        self.profile = helpers.shift_series_indices(self.profile, shift_value)

        self.center_of_blade_x += shift_value
        
        for segment in self.segments:
            segment.x_shift(shift_value)

    # @helpers.timeit
    def perpendicular_shift(self, shift_value, recreate_segments = False):

        # New approach at perpendicularly shifting segments
        # Instead of directly shifting the segments and dealing with 
        # the fallout (disjoint, derivative mismatch) we instead shift our
        # profile and then recreate our segments.

        # Create two lists of new values for our profile series
        shifted_profile_x = []
        shifted_profile_y = []
        diff_series = self.profile.diff()

        # Iterate through series points
        # Create new x and y coordinates using the mean inverse derivative of this point and the next
        # Add new point to shifted profile lists
        for index, x in enumerate(diff_series.index.values):
            # Ignore first and last point
            # First point will not have a derivative
            # Last point will not have a next point to compare
            if ((index == 0) or (index == diff_series.size-1)):
                continue                    
            else:
                # x = index
                # y = self.profile[x]
                # derivative_at_point = diff_series[index]
                derivative_1_y = diff_series.values[index]
                derivative_1_x_delta = diff_series.index.values[index] - diff_series.index.values[index-1]
                derivative_1 = derivative_1_y / derivative_1_x_delta

                derivative_2_y = diff_series.values[index+1]
                derivative_2_x_delta = diff_series.index.values[index+1] - diff_series.index.values[index]
                derivative_2 = derivative_2_y / derivative_2_x_delta

                derivative = (derivative_1 + derivative_2)/2
                true_shift_value = shift_value if derivative < 0 else -shift_value # Reverse shift direction
                inverse_derivative = -1 / derivative
                inverse_vector = helpers.derivative_to_vector_components(inverse_derivative, true_shift_value)
        
                shifted_x = x + inverse_vector["x"]
                shifted_y = self.profile[x] + inverse_vector["y"]

                shifted_profile_x.append(shifted_x)
                shifted_profile_y.append(shifted_y)

        shifted_profile = pd.Series(data=shifted_profile_y, index=shifted_profile_x)

        # Runtime of patch_profile seems a lot better when the shift_value is smaller
        # TODO: Investigate why patch_profile method performance degrades with larger shift values.
        # maybe with larger shift values there are more issues that it catches?
        patched_profile = None
        if shift_value > 10.0:
            # helpers.timer_instance.add_manual_timestamp("new patch profile")
            patched_profile = self.new_patch_profile(shifted_profile, 0.075)
            # helpers.timer_instance.update_manual_timestamp("new patch profile")
        else:
            # helpers.timer_instance.add_manual_timestamp("patch profile")
            patched_profile = self.patch_profile(shifted_profile)
            # helpers.timer_instance.update_manual_timestamp("patch profile")

        # helpers.timer_instance.print_timestamps()

        self.profile = patched_profile
        if recreate_segments:
            self.segments = self.create_segments(patched_profile)
            
    # Make sure segments derivatives are continuous
    def update_segment_derivatives(self, segments, max_circle_difference=0.000005):

        local_segments = list(map(lambda segment: segment.copy(), segments))

        number_of_segments = len(local_segments)
        for segment_index, segment in enumerate(local_segments):

            next_segment_index = segment_index + 1
            if next_segment_index < number_of_segments:
                
                next_segment = local_segments[next_segment_index]
                next_segment_shape = next_segment.get_shape()

                current_segment = segment
                current_segment_shape = current_segment.get_shape()

                if current_segment_shape == "Circle" and next_segment_shape == "Circle":
                    # Do nothing
                    # Our profile circle generation already confirms our derivatives
                    circle_difference = helpers.calculate_segment_end_derivatives_difference(current_segment, next_segment)
                    if circle_difference > max_circle_difference:
                        print(f"From segment {segment_index} to segment {next_segment_index} there is a derivative of {circle_difference}")

                elif current_segment_shape == "Circle" and next_segment_shape == "CubicBezier":

                    # Update next segment only
                    current_segment_end_derivative = current_segment.derivative(current_segment.get_end())
                    next_segment_P1_P0_delta_x = next_segment.raw_shape.P1["x"] - next_segment.raw_shape.P0["x"]  
                    new_next_segment_P1_P0_delta_y = current_segment_end_derivative * next_segment_P1_P0_delta_x
                    next_segment.raw_shape.P1["y"] = next_segment.raw_shape.P0["y"] + new_next_segment_P1_P0_delta_y

                elif current_segment_shape == "CubicBezier" and next_segment_shape == "Circle":

                    # Update current segment only
                    next_segment_start_derivative = next_segment.derivative(next_segment.get_start())
                    current_segment_P2_P3_delta_x = current_segment.raw_shape.P2["x"] - current_segment.raw_shape.P3["x"]
                    new_current_segment_P2_P3_delta_y = next_segment_start_derivative * current_segment_P2_P3_delta_x
                    current_segment.raw_shape.P2["y"] = current_segment.raw_shape.P3["y"] + new_current_segment_P2_P3_delta_y

                elif current_segment_shape == "CubicBezier" and next_segment_shape == "CubicBezier":

                    # Update both segments
                    current_segment_end_derivative = current_segment.derivative(current_segment.get_end())
                    next_segment_start_derivative = next_segment.derivative(next_segment.get_start())
                    segment_join_derivative = (current_segment_end_derivative + next_segment_start_derivative) / 2

                    # Update current segment
                    current_segment_P2_P3_delta_x = current_segment.raw_shape.P2["x"] - current_segment.raw_shape.P3["x"]
                    new_current_segment_P2_P3_delta_y = segment_join_derivative * current_segment_P2_P3_delta_x
                    current_segment.raw_shape.P2["y"] = current_segment.raw_shape.P3["y"] + new_current_segment_P2_P3_delta_y

                    # Update new segment
                    next_segment_P1_P0_delta_x = next_segment.raw_shape.P1["x"] - next_segment.raw_shape.P0["x"]  
                    new_next_segment_P1_P0_delta_y = segment_join_derivative * next_segment_P1_P0_delta_x
                    next_segment.raw_shape.P1["y"] = next_segment.raw_shape.P0["y"] + new_next_segment_P1_P0_delta_y

        return local_segments

    def patch_segments(self, raw_segments, minimum_patch_segment_length=1.0):

        patched_segments = []

        # Look for gaps and overlaps
        # This algorithm compares the nth element with the nth+1 element
        # So no need to iterate to the last element

        segment_index = 0
        # TODO: Should stop before last element
        # This algorithm needs to be reworked
        while segment_index < (len(raw_segments) - 1):

            segment = raw_segments[segment_index].copy()
            next_segment = raw_segments[segment_index + 1]
            segment_end_x = segment.get_end()
            next_segment_start_x = next_segment.get_start()
            
            # If segment end and next segment start match within half of 3 decimal points
            # Just append this segment without modifications
            if helpers.float_equivalence(segment_end_x, next_segment_start_x, precision=0.0005):
                
                patched_segments.append(segment)

            else:
                
                # If this segment end is past next segment start
                # This is considered an overlap
                if segment_end_x > next_segment_start_x:

                    # Overlap
                    # Find potential next segment start x
                    # TODO:
                    # This should be potential next segment END x not start
                    # Review if this is correct
                    target_next_segment_start_x = segment_end_x + minimum_patch_segment_length
                    
                    # Make sure a segment exists such that target next segment start x is in it's domain
                    last_segment = raw_segments[-1]
                    if last_segment.get_end() < target_next_segment_start_x:
                        # If it doesn't just use the middle of our last segment
                        target_next_segment_start_x = (last_segment.get_end() - last_segment.get_start()) / 2 + last_segment.get_start()

                    # Subset segments past current segment
                    next_segments = raw_segments[segment_index:]

                    # This can cause an infinite loop
                    # Find segment with target x value
                    true_next_segment, true_next_segment_index = helpers.find_segment_with_x_value(next_segments, target_next_segment_start_x)

                    # Set that as next_segment
                    next_segment = true_next_segment

                    # Update next_segment domain
                    next_segment.set_start(target_next_segment_start_x)

                    # Update segment_index
                    # if true_next_segment_index is 0 then we create an infinite loop
                    if true_next_segment_index != 0:
                        segment_index += true_next_segment_index - 1

                    # we now have a gap to fill

                # Gap
                # Gaps should be fixed by inserting a new segment

                if (definitions.APP_CONFIG["toolPathGenerator"]["segmentType"] == "line"):
                    
                    gap_segment = create_line_segment_from_gap(segment, next_segment)

                else:
                    
                    # The circle should be generated using the start and end derivatives of the adjacent segments
                    gap_segment = create_circle_segment_from_gap(segment, next_segment)
                
                patched_segments.append(segment)
                patched_segments.append(gap_segment)

            segment_index += 1
                
        return patched_segments

    def new_patch_profile(self, series, tolerance = 0.5):
        local_series = series.copy()
        profile_size = local_series.size

        # skip first point
        index = 1
        # skip last point
        while index < (profile_size - 1):
            skip = 0
            next_index = index + 1

            derivative_1_y = local_series[local_series.index.values[index]] - local_series[local_series.index.values[index-1]]
            derivative_1_x_delta = local_series.index.values[index] - local_series.index.values[index-1]
            derivative_1 = derivative_1_y / derivative_1_x_delta

            next_index = index + 1
            derivative_2_y = local_series[local_series.index.values[next_index]] - local_series[local_series.index.values[index]]
            derivative_2_x_delta = local_series.index.values[next_index] - local_series.index.values[index]
            derivative_2 = derivative_2_y / derivative_2_x_delta

            while(abs(derivative_1 - derivative_2) > tolerance):
                # find next index
                skip += 1
                # next_index = index + skip + 1
                next_index += 1
                # if next point out of bounds:
                if(next_index == profile_size-1):
                    skip = -1
                    break
                else:
                    # reevaluate derivative_2 with new index
                    derivative_2_y = local_series[local_series.index.values[next_index]] - local_series[local_series.index.values[index]]
                    derivative_2_x_delta = local_series.index.values[next_index] - local_series.index.values[index]
                    derivative_2 = derivative_2_y / derivative_2_x_delta
            if(skip < 0):
                break
            if(skip > 0):
                # Create a step size to evenly space points between trim start and end
                # Add some points to fill in gap
                trim_start_x = local_series.index.values[index]
                trim_end_x = local_series.index.values[next_index]

                # Create a step size to evenly space points between trim start and end
                step_size = (trim_end_x - trim_start_x) / (skip + 1)
                gap_series_x = []
                gap_series_y = []
                point_counter = 0
                while point_counter < skip :
                    point_counter += 1
                    gap_series_x_value = trim_start_x + point_counter * step_size
                    gap_series_x.append(gap_series_x_value)
                    gap_series_y.append(np.NaN)

                gap_series = pd.Series(data=gap_series_y, index=gap_series_x)
                local_series = pd.concat([local_series.iloc[:index + 1], gap_series, local_series.iloc[next_index:]])
                local_series = local_series.interpolate(method="slinear")

            index = next_index
        return local_series

    def patch_profile(self, series, patch_index_length=2, gap_points=3):
        local_series = series.copy()

        # iterate through series indices
        # watch for series values that go backwards
        for index, x in enumerate(local_series.index.values):

            # Skip first index
            if index == 0:
                continue

            last_index = index - 1
            last_x = local_series.index.values[last_index]
            # Backwards move detected
            if x < last_x:

                # Trim and interpolate series to remove backwards move
                trimmed_series = helpers.interpolate_series_at_index(local_series, last_index, interpolation_window=patch_index_length*2, gap_points=gap_points)

                # recursive call to patch_profile again to catch any other instances of backwards stepping
                local_series = self.patch_profile(trimmed_series, patch_index_length=patch_index_length, gap_points=gap_points)
                break

        return local_series

    #############
    # Profiling #
    #############

    def apply_profiling(self, segments):

        # Split up profiling specifications into center, toe, and heel
        center_profiling_specification_index, center_profiling_specification = helpers.find_center_profiling_specification(self.profiling_specifications)
        heel_profiling_specifications = self.profiling_specifications[:center_profiling_specification_index]
        toe_profiling_specifications = self.profiling_specifications[center_profiling_specification_index + 1:]

        # Create center section
        center_profile_segment_start_index = self.find_profiling_index_at_percentage(center_profiling_specification.start_percentage)
        center_profile_segment_start_y = self.profile[center_profile_segment_start_index]
        center_profile_segment_end_index = self.find_profiling_index_at_percentage(center_profiling_specification.end_percentage)
        center_profile_segment_end_y = self.profile[center_profile_segment_end_index]

        # If no target pitch is set
        # Use the start, middle and end points to generate our center profile segment
        # Then set the radius
        if center_profiling_specification.pitch is None:
            # Find middle point
            center_profile_segment_middle_percentage = (center_profiling_specification.start_percentage + center_profiling_specification.end_percentage) / 2
            center_profile_segment_middle_index = self.find_profiling_index_at_percentage(center_profile_segment_middle_percentage)
            center_profile_segment_middle_y = self.profile[center_profile_segment_middle_index]

            # Generate circle using start, middle, and end points
            center_profile_segment = segments_module.CircleBladeSegment.from_3_points(\
                center_profile_segment_start_index, center_profile_segment_start_y, \
                center_profile_segment_middle_index, center_profile_segment_middle_y, \
                center_profile_segment_end_index, center_profile_segment_end_y, \
                raw_series=self.profile, profiling=True
            )

            # Shift our circle in Y by the difference of our current and target radius
            # This moves the apex of the circle to the same Y level as before
            center_profile_radius_delta = center_profile_segment.raw_shape.radius - center_profiling_specification.radius
            center_profile_segment.raw_shape.y0 += center_profile_radius_delta

            # Set radius to our profiling target
            center_profile_segment.raw_shape.radius = center_profiling_specification.radius

        else:
            
            # When creating the circle segment with a pitch we explicitly set the x0 of the circle segment
            x0_pitch_offset = (center_profiling_specification.pitch / 100) * self.size
            center_profile_segment_x0 = helpers.find_closest_index(self.center_of_blade_x + x0_pitch_offset, self.profile)
            center_profile_length = abs(center_profile_segment_end_index - center_profile_segment_start_index)
            center_profile_segment_start_index = helpers.find_closest_index(self.center_of_blade_x - center_profile_length / 2, self.profile)
            center_profile_segment_end_index = helpers.find_closest_index(self.center_of_blade_x + center_profile_length / 2, self.profile)
            center_profile_segment = segments_module.CircleBladeSegment.for_profiling_center(\
                self.profile, \
                center_profile_segment_start_index, \
                center_profile_segment_end_index, \
                center_profiling_specification.radius, \
                center_profile_segment_x0 \
            )
            
        center_profile_segment_left_derivative = center_profile_segment.derivative(center_profile_segment_start_index)
        center_profile_segment_right_derivative = center_profile_segment.derivative(center_profile_segment_end_index)
        center_profile_segment_offset = center_profile_segment_start_index - self.find_profiling_index_at_percentage(center_profiling_specification.start_percentage)

        # Create toe profile segments while maintaining a continuos derivative
        toe_profile_segments = []
        for toe_profiling_specification in toe_profiling_specifications:
            # If first toe profile segment use the center segment derivative
            if len(toe_profile_segments) == 0:
                key_derivative = center_profile_segment_right_derivative
            else:
                last_toe_profile_segment = toe_profile_segments[-1]
                key_derivative = last_toe_profile_segment.derivative(last_toe_profile_segment.get_end())
            
            toe_profile_segments.append(self.create_end_profiling_segments(toe_profiling_specification, 0, key_derivative, center_profile_segment_offset))
            
        # Create heel profile segments while maintaining a continuos derivative
        heel_profile_segments = []
        for heel_profiling_specification in heel_profiling_specifications:
            # If first heel profile segment use the center segment derivative
            if len(heel_profile_segments) == 0:
                key_derivative = center_profile_segment_left_derivative
            else:
                last_heel_profile_segment = heel_profile_segments[-1]
                key_derivative = last_heel_profile_segment.derivative(last_toe_profile_segment.get_start())
            
            heel_profile_segments.append(self.create_end_profiling_segments(heel_profiling_specification, -1, key_derivative, center_profile_segment_offset))
        
        reversed_heel_profile_segments = heel_profile_segments[::-1]

        profile_segments = reversed_heel_profile_segments + [center_profile_segment] + toe_profile_segments

        updated_segments = self.insert_and_replace_segments(profile_segments, segments)

        joint_segments = self.update_disjoint_segments(updated_segments)

        adjusted_segments = self.adjust_adjacent_segments(profile_segments, joint_segments)
        
        shallowest_cut_segments = self.shift_segments_to_shallowest_cut(adjusted_segments)

        # Make sure segment derivatives are continuous
        final_segments = self.update_segment_derivatives(shallowest_cut_segments)

        return final_segments

    def update_disjoint_segments(self, segments):
        # Update segments to ensure continuity
        center_index = self.find_profiling_index_at_percentage(50.0)
        center_segment, center_segment_index = helpers.find_segment_with_x_value(segments, center_index)
        joint_segments = helpers.join_segments_to_circle_segment(segments, center_segment_index)
        return joint_segments

    # TODO: Update this method to use a ray projection approach and apply mutation via perpendicular shift
    # This may cause an infinite loop
    def shift_segments_to_shallowest_cut(self, segments):
        # Move to shallowest cut
        max_y_delta = helpers.find_segments_max_y_delta(segments, self.profile)
        shifted_segments = helpers.y_shift_segments(segments, max_y_delta * -1)
        return shifted_segments

        # Not working as expected
        # Also perpendicular shifts for circles will change the radius
        # max_negative_perpendicular_shift = helpers.find_segments_max_perpendicular_deltas(self.segments, self.profile)[0]
        # self.perpendicular_shift(max_negative_perpendicular_shift)
        
    def create_end_profiling_segments(self, profiling_specification, key_derivative_index, key_derivative, center_profile_segment_offset):

        # Find x domain
        profile_segment_start_index_raw = self.find_profiling_index_at_percentage(profiling_specification.start_percentage) 
        profile_segment_start_index = helpers.find_closest_index(profile_segment_start_index_raw + center_profile_segment_offset, self.profile)
        profile_segment_end_index_raw = self.find_profiling_index_at_percentage(profiling_specification.end_percentage)
        profile_segment_end_index = helpers.find_closest_index(profile_segment_end_index_raw + center_profile_segment_offset, self.profile)
        return segments_module.CircleBladeSegment.for_profiling(self.profile, profile_segment_start_index, profile_segment_end_index, profiling_specification.radius, key_derivative_index, key_derivative)

    def insert_and_replace_segments(self, insert_segments, segments):

        local_segments = list(map(lambda segment: segment.copy(), segments))
        insert_segments_start_index = insert_segments[0].get_start()
        insert_segments_end_index = insert_segments[-1].get_end()
        # Find key segments and their indices at start_x and end_x
        start_segment, start_segment_index = helpers.find_segment_with_x_value(local_segments, insert_segments_start_index)
        end_segment, end_segment_index = helpers.find_segment_with_x_value(local_segments, insert_segments_end_index)

        # If our detected start and end segment are the same segment, copy the segment so that we can
        # insert our segments in the middle of it without losing the ends of this segment
        if start_segment_index == end_segment_index:
            # Create copy of key segment
            end_segment = end_segment.copy()
            end_segment_index += 1
            local_segments.insert(end_segment_index, end_segment)
        
        # Adjust start and end segments' x domains
        start_segment.set_end(insert_segments_start_index)
        end_segment.set_start(insert_segments_end_index)

        # This is quite unlikely
        if start_segment.get_start() == start_segment.get_end():
            # If our start key segment start and end are equal
            # Remove the segment and use the previous segment
            start_segment_index -= 1
            start_segment = local_segments[start_segment_index]

        if end_segment.get_start() == end_segment.get_end():
            # If our end key segment start and end are equal
            # Remove the segment and use the previous segment
            end_segment_index += 1
            end_segment = local_segments[end_segment_index]

        profile_segment_insert_index = start_segment_index + 1

        # Remove segments between our start and end segments and insert our insert segments
        new_segments = local_segments[:profile_segment_insert_index] + insert_segments + local_segments[end_segment_index:]
        return new_segments

    def adjust_adjacent_segments(self, profile_segments, segments):

        # Find joining derivatives
        left_most_profiled_segment = profile_segments[0]
        right_most_profiled_segment = profile_segments[-1]

        left_adjusted_segments = self.adjust_adjacent_segment(left_most_profiled_segment, -1, segments)
        adjusted_segments = self.adjust_adjacent_segment(right_most_profiled_segment, 1, left_adjusted_segments)
        return adjusted_segments

    # direction should be an integer
    #  - if -1 adjust the left side
    #  - if 1 adjust the right side
    # TODO: Split up into simplier functions
    def adjust_adjacent_segment(self, key_segment, direction, segments, minimum_segment_length = 10.0, control_point_x_delta_percentage=35):

        local_segments = list(map(lambda segment: segment.copy(), segments))

        if direction == -1:
            profile_join_point_x = key_segment.get_start()
        elif direction == 1:
            profile_join_point_x = key_segment.get_end()

        key_derivative = key_segment.derivative(profile_join_point_x)

        # Find joining point
        # Our joining point should have a similar derivative to the end/start of our profile segment
        # This way we can ensure our joining segment will be smooth
        join_segment_x_index_raw = self.find_segment_with_derivative(key_derivative, direction, local_segments)

        # Make sure joining point is at least as long as our minimum segment length
        # If not move it over
        if abs(join_segment_x_index_raw - profile_join_point_x) < minimum_segment_length:
            join_segment_x_index_raw = direction * minimum_segment_length + profile_join_point_x

        join_segment_x_index = helpers.find_closest_index(join_segment_x_index_raw, self.profile)
        join_segment, join_segment_index = helpers.find_segment_with_x_value(local_segments, join_segment_x_index)
        
        # Trim join segment domain
        # If trimming the join segment makes the segment too small use the next segment
        # and remove the join_segment from our segments
        if direction == -1:

            if abs(join_segment.get_start() - join_segment_x_index) < minimum_segment_length:
                join_segment_x_index = join_segment.get_start()
                join_segment = local_segments[join_segment_index - 1]
                local_segments = local_segments[:join_segment_index] + local_segments[join_segment_index + 1:]
                join_segment_index -= 1
            else:
                join_segment.set_end(join_segment_x_index)

        elif direction == 1:

            if abs(join_segment.get_end() - join_segment_x_index) < minimum_segment_length:
                join_segment_x_index = join_segment.get_end()
                join_segment = local_segments[join_segment_index + 1]
                local_segments = local_segments[:join_segment_index] + local_segments[join_segment_index + 1:]
                # No need to change our join segment index since we trimmed our segments

            else:
                join_segment.set_start(join_segment_x_index)

        # Find profile segment index
        profile_segment_index = helpers.find_segment_index_in_segments(local_segments, key_segment)

        # Remove all segments in between
        if direction == -1:
            local_segments = local_segments[:join_segment_index + 1] + local_segments[profile_segment_index:]
        elif direction == 1:
            local_segments = local_segments[:profile_segment_index + 1] + local_segments[join_segment_index:]

        if join_segment.get_shape() == "CubicBezier":
            join_segment.reset_control_points()

        # Create joining segment
        if direction == -1: 
            
            joining_segment_length = profile_join_point_x - join_segment.get_end()
            control_point_x_delta = joining_segment_length * control_point_x_delta_percentage / 100

            joining_segment_P3 = {
                "x": key_segment.get_start(),
                "y": key_segment.y(key_segment.get_start())
            }

            joining_segment_P2 = {
                "x": key_segment.get_start() - control_point_x_delta,
                "y": key_segment.y(key_segment.get_start()) - key_derivative * control_point_x_delta
            }

            join_segment_end_derivative = join_segment.derivative(join_segment.get_end())

            joining_segment_P1 = {
                "x": join_segment.get_end() + control_point_x_delta,
                "y": join_segment.y(join_segment.get_end()) + join_segment_end_derivative * control_point_x_delta
            }

            # For the left direction if P2y < P1y
            # y shift P1 and P0 by y_shift_delta
            # y shift all segments to the left by y_shift_delta
            segment_half_length = joining_segment_length / 2
            P2y_to_center = key_segment.y(key_segment.get_start()) - key_derivative * segment_half_length
            P1y_to_center = join_segment.y(join_segment.get_end()) + join_segment_end_derivative * segment_half_length

            control_point_delta = P2y_to_center - P1y_to_center
            for segment_index, segment in enumerate(local_segments):
                if segment_index < (join_segment_index + 1):
                    segment.y_shift(control_point_delta)

            joining_segment_P1["y"] += control_point_delta

            joining_segment_P0 = {
                "x": join_segment.get_end(),
                "y": join_segment.y(join_segment.get_end())
            }

            joining_segment_raw_shape = segments_module.CubicBezier(joining_segment_P0, joining_segment_P1, joining_segment_P2, joining_segment_P3)
            joining_segment = segments_module.CubicBezierBladeSegment(joining_segment_P0["x"], joining_segment_P3["x"], self.profile, joining_segment_raw_shape)
            joining_circle_segment = segments_module.CircleBladeSegment.from_cubic_bezier_segment(self.profile, joining_segment)

            local_segments.insert(join_segment_index+1, joining_circle_segment)

        elif direction == 1:

            joining_segment_length = join_segment.get_start() - profile_join_point_x
            control_point_x_delta = joining_segment_length * control_point_x_delta_percentage / 100

            joining_segment_P0 = {
                "x": key_segment.get_end(),
                "y": key_segment.y(key_segment.get_end())
            }

            joining_segment_P1 = {
                "x": key_segment.get_end() + control_point_x_delta,
                "y": key_segment.y(key_segment.get_end()) + key_derivative * control_point_x_delta
            }

            join_segment_start_derivative = join_segment.derivative(join_segment.get_start())

            joining_segment_P2 = {
                "x": join_segment.get_start() - control_point_x_delta,
                "y": join_segment.y(join_segment.get_start()) - join_segment_start_derivative * control_point_x_delta
            }

            # For the right direction if P1y < P2y
            # y shift P2 and P3 by y_shift_delta
            # y shift all segments to the right by y_shift_delta
            segment_half_length = joining_segment_length / 2
            P1y_to_center = key_segment.y(key_segment.get_end()) + key_derivative * segment_half_length
            P2y_to_center = join_segment.y(join_segment.get_start()) - join_segment_start_derivative * segment_half_length

            control_point_delta = P1y_to_center - P2y_to_center
            for segment_index, segment in enumerate(local_segments):
                if segment_index > profile_segment_index:
                    segment.y_shift(control_point_delta)

            joining_segment_P2["y"] += control_point_delta

            joining_segment_P3 = {
                "x": join_segment.get_start(),
                "y": join_segment.y(join_segment.get_start())
            }

            joining_segment_raw_shape = segments_module.CubicBezier(joining_segment_P0, joining_segment_P1, joining_segment_P2, joining_segment_P3)
            joining_segment = segments_module.CubicBezierBladeSegment(joining_segment_P0["x"], joining_segment_P3["x"], self.profile, joining_segment_raw_shape)
            joining_circle_segment = segments_module.CircleBladeSegment.from_cubic_bezier_segment(self.profile, joining_segment)
            local_segments.insert(profile_segment_index + 1, joining_circle_segment)
        
        return local_segments

    # TODO: Move to helpers module
    def find_segment_with_derivative(self, key_derivative, direction, segments, additional_offset=0.01):
        # Find segments to traverse
        segments_to_traverse = self.filter_segments_by_direction(direction, segments)

        for segment in segments_to_traverse:

            # Create 40 points evenly spaced out to test
            number_of_points = 40
            # Lines will have a consistent derivative so no need to create as many points
            if (segment.get_shape() == "Line"):
                number_of_points = 3

            x_series, y_series, derivative_series = helpers.create_segment_debug_series(segment, number_of_points = number_of_points)

            for derivative_value_index, derivative_value in enumerate(derivative_series):
                
                if direction == -1 and derivative_value > key_derivative + additional_offset:

                    return x_series[derivative_value_index]
                
                elif direction == 1 and derivative_value < key_derivative - additional_offset:

                    return x_series[derivative_value_index]

    # TODO: Move to helpers module
    def filter_segments_by_direction(self, direction, segments):
        
        # Iterate through segments and subset the non profiling segments
        # If direction is -1 we want the left segments
        left_segments = []
        right_segments = []

        # Loop through the segments and look for segments that are created for profiling
        # Before we hit our first profiling segment store segments in left_segments
        # Ignore profiling segments
        # After we hit our first profiling segment all non-profiling segments will be stored in
        # the right_segments list
        past_profiling_segments = False
        for segment in segments:
            
            if getattr(segment, 'profiling', False):

                past_profiling_segments = True

            else:

                if past_profiling_segments == False:
                    left_segments.append(segment)
                else:
                    right_segments.append(segment)
        
        if direction == -1:
            # Reverse the order of segments so we can iterate through them in order
            return left_segments[::-1]
        else:
            return right_segments

    def recreate_adjacent_segments(self, profiling_specifications, replacement_length=10.0):

        self.update_adjacent_segment(profiling_specifications, True, replacement_length)
        self.update_adjacent_segment(profiling_specifications, False, replacement_length)

        # # Update the segments adjacent to our profiling specification boundaries
        # first_profiling_specification = profiling_specifications[0]
        # last_profiling_specification = profiling_specifications[-1]

        # left_adjacent_end_index = self.find_profiling_index_at_percentage(first_profiling_specification.start_percentage)
        # left_adjacent_subset_end_index = left_adjacent_end_index + replacement_length
        # left_adjacent_start_index = left_adjacent_end_index - replacement_length
        # left_adjacent_subset_start_index = left_adjacent_start_index - replacement_length

        # left_adjacent_subset_series = helpers.create_subset_series_from_segments(self.segments, left_adjacent_subset_start_index, left_adjacent_subset_end_index)
        
        # left_adjacent_subset_trimmed_series = helpers.replace_series_values(left_adjacent_subset_series, left_adjacent_start_index, left_adjacent_end_index)

        # left_adjacent_subset_interpolated_series = left_adjacent_subset_trimmed_series.interpolate(method='akima')

        # # Get points to create circle from 3 points
        # left_replacement_circle_point_1_x = helpers.find_closest_index(left_adjacent_start_index, left_adjacent_subset_interpolated_series)
        # left_replacement_circle_point_1_y = left_adjacent_subset_interpolated_series[left_replacement_circle_point_1_x]

        # left_replacement_circle_point_2_x = helpers.find_closest_index((left_adjacent_start_index + left_adjacent_end_index) / 2, left_adjacent_subset_interpolated_series)
        # left_replacement_circle_point_2_y = left_adjacent_subset_interpolated_series[left_replacement_circle_point_2_x]

        # left_replacement_circle_point_3_x = helpers.find_closest_index(left_adjacent_end_index, left_adjacent_subset_interpolated_series)
        # left_replacement_circle_point_3_y = left_adjacent_subset_interpolated_series[left_replacement_circle_point_3_x]

        # left_replacement_circle_segment = segments_module.CircleBladeSegment.from_3_points(
        #         left_replacement_circle_point_1_x, left_replacement_circle_point_1_y, \
        #         left_replacement_circle_point_2_x, left_replacement_circle_point_2_y, \
        #         left_replacement_circle_point_3_x, left_replacement_circle_point_3_y, \
        #         raw_series=self.profile
        #     )

        # # Replace segment in list of segments
        # self.inject_segment(left_replacement_circle_segment, left_adjacent_start_index, left_adjacent_end_index)

    def update_adjacent_segment(self, profiling_specifications, left, replacement_length):

        min_index = self.segments[0].start
        max_index = self.segments[-1].end

        # Find our boundaries
        # We add an extra unit of replacement_length around our boundaries to help with interpolation
        if left:
            profiling_specification = profiling_specifications[0]

            adjacent_end_index = self.find_profiling_index_at_percentage(profiling_specification.start_percentage)
            adjacent_subset_end_index = adjacent_end_index + replacement_length
            adjacent_start_index = adjacent_end_index - replacement_length
            adjacent_subset_start_index = adjacent_start_index - replacement_length

        else:
            profiling_specification = profiling_specifications[-1]
        
            adjacent_start_index = self.find_profiling_index_at_percentage(profiling_specification.end_percentage)
            adjacent_subset_start_index = adjacent_start_index - replacement_length
            adjacent_end_index = adjacent_start_index + replacement_length
            adjacent_subset_end_index = adjacent_end_index + replacement_length


        # TODO: Add some tests
        # This method does not produce a good result since it breaks interpolation
        # Better to have the program crash
        # if adjacent_end_index > max_index:
        #     adjacent_end_index = max_index
        
        # if adjacent_subset_end_index > max_index:
        #     adjacent_subset_end_index = max_index
            
        # if adjacent_start_index < min_index:
        #     adjacent_start_index = min_index
        
        # if adjacent_subset_start_index < min_index:
        #     adjacent_subset_start_index = min_index

        # Create our series subset for interpolation
        adjacent_subset_series = helpers.create_subset_series_from_segments(self.segments, adjacent_subset_start_index, adjacent_subset_end_index)
        # Remove the area we want to interpolate over
        cleared_adjacent_subset_series = helpers.replace_series_values(adjacent_subset_series, adjacent_start_index, adjacent_end_index)
        interpolated_adjacent_subset_series = cleared_adjacent_subset_series.interpolate(method='akima')

        # Get points to create circle from 3 points
        replacement_circle_point_1_x = helpers.find_closest_index(adjacent_start_index, interpolated_adjacent_subset_series)
        replacement_circle_point_1_y = interpolated_adjacent_subset_series[replacement_circle_point_1_x]

        replacement_circle_point_2_x = helpers.find_closest_index((adjacent_start_index + adjacent_end_index) / 2, interpolated_adjacent_subset_series)
        replacement_circle_point_2_y = interpolated_adjacent_subset_series[replacement_circle_point_2_x]

        replacement_circle_point_3_x = helpers.find_closest_index(adjacent_end_index, interpolated_adjacent_subset_series)
        replacement_circle_point_3_y = interpolated_adjacent_subset_series[replacement_circle_point_3_x]

        replacement_circle_segment = segments_module.CircleBladeSegment.from_3_points(
                replacement_circle_point_1_x, replacement_circle_point_1_y, \
                replacement_circle_point_2_x, replacement_circle_point_2_y, \
                replacement_circle_point_3_x, replacement_circle_point_3_y, \
                raw_series=self.profile
            )

        # Replace segment in list of segments
        self.inject_segment(replacement_circle_segment, adjacent_start_index, adjacent_end_index)

    def inject_segment(self, segment, start, end):
        # Find x domain
        new_segment_start_index = helpers.find_closest_index(start, self.profile)
        new_segment_end_index = helpers.find_closest_index(end, self.profile)

        # Find key segments and their indices at start_x and end_x
        start_segment, start_segment_index = helpers.find_segment_with_x_value(self.segments, new_segment_start_index)
        end_segment, end_segment_index = helpers.find_segment_with_x_value(self.segments, new_segment_end_index)

        # If our start and end segment are the same segment split up the segment into two
        if start_segment_index == end_segment_index:
            # Create copy of segment
            end_segment = end_segment.copy()
            end_segment_index += 1
            self.segments.insert(end_segment_index, end_segment)
        
        # Adjust start and end segments' x domains
        start_segment.end = new_segment_start_index
        end_segment.start = new_segment_end_index

        if start_segment.get_start() == start_segment.get_end():
            # If our segment start and end are equal
            # Remove the segment and use the previous segment
            start_segment_index -= 1
            start_segment = self.segments[start_segment_index]

        if end_segment.get_start() == end_segment.get_end():
            # If our segment start and end are equal
            # Remove the segment and use the previous segment
            end_segment_index += 1
            end_segment = self.segments[end_segment_index]

        segment_insert_index = start_segment_index + 1

        # If there are segments between our start and end segments remove them
        if segment_insert_index < end_segment_index:
            self.segments = self.segments[:segment_insert_index] + self.segments[end_segment_index:]
        
        # Insert segment into segments
        self.segments.insert(segment_insert_index, segment)

    def apply_profiling_update_adjacent_segments(self, profiling_specifications):
        # Update the segments adjacent to our profiling specification boundaries
        first_profiling_specification = profiling_specifications[0]
        last_profiling_specification = profiling_specifications[-1]

        left_adjacent_index = self.find_profiling_index_at_percentage(first_profiling_specification.start_percentage)
        left_adjacent_segment, left_adjacent_segment_index = helpers.find_segment_with_x_value(self.segments, left_adjacent_index)
        
        right_adjacent_index = self.find_profiling_index_at_percentage(last_profiling_specification.end_percentage)
        right_most_profiled_segment, right_most_profiled_segment_index = helpers.find_segment_with_x_value(self.segments, right_adjacent_index)
        # Due to the way find_segment_with_x_value works we need to move one index over
        right_adjacent_segment_index = right_most_profiled_segment_index + 1
        right_adjacent_segment = self.segments[right_adjacent_segment_index]

        # Create new left adjacent segment
        left_adjacent_segment_series = self.profile.loc[left_adjacent_segment.get_start():left_adjacent_segment.get_end()]
        left_adjacent_key_derivative = left_adjacent_segment.derivative(left_adjacent_segment.get_start()) # Start derivative should still be accurate
        new_left_adjacent_segments = segments_module.CircleBladeSegment.convert_series_to_circle_segments(left_adjacent_segment_series, key_derivative_index=0, key_derivative=left_adjacent_key_derivative, start_index=left_adjacent_segment.get_start(), end_index=left_adjacent_segment.get_end())
        self.segments = self.segments[:left_adjacent_segment_index] + new_left_adjacent_segments + self.segments[left_adjacent_segment_index + 1:]

        # Create new right adjacent segment
        right_adjacent_segment_series = self.profile.loc[right_adjacent_segment.get_start():right_adjacent_segment.get_end()]
        right_adjacent_key_derivative = right_most_profiled_segment.derivative(right_most_profiled_segment.get_end()) # Get the end derivative of the right_most_profiled_segment
        new_right_adjacent_segments = segments_module.CircleBladeSegment.convert_series_to_circle_segments(right_adjacent_segment_series, key_derivative_index=0, key_derivative=right_adjacent_key_derivative, start_index=right_adjacent_segment.get_start(), end_index=right_adjacent_segment.get_end())
        self.segments = self.segments[:right_adjacent_segment_index] + new_right_adjacent_segments + self.segments[right_adjacent_segment_index + 1:]

    # This method is used to make sure we always use the trimmed profile for finding profiling indices
    def find_profiling_index_at_percentage(self, percentage):
        return helpers.find_series_index_at_percentage(percentage, self.profile)

    def generate_toolpath_parameters(self, tool_instance):

        # Setup some feeds and speeds
        rough_cut_depth = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["profile"]["maxDepth"]
        rough_cut_SFM = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["profile"]["SFM"]
        rough_cut_IPT = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["profile"]["IPT"]
        rough_cut_RPM_multiplier = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["profile"]["RPMMultiplier"]
        rough_cut_CLF = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["profile"]["CLF"]
        rough_cut_feeds_and_speeds = tool_instance.generate_feed_and_speed(rough_cut_SFM, rough_cut_IPT, rough_cut_RPM_multiplier, rough_cut_CLF)

        rough_cut_feedrate = rough_cut_feeds_and_speeds["feed_rate_mm_min"]
        rough_cut_spindle_rpm = rough_cut_feeds_and_speeds["spindle_rpm"]
        
        finishing_cut_depth = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["profile"]["finishingDepth"]
        finishing_cut_SFM = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["SFM"]
        finishing_cut_IPT = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["IPT"]
        finishing_cut_RPM_multiplier = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["RPMMultiplier"]
        finishing_cut_CLF = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["CLF"]
        finishing_cut_feeds_and_speeds = tool_instance.generate_feed_and_speed(finishing_cut_SFM, finishing_cut_IPT, finishing_cut_RPM_multiplier, finishing_cut_CLF)

        finishing_cut_feedrate = finishing_cut_feeds_and_speeds["feed_rate_mm_min"]
        finishing_cut_spindle_rpm = finishing_cut_feeds_and_speeds["spindle_rpm"]

        # OLD METHOD
        # Find max y delta
        # This will be our initial y offset
        # While offset till final cut > 0
        # If offset left is greater than our profiling depth of cut
        # offset left -= profiling depth of cut
        # Create a new blade instance with a y offset of offset left
        # Else offset is less than our profiling depth of cut so we need to use our sharpening depth of cut

        # NEW METHOD
        # Create a list of cut depths that need to be applied from our profiling segments to get to our initial blade shape
        # old method might have actually been easier to understand

        # max_y_delta = helpers.find_segments_max_y_delta(self.segments, self.profile)
        max_perp_delta = helpers.find_segments_max_perpendicular_delta(self.segments, self.profile)
        left_to_cut = max_perp_delta

        # assert left_to_cut > finishing_cut_depth, f"Max perpendicular delta ({round(max_perp_delta, 5)}) less than our finishing cut depth ({finishing_cut_depth})!"

        # Find our depth of cuts
        # It is quite unlikely that our max_perp_delta will be a factor of our roughing or finishing depth of cut
        cut_depths = []
        while left_to_cut > 0.001:
            rough_cut_factor = left_to_cut / rough_cut_depth

            if rough_cut_factor > 2:

                left_to_cut -= rough_cut_depth
                cut_depths.append(rough_cut_depth)

            elif rough_cut_factor > 1:

                pre_final_cut_depth = left_to_cut - rough_cut_depth - finishing_cut_depth

                if pre_final_cut_depth == 0:

                    left_to_cut -= rough_cut_depth
                    cut_depths.append(rough_cut_depth)

                else:

                    split_cut_depth = (pre_final_cut_depth + rough_cut_depth) / 2
                    left_to_cut -= (split_cut_depth * 2)
                    cut_depths.append(split_cut_depth)
                    cut_depths.append(split_cut_depth)
            
            else:
                
                pre_final_cut_depth = left_to_cut - finishing_cut_depth

                if pre_final_cut_depth > 0.001:
                    left_to_cut -= pre_final_cut_depth
                    cut_depths.append(pre_final_cut_depth)

                if left_to_cut > 0:
                    left_to_cut -= finishing_cut_depth
                    cut_depths.append(finishing_cut_depth)

        # reverse our cut depths so that the finishing cut is applied first
        cut_depths.reverse()
        
        toolpath_parameters = []

        # Add current position
        # This will be our final and only finishing cut
        final_blade_cut_instance = self.copy()
        toolpath_parameters.append(ToolpathParameter(final_blade_cut_instance, finishing_cut_feedrate, finishing_cut_spindle_rpm))

        # Add our roughing passes
        roughing_blade = self.copy()
        for cut_depth in cut_depths:

            # mutate local_blade with our depth of cut
            roughing_blade.perpendicular_shift(cut_depth)
            blade_cut_instance = roughing_blade.copy()
            toolpath_parameters.append(ToolpathParameter(blade_cut_instance, rough_cut_feedrate, rough_cut_spindle_rpm))

        # We finally reverse the toolpath parameters to our cutting order
        # We will start with our roughing cuts then finish with a finishing cut
        toolpath_parameters.reverse()
        # Remove the first cut since it just gets us back to our target profile
        trimmed_toolpath_parameters = toolpath_parameters[1:]
        return trimmed_toolpath_parameters

    def create_web_profile_json(self, number_of_points=1000):
        return helpers.pd_series_to_json(self.profile, number_of_points)

    def to_json(self):

        # profiling_specifications = None if self.profiling_specifications is None else self.profiling_specifications.copy()

        return {
            "size": self.size,
            "profile": helpers.pd_series_to_json(self.profile),
            "z_shape": self.z_shape.to_json(),
            "center_of_blade_x": self.center_of_blade_x,
            "segments": list(map(lambda segment: segment.to_json(), self.segments)),
            "wedge": self.wedge_instance.to_json()
        }

    def segments_web_profiles(self):
        return_json = []
        for segment in self.segments:
            return_json.append(segment.to_web_profile())
        return return_json

    @staticmethod
    def from_stored_blades(stored_blades):
        # Return pair of blades where first index is the top blade
        blade_instances = []
        top_blade_index = None
        for stored_blade_index, stored_blade in enumerate(stored_blades):
            
            profile_series = pd.Series(data=stored_blade["class_instance"]["profile"]["y"], index=stored_blade["class_instance"]["profile"]["x"])
            z_shape = segments_module.create_shape_from_stored_shape(stored_blade["class_instance"]["z_shape"])
            center_of_blade_x = stored_blade["class_instance"]["center_of_blade_x"]
            size = stored_blade["class_instance"]["size"]
            wedge_instance = wedge_module.Wedge.from_stored_wedge(stored_blade["class_instance"]["wedge"])

            blade_segments = list(map(lambda stored_segment: segments_module.create_segment_from_stored_segment(stored_segment, profile_series), stored_blade["class_instance"]["segments"]))
            blade_instance = Blade(size, profile_series, z_shape, center_of_blade_x, wedge_instance, segments=blade_segments)

            # Make sure the top blade is first
            if stored_blade["top"] == True:
                blade_instances.insert(0, blade_instance)
            else:    
                blade_instances.append(blade_instance)

        return blade_instances

    @staticmethod
    def parse_from_scan_instance(scan_instance):

        # fixture_instance = FixtureM0P2(scan_instance)
        fixture_instance = FixtureM0P3(scan_instance)
        x_axis_reference_feature = fixture_instance.reference_features()["x"]
        x_axis_offset = x_axis_reference_feature * definitions.SPACING_PER_X_INDEX * -1

        # TODO: Remove splitting this seems to have no purpose anymore
        # Subset scan data around the static x axis reference features to get the center section of the cartridge
        blades_dataframe = scan_instance.trimmed_scan_dataframe.copy()
        blade_center_dataframe = blades_dataframe.loc[(x_axis_reference_feature + 1400) : (x_axis_reference_feature + 1500), :]
        blade_center_median = blade_center_dataframe.median()

        # Split dataframe in half
        blade_center_series = helpers.split_series_in_half(blade_center_median)

        # TODO: Add output requirements handling to support only exporting bottom or top blade
        # subset our blades
        blades = []
        for blade_index, blade_series in enumerate(blade_center_series):
            
            # TODO: Move away from hardcoding
            # Blade center z index is used to help detect the profile of the blade
            # It gives us a starting point in the z axis to subset our blade data
            top = (blade_index == 1)
            blade_center_z_index = 660 if top else 140
            blade_profile = fixture_instance.blade_smooth_profile(blade_center_z_index)
            blade_profile_mm = helpers.rename_series_indices(blade_profile.copy(), definitions.SPACING_PER_X_INDEX)            
            
            center_of_blade_x = fixture_instance.blade_center_x(blade_center_z_index)
            true_center_of_blade_x = center_of_blade_x + x_axis_offset
            blade_size = fixture_instance.blade_size(blade_center_z_index)

            # Apply scanner offsets to convert data to true machine coordinates
            adjusted_blade_profile_mm = helpers.apply_scanner_offsets(blade_profile_mm, x_axis_offset)
            
            # For each cartridge these values will differ
            # Top blade is index 1 and bottom blade is index 0
            
            blade_z_shape_json = fixture_instance.blade_z_level(blade_center_z_index, top = top)
            z_shape = segments_module.create_shape_from_stored_shape(blade_z_shape_json)
            wedge_instance = wedge_module.Wedge.parse_from_fixture_instance(fixture_instance, blade_center_z_index, x_axis_offset, top)
            blade_instance = Blade(blade_size, adjusted_blade_profile_mm, z_shape, true_center_of_blade_x, wedge_instance)

            # helpers.plotter_instance.plot_series_set([
            #     {
            #         "name": "wedge_instance",
            #         "data": wedge_instance.profile
            #     },
            #     {
            #         "name": "blade_instance",
            #         "data": blade_instance.profile
            #     }
            # ], lock_aspect_ratio=False)

            blades.append(blade_instance)

        return blades   

    @staticmethod
    def parse_blades_from_scan_file(scan_file_path):
        # Process our blades
        scan_file = scan.ScanFile(scan_file_path)
        scan_instance = scan.Scan(scan_file.to_dataframe())
        # Reverse the order of the blades so we cut the top blade first
        blade_instances = Blade.parse_from_scan_instance(scan_instance)[::-1]
        return blade_instances

class ProfilingSpecification():
    def __init__(self, start_percentage, end_percentage, radius, pitch=None):
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage
        self.radius = radius
        self.pitch = pitch

    def to_json(self):
        return {
            "radius": self.radius,
            "start_percentage": self.start_percentage,
            "end_percentage": self.end_percentage,
            "pitch": self.pitch
        }

    def copy(self):
        return ProfilingSpecification(self.start_percentage, self.end_percentage, self.radius, pitch = self.pitch)

    @staticmethod
    def multiple_to_json(profiling_specifications):
        return_jsons = list(map(lambda profiling_specification: profiling_specification.to_json(),profiling_specifications))
        return return_jsons

    @staticmethod
    def multiple_from_request(profile_parameters):

        # TODO: Order the specifications from heel to toe

        profiling_specifications = []

        for profile_parameter in profile_parameters:

            profiling_specification = ProfilingSpecification(profile_parameter["start_percentage"], profile_parameter["end_percentage"], profile_parameter["radius"])

            if "pitch" in profile_parameter.keys():
                # The request pitch is relative to the perceived effect of the pitch
                # This will require modification to the pitch in the opposite direction
                profiling_specification.pitch = profile_parameter["pitch"] * -1

            profiling_specifications.append(profiling_specification)

        return profiling_specifications

    @staticmethod
    def multiple_from_config(profile_sections):

        # TODO: Order the specifications from heel to toe

        profiling_specifications = []

        for profile_section in profile_sections:

            profiling_specification = ProfilingSpecification(profile_section["startPercentage"], profile_section["endPercentage"], profile_section["radius"])
            if "pitch" in profile_section.keys():
                profiling_specification.pitch = profile_section["pitch"]

            profiling_specifications.append(profiling_specification)

        return profiling_specifications

# TODO: Update name.
# This name now conflicts with our database toolpath.parameters column.
class ToolpathParameter():

    def __init__(self, blade_instance, feedrate, spindle_rpm):
        self.blade_instance = blade_instance
        self.feedrate = feedrate
        self.spindle_rpm = spindle_rpm


def create_line_segment_from_gap(left_segment, right_segment):
    x1 = left_segment.get_end()
    y1 = left_segment.y(x1)
    x2 = right_segment.get_start()
    y2 = right_segment.y(x2)

    return segments_module.LineBladeSegment.from_two_points(x1, y1, x2, y2)


def create_circle_segment_from_gap(left_segment, right_segment):
    left_segment_end_x = left_segment.get_end()
    right_segment_start_x = right_segment.get_start()

    # Create cubic bezier between gap
    left_segment_end_y = left_segment.y(left_segment_end_x)
    P0 = {
        "x": left_segment_end_x,
        "y": left_segment_end_y
    }
    left_segment_end_derivative = left_segment.derivative(left_segment_end_x)

    
    right_segment_start_y = right_segment.y(right_segment_start_x)
    P3 = {
        "x": right_segment_start_x,
        "y": right_segment_start_y
    }
    right_segment_start_derivative = right_segment.derivative(right_segment_start_x)

    gap_cubic_bezier_segment = segments_module.CubicBezierBladeSegment.from_two_points_and_derivatives(left_segment.raw_series, P0, P3, left_segment_end_derivative, right_segment_start_derivative)
    
    # Confirm we can test our radicand
    test_radicand = True
    if left_segment_end_x < left_segment.raw_series.index[0] or right_segment_start_x > left_segment.raw_series.index[-1]:
        test_radicand = False

    # Convert cubic bezier to circle segment
    gap_circle_segment = segments_module.CircleBladeSegment.from_cubic_bezier_segment(left_segment.raw_series, gap_cubic_bezier_segment, test_radicand=test_radicand)

    return gap_circle_segment