import abc
import math
import copy
import numpy as np
import pandas as pd

from modules import helpers
import definitions

##########
# Models #
##########

class Circle():
    """
    Represents a circle with equation r^2 = (x-x0)^2 + (y-y0)^2 in length space
    """

    def __init__(self, x0, y0, radius, radicand=1):
        self.x0 = x0
        self.y0 = y0
        self.radius = radius
        self.radicand = radicand
        
    def y(self, x):
        """
        r^2 = (x-x0)^2 + (y-y0)^2
        (r^2 - (x-x0)^2)^1/2 + y0 = y
        """
        return self.y0 + self.radicand * ( self.radius ** 2 - (x - self.x0) ** 2) ** (1/2)

    def derivative(self, x):
        """
        r^2 = (x-x0)^2 + (y-y0)^2
        0 = 2(x-x0) + 2(y-y0)dy/dx
        dy/dx = x0 - x / (y-y0)
        """
        numerator = self.x0 - x
        denominator = self.y(x) - self.y0

        return numerator / denominator

    def identifier_string(self):
        return f"({round(self.x0, 2)}, {round(self.y0, 2)}) r:{round(self.radius, 2)}"

    ############
    # Mutators #
    ############

    def y_shift(self, shift_value):
        self.y0 += shift_value
    
    def x_shift(self, shift_value):
        self.x0 += shift_value

    ##############
    # Generators #
    ##############

    def copy(self):
        return Circle(self.x0, self.y0, self.radius, radicand = self.radicand)

    def to_json(self):
        return {
            "class_name":  self.__class__.__name__,
            "x0": self.x0,
            "y0": self.y0,
            "radius": self.radius,
            "radicand": self.radicand
        }

    @staticmethod
    def from_stored_shape(stored_shape):
        return Circle(
            x0=stored_shape['x0'],
            y0=stored_shape['y0'],
            radius=stored_shape['radius'],
            radicand=stored_shape['radicand']
        )

    @staticmethod
    def from_3_points(x1, y1, x2, y2, x3, y3):
        # Use the intersections of two orthoganal lines to determine the center
        p12_line = Line.from_two_points(x1, y1, x2, y2)
        p12_x = (x2 + x1) / 2
        p12_y = (y2 + y1) / 2
        p12_inverse_line = p12_line.inverse(new_x1=p12_x, new_y1=p12_y)

        p23_line = Line.from_two_points(x2, y2, x3, y3)
        p23_x = (x3 + x2) / 2
        p23_y = (y3 + y2) / 2
        p23_inverse_line = p23_line.inverse(new_x1=p23_x, new_y1=p23_y)

        center_point = p12_inverse_line.intersection_point(p23_inverse_line)

        # Find radius from center point
        radius = ((x1 - center_point["x"]) ** 2 + (y1 - center_point["y"]) ** 2) ** (1/2)

        return Circle(center_point["x"], center_point["y"], radius)

    @staticmethod
    def from_3_points_old(x1, y1, x2, y2, x3, y3):
        """
        https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
        """
        x12 = x1 - x2  
        x13 = x1 - x3  
    
        y12 = y1 - y2  
        y13 = y1 - y3  
    
        y31 = y3 - y1
        y21 = y2 - y1
    
        x31 = x3 - x1
        x21 = x2 - x1 
    
        # x1^2 - x3^2  
        sx13 = pow(x1, 2) - pow(x3, 2)
    
        # y1^2 - y3^2  
        sy13 = pow(y1, 2) - pow(y3, 2)
    
        sx21 = pow(x2, 2) - pow(x1, 2)
        sy21 = pow(y2, 2) - pow(y1, 2)
    
        f = (((sx13) * (x12) + (sy13) * 
            (x12) + (sx21) * (x13) + 
            (sy21) * (x13)) // (2 * 
            ((y31) * (x12) - (y21) * (x13))))
                
        g = (((sx13) * (y12) + (sy13) * (y12) + 
            (sx21) * (y13) + (sy21) * (y13)) // 
            (2 * ((x31) * (y12) - (x21) * (y13))))
    
        c = (-pow(x1, 2) - pow(y1, 2) - 
            2 * g * x1 - 2 * f * y1)
    
        # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0  
        # where centre is (x0 = -g, y0 = -f) and  
        # radius r as r^2 = x0^2 + y0^2 - c  
        x0 = -g
        y0 = -f
        sqr_of_r = x0 * x0 + y0 * y0 - c
    
        # r is the radius  
        r = math.sqrt(sqr_of_r)

        potential_circle = Circle(x0, y0, r)

        return potential_circle

class Polynomial():
    """
    Represents a polynomial.

    Consider the form y = Ax^5 + Bx^4 + Cx^3 + Dx^2 + Ex + F 
    """

    def __init__(self, coefficients, order, x_translate = 0):
        self.coefficients = coefficients
        self.order = order
        self.x_translate = x_translate

    def y(self, x):
        """
        return y value given an x
        """
        accumulator = 0
        for index, value in enumerate(self.coefficients):
            accumulator += value * (x - self.x_translate) ** (self.order - index)

        return accumulator

    def derivative(self, x):
        """
        return derivative value given an x

        first index will be nan
        """
        accumulator = 0
        for index, value in enumerate(self.coefficients):
            accumulator += (self.order - index) * value * (x - self.x_translate) ** (self.order - index - 1)
        return accumulator
    
    def identifier_string(self):
        coefficients_string = "["
        for coefficient in self.coefficients:
            coefficients_string += str(round(coefficient,2)) + ","

        # Trim last comma
        coefficients_string = coefficients_string[:len(coefficients_string) - 1] + "]"


        return f"o:{self.order} c: {coefficients_string}"

    ############
    # Mutators #
    ############

    def y_shift(self, shift_value):
        self.coefficients[-1] += shift_value

    def x_shift(self, shift_value):
        self.x_translate += shift_value

    ##############
    # Generators #
    ##############

    def copy(self):
        return Polynomial(self.coefficients, self.order, x_translate = self.x_translate)

    def to_json(self):
        return {
            "class_name":  self.__class__.__name__,
            "coefficients": self.coefficients,
            "order": self.order,
            "x_translate": self.x_translate
        }

    @staticmethod
    def from_stored_shape(stored_shape):
        return Polynomial(stored_shape["coefficients"], stored_shape["order"], stored_shape["x_translate"])

    @staticmethod
    def from_points(points, order):
        # Create our system of equations and solve for polynomial coefficients
        # Follows: https://www.mathsisfun.com/algebra/systems-linear-equations-matrices.html
        # Create matrix of x values
        x_matrix = []
        y_matrix = []
        for index, value in enumerate(points):
            x_matrix_row = []
            current_iteration = order
            while current_iteration >= 0:
                x_matrix_row.append(value["x"] ** current_iteration)
                current_iteration -= 1

            x_matrix.append(x_matrix_row)
            y_matrix.append(value["y"])

        # Inverse x matrix
        inversed_x_matrix = np.linalg.inv(x_matrix)       
        coefficients = np.dot(inversed_x_matrix, y_matrix)
        return Polynomial(coefficients, order)

    @staticmethod
    def find_best_polynomial(series, min_order=2, max_order=10):

        order_range = range(min_order, max_order + 1, 1)
        possible_polynomials = []

        for order in order_range:
            test_polynomial = Polynomial.from_series_using_points(series, order)
                
            test_segment_difference = helpers.calculate_segment_to_series_difference(series, test_polynomial)

            possible_polynomials.append({
                "difference": test_segment_difference,
                "shape": test_polynomial
            })

        possible_polynomials.sort(key=lambda element: element["difference"])
        best_polynomial = possible_polynomials[0]["shape"]
        return best_polynomial

    @staticmethod
    def from_series_using_points(series, order):
        local_series = series.copy()
        # We interpolate and smooth to remove any odd features in the series
        interpolated_series = local_series.interpolate(method='akima')
        # TODO: Will be more accurate to smooth over 2 axes
        smoothed_interpolated_series = interpolated_series.ewm(alpha=0.1).mean()

        # Need need to solve for order + 1 coefficients
        points_required = order + 1
        points = []
        total_number_of_values = smoothed_interpolated_series.size
        step_size = math.floor(total_number_of_values / points_required)

        # Add start point
        start_x = smoothed_interpolated_series.index[0]
        start_y = smoothed_interpolated_series[start_x]
        points.append({
            "x": start_x, 
            "y": start_y
        })
        
        # Add mid points
        current_step = 1
        while len(points) < (points_required - 1):

            x_index = current_step * step_size
            x_value = smoothed_interpolated_series.index.values[x_index]
            y_value = smoothed_interpolated_series[x_value]

            points.append({
                "x": x_value, 
                "y": y_value
            })

            current_step += 1

        # Add end point
        end_x = smoothed_interpolated_series.index[-1]
        end_y = smoothed_interpolated_series[end_x]
        points.append({
            "x": end_x, 
            "y": end_y
        })
        
        return Polynomial.from_points(points, order)

    @staticmethod
    def from_series_using_derivatives(series, order, key_index, key_derivative):
        local_series = series.copy()
        # We interpolate and smooth to remove any odd features in the series
        interpolated_series = local_series.interpolate(method='akima')
        # TODO: Will be more accurate to smooth over 2 axes
        smoothed_interpolated_series = interpolated_series.ewm(alpha=0.1).mean()
        smoothed_interpolated_derivative_series = smoothed_interpolated_series.diff()

        points = []

        # Add our key index
        if key_index == 0:
            key_index = 1 #We will not have a derivative at the first point so use the second point

        # Add key derivative and regular point
        key_x = smoothed_interpolated_derivative_series.index[key_index]
        derivative_point ={
            "x": key_x, 
            "derivative": key_derivative
        }

        points.append({
            "x": key_x, 
            "y": smoothed_interpolated_series[key_x]
        })

        # Add end point
        end_x_index = key_index * (-1/2) - 1/2
        end_x = smoothed_interpolated_series.index.values[int(end_x_index)]
        end_y = smoothed_interpolated_series[end_x]
        points.append({
            "x": end_x,
            "y": end_y
        })

        # Add mid points
        points_required = order + 1 # We will need to solve for order + 1 coefficents
        points_required -= 3 # Minus 3 since we already have 3 points
        total_number_of_values = smoothed_interpolated_series.size
        step_size = math.floor(total_number_of_values / (points_required + 1))
        
        # Add mid points
        current_step = 1 # start at 1 since we alreayd included our start and end points
        while points_required > 0:

            x_index = current_step * step_size
            x_value = smoothed_interpolated_series.index.values[x_index]
            y_value = smoothed_interpolated_series[x_value]

            points.append({
                "x": x_value, 
                "y": y_value
            })

            current_step += 1
            points_required -= 1

        # Create our system of equations and solve for polynomial coefficients
        # Follows: https://www.mathworks.com/matlabcentral/answers/51206-determining-polynomial-coefficients-by-known-derivatives
        # Create matrix of x values
        x_matrix = []
        y_matrix = []
        for index, point in enumerate(points):
            x_matrix_row = []
            current_iteration = order
            while current_iteration >= 0:
                x_matrix_row.append(point["x"] ** current_iteration)
                current_iteration -= 1

            x_matrix.append(x_matrix_row)
            y_matrix.append(point["y"])
        
        derivative_x_matrix_row = []
        current_iteration = order - 1 # Minus one since we are considering the derivative function of the polynomial
        while current_iteration >= 0:
            derivative_x_matrix_row.append((current_iteration + 1) * derivative_point["x"] ** current_iteration)
            current_iteration -= 1
        
        derivative_x_matrix_row.append(0.0) # Add one 0 for the last coefficent
        x_matrix.append(derivative_x_matrix_row)
        y_matrix.append(derivative_point["derivative"])
        
        # Inverse x matrix
        inversed_x_matrix = np.linalg.inv(x_matrix)
        coefficients = np.dot(inversed_x_matrix, y_matrix)

        return Polynomial(coefficients, order)

class CubicBezier():
    """
    https://en.wikipedia.org/wiki/B%C3%A9zier_curve

    A Cubic Bézier curve is a parametric curve with the form:

    B(t) = (1 - t)^3 * P0 + 3(1-t)^2 * tP1 + 3(1-t)t^2 * P2 + t^3 * P3,  where 0 <= t <= 1
    B'(t) = 3(1 - t)^2 * (P1 - P0) + 6(1 - t)t(P2 - P1) + 3t^2 * (P3 - P2),  where 0 <= t <= 1

    and P0, P1, P2, P3, are anchor points which define the curve.

    P1 and P2 are the handle control points for P0 and P3 respectively.

    NOTE: This is only for a monotonic curve

    """
    def __init__(self, P0, P1, P2, P3, t_set=None):
        self.P0 = P0
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3

        # Keep track of previously found t values
        if t_set is None:
            self.t_set = {}
        else:
            self.t_set = t_set

    def test_x_in_bounds(self, x, tolerance=0.000000001):
        # Check out of bound cases
        if (x - self.P3["x"]) > tolerance:
            raise Exception(f'error t > 1, due to x: ({x}) being greater than P3_x: ({self.P3["x"]})')
        elif (x - self.P0["x"]) < tolerance * -1:
            raise Exception(f'error t < 0, due to x: ({x}) being less than P0_x: ({self.P0["x"]})')

    def t(self, x, x_tolerance=0.000000001):
        
        self.test_x_in_bounds(x)

        # Check if t value has already been calculated
        if x in self.t_set:
            return self.t_set[x]

        # Binary search solution from https://stackoverflow.com/questions/7348009/y-coordinate-for-a-given-x-cubic-bezier
        # The domain of t is [0, 1] giving us our lower and upper bounds
        lower_bound = 0
        upper_bound = 1
        test_t = (upper_bound + lower_bound) / 2

        test_x = self.point_from_t(test_t)["x"]

        # Binary search
        while (abs(x - test_x) > x_tolerance):
            if x > test_x:
                lower_bound = test_t
            else:
                upper_bound = test_t
            
            test_t = (upper_bound + lower_bound) / 2
            test_x = self.x_from_t(test_t)

        self.t_set[x] = test_t
        return test_t

    # Potential runtime issues
    # Compare with t method
    # Creating imaginary roots
    def t_new(self, x):

        self.test_x_in_bounds(x) 

        # Check if t value has already been calculated
        if x in self.t_set:
            return self.t_set[x]

        # Simplify for t
        # B(t) = (1 - t)^3 * P0 + 3(1-t)^2 * tP1 + 3(1-t)t^2 * P2 + t^3 * P3,  where 0 <= t <= 1
        # B(t) = t^3(-P0 + 3P1 - 3P2 + P3) + t^2(3P0 - 6P1 + 3P2) + t ( -3P0 + 3P1 ) + P0
        # 0 = t^3(-P0 + 3P1 - 3P2 + P3) + t^2(3P0 - 6P1 + 3P2) + t ( -3P0 + 3P1 ) + P0 - B(t)
        # Solve for t

        first_coefficient = -self.P0["x"] + 3 * self.P1["x"] - 3 * self.P2["x"] + self.P3["x"]
        second_coefficient = 3 * self.P0["x"] - 6 * self.P1["x"] + 3 * self.P2["x"]
        third_coefficient = -3 * self.P0["x"] + 3 * self.P1["x"]
        fourth_coefficient = self.P0["x"] - x

        coefficients = [
            first_coefficient,
            second_coefficient,
            third_coefficient,
            fourth_coefficient
        ]

        roots = np.roots(coefficients)

        for root in roots:

            rounded_root = round(root, 12)

            if rounded_root >= 0 and rounded_root <= 1:
                self.t_set[x] = rounded_root
                return rounded_root

        raise Exception(f'No viable roots found for x:{x}. Roots:{roots}')

    def point_from_t(self, t):
        return {
            "x": self.x_from_t(t),
            "y": self.y_from_t(t)
        }

    def y(self, x):
        t = self.t(x)

        return self.y_from_t(t)

    def derivative(self, x):
        t = self.t(x)

        first_term = 3 * (1 - t)**2 * (self.P1["y"] - self.P0["y"])
        second_term =  6 * (1 - t) * t * (self.P2["y"] - self.P1["y"])
        third_term = 3 * t**2 * (self.P3["y"] - self.P2["y"])

        derivative_numerator = first_term + second_term + third_term

        first_term = 3 * (1 - t)**2 * (self.P1["x"] - self.P0["x"])
        second_term =  6 * (1 - t) * t * (self.P2["x"] - self.P1["x"])
        third_term = 3 * t**2 * (self.P3["x"] - self.P2["x"])

        derivative_denomenator = first_term + second_term + third_term
        derivative = derivative_numerator / derivative_denomenator
        return derivative
    
    def x_from_t(self, t):

        # Q0 is the line from P0 to P1
        Q0_x = (1 - t) * self.P0["x"] + t * self.P1["x"]
        Q1_x = (1 - t) * self.P1["x"] + t * self.P2["x"]
        Q2_x = (1 - t) * self.P2["x"] + t * self.P3["x"]

        # N0 is the line from Q0 to Q1
        N0_x = (1 - t) * Q0_x + t * Q1_x
        N1_x = (1 - t) * Q1_x + t * Q2_x

        M0_x = (1 - t) * N0_x + t * N1_x
        return M0_x

    def y_from_t(self, t):

        # Q0 is the line from P0 to P1
        Q0_y = (1 - t) * self.P0["y"] + t * self.P1["y"]
        Q1_y = (1 - t) * self.P1["y"] + t * self.P2["y"]
        Q2_y = (1 - t) * self.P2["y"] + t * self.P3["y"]

        # N0 is the line from Q0 to Q1
        N0_y = (1 - t) * Q0_y + t * Q1_y
        N1_y = (1 - t) * Q1_y + t * Q2_y

        M0_y = (1 - t) * N0_y + t * N1_y
        return M0_y

    def y_from_t_old(self, t):

        first_term = (1 - t)**3 * self.P0["y"] 
        second_term = 3 * (1 - t)**2 * t * self.P1["y"]
        third_term = 3 * (1 - t) * t**2 * self.P2["y"]
        fourth_term = t**3 * self.P3["y"]

        y = first_term + second_term + third_term + fourth_term
        return y

    def x_from_t_old(self, t):
        
        first_term = (1 - t)**3 * self.P0["x"] 
        second_term = 3 * (1 - t)**2 * t * self.P1["x"]
        third_term = 3 * (1 - t) * t**2 * self.P2["x"]
        fourth_term = t**3 * self.P3["x"]

        x = first_term + second_term + third_term + fourth_term
        return x

    def identifier_string(self):
        return f'({round(self.P0["x"],2)},{round(self.P0["y"],2)}) \
        ({round(self.P1["x"],2)},{round(self.P1["y"],2)}) \
        ({round(self.P2["x"],2)},{round(self.P2["y"],2)}) \
        ({round(self.P3["x"],2)},{round(self.P3["y"],2)})'

    ############
    # Mutators #
    ############

    def y_shift(self, shift_value):
        self.P0["y"] += shift_value
        self.P1["y"] += shift_value
        self.P2["y"] += shift_value
        self.P3["y"] += shift_value

    def x_shift(self, shift_value):
        self.P0["x"] += shift_value
        self.P1["x"] += shift_value
        self.P2["x"] += shift_value
        self.P3["x"] += shift_value

        # Reset t_set values
        self.t_set = {}

    #############
    # Exporting #
    #############

    def to_json(self):
        return {
            "class_name":  self.__class__.__name__,
            "P0": self.P0,
            "P1": self.P1,
            "P2": self.P2,
            "P3": self.P3
        }

    ##############
    # Generators #
    ##############

    def copy(self):
        return CubicBezier( \
            copy.deepcopy(self.P0), \
            copy.deepcopy(self.P1), \
            copy.deepcopy(self.P2), \
            copy.deepcopy(self.P3), \
            t_set = copy.deepcopy(self.t_set))

    def split_at_x(self, x):
        t = self.t(x)

        # Split up the components of the Bezier curve
        # https://imgur.com/gs3ybKr

        M0 = {
            "x": self.P0["x"] + (self.P1["x"] - self.P0["x"]) * t,
            "y": self.P0["y"] + (self.P1["y"] - self.P0["y"]) * t,
        }

        M1 = {
            "x": self.P1["x"] + (self.P2["x"] - self.P1["x"]) * t,
            "y": self.P1["y"] + (self.P2["y"] - self.P1["y"]) * t,
        }

        M2 = {
            "x": self.P2["x"] + (self.P3["x"] - self.P2["x"]) * t,
            "y": self.P2["y"] + (self.P3["y"] - self.P2["y"]) * t,
        }

        Q0 = {
            "x": M0["x"] + (M1["x"] - M0["x"]) * t,
            "y": M0["y"] + (M1["y"] - M0["y"]) * t,
        }

        Q1 = {
            "x": M1["x"] + (M2["x"] - M1["x"]) * t,
            "y": M1["y"] + (M2["y"] - M1["y"]) * t,
        }

        Bt = {
            "x": x,
            "y": self.y(x)
        }

        return [
            CubicBezier(self.P0, M0, Q0, Bt),
            CubicBezier(Bt, Q1, M2, self.P3)
        ]

    @staticmethod
    def from_stored_shape(stored_shape):
        return CubicBezier(
            P0=stored_shape['P0'],
            P1=stored_shape['P1'],
            P2=stored_shape['P2'],
            P3=stored_shape['P3']
        )

    @staticmethod
    def confirm_handle_control_point_conditions_default(cubic_bezier_instance):
        # Default handle point conditions
        # P1_x >= P0_x
        # P1_x <= P2_x
        # P2_x <= P3_x

        if cubic_bezier_instance.P1["x"] < cubic_bezier_instance.P0["x"]:
            # Our start control point should not be behind our start point
            # This would cause an unacceptable shape (https://imgur.com/2mx46Cr)
            return False
        elif cubic_bezier_instance.P1["x"] > cubic_bezier_instance.P2["x"]:
            # Our start control point should not be in front of our end control point
            # This would cause an unacceptable shape (https://imgur.com/yMhsGfG)
            return False
        elif cubic_bezier_instance.P2["x"] > cubic_bezier_instance.P3["x"]:
            # Our end control point should not be in in front of our end point
            # This would cause an unacceptable shape (https://imgur.com/OhSnYfP)
            return False

        return True

    @staticmethod
    def confirm_handle_control_point_conditions_start(cubic_bezier_instance):
        # Start handle point conditions
        # P1_y >= P0_y
        # P1_y <= P2_y
        # P2_y <= P3_y

        if CubicBezier.confirm_handle_control_point_conditions_default(cubic_bezier_instance) == False:
            return False
        elif cubic_bezier_instance.P1["y"] < cubic_bezier_instance.P0["y"]:
            # Our start control point should not be in below our start point
            # This would cause an unacceptable shape (https://imgur.com/bDGaSWP)
            return False
        elif cubic_bezier_instance.P1["y"] > cubic_bezier_instance.P2["y"]:
            # Our start control point should not be in above our end control point
            # This would cause an unacceptable shape (https://imgur.com/hQEujie)
            return False
        elif cubic_bezier_instance.P2["y"] > cubic_bezier_instance.P3["y"]:
            # Our end control point should not be in above our end point
            # This would cause an unacceptable shape (https://imgur.com/1lV5jdM)
            return False

        return True

    @staticmethod
    def confirm_handle_control_point_conditions_end(cubic_bezier_instance):
        # End handle point conditions
        # P1_y <= P0_y
        # P1_y >= P2_y
        # P2_y >= P3_y

        if CubicBezier.confirm_handle_control_point_conditions_default(cubic_bezier_instance) == False:
            return False
        elif cubic_bezier_instance.P1["y"] > cubic_bezier_instance.P0["y"]:
            # Our start control point should not be above our start point
            # This would cause an unacceptable shape (https://imgur.com/PFT0x8J)
            return False
        elif cubic_bezier_instance.P1["y"] < cubic_bezier_instance.P2["y"]:
            # Our start control point should not be in below our end control point
            # This would cause an unacceptable shape (https://imgur.com/9Hojk2q)
            return False
        elif cubic_bezier_instance.P2["y"] < cubic_bezier_instance.P3["y"]:
            # Our end control point should not be in below our end point
            # This would cause an unacceptable shape (https://imgur.com/snGMxg6)
            return False

        return True

    # Not correctly implemented
    # DO NOT USE
    @staticmethod
    def from_series(series, curve_factor, circle_segment):
        # Find boundary conditions
        x0 = series.index[0] #TODO: fix negative indexing bug when it comes to doing the last segment
        x3 = series.index[-1]

        y0 = series[x0]
        y3 = ((circle_segment.raw_shape.radius**2 - (x3 - circle_segment.raw_shape.x0)**2)**(1/2) + circle_segment.raw_shape.y0)

        """
        Find points P0, P1, P2, P3

        where P0 = [x0, y0] is from the series
        where P3 = [x3, y3] is from the circle

        Given Boundary Condition of a finite smooth concave downward curve joining the series with the circle

        y(x0) = y0
        y(x3) = y3
        dy_series/dx|x=x3 = dy_circle/dx|x=x3
        x1 = x0
        y1 = (y3 - y0) / curve_factor
        
        How to solve BVP: http://tutorial.math.lamar.edu/Classes/DE/BoundaryValueProblem.aspx

        """
        # arbitrary curve factor

        P0 = [x0, y0]
        
        y_x3 = (circle_segment.raw_shape.radius**2 - (x3 - circle_segment.raw_shape.x0)**2)**(1/2) + circle_segment.raw_shape.y0

        x2 = (x0 + x3) / 2        
        
        # we want our P2 to have the same x as P0 to ensure concavity
        y2 = y3 + (x3 - circle_segment.raw_shape.x0)**2 / (circle_segment.raw_shape.y0 - y_x3)
        y1 = (y2 - y0) / curve_factor
        # we want our P1 to have the same x as P0 to ensure concavity

        P1 = [x0, y1]
        P2 = [x2, y2]
        P3 = [x3, y3]

        return CubicBezier(P0, P1, P2, P3)

    @staticmethod
    def from_series_new(series, start_index, end_index):
        P0 = {
            "x": start_index,
            "y": series[start_index]
        }

        P3 = {
            "x": end_index,
            "y": series[end_index]
        }

        domain_length = P3["x"] - P0["x"]
        control_point_x_delta = domain_length * 0.25

        series_diff = series.diff() / definitions.SPACING_PER_X_INDEX
        P0_diff = series_diff[P0["x"]]
        P3_diff = series_diff[P3["x"]]

        # We do not need to match our series indices for the control points
        # P1_x = helpers.find_closest_index(P0["x"] + control_point_x_delta, series)
        P1_x = P0["x"] + control_point_x_delta
        P1 = {
            "x": P1_x,
            "y": P0["y"] + P0_diff * control_point_x_delta
        }

        # We do not need to match our series indices for the control points
        # P2_x = helpers.find_closest_index(P3["x"] - control_point_x_delta, series)
        P2_x = P3["x"] - control_point_x_delta
        P2 = {
            "x": P2_x,
            "y": P3["y"] - P3_diff * control_point_x_delta
        }

        return CubicBezier(P0, P1, P2, P3)

class Line():
    """
    Represents a line for the lead in and lead out tool path

    Given the slope 

    input our distance in length and have an output in length
    slope has to be converted

    x0, y0 should be your first point or last point

    """

    def __init__(self, x1, y1, m):
        self.x1 = x1
        self.y1 = y1
        self.m = m
        self.b = self.y1 - (self.m * self.x1)  

    def y(self, x):
        y = self.m * x + self.b
        return y

    def derivative(self, x):
        return self.m

    def identifier_string(self):
        return f"m:{round(self.m,2)} ({round(self.x1,2)},{round(self.y1,2)})"

    ############
    # Mutators #
    ############

    def y_shift(self, shift_value):
        self.b += shift_value
        self.y1 += shift_value

    def x_shift(self, shift_value):
        self.x1 += shift_value
        self.b = self.y1 - (self.m * self.x1)

    def intersection_point(self, another_line):
        b_diff = another_line.b - self.b
        new_m = self.m - another_line.m
        x_point = b_diff / new_m

        return {
            "x": x_point,
            "y": self.y(x_point)
        }

    def inverse(self, new_x1=None, new_y1=None):
        inverse_slope = -1/self.m

        if new_x1 is None:
            new_x1 = self.x1
        
        if new_y1 is None:
            new_y1 = self.y1

        return Line(new_x1, new_y1, inverse_slope)

    ##############
    # Generators #
    ##############

    def copy(self):
        return Line(self.x1, self.y1, self.m)

    def to_json(self):
        return {
            "class_name":  self.__class__.__name__,
            "x1": self.x1,
            "y1": self.y1,
            "m": self.m,
            "b": self.b
        }

    @staticmethod
    def from_two_points(x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        return Line(x1, y1, m)

    @staticmethod
    def from_stored_shape(stored_shape):
        return Line(stored_shape["x1"], stored_shape["y1"], stored_shape["m"])

#############
# Interface #
#############

class BladeSegment(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def get_start(self):
        """
        The start of the segment in length space.
        """
        raise NotImplementedError('Must define get_start property to use this base class')

    @property
    @abc.abstractmethod
    def get_end(self):
        """
        The end of the segment in length space
        """
        raise NotImplementedError('Must define get_end property to use this base class')

    @property
    @abc.abstractmethod
    def get_shape(self):
        """
        The mathematical shape of the segment
        """
        raise NotImplementedError('Must define get_shape property to use this base class')

    @classmethod
    @abc.abstractmethod
    def y(self, x):
        """
        Y value at a given x position in length space
        """
        raise NotImplementedError('Must define y method to use this base class')

    @classmethod
    @abc.abstractmethod
    def derivative(self, x):
        """
        Rate of change of Y value at a given x position in length space
        """
        raise NotImplementedError('Must define derivative method to use this base class')

###########
# Useable #
###########

class LineBladeSegment(BladeSegment):
    def __init__(self, start, end, raw_shape):
        self.start = start
        self.end = end
        self.raw_shape = raw_shape.copy()

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_shape(self):
        # TODO: Our parent class should implement this method and return the raw_shape class name
        return self.raw_shape.__class__.__name__

    def y(self, x):
        return self.raw_shape.y(x)

    def derivative(self, x):
        return self.raw_shape.derivative(x)

    ############
    # Mutators #
    ############

    def y_shift(self, shift_value):
        self.raw_shape.y_shift(shift_value)

    def x_shift(self, shift_value):
        self.start += shift_value
        self.end += shift_value
        self.raw_shape.x_shift(shift_value)

    def perpendicular_shift(self, shift_value):

        derivative = self.derivative(self.get_start())
        true_shift_value = shift_value if derivative < 0 else -shift_value # Reverse shift direction
        inverse_derivative = -1 / derivative
        inverse_vector = helpers.derivative_to_vector_components(inverse_derivative, true_shift_value)

        new_segment = self.copy()
        new_segment.start += inverse_vector["x"]
        new_segment.end += inverse_vector["x"]

        new_x1 = new_segment.raw_shape.x1 + inverse_vector["x"]
        new_y1 = new_segment.raw_shape.y1 + inverse_vector["y"]
        new_segment.raw_shape = Line(new_x1, new_y1, new_segment.raw_shape.m)

        return new_segment

    def set_start(self, start_value):
        # For lines we can just mutate the start value
        self.start = start_value

    def set_end(self, end_value):
        # For lines we can just mutate the end value
        self.end = end_value

    ##############
    # Generators #
    ##############

    def copy(self):
        return LineBladeSegment(self.start, self.end, self.raw_shape.copy())

    @staticmethod
    def from_two_points(x1, y1, x2, y2):
        raw_shape = Line.from_two_points(x1, y1, x2, y2)
        return LineBladeSegment(x1, x2, raw_shape)

    @staticmethod
    def from_two_x_values_of_series(series, x1, x2):
        y1 = series[x1]
        y2 = series[x2]

        return LineBladeSegment.from_two_points(x1, y1, x2, y2)

    @staticmethod
    def convert_series_to_line_segments(series, start_index=None, end_index=None, min_segment_size=5):

        if start_index is None:
            start_index = series.index.values[0]

        if end_index is None:
            end_index = series.index.values[-1]

        # Make sure min_segment_size is positive
        assert min_segment_size > 0, f"LineBladeSegment.convert_series_to_line_segments, Step size ({min_segment_size}) needs to be at greater than 0"

        local_series =  series.loc[start_index:end_index]

        # Make sure min_segment_size is smaller than our series x axis length
        local_series_x_axis_length = local_series.index.values[-1] - local_series.index.values[0]
        assert min_segment_size <= local_series_x_axis_length, f"LineBladeSegment.convert_series_to_line_segments, Step size of {min_segment_size} is too large"

        line_segments = []

        x1_index = 0

        for x2_index in range(1, local_series.size):
            
            # calculate the size of potential segment from x1 to x2
            x1 = local_series.index.values[x1_index]
            x2 = local_series.index.values[x2_index]
            diff_x = x2 - x1
            diff_y = local_series[x2] - local_series[x1]
            hypotenuse_of_segment = math.sqrt((diff_x ** 2) + (diff_y ** 2))

            # create a line segment if the hypotenuse meets the requirement for the minimum size
            if(hypotenuse_of_segment >= min_segment_size):
                line_segment = LineBladeSegment.from_two_x_values_of_series(local_series, x1, x2)
                line_segments.append(line_segment)
                x1_index = x2_index
                
        return line_segments

    def to_json(self):
        return {
            "class_name": self.get_shape(),
            "start": self.start,
            "end": self.end,
            "raw_shape": self.raw_shape.to_json()
        }

    def to_web_profile(self, number_of_points=None):
        
        x_1 = self.get_start()
        x_2 = self.get_end()

        y_1 = self.y(x_1)
        y_2 = self.y(x_2)

        points_x = [x_1, x_2]
        points_y = [y_1, y_2]

        return {
            "x": points_x,
            "y": points_y
        }

    @staticmethod
    def from_stored_segment(stored_segment, raw_series):
        return LineBladeSegment(
            start=stored_segment['start'],
            end=stored_segment['end'],
            raw_shape=Line.from_stored_shape(stored_segment['raw_shape'])
        )

class CircleBladeSegment(BladeSegment):

    def __init__(self, start, end, raw_shape, raw_series, test_radicand=True, profiling=False):
        self.start = start
        self.end = end
        self.raw_shape = raw_shape.copy()
        self.raw_series = raw_series.copy()
        self.profiling = profiling

        if test_radicand:
            self.confirm_radicand()

    def confirm_radicand(self):
        # Check if we have the correct radicand
        # In some cases our circle is actually the negative radicand
        # Compare y delta between positive and negative radicand
        circle_start_x = self.get_start()
        circle_start_y = self.y(circle_start_x)
        series_start_y = self.raw_y(circle_start_x)
        generated_circle_y_delta = abs(series_start_y - circle_start_y)

        reversed_raw_circle = self.raw_shape.copy()
        reversed_raw_circle.radicand *= -1
        reversed_circle_start_y = reversed_raw_circle.y(circle_start_x)
        reversed_generated_circle_y_delta = abs(series_start_y - reversed_circle_start_y)

        if reversed_generated_circle_y_delta < generated_circle_y_delta:
            self.raw_shape = reversed_raw_circle

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_shape(self):
        # TODO: Our parent class should implement this method and return the raw_shape class name
        return self.raw_shape.__class__.__name__

    def y(self, x):
        return self.raw_shape.y(x)

    def raw_y(self, x):
        if self.raw_series is None:
            assert("No raw series included during instantiation!")
        else:
            return helpers.find_best_series_value_at_index(self.raw_series, x)

    def derivative(self, x):
        return self.raw_shape.derivative(x)

    ############
    # Mutators #
    ############

    def y_shift(self, shift_value):
        self.raw_shape.y_shift(shift_value)
        
        if self.raw_series is not None:
            self.raw_series += shift_value

    def x_shift(self, shift_value):
        self.start += shift_value
        self.end += shift_value
        self.raw_shape.x_shift(shift_value)

        if self.raw_series is not None:
            self.raw_series = helpers.shift_series_indices(self.raw_series, shift_value)

    def set_start(self, start_value):
        # For circles we can just mutate the start value
        self.start = start_value

    def set_end(self, end_value):
        # For circles we can just mutate the end value
        self.end = end_value
        
    def to_json(self):
        return {
            "class_name": self.get_shape(),
            "start": self.start,
            "end": self.end,
            "profile": self.profiling,
            "raw_shape": self.raw_shape.to_json()
        }

    def to_web_profile(self, number_of_points=None):
        
        domain_length = self.end - self.start
        if number_of_points is None:
            resolution = definitions.SPACING_PER_X_INDEX
            number_of_points = round(domain_length / resolution)
        
        point_increment = domain_length / number_of_points

        point_counter = 0
        points_x = []
        points_y = []

        while(point_counter <= number_of_points):
            point_x = point_counter * point_increment + self.start
            point_y = self.y(point_x)

            points_x.append(round(point_x, 3))
            points_y.append(round(point_y, 3))
            point_counter += 1

        return {
            "x": points_x,
            "y": points_y
        }

    ##############
    # Generators #
    ##############

    def copy(self):
        return CircleBladeSegment(self.start, self.end, self.raw_shape.copy(), self.raw_series.copy(), profiling=self.profiling, test_radicand=False)

    def perpendicular_shift(self, shift_value):

        # Confirm shift is possible before making shift
        if confirm_perpendicular_shift(self, shift_value) == False:
            # Segment should just be removed at this point
            # Returning None here would be sufficient but it will not perserve our segments if we need to shift back
            # Better to return a circle with a skip property
            # When creating gcode this skip property should be checked before the segment is output to gcode
            return None

        start_derivative = self.derivative(self.get_start())
        start_shift_value = shift_value if start_derivative < 0 else -shift_value # Reverse shift direction
        start_inverse_derivative = -1 / start_derivative
        start_inverse_vector = helpers.derivative_to_vector_components(start_inverse_derivative, start_shift_value)

        new_start = self.start + start_inverse_vector["x"]

        end_derivative = self.derivative(self.get_end())
        end_shift_value = shift_value if end_derivative < 0 else -shift_value # Reverse shift direction
        end_inverse_derivative = -1 / end_derivative
        end_inverse_vector = helpers.derivative_to_vector_components(end_inverse_derivative, end_shift_value)

        new_end = self.end + end_inverse_vector["x"]

        if self.raw_shape.radicand == 1:
            new_radius = self.raw_shape.radius + shift_value
        else:
            new_radius = self.raw_shape.radius - shift_value

        new_raw_shape = Circle(self.raw_shape.x0, self.raw_shape.y0, new_radius, radicand=self.raw_shape.radicand)
        new_blade_segment = CircleBladeSegment(new_start, new_end, new_raw_shape, self.raw_series, profiling=self.profiling, test_radicand=False)

        return new_blade_segment

    def perpendicular_shift_old(self, shift_value):

        # Confirm shift is possible before making shift
        if confirm_perpendicular_shift(self, shift_value) == False:
            # Segment should just be removed at this point
            # Returning None here would be sufficient but it will not perserve our segments if we need to shift back
            # Better to return a circle with a skip property
            # When creating gcode this skip property should be checked before the segment is output to gcode
            return None

        start_x_point = self.get_start()
        start_y_point = self.y(start_x_point)
        end_x_point = self.get_end()
        end_y_point = self.y(end_x_point)
        mid_x_point = (start_x_point + end_x_point) / 2
        mid_y_point = self.y(mid_x_point)

        points = [
            {
                "x": start_x_point,
                "y": start_y_point,
            },
            {
                "x": mid_x_point,
                "y": mid_y_point,
            },
            {
                "x": end_x_point,
                "y": end_y_point,
            },
        ]

        new_points = []

        for point in points:

            derivative = self.derivative(point["x"])

            if derivative == 0:
                x_shift = 0
                y_shift = shift_value

            else:

                inverse_derivative = -1/derivative
                x_axis_theta = math.atan(inverse_derivative)

                true_shift_value = shift_value if derivative < 0 else -shift_value

                x_shift = true_shift_value * math.cos(x_axis_theta)
                y_shift = true_shift_value * math.sin(x_axis_theta)

            new_points.append({
                "x": point["x"] + x_shift,
                "y": point["y"] + y_shift,
            })

        new_start_x_point = new_points[0]["x"]
        new_end_x_point = new_points[-1]["x"]

        return CircleBladeSegment.from_3_points( \
            new_points[0]["x"], \
            new_points[0]["y"], \
            new_points[1]["x"], \
            new_points[1]["y"], \
            new_points[2]["x"], \
            new_points[2]["y"], \
            start = new_start_x_point, \
            end = new_end_x_point, \
            raw_series = self.raw_series, \
            test_radicand=False \
        )

    def to_line_segments(self, number_of_line_segments=10):

        line_x_size = (self.get_end() - self.get_start()) / (number_of_line_segments)
        key_x_points = []
        while len(key_x_points) < (number_of_line_segments + 1):
            x_point = self.get_start() + line_x_size * len(key_x_points)
            key_x_points.append(x_point)

        line_segments = []
        for key_point_index, key_point_x in enumerate(key_x_points):

            # Ignore last point
            if key_point_index > (len(key_x_points) - 2):
                continue

            # Create line
            x_1 = key_point_x
            x_2 = key_x_points[key_point_index + 1]

            y_1 = self.y(x_1)
            y_2 = self.y(x_2)

            line_segment = LineBladeSegment.from_two_points(x_1, y_1, x_2, y_2)
            line_segments.append(line_segment)
        
        return line_segments


    @staticmethod
    def from_stored_segment(stored_segment, raw_series):
        return CircleBladeSegment(
            start=stored_segment['start'],
            end=stored_segment['end'],
            profiling=stored_segment['profiling'],
            raw_series=raw_series,
            raw_shape=Circle.from_stored_shape(stored_segment['raw_shape'])
        )

    @staticmethod
    def from_series(series):
        start_x = series.index[0]
        end_x = series.index[-1]

        segment_length = series.size
        mid_x = series.index[int(segment_length/2)]

        start_y = series[start_x]
        mid_y = series[mid_x]
        end_y = series[end_x]

        return CircleBladeSegment.from_3_points( \
            start_x, start_y, \
            mid_x, mid_y, \
            end_x, end_y, \
            raw_series = series \
        )

    @staticmethod
    def from_3_points(x1, y1, x2, y2, x3, y3, start=None, end=None, raw_series=None, profiling=False, test_radicand=True):

        if start is None:
            start = x1

        if end is None:
            end = x3

        raw_shape = Circle.from_3_points(x1, y1, x2, y2, x3, y3)
        return CircleBladeSegment(start, end, raw_shape, raw_series, profiling=profiling, test_radicand=test_radicand)

    @staticmethod
    def convert_series_to_circle_segments(series, start_index=None, end_index=None, key_derivative_index=0, key_derivative=None):
        if start_index is None:
            start_index = series.index.values[0]

        if end_index is None:
            end_index = series.index.values[-1]

        if key_derivative is None:
            key_derivative = series.diff()[start_index] / definitions.SPACING_PER_X_INDEX

        segments = CircleBladeSegment.series_to_circle_segments(series, key_derivative_index, key_derivative)

        return segments

    @staticmethod
    def series_to_circle(series, start, end, key_derivative_index, key_derivative):

        circle_start_x = start
        circle_start_y = series[circle_start_x]

        circle_end_x = end
        circle_end_y = series[circle_end_x]

        series_derivative = series.diff() / definitions.SPACING_PER_X_INDEX 

        if key_derivative_index == 0:
            circle_start_derivative = key_derivative
            circle_end_derivative = series_derivative[circle_end_x]
        else:
            circle_start_derivative = series_derivative[circle_start_x]
            circle_end_derivative = key_derivative
            

        # circle_start_derivative = series_derivative[circle_start_x]
        # circle_end_derivative = series_derivative[circle_end_x]

        circle_start_inverse_derivative = -1 * circle_start_derivative ** -1
        circle_start_orthogonal_line = Line(circle_start_x, circle_start_y, circle_start_inverse_derivative)

        circle_end_inverse_derivative = -1 * circle_end_derivative ** -1
        circle_end_orthogonal_line = Line(circle_end_x, circle_end_y, circle_end_inverse_derivative)

        # Our intersection point is the center of the circle
        intersection_point = circle_start_orthogonal_line.intersection_point(circle_end_orthogonal_line)

        # TODO: Find a better way that uses both points
        # Might be better to create circles using one derivative and two points
        # The point we use for our derivative should be defined by the side of the cut we are on
        # If we are on the start of the cut we should use the end derivative
        # If we are on the end of the cut we should use the start derivative
        intersection_start_point_x_delta = abs(circle_start_x - intersection_point["x"])
        intersection_start_point_y_delta = abs(circle_start_y - intersection_point["y"])

        radius = (intersection_start_point_x_delta ** 2 + intersection_start_point_y_delta ** 2 ) ** (1/2)
        circle = Circle(intersection_point["x"], intersection_point["y"], radius)

        return CircleBladeSegment(circle_start_x, circle_end_x, circle, series)

    @staticmethod
    def series_to_circle_new(series, start, end, key_derivative_index, key_derivative):
     
        # For points A (start_x, start_y) and B (end_x, end_y) with tangent line L
        circle_start_x = start
        circle_start_y = series[circle_start_x]

        circle_end_x = end
        circle_end_y = series[circle_end_x]

        # If key_derivative_index is 0
        # L goes through A
        # Else L goes through B
        if key_derivative_index == 0:
            key_line = Line(circle_start_x, circle_start_y, key_derivative)
        else:
            key_line = Line(circle_end_x, circle_end_y, key_derivative)
            
        # Find inverse of L
        inverse_key_line = key_line.inverse()
        
        # Find biarc between A and B name it K
        point_biarc = Line.from_two_points(circle_start_x, circle_start_y, circle_end_x, circle_end_y)
        biarc_point = {
            "x": (circle_end_x - circle_start_x) / 2 + circle_start_x,
            "y": (circle_end_y - circle_start_y) / 2 + circle_start_y,
        }

        # Find inverse of K
        inverse_point_biarc = point_biarc.inverse(new_x1 = biarc_point["x"], new_y1 = biarc_point["y"])

        # Find intersection of L and K
        # This is the center of the circle
        intersection_point = inverse_key_line.intersection_point(inverse_point_biarc)
        intersection_start_point_x_delta = abs(circle_start_x - intersection_point["x"])
        intersection_start_point_y_delta = abs(circle_start_y - intersection_point["y"])

        radius = (intersection_start_point_x_delta ** 2 + intersection_start_point_y_delta ** 2 ) ** (1/2)
        circle = Circle(intersection_point["x"], intersection_point["y"], radius)

        return CircleBladeSegment(circle_start_x, circle_end_x, circle, series)
        
    @staticmethod
    def series_to_circle_segments(series, key_derivative_index, key_derivative, start_index=None, end_index=None):
        if start_index is None:
            start_index = series.index.values[0]

        if end_index is None:
            end_index = series.index.values[-1]

        max_circle_segment_difference = definitions.APP_CONFIG["toolPathGenerator"]["maxCircleSegmentDifference"]

        # Break up the segment into multiple circles    
        segment_length = end_index - start_index
        segment_minimum_length = definitions.SPACING_PER_X_INDEX
        
        # TODO: Investigate cases where very small circles act as bumps
        # This seems to be caused by negative radicand circles and the derivatives they provide
        if segment_length < segment_minimum_length:
            print(f"Segment length is less than our segment minimum length!")

        circle_segment = CircleBladeSegment.series_to_circle(series, start_index, end_index, key_derivative_index, key_derivative)
        # circle_segment = CircleBladeSegment.series_to_circle_new(series, start_index, end_index, key_derivative_index, key_derivative)

        series_test_range = [start_index, end_index]
        circle_segment_difference = helpers.calculate_segment_series_difference(series, circle_segment, series_test_range)

        circle_segments = []
        if circle_segment_difference < max_circle_segment_difference or segment_length < segment_minimum_length or helpers.float_equivalence(segment_length, segment_minimum_length):
            circle_segments.append(circle_segment)        
        else:
            
            left_start = start_index
            left_end = helpers.find_closest_index(left_start + segment_length / 2, series)

            if key_derivative_index == 0:
                
                left_circle_segments = CircleBladeSegment.series_to_circle_segments(series, key_derivative_index, key_derivative, start_index=left_start, end_index=left_end)

                joining_circle_segment = left_circle_segments[-1]
                joining_circle_segment_end = joining_circle_segment.get_end()
                joining_circle_segment_end_derivative = joining_circle_segment.derivative(joining_circle_segment_end)

                right_start = left_end
                right_end = end_index

                right_circle_segments = CircleBladeSegment.series_to_circle_segments(series, key_derivative_index, joining_circle_segment_end_derivative, start_index=right_start, end_index=right_end)

            else:                

                right_start = left_end
                right_end = end_index
                right_circle_segments = CircleBladeSegment.series_to_circle_segments(series, key_derivative_index, key_derivative, start_index=right_start, end_index=right_end)

                joining_circle_segment = right_circle_segments[0]
                joining_circle_segment_start = joining_circle_segment.get_start()
                joining_circle_segment_start_derivative = joining_circle_segment.derivative(joining_circle_segment_start)

                left_circle_segments = CircleBladeSegment.series_to_circle_segments(series, key_derivative_index, joining_circle_segment_start_derivative, start_index=left_start, end_index=left_end)

            circle_segments = left_circle_segments + right_circle_segments
        
        return circle_segments
    
    @staticmethod
    def for_profiling(series, start, end, radius, key_derivative_index, key_derivative):

        inverse_key_derivative = -1 / key_derivative
        to_circle_center_theta = math.atan(inverse_key_derivative)

        circle_center_x_delta = radius * math.cos(to_circle_center_theta)
        circle_center_y_delta = radius * math.sin(to_circle_center_theta)

        if key_derivative < 0:
            circle_center_x_delta *= -1
            circle_center_y_delta *= -1

        if key_derivative_index == 0:
            circle_reference_x = start
        else:
            circle_reference_x = end

        circle_reference_y = series[circle_reference_x]
        circle_center_x = circle_reference_x + circle_center_x_delta
        circle_center_y = circle_reference_y + circle_center_y_delta

        raw_circle = Circle(circle_center_x, circle_center_y, radius)

        return CircleBladeSegment(start, end, raw_circle, series, profiling=True)

    @staticmethod
    def for_profiling_center(series, start, end, radius, x0):
        x0_index =  helpers.find_closest_index(x0, series)

        circle_apex_y = series[x0_index]
        # We subtract the radius at the apex of the circle to find the center of the circle in Y
        y0 = circle_apex_y - radius

        raw_circle = Circle(x0_index, y0, radius)
        return CircleBladeSegment(start, end, raw_circle, series, profiling=True)

    @staticmethod
    def from_cubic_bezier_segment(series, segment, test_radicand=True):

        x1 = segment.get_start()
        x3 = segment.get_end()
        x2 = (x1 + x3) / 2

        y1 = segment.y(x1)
        y2 = segment.y(x2)
        y3 = segment.y(x3)

        return CircleBladeSegment.from_3_points(x1, y1, x2, y2, x3, y3, start=x1, end=x3, raw_series=series, test_radicand=test_radicand)


class PolynomialBladeSegment(BladeSegment):
    def __init__(self, series, raw_shape):
        self.raw_shape = raw_shape
        self.raw_series = series

        self.start = series.index.values[0]
        self.end = series.index.values[-1]

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_shape(self):
        # TODO: Our parent class should implement this method and return the raw_shape class name
        return self.raw_shape.__class__.__name__

    def y(self, x):
        return self.raw_shape.y(x)

    def raw_y(self, x):
        return helpers.find_best_series_value_at_index(self.raw_series, x)

    def derivative(self, x):
        return self.raw_shape.derivative(x)

    ############
    # Mutators #
    ############

    def y_shift(self, shift_value):
        self.raw_shape.y_shift(shift_value)

        if self.raw_series is not None:
            self.raw_series += shift_value

    def x_shift(self, shift_value):
        self.start += shift_value
        self.end += shift_value
        self.raw_shape.x_shift(shift_value)

        if self.raw_series is not None:
            self.raw_series = helpers.shift_series_indices(self.raw_series, shift_value)

    ##############
    # Generators #
    ##############

    def copy(self):
        local_raw_shape = self.raw_shape.copy()
        local_raw_series = self.raw_series.copy()
        return PolynomialBladeSegment(local_raw_series, local_raw_shape)

    @staticmethod
    def from_series_using_points(series, order):
        rawPolynomial = Polynomial.from_series_using_points(series, order)
        return PolynomialBladeSegment(series, rawPolynomial)

    @staticmethod
    def from_series_using_derivatives(series, order, key_index, key_derivative):
        rawPolynomial = Polynomial.from_series_using_derivatives(series, order, key_index, key_derivative)
        return PolynomialBladeSegment(series, rawPolynomial)

class CubicBezierBladeSegment(BladeSegment):
    """
    Fits series over a segmented domain as a cubic bezier curve
    """

    def __init__(self, start, end, raw_series, raw_shape):
        self.start = start
        self.end = end
        self.raw_shape = raw_shape.copy()
        self.raw_series = raw_series.copy()

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_shape(self):
        # TODO: Our parent class should implement this method and return the raw_shape class name
        return self.raw_shape.__class__.__name__

    def y(self, x):
        return self.raw_shape.y(x)

    def raw_y(self, x):
        if self.raw_series is None:
            assert("No raw series included during instantiation!")
        else:
            return helpers.find_best_series_value_at_index(self.raw_series, x)

    def derivative(self, x):
        return self.raw_shape.derivative(x)

    ############
    # Mutators #
    ############

    def y_shift(self, shift_value):
        self.raw_shape.y_shift(shift_value)

        if self.raw_series is not None:
            self.raw_series += shift_value

    def x_shift(self, shift_value):
        self.start += shift_value
        self.end += shift_value
        self.raw_shape.x_shift(shift_value)

        if self.raw_series is not None:
            self.raw_series = helpers.shift_series_indices(self.raw_series, shift_value)

    def set_start(self, start_value):
        
        if start_value < self.start:
            assert("Cannot set start value of Bezier curve to less than original start. Create a new \
                Bezier curve instead.")

        self.start = start_value

        # We also need to update the raw shape
        # Split the Bezier at start_value and take the right most curve
        self.raw_shape = self.raw_shape.split_at_x(start_value)[1]

    def set_end(self, end_value):

        if end_value > self.end:
            assert("Cannot set end value of Bezier curve to more than original end. Create a new \
                Bezier curve instead.")

        self.end = end_value

        # We also need to update the raw shape
        # Split the Bezier at end_value and take the left most curve
        self.raw_shape = self.raw_shape.split_at_x(end_value)[0]

    def reset_control_points(self, control_point_x_percentage=25, start_derivative=None, end_derivative=None):

        segment_length = self.get_end() - self.get_start()
        control_point_x_delta = segment_length * control_point_x_percentage / 100

        if start_derivative is None:
            start_derivative = self.derivative(self.get_start())

        if end_derivative is None:
            end_derivative = self.derivative(self.get_end())

        self.raw_shape.P1 = {
            "x": self.raw_shape.P0["x"] + control_point_x_delta,
            "y": self.raw_shape.P0["y"] + start_derivative * control_point_x_delta
        }

        self.raw_shape.P2 = {
            "x": self.raw_shape.P3["x"] - control_point_x_delta,
            "y": self.raw_shape.P3["y"] - end_derivative * control_point_x_delta
        }

    #############
    # Exporting #
    #############

    def to_json(self):
        return {
            "class_name": self.get_shape(),
            "start": self.start,
            "end": self.end,
            "raw_shape": self.raw_shape.to_json()
        }

    def to_web_profile(self, number_of_points=None):
        
        domain_length = self.end - self.start
        if number_of_points is None:
            resolution = definitions.SPACING_PER_X_INDEX
            number_of_points = round(domain_length / resolution)
        
        point_increment = domain_length / number_of_points

        point_counter = 0
        points_x = []
        points_y = []

        while(point_counter <= number_of_points):
            point_x = point_counter * point_increment + self.start
            point_y = self.y(point_x)

            points_x.append(round(point_x, 3))
            points_y.append(round(point_y, 3))
            point_counter += 1

        return {
            "x": points_x,
            "y": points_y
        }

    ##############
    # Generators #
    ##############

    def copy(self):
        return CubicBezierBladeSegment(self.start, self.end, self.raw_series, self.raw_shape)


    def perpendicular_shift(self, shift_value):

        # the control points will be set arbitrarily since they will be reset at the end of this method
        new_P1  = {
            "x": 0.0,
            "y": 0.0
        }

        new_P2 = {
            "x": 0.0,
            "y": 0.0
        }

        # Calculate control point x percentage for when we reset our control points
        segment_length = self.get_end() - self.get_start()
        control_point_x_percentage = abs(self.raw_shape.P1["x"] - self.raw_shape.P0["x"]) / segment_length * 100

        # Find new P0 and P3
        start_derivative = self.derivative(self.get_start())
        start_inverse_derivative = -1 / start_derivative
        start_inverse_vector = helpers.derivative_to_vector_components(start_inverse_derivative, shift_value)
        
        if start_derivative > 0:
            start_inverse_vector["x"] *= -1
            start_inverse_vector["y"] *= -1

        new_P0 = {
            "x": self.raw_shape.P0["x"] + start_inverse_vector["x"],
            "y": self.raw_shape.P0["y"] + start_inverse_vector["y"]
        }

        end_derivative = self.derivative(self.get_end())
        end_inverse_derivative = -1 / end_derivative
        end_inverse_vector = helpers.derivative_to_vector_components(end_inverse_derivative, shift_value)

        if end_derivative > 0:
            end_inverse_vector["x"] *= -1
            end_inverse_vector["y"] *= -1

        new_P3 = {
            "x": self.raw_shape.P3["x"] + end_inverse_vector["x"],
            "y": self.raw_shape.P3["y"] + end_inverse_vector["y"]
        }

        new_segment = CubicBezierBladeSegment.from_series_with_points(self.raw_series, new_P0, new_P1, new_P2, new_P3)

        new_segment.reset_control_points(control_point_x_percentage=control_point_x_percentage, start_derivative=start_derivative, end_derivative=end_derivative)

        return new_segment

    @staticmethod
    def from_stored_segment(stored_segment, raw_series):
        return CubicBezierBladeSegment(
            start=stored_segment['start'],
            end=stored_segment['end'],
            raw_series=raw_series,
            raw_shape=CubicBezier.from_stored_shape(stored_segment['raw_shape'])
        )

    @staticmethod
    def from_series_with_points(series, P0, P1, P2, P3):
        raw_shape = CubicBezier(P0, P1, P2, P3)
        return CubicBezierBladeSegment(P0["x"], P3["x"], series, raw_shape)

    @staticmethod
    def from_two_points_and_derivatives(series, P0, P3, left_derivative, right_derivative, control_point_x_delta_percentage=35):

        x_delta_between_points = P3["x"] - P0["x"]
        x_delta_to_control_points = x_delta_between_points * control_point_x_delta_percentage / 100

        P1 = {
            "x": P0["x"] + x_delta_to_control_points,
            "y": P0["y"] + left_derivative * x_delta_to_control_points
        }

        P2 = {
            "x": P3["x"] - x_delta_to_control_points,
            "y": P3["y"] - right_derivative * x_delta_to_control_points
        }

        return CubicBezierBladeSegment.from_series_with_points(series, P0, P1, P2, P3)

    @staticmethod
    def from_series(series, curve_factor, circle_segment):
        start = series.index.values[0]
        end = series.index.values[-1]
        raw_shape = CubicBezier.from_series(series, curve_factor, circle_segment)
        return CubicBezierBladeSegment(start, end, series, raw_shape)

    @staticmethod
    def from_series_new(series, start_index, end_index):
        raw_shape = CubicBezier.from_series_new(series, start_index, end_index)
        return CubicBezierBladeSegment(start_index, end_index, series, raw_shape)

    @staticmethod
    def convert_series_to_segments(series, start_index, end_index):

        max_segment_difference = definitions.APP_CONFIG["toolPathGenerator"]["maxCircleSegmentDifference"]

        segment_length = end_index - start_index
        segment_minimum_length = definitions.SPACING_PER_X_INDEX
        test_segment = CubicBezierBladeSegment.from_series_new(series, start_index, end_index)

        if abs(segment_length - segment_minimum_length) < 0.001:
            print(f"Segment length is less than our segment minimum length!")
            return [test_segment]

        series_test_range = [start_index, end_index]
        segment_difference = helpers.calculate_segment_series_difference(series, test_segment, series_test_range)

        return_segments = []
        if segment_difference < max_segment_difference or segment_length < segment_minimum_length:
            
            return_segments.append(test_segment)

        else:
            
            center_index = helpers.find_closest_index(start_index + segment_length / 2, series)
            left_segments = CubicBezierBladeSegment.convert_series_to_segments(series, start_index, center_index)
            right_segments = CubicBezierBladeSegment.convert_series_to_segments(series, center_index, end_index)

            return_segments = left_segments + right_segments

        return return_segments

def create_segment_from_stored_segment(stored_segment, raw_series):
    
    # Confirm class type
    segment_class_switcher = {
        "CubicBezier": CubicBezierBladeSegment,
        "Circle": CircleBladeSegment,
        "Line": LineBladeSegment
    }

    target_class = segment_class_switcher.get(stored_segment["class_name"])

    return target_class.from_stored_segment(stored_segment, raw_series)

def create_shape_from_stored_shape(stored_shape):

    # Confirm class type
    shape_class_switcher = {
        "CubicBezier": CubicBezier,
        "Line": Line,
        "Polynomial": Polynomial,
        "Circle": Circle
    }

    target_class = shape_class_switcher.get(stored_shape["class_name"])

    return target_class.from_stored_shape(stored_shape)

def confirm_perpendicular_shift(segment, shift_value):
    # Confirm shift is possible before making shift
    # There are cases where our perpendicular flips the circle
    # We could use the circle center for this calculation but we should
    # create a function that can handle this for all types of segments
    segment_start_point_x = segment.get_start()
    segment_start_point_y = segment.y(segment_start_point_x)
    segment_start_derivative = segment.derivative(segment_start_point_x)
    segment_start_inverse_derivative = -1/segment_start_derivative
    segment_start_inverse_line = Line(segment_start_point_x, segment_start_point_y, segment_start_inverse_derivative)

    segment_end_point_x = segment.get_end()
    segment_end_point_y = segment.y(segment_end_point_x)
    segment_end_derivative = segment.derivative(segment_end_point_x)
    segment_end_inverse_derivative = -1/segment_end_derivative
    segment_end_inverse_line = Line(segment_end_point_x, segment_end_point_y, segment_end_inverse_derivative)

    intersection_point = segment_start_inverse_line.intersection_point(segment_end_inverse_line)

    if intersection_point is None:
        # Derivatives are parallel
        return True

    displacement_from_start_point = ((intersection_point["x"] - segment_start_point_x)**2 + (intersection_point["y"] - segment_start_point_y)**2) ** (1/2)
    intersection_point_y_direction = -1 if segment_start_point_y > intersection_point["y"] else 1
    shift_direction = -1 if shift_value < 0 else 1

    if abs(shift_value) >= displacement_from_start_point and shift_direction == intersection_point_y_direction:
        print(f"Can not shift segment (start: {segment_start_point_x} end: {segment_end_point_x}) by shift_value: {shift_value}. Max shift in direction is {displacement_from_start_point}")
        return False