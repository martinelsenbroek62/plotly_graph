from datetime import datetime
import definitions

class Gcode():
    """ 
    Stores a Gcode program in an array where each index contains one line of 
    Gcode text. Can be converted to a string via it’s __str__ method.

    Keyword Arguments:
        debug_mode {boolean} -- Enable extra output for troubleshooting.
         (default: {False})
    """

    def __init__(self, mock=False, debug_mode=definitions.APP_CONFIG["toolPathGenerator"]["debug"]):
        self.debug_mode = debug_mode
        self.lines = []

        now = datetime.now()
        file_name_timestamp = now.strftime("%Y%m%d%H%M%S%f")
        file_name_prefix = file_name_timestamp + "_output"
        file_name_suffix = ".nc" if mock == False else "_mock.nc"
        self.file_name = file_name_prefix + file_name_suffix

    def update_line(self, command, line_number=0):
        """ 
        Updates a line in our Gcode program.

        Arguments:
            command {string} -- The gcode line of text to add. You do not need
             to include a semicolon.

        Keyword Arguments:
            line_number {int} -- Enable extra output for troubleshooting.
            (default: {False})

        Raises:
            Exception -- line number index not found.
        """

        # Check line number exists or is next available line
        max_acceptable_line_number = len(self.lines) - 1
        if (line_number > max_acceptable_line_number):
            raise Exception(f"Line number {line_number} does not exists in current Gcode program!")

        self.lines[line_number] = command

    def append_line(self, command):
        """ 
        Append a line to our Gcode program.

        Arguments:
            command {string} -- The gcode line of text to add. You do not need
             to include a semicolon.
        """
        # nextLineNumber = len(self.lines)
        self.lines.append(command)

    def append(self, text):
        """ 
        Append to last line in our Gcode program.

        Arguments:
            text {string} -- The gcode text to append. You do not need
             to include a semicolon.
        """
        
        self.lines[-1] += text

    def export(self):
        gcode_file = open(f"{definitions.OUTPUT_DIR_PATH}/{self.file_name}", 'w')
        gcode_file.write(str(self))

    def __str__(self):
        """ 
        Outputs our Gcode program with each index being a new line delimited 
        by a semicolon and newline character.
        """
        delimiter = ';\n'
        return delimiter.join(self.lines) + delimiter
