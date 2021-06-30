from xmlrpc.client import ServerProxy, Error, ProtocolError

import definitions
from modules import helpers

class Mill():
    """ 
    Controls for our CnC mill hardware.

    Still need a lot of investigating to implement this integration.

    Keyword Arguments:
        debug {boolean} -- Enable extra output for troubleshooting.
         (default: {False})
    """

    def __init__(self):
        self.host = definitions.APP_CONFIG["mill"]["host"]
        self.port = definitions.APP_CONFIG["mill"]["port"]
        self.server_address = f"{self.host}:{str(self.port)}"

        # Connection testing will be done at method calltime
        self.rpc_connection = ServerProxy(self.server_address)

    def send_request(self, function_reference, *args, **kwargs):

        try:
            raw_mill_response = function_reference(*args, **kwargs)
            mill_response = MillResponse(raw_mill_response)
            # Not sure if we need to do any more processing of the response here
            # response structure should be:
            # success : <bool>
            # message : <string>
            # data : <mixed>
            return mill_response

        except ProtocolError as err:
            print("A protocol error occurred")
            print("URL: %s" % err.url)
            print("HTTP/HTTPS headers: %s" % err.headers)
            print("Error code: %d" % err.errcode)
            print("Error message: %s" % err.errmsg)
            return False, err

    def mdi(self, command, wait = True):
        """ 
        Runs a set of Gcode commands in MDI mode.

        Arguments:
            gcode {Gcode || string} -- Gcode class instance or text to run.
        """
        return self.send_request(self.rpc_connection.mdi, command, wait)
    
    def gcode(self, file_path, wait = True):
        return self.send_request(self.rpc_connection.gcode, file_path, wait)

    def reset(self):
        return self.send_request(self.rpc_connection.reset)

    def status(self):
        return self.send_request(self.rpc_connection.status)

    def confirm_work_offsets(self):
        expected_cut_work_offsets = definitions.APP_CONFIG['toolPathGenerator']['workOffsets']['cut']
        expected_scan_work_offsets = definitions.APP_CONFIG['toolPathGenerator']['workOffsets']['scan']

        actual_cut_work_offsets_response = self.send_request(self.rpc_connection.get_work_offsets, expected_cut_work_offsets['number'])
        actual_cut_work_offsets = actual_cut_work_offsets_response.data
        actual_scan_work_offsets_response = self.send_request(self.rpc_connection.get_work_offsets, expected_scan_work_offsets['number'])
        actual_scan_work_offsets = actual_scan_work_offsets_response.data

        if check_work_offset_equivalence(expected_scan_work_offsets, actual_scan_work_offsets) == False:
            return "Scan work offset mismatched!"

        if check_work_offset_equivalence(expected_cut_work_offsets, actual_cut_work_offsets) == False:
            return "Cut work offset mismatched!"

        return None

class MillResponse():

    def __init__(self, raw_mill_response):
        self.data = raw_mill_response["data"]
        self.message = raw_mill_response["message"]
        self.success = raw_mill_response["success"]

def check_work_offset_equivalence(expected_offset, actual_offset):

    x_match = helpers.float_equivalence(expected_offset["x"], actual_offset[0] * 25.4)
    y_match = helpers.float_equivalence(expected_offset["y"], actual_offset[1] * 25.4)
    z_match = helpers.float_equivalence(expected_offset["z"], actual_offset[2] * 25.4)

    return (x_match and y_match and z_match)
