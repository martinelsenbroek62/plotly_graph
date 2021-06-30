import time

from datetime import datetime
import requests
from requests.exceptions import HTTPError

import definitions

class Scanner():
    """ 
    Controls for our laser scanner hardware.

    Keyword Arguments:
        debug {boolean} -- Enable extra output for troubleshooting.
         (default: {False})
    """

    def __init__(self):
        self.batch_profiles = definitions.APP_CONFIG["scanner"]["batchProfiles"]
        self.host = definitions.APP_CONFIG["scanner"]["host"]
        self.port = definitions.APP_CONFIG["scanner"]["port"]
        self.server_address = f"{self.host}:{str(self.port)}"

        # State should be used to make sure we do not send any requests until
        # our connection to the scanner is ready
        self.state = "Initializing"
        
        self.start_connection()

        self.state = "Ready"

    # TODO: Attempt to connect 5 times with increasing delays
    # If already attempting to connect reset the delay and connection counter
    def start_connection(self, connection_attempts=5, connection_delay=0.5):
        self.attempt_connection()

    def attempt_connection(self):
        webserver_connection_success = self.confirm_web_server_connection()

        assert webserver_connection_success, "Scanner webserver connection failed"

        scanner_connection_success = self.initialize_scanner()

        assert scanner_connection_success, "Scanner connection failed"


    def request_handler(self, method, path, payload=None, params=None):

        request_url = self.server_address + path

        if method == "get":
            request_response = requests.get(request_url, params=params)
        elif method == "post":
            request_response = requests.post(request_url, payload, params=params)
        else:
            raise Exception("Invalid request method")
        
        try:
            request_response.raise_for_status()
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')  # Python 3.6
        except Exception as err:
            print(f'Other error occurred: {err}')  # Python 3.6
        
        return request_response

    def confirm_web_server_connection(self):
        response = self.request_handler("get", "/version")

        if response.status_code != 200:
            return False

        print(f"Scanner API version: f{response}")
        return True

    def initialize_scanner(self):
        initialize_response = self.request_handler("post", "/initialize")

        if initialize_response.status_code != 200:

            if initialize_response.content is not None:
                print(initialize_response.content.message)

            return False

        if definitions.APP_CONFIG["scanner"]["connection"]["type"] == "eth":
            params = {
                "host":definitions.APP_CONFIG["scanner"]["connection"]["host"],
                "port":definitions.APP_CONFIG["scanner"]["connection"]["port"]
            }
            connection_response = self.request_handler("post", "/coms/eth/open", params=params)
        else:
            print("usb connection not configured")

        if connection_response.status_code != 200:

            if connection_response.content is not None:
                print(connection_response.content.message)

            return False

        return True
  
    def download_scan_data(self):
        params = {
            "profiles": self.batch_profiles
        }

        response = self.request_handler("get", "/download/batch/advanced", params = params)

        now = datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S%f")
        file_path = f"{definitions.APP_DIR_PATH}/scans/{current_time}.csv"
        open(file_path, "wb").write(response.content)

        return file_path

    def dummy_scan(self, delay):
        self.request_handler("post", "/scan/start")
        time.sleep(delay)
        self.request_handler("post", "/scan/stop")
        return self.request_handler("get", "/download/batch/advanced")

    # Check if Keyence api webserver is up, scanner is responsive, and scanner is not outputting duplicate data
    def status(self):
        webserver_connection_success = self.confirm_web_server_connection()
        if webserver_connection_success == False:
            return {
                "status": False,
                "message": "Scanner API connection error"
            }

        # TODO: Update this section
        # Start scanner for 0.1 seconds
        # self.dummy_scan(0.1)

        # Start scanner for 0.2 seconds to make sure checksum is different
        # dummy_scan_response = self.dummy_scan(0.2)

        # return {
        #     "status": True if dummy_scan_response.status_code == 200 else False,
        #     "message": dummy_scan_response.message
        # }

        return {
            "status": True,
            "message": "ok"
        }