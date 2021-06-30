from smb.SMBConnection import SMBConnection
import tempfile

import definitions

class SmbApi():

    def __init__(self):

        self.server = definitions.APP_CONFIG["mill"]["smb"]["server"]
        self.share = definitions.APP_CONFIG["mill"]["smb"]["share"]
        self.address = f"//{self.server}/{self.share}"
        self.port = definitions.APP_CONFIG["mill"]["smb"]["port"]

        self.status = "Initialized"
        # TODO: Add a test connection method
        # Keep in mind each connection should only be opened to do one thing
        # Most SMB servers have a client timeout built in

    # TODO: Attempt to connect 5 times with increasing delays
    # If already attempting to connect reset the delay and connection counter
    def start_connection(self, connection_attempts=5, connection_delay=0.5):
        self.attempt_connection()

    def attempt_connection(self):
        try:
            potential_connection = SMBConnection("python", "python", "pythonpc", self.address, use_ntlm_v2=True)
            assert potential_connection.connect(self.server, self.port)
            self.connection = potential_connection
            self.status = "Connected"
            return True
        except:
            print("SMB connection error!")
            self.status = "Error - Connection"
            return False

    def write_bytes(self, path, data):
        temp_file = tempfile.TemporaryFile()
        temp_file.write(data)
        temp_file.seek(0)

        self.connection.storeFile(self.share, path, temp_file)

    def read_bytes(self, path):
        temp_file = tempfile.TemporaryFile()

        self.connection.retrieveFile(self.share, path, temp_file)

        return temp_file.read()

    def copy_file(self, file_object, path):
        self.start_connection()
        self.write_bytes(path, file_object.read())
        self.connection.close()
