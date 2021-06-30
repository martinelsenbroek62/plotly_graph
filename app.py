import time, json

from flask import Flask, request, make_response
from flask_cors import CORS

from modules import scanner, mill, smb, scan, blade, tool_path, tool, helpers, db_connection, mill_sessions

import definitions

class Server_Response_Factory():

    @staticmethod
    def create_server_response(status_code, message, data=None):
        return make_response(({
            "status_code": status_code,
            "message": message,
            "data": data,
        }, status_code, None))

# Services
db_interface_instance = None
scanner_instance = None
mill_instance = None
smb_instance = None

# Flask web server definition
webserver = Flask(__name__)
CORS(webserver)

@webserver.route('/status', methods=['GET'])
def get_status():

    scanner_status = None
    mill_status = None

    if definitions.APP_CONFIG["toolPathGenerator"]["localDevelopmentMode"]:
        scanner_status = {
            "status": True,
            "message": "Dummy response. Server in local development mode."
        }

        mill_status = {
            "status": True,
            "message": "Dummy response. Server in local development mode."
        }

    else:
        scanner_status = scanner_instance.status()
        mill_status = mill_instance.status()

    hardware_status = {
        "scanner": scanner_status,
        "mill": mill_status,
    }
    return Server_Response_Factory.create_server_response(200, "Success", hardware_status)

@webserver.route('/scans', methods=['GET'])
def get_scans():
    scans = db_interface_instance.get_scans()
    return Server_Response_Factory.create_server_response(200, "Success", scans)

@webserver.route('/scan/<scan_id>', methods=['GET'])
def get_scan(scan_id):
    scan = db_interface_instance.get_scan(scan_id, with_blades=True, with_toolpaths=True)
    
    # Remove class_instance key as it is only used internally
    for db_blade in scan["blades"]:
        del db_blade["class_instance"]

    if scan is None:
        response = Server_Response_Factory.create_server_response(404, f"Scan with id {scan_id} not found")
    else:
        response = Server_Response_Factory.create_server_response(200, "Success", scan)

    return response

@webserver.route('/toolpaths', methods=['GET'])
def get_toolpaths():

    scan_id = request.args.get('scan_id')
    toolpaths = db_interface_instance.get_toolpaths(scan_id=scan_id)
    return Server_Response_Factory.create_server_response(200, "Success", toolpaths)

@webserver.route('/toolpath/<toolpath_id>', methods=['GET'])
def get_toolpath(toolpath_id):

    toolpath = db_interface_instance.get_toolpath(toolpath_id, with_scan=True, with_blades=True, with_segments=True)
    # Remove class_instance key as it is only used internally
    for db_blade in toolpath["blades"]:
        del db_blade["class_instance"]
        
    if toolpath is None:
        response = Server_Response_Factory.create_server_response(404, f"Toolpath with id {toolpath_id} not found")
    else:
        response = Server_Response_Factory.create_server_response(200, "Success", toolpath)
    return response

@webserver.route('/scanner/scan', methods=['POST'])
def run_scan_program():

    # This section is currently commented out to help develop offsite
    file_path = None
    if definitions.APP_CONFIG["toolPathGenerator"]["localDevelopmentMode"]:

        file_path = f"{definitions.APP_DIR_PATH}/scans/dummy.csv.bz2"

    else:

        # Make sure work offsets match expected values in config
        confirm_work_offsets_response = mill_instance.confirm_work_offsets()
        if confirm_work_offsets_response is not None:
            return Server_Response_Factory.create_server_response(409, confirm_work_offsets_response)

        # Run scan program
        mill_instance.gcode(definitions.APP_CONFIG["mill"]["scanProgramPath"], wait = True)
        
        # Download scan data
        file_path = scanner_instance.download_scan_data()

        print("Download done")

    # This section should be commented out in production
    # file_path = definitions.APP_DIR_PATH + "/scans/test.csv"

    scan_insert_values = [file_path]
    scan_insert_result = db_interface_instance.insert_scan(scan_insert_values)
    scan_id = scan_insert_result["id"]

    blade_instances = blade.Blade.parse_blades_from_scan_file(file_path)
    top_blade_values = [scan_id, True, blade_instances[0].to_json(), blade_instances[0].create_web_profile_json()]
    top_blade_insert_result = db_interface_instance.insert_blade(top_blade_values)
    bottom_blade_values = (scan_id, False, blade_instances[1].to_json(), blade_instances[1].create_web_profile_json())
    bottom_blade_insert_result = db_interface_instance.insert_blade(bottom_blade_values)

    scan = db_interface_instance.get_scan(scan_id)
    if scan is None:
        response = Server_Response_Factory.create_server_response(410, f"Scan {scan_id} not found after insert.", None)
    else:
        response = Server_Response_Factory.create_server_response(200, "Success", scan)

    return response

@webserver.route('/scanner/test', methods=['POST'])
def run_scanner_test():
    return Server_Response_Factory.create_server_response(200, "Not ready")

@webserver.route('/scanner/reboot', methods=['POST'])
def run_scanner_reboot():
    return Server_Response_Factory.create_server_response(200, "Not ready")

@webserver.route('/toolpath/generate', methods=['POST'])
def generate_toolpath():
    
    scan_id = request.json['scan_id']
    # TODO: Run test to make sure we can support this hollow radius
    # Requires quering tools available
    # We should also update the mill_session instances to accept a tool index instead of a hollow_radius
    hollow_radius = request.json['hollow_radius']
    # Convert hollow_radius string to float
    if isinstance(hollow_radius, str):

        # Check to see if we need to remove a back slash character
        split_tool_hollow_radius = hollow_radius.split("/")
        if len(split_tool_hollow_radius) > 1:
            target_hollow_radius = int(split_tool_hollow_radius[0]) / int(split_tool_hollow_radius[1])
        else:
            target_hollow_radius = float(hollow_radius)
    else:
        target_hollow_radius = hollow_radius

    tool_instance = tool.Tool.from_hollow_radius(target_hollow_radius)

    # The from hollow radius method already has an assert in place to handle this, but
    if tool_instance is None:
        return Server_Response_Factory.create_server_response(400, f"No tool available with a radius of {target_hollow_radius}")

    sharpening = request.json['sharpening']

    # Find scan
    scan = db_interface_instance.get_scan(scan_id, with_blades=True)
    if scan is None:
        return Server_Response_Factory.create_server_response(404, f"Scan with id {scan_id} not found")
    
    # Recreate blade instances
    blade_instances = blade.Blade.from_stored_blades(scan["blades"])


    if sharpening == False:
        profile_parameters = request.json['profile_parameters']
        profiling_specifications = blade.ProfilingSpecification.multiple_from_request(profile_parameters)
        mill_session = mill_sessions.ProfilingSession(blade_instances, tool_instance, profiling_specifications)
    else:
        mill_session = mill_sessions.SharpeningSession(blade_instances, tool_instance)

    # Insert toolpath entry
    toolpath_file_path = f"{definitions.OUTPUT_DIR_PATH}/{mill_session.tool_path_instance.gcode_instance.file_name}"
    toolpath_insert_values = [scan_id, definitions.SOFTWARE_VERSION, mill_session.to_json(), toolpath_file_path]
    toolpath_insert_result = db_interface_instance.insert_toolpath(toolpath_insert_values)
    toolpath_id = toolpath_insert_result["id"]

    toolpath = db_interface_instance.get_toolpath(toolpath_id)
    if toolpath is None:
        return Server_Response_Factory.create_server_response(410, f"Toolpath {toolpath_id} not found after insert.", None)
    else:
        response = Server_Response_Factory.create_server_response(200, "Success", toolpath)

    # Insert toolpath segments
    for db_blade_index, db_blade in enumerate(scan["blades"]):
        blade_id = db_blade["id"]
        cut_blade_instances = mill_session.get_cut_blade_instances(db_blade_index)

        # If profiling we only need the last cut
        last_cut_blade_instance = None
        if sharpening == False:
            last_cut_blade_instance = cut_blade_instances[-1]
        else:
            last_cut_blade_instance = cut_blade_instances[0]

        if(mill_session.clears_wedge(last_cut_blade_instance.profile, last_cut_blade_instance.wedge_instance.profile) == False):
            return Server_Response_Factory.create_server_response(409, f"Cannot cut. Cut is too deep, will cut wedge", None)

        segments_web_profiles = last_cut_blade_instance.segments_web_profiles()

        for segment_web_profile in segments_web_profiles:
            toolpath_segment_insert_values = [toolpath_id, blade_id, segment_web_profile]
            toolpath_segment_insert_result = db_interface_instance.insert_toolpath_segment(toolpath_segment_insert_values)

    return response

@webserver.route('/toolpath/generate/toollife', methods=['POST'])
def generate_toollife_toolpath():
    
    scan_id = request.json['scan_id']
    # TODO: Run test to make sure we can support this hollow radius
    # Requires quering tools available
    # We should also update the mill_session instances to accept a tool index instead of a hollow_radius
    hollow_radius = request.json['hollow_radius']
    # Convert hollow_radius string to float
    if isinstance(hollow_radius, str):

        # Check to see if we need to remove a back slash character
        split_tool_hollow_radius = hollow_radius.split("/")
        if len(split_tool_hollow_radius) > 1:
            target_hollow_radius = int(split_tool_hollow_radius[0]) / int(split_tool_hollow_radius[1])
        else:
            target_hollow_radius = float(hollow_radius)
    else:
        target_hollow_radius = hollow_radius

    tool_instance = tool.Tool.from_hollow_radius(target_hollow_radius)

    # The from hollow radius method already has an assert in place to handle this, but
    if tool_instance is None:
        return Server_Response_Factory.create_server_response(400, f"No tool available with a radius of {target_hollow_radius}")

    # Find scan
    scan = db_interface_instance.get_scan(scan_id, with_blades=True)
    if scan is None:
        return Server_Response_Factory.create_server_response(404, f"Scan with id {scan_id} not found")
    
    # Recreate blade instances
    blade_instances = blade.Blade.from_stored_blades(scan["blades"])

    # New Tool Life Session 
    DoC = request.json["DoC"]
    SFM = request.json["SFM"]
    IPT = request.json["IPT"]
    RPM = request.json["RPM"]
    CLF = request.json["CLF"]
    NoC = request.json["NoC"]
    mill_session = mill_sessions.ToolLifeSession(blade_instances, tool_instance, DoC, SFM, IPT, RPM, CLF, NoC)

    # Insert toolpath entry
    toolpath_file_path = f"{definitions.OUTPUT_DIR_PATH}/{mill_session.tool_path_instance.gcode_instance.file_name}"
    toolpath_insert_values = [scan_id, definitions.SOFTWARE_VERSION, mill_session.to_json(), toolpath_file_path]
    toolpath_insert_result = db_interface_instance.insert_toolpath(toolpath_insert_values)
    toolpath_id = toolpath_insert_result["id"]

    toolpath = db_interface_instance.get_toolpath(toolpath_id)
    if toolpath is None:
        return Server_Response_Factory.create_server_response(410, f"Toolpath {toolpath_id} not found after insert.", None)
    else:
        response = Server_Response_Factory.create_server_response(200, "Success", toolpath)

    # Insert toolpath segments
    for db_blade_index, db_blade in enumerate(scan["blades"]):
        blade_id = db_blade["id"]
        cut_blade_instances = mill_session.get_cut_blade_instances(db_blade_index)
        last_cut_blade_instance = cut_blade_instances[-1]

        if(mill_session.clears_wedge(last_cut_blade_instance.profile, last_cut_blade_instance.wedge_instance.profile) == False):
            return Server_Response_Factory.create_server_response(409, f"Cannot cut. Cut is too deep, will cut wedge", None)

        segments_web_profiles = last_cut_blade_instance.segments_web_profiles()

        for segment_web_profile in segments_web_profiles:
            toolpath_segment_insert_values = [toolpath_id, blade_id, segment_web_profile]
            toolpath_segment_insert_result = db_interface_instance.insert_toolpath_segment(toolpath_segment_insert_values)

    return response

@webserver.route('/toolpath/<toolpath_id>/run', methods=['POST'])
def run_toolpath(toolpath_id):

    # Get gcode program file_path
    toolpath = db_interface_instance.get_toolpath(toolpath_id)

    if definitions.APP_CONFIG["toolPathGenerator"]["localDevelopmentMode"] == False:

        # Make sure work offsets match expected values in config
        confirm_work_offsets_response = mill_instance.confirm_work_offsets()
        if confirm_work_offsets_response is not None:
            return Server_Response_Factory.create_server_response(409, confirm_work_offsets_response)

        # Copy output.nc over to mill using smb instance
        file_to_copy = open(f"{toolpath['file_path']}", "rb")
        smb_instance.copy_file(file_to_copy, "/output.nc")

        # Run output.nc using mill gcode method
        mill_instance.gcode(f"{definitions.APP_CONFIG['mill']['gcodeDirectory']}/output.nc")

    return Server_Response_Factory.create_server_response(200, "Success")

if __name__ == '__main__':

    db_interface_instance = db_connection.Interface()

    if definitions.APP_CONFIG["toolPathGenerator"]["localDevelopmentMode"] == False:
        scanner_instance = scanner.Scanner()
        mill_instance = mill.Mill()
        smb_instance = smb.SmbApi()

    # Test scanner connection
    webserver.run(host="localhost", port=definitions.APP_CONFIG["flask"]["port"], debug=definitions.APP_CONFIG["flask"]["debug"])