import definitions
from modules import helpers, smb, scan, tool_path, tool, blade, mill_sessions
from glob import glob

#############
# Constants #
#############
if __name__ == '__main__':

    # smb_instance = smb.SmbApi()

    # file_to_copy = open("./script.py", "rb")
    # smb_instance.copy_file(file_to_copy, "/output.nc")

    target_tool_number = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["toolNumber"]
    tool_instance = tool.Tool.from_tool_number(target_tool_number)
    assert tool_instance is not None, f"No tool available with a number of {target_tool_number}"
    
    directory_path = f"{definitions.APP_DIR_PATH}/scans"
    scan_files = glob(f"{directory_path}/*.csv*") 

    for scan_file in scan_files:

        selected_scan = scan.ScanFile(scan_file, 0.0)
        blade_instances = blade.Blade.parse_blades_from_scan_file(selected_scan.file_path)

        if definitions.APP_CONFIG["toolPathGenerator"]["cut"]["profile"]["enable"]:
        
            profiling_specifications = blade.ProfilingSpecification.multiple_from_config(definitions.APP_CONFIG["toolPathGenerator"]["cut"]["profile"]["sections"])
            mill_sessions.ProfilingSession(blade_instances, tool_instance, profiling_specifications)
        else:
            mill_sessions.SharpeningSession(blade_instances, tool_instance)