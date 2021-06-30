import definitions
from modules import helpers, smb, scan, tool_path, tool, blade, mill_sessions

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

    selected_scan = scan.ScanFile.from_directory(f"{definitions.APP_DIR_PATH}/scans")
    blade_instances = blade.Blade.parse_blades_from_scan_file(selected_scan.file_path)

    if definitions.APP_CONFIG["toolPathGenerator"]["cut"]["toolLife"]["enable"]:

        depth_of_cut_per_pass = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["toolLife"]["depth"]
        SFM = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["toolLife"]["SFM"]
        IPT = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["toolLife"]["IPT"]
        rpm_factor = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["toolLife"]["RPMMultiplier"]
        chip_load_factor = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["toolLife"]["CLF"]
        number_of_passes = definitions.APP_CONFIG["toolPathGenerator"]["cut"]["toolLife"]["passes"]
        mill_sessions.ToolLifeSession(blade_instances, tool_instance, depth_of_cut_per_pass, SFM, IPT, rpm_factor, chip_load_factor, number_of_passes)

    elif definitions.APP_CONFIG["toolPathGenerator"]["cut"]["profile"]["enable"]:

        profiling_specifications = blade.ProfilingSpecification.multiple_from_config(definitions.APP_CONFIG["toolPathGenerator"]["cut"]["profile"]["sections"])
        mill_sessions.ProfilingSession(blade_instances, tool_instance, profiling_specifications)

    else:
        mill_sessions.SharpeningSession(blade_instances, tool_instance)
    

    # helpers.timer_instance.update_manual_timestamp("script")
    # helpers.timer_instance.print_timestamps()