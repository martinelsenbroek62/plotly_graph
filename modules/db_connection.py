import psycopg2
from psycopg2.extras import RealDictCursor, Json

import definitions

class Connection():

    def __init__(self):

        self.host = definitions.APP_CONFIG["database"]["host"]
        self.port = definitions.APP_CONFIG["database"]["port"]
        self.database = definitions.APP_CONFIG["database"]["database"]
        self.user = definitions.APP_CONFIG["database"]["user"]
        self.password = definitions.APP_CONFIG["database"]["password"]

        # Attempt a connection
        self.connection = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )

        self.connection.set_session(autocommit=True)

        self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)

    # Interface

    def fetch_all(self, query, values=None):
        self.run_query(query, values)
        result = self.cursor.fetchall()
        return result

    def fetch_one(self, query, values=None):
        self.run_query(query, values)
        result = self.cursor.fetchone()
        return result

    def insert(self, query, values=None):
        self.run_query(query, values)
        return True

    # Helpers

    def run_query(self, query, values):
        if values is not None:
            converted_values = Connection.convert_values(values)
        else:
            converted_values = values
        self.cursor.execute(query, converted_values)

    @staticmethod
    def convert_values(values):
        return list(map(lambda value: Json(value) if isinstance(value, dict) else value, values))

class Interface():
    def __init__(self):
        self.connection_instance = Connection()

    def get_toolpath_segments(self, toolpath_id=None):
        query = f"SELECT \
            id, \
            toolpath_id, \
            blade_id, \
            web_profile, \
            created_at \
            FROM toolpath_segments"

        query = Interface.where_toolpath_id(query, toolpath_id)

        toolpath_segments = self.connection_instance.fetch_all(query)
        return toolpath_segments

    def get_toolpaths(self, scan_id=None, with_segments=False):
        query = f"SELECT \
            id, \
            scan_id, \
            software_version, \
            parameters, \
            file_path, \
            created_at \
            FROM toolpaths"

        query = Interface.where_scan_id(query, scan_id)

        toolpaths = self.connection_instance.fetch_all(query)

        if with_segments:
            for toolpath in toolpaths:
                toolpath["toolpath_segments"] = self.get_toolpath_segments(toolpath_id = toolpath["id"])

        return toolpaths

    def get_toolpath(self, toolpath_id, with_scan=False, with_blades=False, with_segments=False):
        query = f"SELECT \
            id, \
            scan_id, \
            software_version, \
            parameters, \
            file_path, \
            created_at \
            FROM toolpaths \
            WHERE toolpaths.id = {toolpath_id}"

        toolpath = self.connection_instance.fetch_one(query)

        if toolpath is None:
            return None
        
        scan_id = toolpath["scan_id"]
        if with_scan:
            toolpath["scan"] = self.get_scan(scan_id)
        if with_blades:
            toolpath["blades"] = self.get_blades(scan_id=scan_id)
        if with_segments:
            toolpath["toolpath_segments"] = self.get_toolpath_segments(toolpath_id = toolpath["id"])

        return toolpath

    def insert_toolpath(self, values):
        insert_query = f"INSERT INTO toolpaths(scan_id, software_version, parameters, file_path) VALUES(%s, %s, %s, %s) RETURNING id"
        insert_result = self.connection_instance.fetch_one(insert_query, values=values)
        return insert_result

    def insert_toolpath_segment(self, values):
        insert_query = f"INSERT INTO toolpath_segments(toolpath_id, blade_id, web_profile) VALUES(%s, %s, %s) RETURNING id"
        insert_result = self.connection_instance.fetch_one(insert_query, values=values)
        return insert_result

    def get_blades(self, scan_id=None):
        query = f"SELECT \
            id, \
            scan_id, \
            top, \
            web_profile, \
            class_instance, \
            created_at \
            FROM blades"

        query = Interface.where_scan_id(query, scan_id)

        blades = self.connection_instance.fetch_all(query)
        return blades

    def insert_blade(self, values):
        insert_query = f"INSERT INTO blades(scan_id, top, class_instance, web_profile) VALUES(%s, %s, %s, %s) RETURNING id"
        insert_result = self.connection_instance.fetch_one(insert_query, values=values)
        return insert_result

    def get_scans(self):
        query = f"SELECT id, file_path, created_at FROM scans"
        scans = self.connection_instance.fetch_all(query)
        return scans

    def get_scan(self, scan_id, with_blades=False, with_toolpaths=False):

        query = f"SELECT \
            id, \
            file_path, \
            created_at \
            FROM scans \
            WHERE scans.id = {scan_id}"

        scan = self.connection_instance.fetch_one(query)

        if scan is None:
            return None

        if with_blades:
            scan["blades"] = self.get_blades(scan_id=scan_id)
        if with_toolpaths:
            scan["toolpaths"] = self.get_toolpaths(scan_id=scan_id)

        return scan

    def insert_scan(self, values):
        insert_query = f"INSERT INTO scans(file_path) VALUES(%s) RETURNING id"
        insert_result = self.connection_instance.fetch_one(insert_query, values=values)
        return insert_result

    @staticmethod
    def where_scan_id(query, scan_id):
        if scan_id is not None:
            query +=  f" WHERE scan_id = {scan_id}"
        return query
    
    @staticmethod
    def where_toolpath_id(query, toolpath_id):
        if toolpath_id is not None:
            query +=  f" WHERE toolpath_id = {toolpath_id}"
        return query
