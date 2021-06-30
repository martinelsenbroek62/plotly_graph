class Fixture():

    def __init__(self, scan_instance):
        self._scan_instance = scan_instance

    @property
    def scan_instance(self):
        return self._scan_instance

    @scan_instance.setter
    def scan_instance(self, value):
        # Add validation logic here
        self._scan_instance = value

    @classmethod
    def reference_features(self):
        """
        Find the reference features for the provided scan
        """
        raise NotImplementedError('Must define reference_features method to use this base class')

    @classmethod
    def blade_profile(self, z):
        """
        Finds a blade profile series at a given z height
        """
        raise NotImplementedError('Must define blade_profile method to use this base class')

    @classmethod
    def blade_center_x(self):
        """
        Finds the x-axis center of blades
        """
        raise NotImplementedError('Must define blade_center_x method to use this base class')