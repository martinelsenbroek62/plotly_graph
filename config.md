# Ares Backend Rebuild - Config

Here we will outline different properties of our config json

## Main Keys

1) toolPathGenerator
    - Manages our scan data to tool path converter algorithm
2) scanner
3) mill
4) database
5) flask
    - Manages our flash REST API instance
    - This is primarily used to enable an interface with a frontend application

### toolPathGenerator

- <b>debug</b> *boolean* : Used to enable extra debugging output.
- <b>localDevelopmentMode</b> *boolean* : Used to disable the scanner, mill, and smb connections for local development.
- <b>outputDirectory</b> *string* : Directory to store generated gcode programs.
- <b>mainWorkingRadiusPercent</b> *float* : Percentage of the blade length that represents the main working radius.
- <b>segmentType</b> *string* : The type of segments to model our scan data with. Can be Line, Circle, or CubicBezier.
- <b>minLineSegmentLength</b> *float* : Used to control the distance between points when modeling our scan data.
- <b>maxCircleSegmentDifference</b> *float* : When converting polynomials to circles what is the maximum amount of acceptable delta between the generated circle and the polynomial. If a circle does not meet requirements the x domain is split into two new circles and evaluate again.
- <b>leadInOutXLength</b> *float* : The length on the x-axis in mm for the lead in and out to generate. To be safe set this to the diameter of the tool being used.
- <b>workOffsets.cut</b> *dict* : The work offset used in our generated gcode program. Check out the readme of our ares-gcodes repo to get more info on what position these offsets refer to.
- <b>workOffsets.scan</b> *dict* : The work offset used in our scanning gcode program. Check out the readme of our ares-gcodes repo to get more info on what position these offsets refer to.

#### Output
- <b>cut.top</b> *boolean* : In our output include a cutting the top blade.
- <b>cut.bottom</b> *boolean* : In our output include a cutting the bottom blade.
- <b>deburr</b> *boolean* : In output gcode include deburring.

#### Cut
- <b>depth</b> *float* : Amount of material to remove from the blade in the y-axis when sharpening. Units are mm.
- <b>toolNumber</b> *int* : What tool index in the ATC our desired cutting head is on.
- <b>SFM</b> *float* : Surface speed feed per minute.
- <b>IPT</b> *float* : Feed per tooth in inches.
- <b>RPMMultiplier</b> *float* : Adjustment to RPM post feeds and speeds calculation.
- <b>CLF</b> *float* : Adjustment to feed rate post feeds and speeds calculation.
- <b>profile.enable</b> *boolean* : If we should profile this blade or just sharpen it.
- <b>profile.finishingDepth</b> *float* : Final depth of cut used for a finishing pass. Units are mm.
- <b>profile.maxDepth</b> *float* : Maximum depth of cut per profiling pass. Units are mm.
- <b>profile.SFM</b> *float* : Surface speed feet per minute.
- <b>profile.IPT</b> *float* : Inches per tooth.
- <b>profile.RPMMultiplier</b> *float* : Adjustment to RPM post feeds and speeds calculation.
- <b>profile.CLF</b> *float* : Adjustment to feed rate post feeds and speeds calculation.
- <b>profile.sections</b> *Array\[object\]* : Objects in this array have keys; radius, startPercentage, and endPercentage.
  - <b>radius</b> *float* : Target radius of the section of blade.
  - <b>startPercentage</b> *float* : Start point of the section to profile as a percentage.
  - <b>endPercentage</b> *float* : End point of the section to profile as a percentage.
  - <b>pitch</b> *float* : A percentage of blade length to move the apex of the center profile. If you want to pitch forward use a negative value. For the 296 blade a pitch value of 1 will move the apex of the circle 2.96mm forward, which then pitches the blade backwards.

#### Deburr
- <b>feedrate</b> *float* : The feedrate in mm/min to write out to our output gcode program.
- <b>toolNumber</b> *int* : What tool index in the ATC our desired deburring head is on.
- <b>spindleRpm</b> *int* : RPM of spindle to use for our deburring pass.

#### Scans
- <b>feedrate</b> *float* : The feedrate at which our scans were taken at in mm/minute.

#### Scanner
- <b>toolNumber</b> *int* : Tool to use when scanning. This tool should be very short or nothing at all.
- <b>yAxisAccuracy</b> *Array\[float\]* : Defines if we should use a set of scans or a single one.
- <b>frequency</b> *int* : The frequency at which our scanner takes new readings in Hz. Our feedrate divided by this will give us the distance between scan slices.
- <b>zSpacing</b> *float* : The distance between scanner diodes in mm. This gives us the distance between our points in the z axis.
- <b>offsets.z</b> *float* : The absolute z value of the scan program. This should include any work offsets used. https://docs.google.com/drawings/d/1W4gzg8iuJN-FfVTq6sYxZ37nge_oFvaeuK8wYXYzUcA/edit?usp=sharing
- <b>offsets.y</b> *{float}* : The absolute y value of the scan program. This should include any work offsets used. https://docs.google.com/drawings/d/1LcXZcGAO8wMNvcnutYqmTofIK2E7FJVjSFNGeEHPGuY/edit?usp=sharing

#### Fixture
- <b>fixture.cartridge.xAxisReference.length</b> *{float}* : The length of the x-axis reference feature along the axis of the movement of the scanner.
- <b>fixture.cartridge.xAxisReference.toEndOfGrip</b> *{float}* : The offset from the center of the x-axis reference to the end of static blade grip closest to the center of the fixture.

### Scanner
- <b>host</b> *string* : The Keyence Python API server host address.
- <b>port</b> *int* : The Keyence Python API server port.
- <b>batchProfiles</b> *int* : The number of profiles to download per request. Values above 5500 usually cause the buffer to overflow and error.
- <b>connection.type</b> *string* : Can be eth or usb. Identifies how to interface with the scanner.
- <b>connection.host</b> *string* : Scanner hardware API host.
- <b>connection.port</b> *int* : Scanner hardware API port.

### Mill
- <b>host</b> *string* : The linuxCNC REST API server host address.
- <b>port</b> *int* : The linuxCNC REST API server port.
- <b>scanProgramPath</b> *string* : Full path on mill to laser gcode program.
- <b>gcodeDirectory</b> *string* : Path to mill directory where gcode files should be stored.
- <b>tools</b> *Array\[object\]* : Array of objects with keys number, form, diameter, numberOfFlutes, and sideProfileRadius.
  - <b>number</b> *int* : The position in the tool changer this tool exists.
  - <b>form</b> *string* : The type of tool. Can be endmill, or deburr_brush.
  - <b>diameter</b> *float* : The diameter of the tool bit in mm.
  - <b>numberOfFlutes</b> *int* : The number of flutes the tool has. Optional
  - <b>sideProfileRadius</b> *float* : The radius of the hollow this tool will cut. Optional

#### SMB
- <b>smb.server</b> *string* : The host server address. Can be a DNS or IP address.
- <b>smb.share</b> *string* : The share path.
- <b>smb.port</b> *int* : The server port.
- <b>smb.username</b> *string* : The connection authentication username.
- <b>smb.password</b> *string* : The connection authentication password.

### Database
- <b>database</b> *string* : The name of the database to connect to.
- <b>host</b> *string* : The host server address.
- <b>port</b> *int* : The server port.
- <b>user</b> *string* : The database user username.
- <b>password</b> *string* : The database user password.

### Flask
- <b>debug</b> *boolean* : If we should output extra debugging messages.
- <b>port</b> *int* : Port the flask REST API should be exposed on.