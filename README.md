# Extended-Kalman-Filter-Based-Localization
These are the files for a simulated robot and its controller built using an open source simulation software called Webots. The robot has four driven wheels and two independent steering motors making it a non-holonomic system. This warrants the use of an Ackerman steering mechanism for accurate positioning and the localization is done using wheel odometry and inertial measurements. As a result of the simulated surface with variable friction, bounce and unevenness, the mapping needs to be strengthened using a Kalman filter.

Given a list of coordinates, the robot will autonomously drive itself to each coordinate in the list in that order and stop when it reaches the final point.. The default robot controller should be "kf_controller_final".

The robot is able to:

Calculate it's current location at any given moment with the help of a Kalman filter.

Traverse to any coordinate in the plane provided the distance between any two consecutive coordinates is slightly larger than the length of the robot.

Check whether it has reached a point, iterate to the next waypoint, calculate odometry and inertial values and can be coded to move at different speeds.
