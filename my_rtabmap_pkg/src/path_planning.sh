#!/bin/bash

gnome-terminal --tab -- bash -c "\
rosbag record -O case4.bag /camera/color/image_raw /rtabmap/mapData /rtabmap/info /camera/depth_registered/image /current_path /rtabmap/localization_pose /rtabmap/cloud_map ; exec bash"


sleep 5s
gnome-terminal --tab -- bash -c "\
source ~/catkin_ws/devel/setup.bash;\
rosrun my_rtabmap_pkg nav.py --language=ch --case=case4 2> /dev/null; \
exec bash"




