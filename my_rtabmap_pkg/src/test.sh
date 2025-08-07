#!/bin/bash

gnome-terminal --tab -- bash -c "roscore;exec bash"

echo “roscore successfully started”


sleep 5s
gnome-terminal --tab -- bash -c "\
source ~/catkin_ws/devel/setup.bash;\
rosrun my_rtabmap_pkg test_cam.py \
exec bash"



# rosrun my_rtabmap_pkg test_cam.py
# rosrun my_rtabmap_pkg test_TTS.py
# rosrun my_rtabmap_pkg test_yolo.py
