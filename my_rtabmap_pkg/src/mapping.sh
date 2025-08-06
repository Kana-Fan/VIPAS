#!/bin/bash

gnome-terminal --tab -- bash -c "roscore;exec bash"

echo “roscore successfully started”

# 两个roslauch之间需要间隔一段时间，否则会相互抢占roscore,导致其中一个roslaunch失败,报runid错误

sleep 3s

gnome-terminal --tab -- bash -c "\
roslaunch realsense2_camera rs_camera.launch align_depth:=true;\
exec bash"

echo “realsense2_camera successfully started”

sleep 5s

# rtabmap_args:="--delete_db_on_start"
# database_path:=~/.ros/14.db

gnome-terminal --tab -- bash -c "\
roslaunch rtabmap_ros rtabmap.launch \
    database_path:=~/.ros/f14_15.db \
    depth_topic:=/camera/aligned_depth_to_color/image_raw \
    rgb_topic:=/camera/color/image_raw \
    camera_info_topic:=/camera/color/camera_info \
    approx_sync:=false; \
exec bash"

echo “rtabmap mapping successfully started”


