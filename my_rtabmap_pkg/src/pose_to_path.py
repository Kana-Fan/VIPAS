#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path

class PoseToPath:
    def __init__(self):
        self.path = Path()
        self.path.header.frame_id = "map"
        self.path_pub = rospy.Publisher('/current_path', Path, queue_size=10)
        self.pose_sub = rospy.Subscriber('/rtabmap/localization_pose', PoseWithCovarianceStamped, self.pose_callback)

    def pose_callback(self, msg):
        # 從 PoseWithCovarianceStamped 提取 PoseStamped
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.path.poses.append(pose_stamped)
        self.path.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path)

if __name__ == '__main__':
    rospy.init_node('pose_to_path', anonymous=True)
    print("pose_to_path")
    PoseToPath()
    rospy.spin()
