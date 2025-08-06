#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String

# 用於儲存標籤的全局變數
labeled_markers = []
# 用於記錄上一次的最近標籤
last_nearest_label = None
# 發布者物件
nearest_label_pub = None

def localization_pose_callback(data):
    global last_nearest_label, nearest_label_pub
    
    # 獲取當前位置
    current_position = data.pose.pose.position
    
    if not labeled_markers:  # 如果沒有儲存的標籤
        if last_nearest_label != "No markers":
            nearest_label_pub.publish("No labeled markers available")
            last_nearest_label = "No markers"
        return

    # 計算與每個標籤的距離並找出最近的
    min_distance = float('inf')
    nearest_label = None
    
    for marker in labeled_markers:
        marker_pos = marker['position']
        distance = math.sqrt(
            (current_position.x - marker_pos.x) ** 2 +
            (current_position.y - marker_pos.y) ** 2 +
            (current_position.z - marker_pos.z) ** 2
        )
        if distance < min_distance:
            min_distance = distance
            nearest_label = marker['label']
    
    # 只有當最近標籤改變時才發布
    if nearest_label != last_nearest_label:
        # 發布最近的標籤
        nearest_label_pub.publish(nearest_label)
        last_nearest_label = nearest_label

        # 印出當前位置和最近的標籤
        print("Current Position:")
        print(f"x={current_position.x}, y={current_position.y}, z={current_position.z}")
        print(f"Nearest Label: {nearest_label}")
        print(f"Distance: {min_distance:.2f} meters")
        print("---")
    
    nearest_label_pub.publish(last_nearest_label)

def get_labels_once():
    # 單次獲取 /rtabmap/labels 話題數據並儲存
    global labeled_markers

    print("Waiting for /rtabmap/labels message...")

    try:
        msg = rospy.wait_for_message('/rtabmap/labels', MarkerArray, timeout=10.0)
        
        print("Storing Labeled Markers:")
        has_labeled = False
        for marker in msg.markers:
            # 檢查 marker.text 是否不為空且不等於 marker.id 的字符串形式
            temp = marker.text[0:3]
            if marker.text != str(marker.id) and temp != "map":
                labeled_markers.append({
                    'id': marker.id,
                    'label': marker.text,
                    'position': marker.pose.position
                })
                print(f"Stored - Marker ID: {marker.id}, Label: {marker.text}, "
                      f"Position: x={marker.pose.position.x}, y={marker.pose.position.y}, z={marker.pose.position.z}")
                has_labeled = True
        if not has_labeled:
            print("No labeled markers found.")
        print("---")
        return 1
    except rospy.ROSException as e:
        print(f"Failed to get labels: {e}")
        return 0

def main():
    # 初始化 ROS 節點
    rospy.init_node('rtabmap_subscriber', anonymous=True)

    # 設置發布者
    global nearest_label_pub
    nearest_label_pub = rospy.Publisher('/nearest_label', String, queue_size=10)

    # 單次獲取並儲存標籤
    get = 0
    while get == 0 :
        get = get_labels_once()
        if rospy.is_shutdown() :
            get = 1

    # 訂閱 /rtabmap/localization_pose 話題
    rospy.Subscriber('/rtabmap/localization_pose', PoseWithCovarianceStamped, localization_pose_callback)

    # 保持節點運行
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass