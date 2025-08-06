#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber:
    def __init__(self):
        # 初始化 ROS 節點
        rospy.init_node('realsense_rgb_viewer', anonymous=True)
        
        # 创建 CvBridge 对象，用于将 ROS 图像消息转换为 OpenCV 格式
        self.bridge = CvBridge()
        
        # 订阅 RGB 图像主题（默认主题为 /camera/color/image_raw）
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        
        self.cv_image = None

    def callback(self, data):
        # 将 ROS 图像消息转换为 OpenCV 格式
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            print(e)

    def display(self):
        # 持续运行并显示图像
        while not rospy.is_shutdown():
            if self.cv_image is not None:
                # 显示图像
                cv2.namedWindow("RealSense RGB Image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("RealSense RGB Image", 640, 480)
                cv2.imshow("RealSense RGB Image", self.cv_image)
                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            rospy.sleep(0.01)

    def run(self):
        try:
            self.display()
        except KeyboardInterrupt:
            print("Shutting down")
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    image_subscriber = ImageSubscriber()
    image_subscriber.run()