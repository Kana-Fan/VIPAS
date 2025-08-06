#!/usr/bin/env python3

#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

# import torch
from ultralytics import YOLO

class YOLOWorldNode:
    def __init__(self, headless=False):
        rospy.init_node('yoloworld_node', anonymous=True)
        self.bridge = CvBridge()
        self.headless = headless

        classes = ['door', 'person', 'stair', 'elevator', 
                    'locker', 'trash_can', 'bench', 'chair', 
                    'fire_extinguisher', 'window', 'handrail',
                    'crutch', 'wheelchair', 'bicycle', 
                    'poster', 'notice_board', 'clock', 'light_switch',
                    'plant', 'mat', 'wet_floor_sign',  'bookshelf',
                    'whiteboard', 'printer',  'fire_exit_door'
                    ]

        # 載入 YOLO-World 模型
        rospy.loginfo("Loading YOLO-World model...")
        self.model = YOLO('yolov8/yolov8s-world.pt')  # 請確保檔案存在
        self.model.set_classes(classes)
        rospy.loginfo("Model loaded.")

        # 訂閱 RealSense color image
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo("YOLO-World ROS node ready.")
        rospy.spin()
        

    def image_callback(self, msg):
        try:
            # ROS Image -> OpenCV Image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("cv_bridge error: %s", e)
            return e

        # YOLO 推論
        results = self.model.predict(source=frame, conf=0.3, iou=0.5, show=False, verbose=False)

        # 繪製結果
        self.result_img = results[0].plot()

        # 顯示影像（除非在 headless 模式）
        if self.headless == False :
            cv2.namedWindow("YOLO-World Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO-World Detection", 640, 480)
            cv2.imshow("YOLO-World Detection", self.result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown("Quit.")

    def get_results(self):
        return self.result_img
        
class YOLOWorldNode2:
    def __init__(self, headless=False):
        rospy.init_node('yoloworld_node', anonymous=True)
        self.bridge = CvBridge()
        self.headless = headless
        self.count = 1

        classes = ['door', 'person', 'stairs', 'elevator_door', 
                    'locker', 'trash_can', 'bench', 'chair', 
                    'fire_extinguisher', 'window', 'handrail',
                    'crutch', 'wheelchair', 'bicycle', 
                    'poster', 'notice_board', 'clock', 'light_switch',
                    'plant', 'mat', 'wet_floor_sign',  'bookshelf',
                    'whiteboard', 'printer',  'fire_exit_door'
                    ]

        # 載入 YOLO-World 模型
        rospy.loginfo("Loading YOLO-World model...")
        self.model = YOLO('yolov8/yolov8s-world.pt')
        self.model.set_classes(classes)
        rospy.loginfo("Model loaded.")

        # 訂閱 RealSense color image
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

        # 辨識結果儲存變數
        self.last_detections = []   # 儲存最近一次的辨識結果（list of dict）
        self.last_image = None      # 儲存最近一次的結果影像（OpenCV 格式）

        rospy.loginfo("YOLO-World ROS node ready.")
        # rospy.spin()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("cv_bridge error: %s", e)
            return

        # YOLO 推論
        results = self.model.predict(source=frame, conf=0.5, iou=0.5, show=False, verbose=False)
        result = results[0]

        # 儲存繪圖影像
        self.last_image = result.plot()

        # 儲存辨識結果
        self.last_detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            score = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()  # [x1, y1, x2, y2]

            self.last_detections.append({
                "label": label,
                "confidence": score,
                "bbox": xyxy
            })

        # 顯示影像（僅為即時觀看）
        if self.headless == False :
            cv2.namedWindow("YOLO-World Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO-World Detection", 640, 480)
            cv2.imshow("YOLO-World Detection", self.last_image)
            # cv2.imwrite("/home/rvl1412/catkin_ws/yolo/YOLO-World Detection_" + str(self.count) + ".jpg", self.last_image)
            self.count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown("Quit.")
                
    def get_detections(self):
        """
        回傳最近一次的辨識結果（list of dict）
        """
        return self.last_detections

    def get_result_image(self):
        """
        回傳最近一次的結果影像（OpenCV BGR 格式）
        """
        return self.last_image

if __name__ == '__main__':
    try:
        # headless模式不會顯示影像
        # obj = YOLOWorldNode(headless=False)
        # result_img = obj.get_results()

        node = YOLOWorldNode2(headless=False)
        print("yolo start")

        rate = rospy.Rate(1)  # 每秒檢查一次結果

        while not rospy.is_shutdown():
            dets = node.get_detections()
            img = node.get_result_image()

            if dets and img is not None:
                print("Detections:")
                for d in dets:
                    print(f"{d['label']} ({d['confidence']:.2f}) @ {d['bbox']}")

                # 顯示影像
                # cv2.imshow("External Result", img)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

            rate.sleep()
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass

