#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

import asyncio
import edge_tts
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr

import openai
import cv2
from cv_bridge import CvBridge
import base64
import numpy as np

from ultralytics import YOLO
import json
import xml.etree.ElementTree as ET
from collections import deque

import argparse
import time

import yaml




class Listener:
    def __init__(self, language):
        # 語音辨識
        # language = "en-us"
        # language = 'zh-tw'
        self.language = language
        self.r = sr.Recognizer()

    def get_voice(self):
        with sr.Microphone() as source:
            try:
                audio = self.r.listen(source, timeout=5, phrase_time_limit=10)
                return self.r.recognize_google(audio, language=self.language)
            except sr.WaitTimeoutError:
                # 沒有聽到聲音
                return "Error01"
            except sr.UnknownValueError:
                # 無法辨識語音
                return "Error02"
            except sr.RequestError as e:
                # 無法連線至語音辨識服務: {e}
                return f"Error03"
            except Exception as e:
                # 發生錯誤: {e}
                print(f"error:{e}")
                return f"Error04"


class EdgeSpeaker:
    def __init__(self, voice="en-US-JennyNeural", rate="+0%", volume="+0%"):
        """
        初始化語音合成器。
        :param voice: 微軟 Edge 語音 ID，如 'en-US-JennyNeural'
        :param rate: 語速調整，如 '+20%'、'-10%'（預設不變）
        :param volume: 音量調整，如 '+10%'（預設不變）
        """
        self.voice = voice
        self.rate = rate
        self.volume = volume

    async def _speak_async(self, text: str):
        stream = BytesIO()
        tts = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
        )

        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                stream.write(chunk["data"])

        stream.seek(0)
        audio = AudioSegment.from_file(stream, format="mp3")
        

        play(audio)

    def speak(self, text: str):
        """
        即時播放語音（非同步封裝成同步使用）。
        :param text: 要朗讀的文字
        """
        print(text)
        asyncio.run(self._speak_async(text))

class YOLOWorldNode:
    def __init__(self, classes, headless=False):
        # rospy.init_node('yoloworld_node', anonymous=True)
        self.bridge = CvBridge()
        self.headless = headless

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
        results = self.model.predict(source=frame, conf=0.3, iou=0.5, show=False, verbose=False)
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

class NavigationNode:
    def __init__(self, osm_file, api_key):  
        # 影像相關
        self.bridge = CvBridge()
        self.latest_image = None
        self.current_label = None

        # 初始化OpenAI客戶端
        self.client = openai.OpenAI(api_key=api_key)

        # 儲存對話歷史以保持上下文
        self.messages = [
            {"role": "system", "content": "You are a navigation assistant for a visually impaired person."}
        ]
        
        # 讀取osmAG地圖
        self.osm_content = self.read_osm_file(osm_file)
        self.graph = self.parse_osmag_to_json(self.osm_content) 
        self.path = {}
        
        # ROS訂閱和發布
        self.label_sub = rospy.Subscriber('/nearest_label', String, self.label_callback)

    # get label by Rtab-map
    def label_callback(self, msg):
        self.current_label = msg.data

    # read osm map
    def read_osm_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    # identification goal
    def identification_goal(self, input ):
        label = list(self.graph.keys())
        # label = ["14 stairs 1", "14 stairs 2", "14 elevator 1", "14 elevator 2", "14 machine room 1"
        #         , "14 machine room 2", "14 balcony", "14 Men restroom", "1421", "1422", "1423", "1424"
        #         ,"15 stairs 1", "15 stairs 2", "15 elevator 1", "15 elevator 2", "15 machine room 1"
        #         , "15 machine room 2",  "15 Women restroom", "1521", "1522"
        #         , "1523", "1524", "1525", "1526", "1527", "1528", "1529", "1530"
        #         , "1531", "1532", "1533", "1534", "1535", "1536"]
        # label = ["2 stairs 1", "2 elevator 1", "231", "232", "234", "235" ,"2 elevator 2", "2 stairs 2", "2 machine room", "243", "240", "239", "238"]
        prompt = f"""
        你的任務是從輸入中的句子裡根據附表找到符合的label然後回傳。
        只要回傳label就好。
        如果沒有符合的就回傳 error。
        比如輸入是:我想去1421
        你要回傳:1421

        輸入:{input}
        附表:{label}
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=50,  # 初始回應不需要太長
                temperature=0.5
            )

            return response.choices[0].message.content

        except openai.APIError as e:
            return f"Error: API request failed - {str(e)}"
        except openai.RateLimitError:
            return "Error: Rate limit exceeded. Please try again later."
        except Exception as e:
            return f"Error: Unable to identification goal - {str(e)}"

    # graph path planning and describe
    def parse_osmag_to_json(self, xml_content):
        
        # 初始化 adjacency list
        graph = {}
        
        try:
            # 解析 XML
            root = ET.fromstring(xml_content)
            
            # 遍歷所有 <way> 元素
            for way in root.findall('way'):
                # 提取標籤
                tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
                
                # 檢查必要標籤
                if 'name' not in tags or 'osmAG:areaType' not in tags:
                    continue
                    
                name = tags['name']
                area_type = tags['osmAG:areaType']
                
                # 初始化鄰居列表
                neighbors = []
                
                # 添加非空的 Left Neighbor
                if 'Left Neighbor' in tags and tags['Left Neighbor']:
                    neighbors.append(tags['Left Neighbor'])
                    
                # 添加非空的 Right Neighbor
                if 'Right Neighbor' in tags and tags['Right Neighbor']:
                    neighbors.append(tags['Right Neighbor'])
                    
                # 對於通道，添加非空的 up
                if area_type == 'channel' and 'up' in tags and tags['up']:
                    neighbors.append(tags['up'])
                
                # 若有鄰居，添加到 graph
                if neighbors:
                    graph[name] = neighbors
                    
            # 儲存為 JSON 檔案
            with open('output.json', 'w') as file:
                json.dump(graph, file, indent=4)
                    
            return graph
        
        except ET.ParseError:
            print("錯誤：XML 格式錯誤")
            return {}
        
    def find_shortest_path(self, start, end):

        if start not in self.graph or end not in self.graph:
            return "Error: Start or end node not found"
        
        # BFS
        queue = deque([(start, [start])])
        visited = set()
        
        while queue:
            node, path = queue.popleft()
            if node == end:
                self.path = {"path": path}
                return {"path": path}
            if node not in visited:
                visited.add(node)
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return "Error: Room not found"

    def describe(self, language):
        prompt = f"""
        You are given two representations of an indoor floor map:

        1. An OSM Area Graph (osmAG) in XML format. Each <way> describes an area (room or vertical channel). It includes:
            - name: the area name (e.g., "1421")
            - level: the floor number (e.g., "14", "15")
            - osmAG:areaType: either "room" or "channel"
            - Left Neighbor and Right Neighbor tags, which indicate direct neighbors on the same floor.
            - up: Indicates passages that lead to different floors.(Only channel have it.)

        2. A simplified connectivity graph in JSON format. This shows which areas are directly connected (bi-directionally), extracted from the OSM tags.
            - Each **key** represents the **name of an area** (e.g., a room, elevator, or stairs).
            - Each **value** is a list of **directly connected neighboring areas**.
            - All connections are bidirectional unless stated otherwise.
            - Area names match those defined in the OSM XML, such as "1421", or "15 stairs 2".
        
        Here is the XML map:{self.osm_content}
        Here is the graph map:{self.graph}

        Your task is to generate a natural language description of the navigation path from the starting point to the end point based on the given path. 
        The path has been calculated by BFS, which is guaranteed to be the shortest path, and stairs are preferred when crossing floors.
        Assuming that the elevator is broken, just pass by the elevator.
        
        {self.path}
        
        """

        # This path has been calculated through breadth-first search (BFS) and is guaranteed to be the shortest path. It does not cross floors this time, but only passes by stairs and elevators. 
        # During the movement, it passes through the room without entering.


        if language == "ch" :
            prompt += "請用繁體中文回答。"
        
        self.messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
                max_tokens=700,
                temperature=0.5
            )
            result = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": result})
            self.path_describe = result
            return result
        
        except openai.APIError as e:
            return f"Error: API request failed - {str(e)}"
        except openai.RateLimitError:
            return "Error: Rate limit exceeded. Please try again later."
        except Exception as e:
            return f"Error: Unable to generate path - {str(e)}"
    

    # guide
    def image_to_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def check_finish(self):
        print(f"{self.current_label},{ self.goal}")
        return self.goal == self.current_label

    def generate_navigation_guidance(self, language):
        # if not self.current_label or not self.latest_image.any():
        #     rospy.logwarn("Missing data for navigation guidance.")
        #     return
        
        # 將影像轉為base64
        image_base64 = self.image_to_base64(self.latest_image)
        
        # 構建提示，包含路徑、當前位置和影像
        prompt = f"""
        Based on the map you analyzed previously and the path(Short path and Describe path) and current location I provided :
        - Current location: {self.current_label} 
        - Short path : {self.path}
        - Describe path: {self.path_describe} 
        - Current RGB image: attached later 
        
        Please limit your answer to one or two short sentences.

        Based on the image, path, and current location, provide simple navigation instructions for the visually impaired, including the current location, the next room, and the recommended direction of walking.
        And determine whether the objects marked in the image will affect walking. If so, please provide additional instructions for whether there are obstacles.
        When encountering stairs, it is best to inform users whether the handrail is on the left or right side, and provide additional prompts to indicate the detailed process (for example, when encountering a turn in the stairs, it is necessary to remind users that they are about to turn) so that users can go upstairs safely.
        """

        # When encountering a closed door, additional prompts are needed to open the door.

        if language == "ch" :
            prompt += "請用繁體中文回答。"
        
        message = self.messages
        message.append({"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=message,
                max_tokens=100,  # 限制輸出為短句
                temperature=0.5
            )
            guidance = response.choices[0].message.content
            # self.messages.append({"role": "assistant", "content": guidance})
            # rospy.loginfo(f"Navigation guidance: {guidance}")
            return guidance
        except Exception as e:
            rospy.logerr(f"Failed to generate navigation guidance: {str(e)}")


    # for call
    def path_planning(self, goal):
        self.goal = goal
        self.path = self.find_shortest_path(self.current_label, goal)
        return self.path

    def guidance(self, img, language):
        self.latest_image = img
        return self.generate_navigation_guidance(language)

    def getLabel(self):
        return self.current_label


def getVoice(listener, speaker, language) :
    has_voice = False

    if language == 'en' :
        speaker.speak("Welcome to use the visually impaired assisted navigation.")
        while( has_voice == False ):
            speaker.speak("What can i help you?")
            goal = listener.get_voice()

            if goal == "Error01" :
                speaker.speak("No sound was heard, please try again.")
            elif goal == "Error02" :
                speaker.speak("Unable to recognize speech, please try again.")
            elif goal == "Error03" :
                speaker.speak("Unable to connect to the speech recognition service.")
            elif goal == "Error04" :
                speaker.speak("An error occurred.")
            else :
                has_voice = True
                print(goal)
        return goal
    else:
        speaker.speak("歡迎使用視障輔助導航。")
        while( has_voice == False ):
            speaker.speak("我能幫您什麼忙？")
            goal = listener.get_voice()

            if goal == "Error01" :
                speaker.speak("沒有聽到聲音，請重試。")
            elif goal == "Error02" :
                speaker.speak("無法辨識語音，請重試。")
            elif goal == "Error03" :
                speaker.speak("無法連線到語音辨識服務。")
            elif goal == "Error04" :
                speaker.speak("發生錯誤。")
            else :
                has_voice = True
                print(goal)
        return goal

def getKey(filepath):
    key = ""

    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
        key = data["OPENAL_API_KEY"]
    
    return key

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default = 'en')
    parser.add_argument('--case', type=str, default = 'case0')
    # --language=ch  中文介面
    # --language=en  英文介面

    args = parser.parse_args()
    language = args.language
    case = args.case

    rospy.init_node('navigation_node')

    classes = ['door', 'person', 'stairs', 'elevator_door', 'trash_can', 'chair', 
                'fire_extinguisher', 'window', 'handrail',
                'crutch', 'wheelchair', 'bicycle', 
                'poster', 'notice_board', 'clock', 'light_switch',
                'plant', 'mat', 'wet_floor_sign',  'bookshelf',
                'Umbrella', 'printer',  'fire_exit_door'
                ]
    
    osm_file = "/home/itri/catkin_ws/src/my_rtabmap_pkg/src/map/osm/merge_test4.osm"  # 替換成你的檔案路徑
    
    api_key = getKey("OPENAL_API_KEY.yaml")
    
    voice = ''
    lan = ''
    # 初始化語音辨識與TTS
    if language == 'en':
        voice = "en-US-JennyNeural"
        lan = 'en-us'
    else :
        voice = "zh-TW-HsiaoChenNeural"
        lan = 'zh-tw'

    speaker = EdgeSpeaker(voice=voice, rate="+0%", volume="+0%")
    listener = Listener(language=lan)
    # listener = WhisperListener(model_name="small")
 

    # 初始化導航系統
    nav = NavigationNode(osm_file, api_key)
    yolo_detections = YOLOWorldNode(classes, False)

    

    getPath = False
    while(getPath == False) :
        # 獲取目標地點
        user_input = getVoice(  listener, speaker, language)


        goal = nav.identification_goal(user_input)
        txt = ''

        if language == 'en':
            print(f"You want to go to {goal}.")
            speaker.speak("Start path planning, please wait a moment.")
        else :
            txt = f"你想到 {goal}."
            print(txt)
            speaker.speak(txt)
            speaker.speak("開始路徑規劃，請稍等。")

        short_path = nav.path_planning(goal)
        print(short_path)

        if( short_path != "Error: Room not found" and short_path != "Error: Start or end node not found" ) :
            getPath = True
        else:
            speaker.speak("Error: Room not found")

    path_describe = nav.describe(language)
    path_describe_clearn = path_describe.replace("*","") # 除掉*號

    if language == 'en':
        speaker.speak("Path planning finish.")
    else :
        speaker.speak("路徑規劃完成。")

    print(path_describe_clearn)

    # speaker.speak(path_describe_clearn)

    # with open("./result/" + case+".txt", "w", encoding="utf-8") as f:
    #     f.write("路徑描述：\n")
    #     f.write( path_describe_clearn + "\n\n")
    #     f.write("================================\n")
    #     f.write("導引過程： \n")


    if language == 'en':
        speaker.speak("Start the guide.")
    else :
        speaker.speak("開始導引。")

    guide = ""
    count = 0
    response_time = 0

    while(nav.check_finish() == False):
        dets = yolo_detections.get_detections()
        img = yolo_detections.get_result_image()
        nowlabel = nav.getLabel()

        speaker.speak(f"您現在位於{nowlabel}。\n")

        start_time = time.time()
        guide = nav.guidance(img, language)
        end_time = time.time() 

        elapsed_time = end_time - start_time  # 計算耗時（秒）
        response_time += elapsed_time
        count += 1
            
        guide_clearn = guide.replace("*","") # 除掉*號

        output = f"第 {count} 次導引計算時間: {elapsed_time:.3f} 秒"
        
        print(output)

        cv2.imwrite("./yolo/"+ case + "/" + str(count) + ".jpg", img)
        speaker.speak(guide_clearn)


    # with open("./result/" + case +".txt", "a", encoding="utf-8") as f:  # 在迴圈外開啟檔案
    #     while(nav.check_finish() == False):
    #         dets = yolo_detections.get_detections()
    #         img = yolo_detections.get_result_image()
    #         nowlabel = nav.getLabel()

    #         speaker.speak(f"您現在位於{nowlabel}。\n")

    #         start_time = time.time()
    #         guide = nav.guidance(img, language)
    #         end_time = time.time() 
    #         elapsed_time = end_time - start_time  # 計算耗時（秒）
    #         response_time += elapsed_time
    #         count += 1
            
    #         guide_clearn = guide.replace("*","") # 除掉*號

    #         output = f"第 {count} 次導引計算時間: {elapsed_time:.3f} 秒"
            
    #         print(output)

    #         cv2.imwrite("./yolo/"+ case + "/" + str(count) + ".jpg", img)
    #         speaker.speak(guide_clearn)
            
    #         f.write(f"第 {count} 次導引: {output} \n")
    #         f.write(guide_clearn + "\n\n\n")  # 每筆導引結果寫入一行
    #         f.flush()
    
    speaker.speak(f"您現在位於{goal}。")

    evg_response_time = response_time / count

    # with open("./result/" + case +".txt", "a", encoding="utf-8") as f:
    #     f.write(f"共導引 {count} 次\n")
    #     f.write(f"平均導引時間: {evg_response_time:.3f} 秒\n")

    if language == 'en':
        speaker.speak("You have reached your destination.")
        rospy.signal_shutdown("Finish.")
    else :
        speaker.speak("您已到達目的地。")
        rospy.signal_shutdown("結束。")

    
    rospy.spin()

