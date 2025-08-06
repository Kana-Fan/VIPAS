#!/usr/bin/env python3

import rospy
import speech_recognition as sr


import asyncio
import edge_tts
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play


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
        asyncio.run(self._speak_async(text))

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

if __name__ == '__main__':
    rospy.init_node('tts_node')
    
    speaker = EdgeSpeaker(voice="en-US-JennyNeural", rate="+0%", volume="+0%")
    listener = Listener(language = 'zh-tw')

    has_voice = False

    while( has_voice == False ):
        speaker.speak("What can i help you?")
        r = listener.get_voice()

        if r == "Error01" :
            speaker.speak("No sound was heard, please try again.")
        elif r == "Error02" :
            speaker.speak("Unable to recognize speech, please try again.")
        elif r == "Error03" :
            speaker.speak("Unable to connect to the speech recognition service.")
        elif r == "Error04" :
            speaker.speak("An error occurred.")
        else :
            has_voice = True
            print(r)
            speaker.speak(r)



