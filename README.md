# Visually impaired people assistance system (VIPAS)

基於環境感知與視覺語言模型之多模態視障導航介面開發

Development of a multimodal visually impairednavigation interface based on environmentalperception and visual language model

# 安裝環境
測試開發是在 **Ubuntu 20.04.6 LTS** 筆電上進行，我們使用 **Intel RealSense D455** 作為感測器。

需要安裝 :
* ROS noetic
* Rtabmap
* Realsense SDK

安裝方法參考 [安裝筆記.pdf](安裝筆記.pdf)


需要安裝的python包可以透過 [requirement.txt](requirement.txt)  查看

將 my_rtabmap_pkg 資料夾整個丟到ROS工作空間中並編譯。
(如果有遇到問題則先創建一個新的pkg，在把my_rtabmap_pkg裡的src複製到新的pkg中)


# 事前準備
1. 須事先繪製拓樸地圖(透過JOSM繪製的osmag)，或想使用其他拓樸地圖也行。
2. 將nav.py中的osm_file改為自己存放地圖的路徑。
3. 將OPENAL_API_KEY檔案中的openAI api改為自己的api。
4. 將mapping.sh與localization.sh中roslaunch的database_path改為自己的點雲地圖地址。
5. 將mapping.sh, localization.sh與path_planning.sh的一些路就調整成符合自己的路徑。
6. 將path_planning.sh中的case改成自己想要的數字，可通過更改language來改變中英文。

# 測試
可以執行 [test.sh](/my_rtabmap_pkg/src/test.sh) 來測試相機、yolo與語音輸出與辨識

# 系統執行流程
1. 在想要使用導航的場域中執行 [mapping.sh](/my_rtabmap_pkg/src/mapping.sh) 在建圖的同時用Rtabmap的圖形化介面加上labal。
2. 建完圖之後再執行 [localization.sh](/my_rtabmap_pkg/src/localization.sh) 開始做定位
3. 最後再執行 [path_planning.sh](/my_rtabmap_pkg/src/path_planning.sh) 做導航

若是已經確認定位不會有問題則可以直接執行 [system.sh](/my_rtabmap_pkg/src/system.sh) 代替步驟2~步驟3。

