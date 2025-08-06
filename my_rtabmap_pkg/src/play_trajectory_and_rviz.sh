#!/bin/bash

# 檢查是否指定了 .bag 檔案
if [ -z "$1" ]; then
  echo "❌ 錯誤：請提供要播放的 .bag 檔案名稱"
  echo "用法：./play_trajectory_and_rviz.sh trajectory.bag"
  exit 1
fi

BAG_FILE=$1

# 啟動 roscore（如果還沒啟動）
if ! pgrep -x "roscore" > /dev/null; then
  echo "[1/3] 啟動 roscore..."
  gnome-terminal --tab -- bash -c "roscore; exec bash"
  sleep 5
fi

# 啟動 RViz 並載入顯示設定（可選）
echo "[2/3] 啟動 RViz..."
gnome-terminal --tab -- bash -c "rviz -d display_path.rviz; exec bash"
sleep 2

# 播放 rosbag
echo "[3/3] 播放 $BAG_FILE ..."
gnome-terminal --tab -- bash -c "rosbag play $BAG_FILE; exec bash"

echo "✅ 播放與 RViz 顯示已啟動！"

