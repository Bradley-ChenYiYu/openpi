# OpenPI 完整工作流程指南（Tracer 機器人）

本文檔提供 OpenPI 框架從資料收集到語音命令集成的完整工作流程，專為 Tracer 移動機器人平台設計。

---

## 目錄

1. [資料收集](#1-資料收集)
2. [數據轉換](#2-數據轉換)
3. [模型訓練](#3-模型訓練)
4. [模型推論](#4-模型推論)
5. [語音命令集成](#5-語音命令集成)
6. [TODO](#todo)

---

## 1. 資料收集

### 1.1 啟動環境和遙控

```bash
# 終端 1: 啟動機器人
ros2 launch tracer_base tracer_bringup.launch.py

# 終端 2: 啟動遙控（手柄或鍵盤）
ros2 launch tracer_base teleop-launch.py
# 或
ros2 run teleop_twist_keyboard teleop_twist_keyboard.py
```

**手柄映射：**
- 左搖桿：控制前進/轉向
- `X` 按鈕：按住禁用安全鎖定

---

### 1.2 記錄數據

```bash
# 終端 3: 記錄數據
ros2 bag record -b 2147483648 -e "/cmd_vel|/odom|.*/color/(image_raw|camera_info)" -s sqlite3
```

- 每段長度：10-60 秒
- 建議收集 30-50 段演示
- 停止：`Ctrl+C`

**參數說明：**
- `-b 2147483648`: 將每個 bag 檔案的最大大小設定為 2GB（2 x 1024^3 bytes），可避免 FAT32 或類似檔案系統的單檔大小限制導致中斷。
- `-e`: 使用正則表達式匹配多個 topic，例如 `"/cmd_vel|/odom|.*/color/(image_raw|camera_info)"` 同時記錄原始影像與 camera_info。
- `-s sqlite3`: 使用 sqlite3 存儲格式以降低磁碟碎片和提高穩定性（可選）。

---

### 1.3 質量檢查

```bash
ros2 bag info rosbag_dir/
```

可選：使用 `rviz2` 播放驗證。

---

## 2. 數據轉換

### 2.1 準備配置檔案

```bash
cp scripts/rosbag-to-lerobot/config/topic_mapping.example.yaml \
  scripts/rosbag-to-lerobot/config/tracer_topic_mapping.yaml

cp scripts/rosbag-to-lerobot/config/metadata.example.yaml \
  scripts/rosbag-to-lerobot/config/tracer_metadata.yaml
```

---

### 2.2 Topic 映射配置 (`tracer_topic_mapping.yaml`)

```yaml
camera_topics:
  front:
    topic: /camera/camera/color/image_raw
    msg_type: sensor_msgs/msg/Image
    resize: [224, 224]

state_topic:
  topic: /odom
  msg_type: nav_msgs/msg/Odometry
  dim: 2
  fields:
    - twist.twist.linear.x
    - twist.twist.angular.z

action_topic:
  topic: /cmd_vel
  msg_type: geometry_msgs/msg/Twist
  dim: 2
  fields:
    - linear.x
    - angular.z

sync:
  reference_stream: action
  tolerance_sec: 0.05
  drop_policy: drop
```

更多欄位說明：

- `camera_topics`: 可定義多個相機（例如 `front`, `wrist`, `rear` 等），對於壓縮影像使用 `sensor_msgs/msg/CompressedImage` 並在處理時解壓。
- `resize`: 選填，建議在轉換階段縮放影像以節省儲存和加速訓練（例如 `[224, 224]`）。
- `state_topic` / `action_topic` 的 `fields`: 支援點選嵌套欄位路徑（例如 `pose.pose.position.x` 或 `twist.twist.linear.x`）。
- `sync.tolerance_sec`: 時間同步最大允許誤差（秒），超過將根據 `drop_policy` 丟棄樣本。

使用 `ros2 bag info your_bag.bag` 檢查實際 topic 名稱與訊息類型，並調整 `tracer_topic_mapping.yaml`。

---

### 2.3 任務描述配置 (`tracer_metadata.yaml`)

**手動定義：**

```yaml
default_task: Navigate around

episodes:
  rosbag2_20260824-10_30_00:
    task: Navigate to the brown cups on the table.
  rosbag2_20260824-10_35_00:
    task: Navigate to the sandwich on the kitchen counter.
```

重要說明：
- **一個 rosbag 資料夾對應一個 episode**；`episodes` 的關鍵字（例如 `rosbag2_20260824-10_30_00`）必須與實際的 rosbag 資料夾名稱完全匹配。
- `default_task` 將套用於未在 `episodes` 中明確定義任務的資料夾。

可選：使用外部工具自動生成任務提示（例如 Ollama / LLM）從影片或摘要生成 `task` 文本。

**自動生成（可選）：**

```bash
uv run scripts/rosbag-to-lerobot/rosbag2video/rosbag2video.py -r 50 rosbag_dir/
uv run scripts/rosbag-to-lerobot/generate_vid_prompt_ollama.py \
  --metadata-path scripts/rosbag-to-lerobot/config/tracer_metadata.yaml \
  --parent-dir rosbag_dir/
```

---

### 2.4 執行轉換

```bash
uv run scripts/rosbag-to-lerobot/convert_rosbag_to_lerobot.py \
  --input-bag-path rosbag_dir/ \
  --repo-id your_hf_username/tracer_dataset \
  --robot-type tracer \
  --fps 50 \
  --config-path scripts/rosbag-to-lerobot/config/tracer_topic_mapping.yaml \
  --metadata-path scripts/rosbag-to-lerobot/config/tracer_metadata.yaml \
    --force-clean-output
```

說明：
- `--input-bag-path` 可為單一 bag 檔案或包含多個 rosbag 的目錄。
- `--force-clean-output` 會在開始轉換前清除舊輸出，避免殘留檔案混淆。

可選（從 rosbag 生成影片與自動任務提示）：

```bash
uv run scripts/rosbag-to-lerobot/rosbag2video/rosbag2video.py -r 50 rosbag_dir/
uv run scripts/rosbag-to-lerobot/generate_vid_prompt_ollama.py \
  --metadata-path scripts/rosbag-to-lerobot/config/tracer_metadata.yaml \
  --parent-dir rosbag_dir/
```

---

### 2.5 驗證

```bash
uv run scripts/rosbag-to-lerobot/visualize_dataset.py \
  --mode distant \
  --repo-id your_hf_username/tracer_dataset \
  --episode-index 0
```

---

### 遠端視覺化與端口轉發（可選）

如果在遠端伺服器上執行視覺化，請使用 SSH 端口轉發將遠端服務映射到本地：

```bash
# SSH 端口轉發示例
ssh -L 7262:localhost:7262 -L 7263:localhost:7263 user@remote_server
```

然後在視覺化命令中指定 web 與 websocket 端口：

```bash
uv run scripts/rosbag-to-lerobot/visualize_dataset.py \
  --mode distant \
  --repo-id your_hf_username/tracer_dataset \
  --web-port 7262 \
  --ws-port 7263 \
  --episode-index 0
```

---

### 常見問題與除錯

以下是常見錯誤與建議的排查步驟：

- **Topic 不匹配**：轉換腳本提示找不到某些 topic 或訊息類型不符。檢查 `scripts/rosbag-to-lerobot/config/tracer_topic_mapping.yaml` 並使用：

```bash
ros2 bag info your_bag.bag
```

確認實際的 topic 路徑與訊息類型後更新配置。

- **影像無法顯示或損壞**：確認 rosbag 同時包含 `image_raw` 與 `camera_info`，並檢查 `topic_mapping.yaml` 中的相機設定與內參是否正確。

- **資料集太大**：考慮降低 FPS（例如 50 → 30）、在轉換時壓縮或縮放影像（`resize`），或只記錄必要的 topics。

- **HuggingFace 上傳失敗**：確保先登入：

```bash
huggingface-cli login
```

並確認網路與磁碟空間足夠。

- **視覺化無法啟動**：檢查指定的 `--web-port` 與 `--ws-port` 是否被佔用；如果在遠端運行，確認 SSH 端口轉發與防火牆設定。

---

### 批量處理（大量 rosbag 時）

如果要對多個 rosbag 資料夾進行批量轉換，可以使用如下簡單腳本：

```bash
#!/bin/bash
# 批量轉換示例

for bag_dir in rosbag_dir/bags_*/; do
    echo "Processing: $bag_dir"
    uv run scripts/rosbag-to-lerobot/convert_rosbag_to_lerobot.py \
        --input-bag-path "$bag_dir" \
        --repo-id your_hf_username/tracer_combined \
        --robot-type tracer \
        --fps 50 \
        --config-path scripts/rosbag-to-lerobot/config/tracer_topic_mapping.yaml \
        --metadata-path scripts/rosbag-to-lerobot/config/tracer_metadata.yaml
done
```


## 3. 模型訓練

### 3.1 設置環境

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run wandb login
```

**硬體要求：**
- 推論：> 8 GB (RTX 4090)
- LoRA 微調：> 22.5 GB (RTX 4090)
- 完整微調：> 70 GB (A100/H100)

---

### 3.2 計算歸一化統計量

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_tracer_finetune
```

---

### 3.3 選擇配置

**推薦配置：** `pi0_tracer_finetune`

**相關文件：**
- 配置定義：`src/openpi/training/config.py` (Pi0TracerFinetuneDataConfig, TrainConfig)
- 策略類：`src/openpi/policies/tracer_policy.py`

---

### 3.4 執行訓練

```bash
# 前台運行
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi0_tracer_finetune \
  --exp-name=my_experiment_tracer \
  --overwrite

# 後台運行
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 nohup uv run scripts/train.py \
  pi0_tracer_finetune \
  --exp-name=my_experiment_tracer \
  --overwrite > train_output.log 2>&1 &
```

---

### 3.5 監控訓練

```bash
# W&B Dashboard
wandb dashboard --entity your_wandb_username

# 日誌
tail -f train_output.log
```

---

### 3.6 檢查點

位置：`checkpoints/pi0_tracer_finetune/my_experiment_tracer/{step}/`

```bash
ls -lt checkpoints/pi0_tracer_finetune/my_experiment_tracer/ | head -5
```

---

## 4. 模型推論

### 4.1 啟動 Policy 服務器

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_tracer_finetune \
  --policy.dir=checkpoints/pi0_tracer_finetune/my_experiment_tracer/20000
```

預設端口：`8000`

---

### 4.2 測試

```bash
uv run examples/simple_client/main.py
```

---

### 4.3 與 Tracer 集成（Docker）

```bash
cd examples/tracer
docker compose -f compose.yml up
```

**登入容器：**
- OpenPI: `docker exec -it tracer-openpi_server-1 bash`
- ROS2: `docker exec -it openpi_tracer bash`

**在 OpenPI 容器中：**
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_tracer_finetune \
  --policy.dir=checkpoints/pi0_tracer_finetune/my_experiment_tracer/20000
```

**在 ROS2 容器中：**
```bash
colcon build --packages-select tracer_base && \
source install/setup.bash && \
ros2 launch tracer_base tracer_bringup.launch.py

colcon build --packages-select pi_bridge && \
source install/setup.bash && \
ros2 launch pi_bridge websocket_bridge.launch.py \
  prompt:="Navigate to the table with the paper cups" \
  inferred_cmd_topic:=/cmd_vel
```

---

## 5. 語音命令集成

```bash
AUDIO_GID=$(getent group audio | cut -d: -f3) \
  docker compose -f scripts/whisper.cpp/examples/command/compose.yml run --build --rm whisper bash
```

如果想保留原本的 `docker run` 行為，這個 compose 檔案會等效地提供同樣的裝置、PulseAudio 和 host network 設定。

### 5.1 編譯 whisper.cpp

```bash
cd scripts/whisper.cpp
sudo apt install -y cmake build-essential

mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON
make -j$(nproc)
```

---

### 5.2 下載模型

```bash
mkdir -p ~/whisper_models
cd ~/whisper_models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/base.en.ggmlf16
```

推薦模型：
- `base.en.ggmlf16`（英文，推薦）
- `base.ggmlf16`（多語種，含繁體中文）

---

### 5.3 啟動 Whisper 服務器

```bash
cd scripts/whisper.cpp/build/bin
./whisper-server -m ~/whisper_models/base.en.ggmlf16 -nt 8 -ps 128 -ct
```

---

### 5.4 啟動語音節點

```bash
cd ros2_ws
colcon build --packages-select pi_bridge
source install/setup.bash
ros2 run pi_bridge voice_command_node.py
```

---

### 5.3b 可選：使用 `whisper-command` 並管道到 `command-ws-server`

在某些環境下，可用下列命令啟動前端識別器並將結果送入命令伺服器（在 repo 根目錄執行）：

```bash
./build/bin/whisper-command --ws-enable| ./build/bin/command-ws-server
```

此命令等同於先啟動識別器再啟動 `command-ws-server`，但以管道方式串接，方便在無需額外代理的情況下直接轉發識別結果。


### 5.5 自定義語音命令

**1. 配置 whisper.cpp 識別關鍵詞：**

編輯 `scripts/whisper.cpp/examples/command/commands.txt`：

```
brown cups
sandwich
kitchen
stop
wait
```

每個關鍵詞佔一行，whisper 將專注識別這些詞彙。

**2. 配置命令映射（`voice_command_node.py`）：**

```python
GO_TO_COMMANDS = {
    "brown cups": "Go to the table with brown cups.",
    "sandwich": "Go to the table with a sandwich.",
    "kitchen": "Go to the kitchen area.",
}
```

---

### 5.6 完整測試流程

```bash
# 終端 1: Whisper 服務器
./build/bin/whisper-command  -m ~/whisper_models/base.en.ggmlf16 -nt 8 --ws-enable -cmd examples/command/commands.txt | ./build/bin/command-ws-server

# 終端 2: Policy 服務器
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_tracer_finetune \
  --policy.dir=checkpoints/pi0_tracer_finetune/my_experiment_tracer/20000

# 終端 3: Tracer
ros2 launch tracer_base tracer_bringup.launch.py

# 終端 4: Bridge
ros2 launch pi_bridge websocket_bridge.launch.py \
  prompt:="Navigate to the brown cups" \
  inferred_cmd_topic:=/cmd_vel

# 終端 5: 語音節點
ros2 run pi_bridge voice_command_node.py
```

測試命令：
- "Go to the brown cups"
- "Navigate to the sandwich"
- "Stop"

---

## 常見問題

**CUDA OOM：**
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# 或使用 --fsdp-devices <n>
```

**Missing norm stats：**
```bash
uv run scripts/compute_norm_stats.py --config-name <your_config>
```

**語音命令未執行：**
- 檢查 `command_probability_threshold` 閾值
- 驗證命令映射表

## TODO

1. 目前主要 git branch 為：
  - tested-feature: 測試成功的 branch 會被 merge 進來
  - robot/voice-command: Voice command feature
  - data/rosbag-to-lerobot: Training 相關
  - robot/tracer: 與 tracer 相關
  - robot/aruco-nav: 為完成整的 ArUco navigation

---

## 其他資源

- [README](../README.md)
- [Docker](docker.md)
- [Remote Inference](remote_inference.md)
- [Norm Stats](norm_stats.md)

---

*2026-08-25: 初始版本*
