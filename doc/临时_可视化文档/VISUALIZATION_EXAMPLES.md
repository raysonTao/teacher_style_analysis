# 可视化功能使用示例

## 快速开始

### 1. 基础使用（自动启用可视化）

```python
from features.video_feature_extractor import VideoFeatureExtractor

# 创建特征提取器
extractor = VideoFeatureExtractor()

# 提取特征（自动生成可视化）
features = extractor.extract_features('path/to/your/video.mp4')

# 查看可视化输出路径
if features.get('visualization_output'):
    print(f"可视化结果目录: {features['visualization_output']['output_dir']}")
    print(f"可视化视频: {features['visualization_output']['video_output_path']}")
```

### 2. 使用测试脚本

```bash
# 激活环境
source teacher_style_env/bin/activate

# 测试单个视频（完整处理）
python test_visualization.py "path/to/video.mp4"

# 快速测试（只处理前100帧）
python quick_test_visualization.py
```

## 配置示例

### 场景1：只保存可视化视频（节省空间）

```python
from config.config import VIDEO_CONFIG

VIDEO_CONFIG['save_visualization_video'] = True
VIDEO_CONFIG['save_visualization_frames'] = False
```

### 场景2：自定义颜色方案（适合深色背景）

```python
from config.config import VIDEO_CONFIG

# 使用高对比度颜色
VIDEO_CONFIG['bbox_color'] = (0, 255, 255)       # 黄色边界框
VIDEO_CONFIG['pose_text_color'] = (0, 255, 255)  # 黄色文本
VIDEO_CONFIG['keypoint_color'] = (255, 0, 255)   # 品红色关键点
VIDEO_CONFIG['skeleton_color'] = (255, 255, 0)   # 青色骨架线
```

### 场景3：调整采样频率（提高处理速度）

```python
from config.config import VIDEO_CONFIG

# 每60帧检测一次（降低计算量）
VIDEO_CONFIG['detection_frame_interval'] = 60
VIDEO_CONFIG['visualization_frame_interval'] = 60
```

### 场景4：高分辨率视频优化

```python
from config.config import VIDEO_CONFIG

# 增加线条粗细和文字大小
VIDEO_CONFIG['bbox_thickness'] = 4
VIDEO_CONFIG['text_thickness'] = 3
VIDEO_CONFIG['text_font_scale'] = 0.9
VIDEO_CONFIG['keypoint_radius'] = 6
VIDEO_CONFIG['skeleton_thickness'] = 4
```

### 场景5：批量处理（禁用可视化以提高速度）

```python
from config.config import VIDEO_CONFIG
import glob

# 禁用可视化
VIDEO_CONFIG['enable_visualization'] = False

# 批量处理视频
video_files = glob.glob('data/videos/*.mp4')
for video in video_files:
    print(f"Processing: {video}")
    features = extractor.extract_features(video)
    # 处理特征...
```

## 输出示例

### 可视化文件结构

```
result/
└── sample_video_a1b2c3d4e5/
    ├── frames/
    │   ├── frame_000030.jpg    # 第30帧
    │   ├── frame_000060.jpg    # 第60帧
    │   ├── frame_000090.jpg    # 第90帧
    │   └── ...
    └── sample_video_a1b2c3d4e5_visualization.mp4
```

### 可视化元素说明

在生成的图片/视频中，你会看到：

1. **红色矩形框** 🟥
   - 标注YOLO检测到的人物位置
   - 框上方显示：`person: 0.85`（类别和置信度）

2. **绿色关键点** 🟢
   - MediaPipe检测到的33个身体关键点
   - 点的大小可通过 `keypoint_radius` 调整

3. **黄色骨架线** 🟡
   - 连接相关关键点的线条
   - 展示人体姿态结构

4. **蓝色信息文本** 🔵（左上角）
   ```
   Frame: 90
   Pose Confidence: 0.87
   Action: standing (0.92)
   ```

## 集成到主程序

### 在 main.py 中使用

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from features.video_feature_extractor import VideoFeatureExtractor
from config.config import VIDEO_CONFIG, logger

def analyze_video_with_visualization(video_path: str, enable_vis: bool = True):
    """
    分析视频并生成可视化结果

    Args:
        video_path: 视频路径
        enable_vis: 是否启用可视化
    """
    # 配置可视化
    VIDEO_CONFIG['enable_visualization'] = enable_vis

    # 创建提取器
    extractor = VideoFeatureExtractor()

    # 提取特征
    logger.info(f"开始分析视频: {video_path}")
    features = extractor.extract_features(video_path)

    # 输出结果
    logger.info(f"检测到动作: {dict(features['action_counts'])}")

    if features.get('visualization_output'):
        vis_info = features['visualization_output']
        logger.info(f"可视化结果已保存到: {vis_info['output_dir']}")

        return features, vis_info

    return features, None

# 使用示例
if __name__ == "__main__":
    video = "src/data/videos/sample.mp4"
    features, vis_info = analyze_video_with_visualization(video)

    if vis_info:
        print(f"\n查看可视化结果:")
        print(f"  视频: {vis_info['video_output_path']}")
        print(f"  帧图片: {vis_info['frames_dir']}")
```

### 在API中使用

```python
from fastapi import FastAPI, UploadFile
from features.video_feature_extractor import VideoFeatureExtractor

app = FastAPI()

@app.post("/analyze")
async def analyze_video(file: UploadFile):
    # 保存上传的视频
    video_path = f"temp/{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # 提取特征和生成可视化
    extractor = VideoFeatureExtractor()
    features = extractor.extract_features(video_path)

    # 返回分析结果和可视化路径
    return {
        "features": {
            "action_counts": dict(features['action_counts']),
            "avg_motion_energy": features['avg_motion_energy']
        },
        "visualization": features.get('visualization_output')
    }
```

## 性能对比

| 配置 | 100帧处理时间 | 1000帧处理时间 | 磁盘占用（10分钟视频） |
|------|--------------|---------------|---------------------|
| 无可视化 | ~8秒 | ~80秒 | 50MB（特征数据） |
| 仅视频 | ~10秒 | ~100秒 | 150MB（+100MB视频） |
| 仅帧图片 | ~11秒 | ~110秒 | 250MB（+200MB图片） |
| 视频+图片 | ~12秒 | ~120秒 | 300MB（全部） |

*测试环境: Apple M4, 1280x720视频, 30fps*

## 常见问题

### Q1: 为什么有些帧没有绿色关键点？

**A:** 当MediaPipe无法检测到清晰的人体姿态时（如人物被遮挡、背对摄像头等），不会绘制关键点。这是正常现象。

### Q2: 如何调整检测灵敏度？

**A:** 修改置信度阈值：
```python
VIDEO_CONFIG['detection_confidence_threshold'] = 0.3  # 默认0.5，降低以检测更多目标
```

### Q3: 可视化视频无法播放？

**A:** 尝试安装ffmpeg或更改编码器：
```bash
brew install ffmpeg  # macOS
```

或修改 `visualization_utils.py` 中的编码器：
```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 尝试不同的编码器
```

### Q4: 如何只可视化特定帧？

**A:** 修改采样间隔：
```python
VIDEO_CONFIG['detection_frame_interval'] = 90  # 每90帧检测一次（每3秒一次，30fps）
```

### Q5: 检测到多个人时会怎样？

**A:** 系统会为每个检测到的人物绘制独立的边界框和姿态关键点。所有人物会显示在同一帧中。

## 最佳实践

1. **开发调试**：启用完整可视化（视频+图片）
2. **生产环境**：仅保存视频或完全禁用
3. **论文展示**：使用高分辨率配置和高对比度颜色
4. **批量处理**：禁用可视化，仅在需要时启用
5. **存储优化**：定期清理旧的可视化文件

## 进阶：自定义绘制逻辑

如果需要自定义可视化效果，可以修改 `src/features/visualization_utils.py`：

```python
def draw_detection_and_pose(self, frame, detection, pose_result, ...):
    vis_frame = frame.copy()

    # 你的自定义绘制逻辑
    # 例如：添加置信度热图、轨迹追踪、统计图表等

    return vis_frame
```

## 相关文件

- [visualization_utils.py](src/features/visualization_utils.py) - 可视化核心模块
- [video_feature_extractor.py](src/features/video_feature_extractor.py#L236-245) - 集成点
- [config.py](src/config/config.py#L114-127) - 配置选项
- [VISUALIZATION_README.md](VISUALIZATION_README.md) - 完整文档
