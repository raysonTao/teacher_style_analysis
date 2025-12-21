# 视频分析可视化功能

## 功能概述

本系统现已集成完整的可视化功能，可以在视频分析过程中实时绘制：
- **YOLO检测框**：红色边界框标注检测到的人物目标
- **姿态关键点**：绿色圆点显示MediaPipe检测到的33个身体关键点
- **骨架连接线**：黄色线条连接关键点形成人体骨架
- **动作信息**：蓝色文字显示识别的动作类型、置信度、姿态置信度等

## 输出结构

可视化结果保存在 `../result/` 目录下，按视频ID分类存储：

```
result/
└── {video_name}_{hash}/
    ├── frames/                          # 可视化帧图片（可选）
    │   ├── frame_000030.jpg
    │   ├── frame_000060.jpg
    │   └── ...
    └── {video_name}_{hash}_visualization.mp4  # 可视化视频
```

- **视频ID格式**：`{视频文件名}_{文件hash前10位}`
- **帧图片**：按检测间隔保存（默认每30帧）
- **可视化视频**：包含所有检测帧的完整视频

## 配置选项

所有可视化配置在 `src/config/config.py` 的 `VIDEO_CONFIG` 中：

### 基础配置

```python
'enable_visualization': True,           # 启用/禁用可视化（默认：True）
'visualization_frame_interval': 30,    # 可视化采样间隔（默认：30帧）
'save_visualization_video': True,      # 保存可视化视频（默认：True）
'save_visualization_frames': True,     # 保存可视化帧图片（默认：True）
```

### 视觉样式配置

#### 检测框样式
```python
'bbox_color': (255, 0, 0),            # 红色边界框 (BGR格式)
'bbox_thickness': 2,                   # 边界框线条粗细
```

#### 文本样式
```python
'pose_text_color': (255, 0, 0),       # 蓝色文本 (BGR格式)
'text_font': cv2.FONT_HERSHEY_SIMPLEX, # 字体
'text_font_scale': 0.6,                # 字体大小
'text_thickness': 2,                   # 文本粗细
```

#### 姿态可视化样式
```python
'keypoint_color': (0, 255, 0),        # 绿色关键点 (BGR格式)
'keypoint_radius': 3,                  # 关键点半径（像素）
'skeleton_color': (0, 255, 255),      # 黄色骨架线 (BGR格式)
'skeleton_thickness': 2                # 骨架线粗细
```

## 使用方法

### 1. 使用测试脚本

```bash
# 激活虚拟环境
source teacher_style_env/bin/activate

# 运行测试脚本
python test_visualization.py <video_path>

# 示例
python test_visualization.py src/data/videos/sample.mp4
```

### 2. 在代码中使用

```python
from features.video_feature_extractor import VideoFeatureExtractor

# 创建特征提取器（自动启用可视化）
extractor = VideoFeatureExtractor()

# 提取特征（自动生成可视化）
features = extractor.extract_features('path/to/video.mp4')

# 获取可视化输出路径
if features.get('visualization_output'):
    vis_info = features['visualization_output']
    print(f"视频ID: {vis_info['video_id']}")
    print(f"输出目录: {vis_info['output_dir']}")
    print(f"可视化视频: {vis_info['video_output_path']}")
```

### 3. 临时禁用可视化

如果需要临时禁用可视化（提高处理速度），可以修改配置：

```python
from config.config import VIDEO_CONFIG

# 禁用可视化
VIDEO_CONFIG['enable_visualization'] = False

# 或只禁用保存视频，只保存帧图片
VIDEO_CONFIG['save_visualization_video'] = False
VIDEO_CONFIG['save_visualization_frames'] = True
```

## 可视化元素说明

### 1. 检测框（红色矩形）
- 标注YOLO检测到的人物目标位置
- 框上方显示类别（person）和置信度
- 示例：`person: 0.95`

### 2. 姿态关键点（绿色圆点）
- MediaPipe检测到的33个身体关键点
- 仅显示可见性 > 0.5 的关键点
- 关键点包括：面部、躯干、手臂、手部、腿部

### 3. 骨架连接线（黄色线条）
- 连接相关关键点形成人体骨架结构
- 仅连接可见性 > 0.5 的关键点对
- 帮助理解人体姿态结构

### 4. 信息文本（蓝色文字）
在视频左上角显示：
- **Frame**: 当前帧编号
- **Pose Confidence**: 姿态检测置信度
- **Action**: 识别的动作类型及置信度
- 示例：
  ```
  Frame: 120
  Pose Confidence: 0.87
  Action: standing (0.92)
  ```

## 性能考虑

### 可视化对性能的影响

| 配置 | 处理速度 | 存储空间 |
|------|---------|---------|
| 禁用可视化 | 最快 | 最小 |
| 仅保存视频 | 较快 | 中等 |
| 仅保存帧图片 | 中等 | 较大 |
| 同时保存视频和帧 | 较慢 | 最大 |

### 优化建议

1. **大规模批处理**：禁用可视化或仅保存视频
   ```python
   VIDEO_CONFIG['enable_visualization'] = False
   ```

2. **调试和展示**：启用完整可视化
   ```python
   VIDEO_CONFIG['enable_visualization'] = True
   VIDEO_CONFIG['save_visualization_video'] = True
   VIDEO_CONFIG['save_visualization_frames'] = True
   ```

3. **减少采样频率**：增加可视化间隔
   ```python
   VIDEO_CONFIG['visualization_frame_interval'] = 60  # 每60帧保存一次
   ```

## 自定义样式示例

### 调整颜色主题

```python
# 深色主题（适合深色背景视频）
VIDEO_CONFIG['bbox_color'] = (0, 255, 255)        # 黄色边界框
VIDEO_CONFIG['pose_text_color'] = (0, 255, 255)   # 黄色文字
VIDEO_CONFIG['keypoint_color'] = (255, 0, 255)    # 品红色关键点
VIDEO_CONFIG['skeleton_color'] = (255, 255, 0)    # 青色骨架线
```

### 调整线条粗细（适合高分辨率视频）

```python
VIDEO_CONFIG['bbox_thickness'] = 3
VIDEO_CONFIG['text_thickness'] = 3
VIDEO_CONFIG['skeleton_thickness'] = 3
VIDEO_CONFIG['keypoint_radius'] = 5
VIDEO_CONFIG['text_font_scale'] = 0.8
```

## 故障排查

### 问题1：未生成可视化文件

**检查**：
1. 确认 `enable_visualization` 为 `True`
2. 检查 `result/` 目录是否有写入权限
3. 查看日志文件中的错误信息

### 问题2：可视化视频无法播放

**原因**：可能缺少视频编码器

**解决**：
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg

# 或更改编码器（在 visualization_utils.py 中）
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 尝试其他编码器
```

### 问题3：检测框或关键点不显示

**检查**：
1. 确认视频中有清晰的人物目标
2. 检查 `detection_confidence_threshold` 设置（默认0.5）
3. 查看日志确认检测是否成功

## 技术细节

### MediaPipe姿态关键点编号

系统使用MediaPipe的33个关键点：
- **0-10**: 面部关键点
- **11-16**: 上肢（肩膀、肘部、手腕）
- **17-22**: 手部关键点
- **23-28**: 下肢（臀部、膝盖、脚踝）
- **29-32**: 脚部关键点

### 骨架连接关系

定义在 `visualization_utils.py` 中的 `POSE_CONNECTIONS` 变量，包含40+条连接关系，构建完整的人体骨架结构。

## 反馈和改进

如果您有更好的可视化建议或发现问题，欢迎：
1. 在 GitHub Issues 中反馈
2. 直接修改配置文件和可视化代码
3. 参考 `src/features/visualization_utils.py` 实现自定义绘制逻辑
