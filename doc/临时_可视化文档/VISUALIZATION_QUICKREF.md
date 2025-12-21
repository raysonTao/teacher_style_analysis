# 可视化功能快速参考

## 🎯 一分钟上手

```bash
# 测试可视化功能
source teacher_style_env/bin/activate
python quick_test_visualization.py
```

查看结果：`result/{video_id}/`

## 📝 基础配置

```python
from config.config import VIDEO_CONFIG

# 启用/禁用
VIDEO_CONFIG['enable_visualization'] = True  # True/False

# 输出选项
VIDEO_CONFIG['save_visualization_video'] = True   # 保存视频
VIDEO_CONFIG['save_visualization_frames'] = True  # 保存图片

# 自定义颜色（BGR格式）
VIDEO_CONFIG['bbox_color'] = (0, 0, 255)      # 红色边界框
VIDEO_CONFIG['pose_text_color'] = (255, 0, 0)  # 蓝色文字
VIDEO_CONFIG['keypoint_color'] = (0, 255, 0)   # 绿色关键点
VIDEO_CONFIG['skeleton_color'] = (0, 255, 255) # 黄色骨架
```

## 🎨 可视化元素

| 元素 | 颜色 | 说明 |
|------|------|------|
| 检测框 | 🟥 红色 | YOLO检测的人物位置 |
| 关键点 | 🟢 绿色 | MediaPipe的33个身体点 |
| 骨架线 | 🟡 黄色 | 连接关键点的线条 |
| 信息文本 | 🔵 蓝色 | 帧号、置信度、动作 |

## 📂 输出结构

```
result/
└── {video_name}_{hash}/
    ├── frames/
    │   └── frame_XXXXXX.jpg
    └── {video_name}_{hash}_visualization.mp4
```

## 💡 常用场景

### 调试开发
```python
VIDEO_CONFIG['enable_visualization'] = True
VIDEO_CONFIG['save_visualization_frames'] = True
```

### 生产环境
```python
VIDEO_CONFIG['enable_visualization'] = False  # 提高速度
```

### 论文展示
```python
VIDEO_CONFIG['bbox_thickness'] = 3
VIDEO_CONFIG['text_font_scale'] = 0.8
VIDEO_CONFIG['keypoint_radius'] = 5
```

## 📖 完整文档

- [VISUALIZATION_README.md](VISUALIZATION_README.md) - 详细说明
- [VISUALIZATION_EXAMPLES.md](VISUALIZATION_EXAMPLES.md) - 使用示例
- [VISUALIZATION_IMPLEMENTATION.md](VISUALIZATION_IMPLEMENTATION.md) - 技术细节

## ⚡ 性能提示

| 配置 | 速度 | 空间 |
|------|------|------|
| 禁用可视化 | 最快 | 最小 |
| 仅视频 | 较快 | 中等 |
| 视频+图片 | 较慢 | 最大 |
