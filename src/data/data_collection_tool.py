#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据收集和标注工具
用于帮助用户收集真实课堂数据并标注风格标签
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

# 导入全局logger
from ..config.config import logger

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# 风格标签定义
STYLE_LABELS = [
    '理论讲授型',
    '启发引导型',
    '互动导向型',
    '逻辑推导型',
    '题目驱动型',
    '情感表达型',
    '耐心细致型'
]

class DataCollectionTool:
    """数据收集和标注工具"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.videos_dir = self.data_dir / 'videos'
        self.annotations_dir = self.data_dir / 'annotations'
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        for dir_path in [self.videos_dir, self.annotations_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def generate_sample_annotation(self, video_id: str, video_path: str) -> Dict:
        """
        生成样本标注文件
        
        Args:
            video_id: 视频ID
            video_path: 视频文件路径
            
        Returns:
            标注字典
        """
        sample_annotation = {
            "video_id": video_id,
            "video_path": str(video_path),
            "teacher_id": "",
            "discipline": "",
            "grade": "",
            "duration": 0.0,
            "annotations": {
                "global_style": {
                    "primary_style": "",
                    "secondary_style": "",
                    "style_scores": {
                        label: 0.0 for label in STYLE_LABELS
                    }
                },
                "segmented_style": [],
                "key_observations": [],
                "overall_evaluation": ""
            },
            "annotator": "",
            "annotation_time": ""
        }
        
        return sample_annotation
    
    def create_annotation_file(self, video_path: str):
        """
        为单个视频创建标注文件
        
        Args:
            video_path: 视频文件路径
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return
        
        # 生成视频ID（基于文件名）
        video_id = video_path.stem
        
        # 生成标注文件路径
        annotation_file = self.annotations_dir / f"{video_id}_annotation.json"
        
        # 检查标注文件是否已存在
        if annotation_file.exists():
            logger.info(f"标注文件已存在: {annotation_file}")
            return
        
        # 生成样本标注
        sample_annotation = self.generate_sample_annotation(video_id, video_path)
        
        # 保存标注文件
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(sample_annotation, f, ensure_ascii=False, indent=2)
        
        logger.info(f"标注文件已创建: {annotation_file}")
    
    def batch_create_annotation_files(self, videos_dir: str):
        """
        批量为目录中的视频创建标注文件
        
        Args:
            videos_dir: 视频目录路径
        """
        videos_dir = Path(videos_dir)
        if not videos_dir.exists():
            logger.error(f"视频目录不存在: {videos_dir}")
            return
        
        # 支持的视频格式
        video_formats = ['.mp4', '.avi', '.mov', '.wmv', '.flv']
        
        # 遍历目录中的视频文件
        video_files = []
        for root, _, files in os.walk(videos_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_formats):
                    video_path = Path(root) / file
                    video_files.append(video_path)
        
        logger.info(f"发现 {len(video_files)} 个视频文件")
        
        # 批量创建标注文件
        for i, video_path in enumerate(video_files, 1):
            logger.info(f"处理视频 {i}/{len(video_files)}: {video_path}")
            self.create_annotation_file(video_path)
        
        logger.info(f"批量处理完成，共创建 {len(video_files)} 个标注文件")
    
    def generate_data_collection_guide(self):
        """
        生成数据收集指南
        """
        guide_content = """# 教师风格画像分析系统 - 数据收集与标注指南

## 一、数据收集要求

### 1. 视频数据要求
- **分辨率**: 建议1920×1080，最低不低于1280×720
- **帧率**: 25fps或30fps
- **时长**: 每节课约45-60分钟
- **画面要求**: 清晰可见教师全身动作，光线充足，无明显遮挡
- **音频要求**: 清晰可辨教师语音，无明显背景噪声

### 2. 数据多样性
- **学科覆盖**: 数学、语文、英语、物理、化学、生物、历史、地理、政治等
- **年级覆盖**: 初中、高中、大学
- **教师类型**: 不同教龄、不同教学风格的教师

## 二、标注规范

### 1. 全局风格标注
为整个视频标注主要风格和次要风格，并为每种风格打分（0-10分）。

**风格定义**:
- **理论讲授型**: 注重系统知识传授和理论讲解
- **启发引导型**: 通过问题引导学生自主思考和探索
- **互动导向型**: 强调师生互动和课堂参与
- **逻辑推导型**: 注重逻辑推理过程和思维训练
- **题目驱动型**: 通过例题讲解帮助学生理解和应用
- **情感表达型**: 教学过程中情感丰富，富有感染力
- **耐心细致型**: 教学节奏适中，注重细节和学生接受度

### 2. 分段风格标注
将视频划分为若干片段（建议每5-10分钟一段），为每个片段标注主导风格。

### 3. 关键观察点标注
记录课堂中的关键观察点，如：
- 典型的教学行为
- 有效的教学策略
- 可改进的教学环节

## 三、标注工具使用

### 1. 创建标注文件
使用本工具为每个视频创建标注文件：
```bash
python data_collection_tool.py create-annotation --video /path/to/video.mp4
```

### 2. 批量创建标注文件
为目录中的所有视频批量创建标注文件：
```bash
python data_collection_tool.py batch-create --dir /path/to/videos
```

### 3. 填写标注文件
使用文本编辑器打开生成的JSON标注文件，按照示例格式填写标注内容。

## 四、示例标注文件格式

```json
{
  "video_id": "example_video",
  "video_path": "path/to/video.mp4",
  "teacher_id": "T001",
  "discipline": "数学",
  "grade": "高中",
  "duration": 45.0,
  "annotations": {
    "global_style": {
      "primary_style": "逻辑推导型",
      "secondary_style": "题目驱动型",
      "style_scores": {
        "理论讲授型": 7.5,
        "启发引导型": 6.0,
        "互动导向型": 5.0,
        "逻辑推导型": 9.0,
        "题目驱动型": 8.5,
        "情感表达型": 4.0,
        "耐心细致型": 6.5
      }
    },
    "segmented_style": [
      {
        "start_time": 0.0,
        "end_time": 10.0,
        "style": "理论讲授型",
        "description": "教师讲解基本概念"
      },
      {
        "start_time": 10.0,
        "end_time": 25.0,
        "style": "逻辑推导型",
        "description": "教师推导数学公式"
      }
    ],
    "key_observations": [
      "教师在推导公式时步骤清晰，逻辑性强",
      "师生互动较少，主要以教师讲解为主",
      "教师对学生的疑问解答耐心细致"
    ],
    "overall_evaluation": "该教师教学风格以逻辑推导为主，讲解清晰，注重例题训练，但互动性有待加强。"
  },
  "annotator": "Annotator001",
  "annotation_time": "2024-11-15T10:30:00"
}
```

## 五、数据验证

标注完成后，使用以下命令验证标注文件格式：
```bash
python data_collection_tool.py validate --dir /path/to/annotations
```

## 六、数据提交

标注完成后，将视频文件和标注文件一起打包提交。
"""
        
        # 保存指南文件
        guide_file = self.data_dir / 'data_collection_guide.md'
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        logger.info(f"数据收集指南已生成: {guide_file}")
    
    def validate_annotation(self, annotation_file: str) -> bool:
        """
        验证标注文件格式
        
        Args:
            annotation_file: 标注文件路径
            
        Returns:
            验证结果
        """
        annotation_file = Path(annotation_file)
        if not annotation_file.exists():
            logger.error(f"标注文件不存在: {annotation_file}")
            return False
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            
            # 检查必要字段
            required_fields = [
                'video_id', 'video_path', 'teacher_id',
                'discipline', 'grade', 'annotations'
            ]
            
            for field in required_fields:
                if field not in annotation:
                    logger.error(f"标注文件缺少必要字段: {field}")
                    return False
            
            # 检查风格分数字段
            style_scores = annotation['annotations']['global_style']['style_scores']
            for label in STYLE_LABELS:
                if label not in style_scores:
                    logger.error(f"风格分数缺少字段: {label}")
                    return False
            
            logger.info(f"标注文件格式验证通过: {annotation_file}")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"标注文件JSON格式错误: {e}")
            return False
        except Exception as e:
            logger.error(f"验证标注文件时出错: {e}")
            return False
    
    def validate_annotations_dir(self, annotations_dir: str):
        """
        验证标注目录中的所有标注文件
        
        Args:
            annotations_dir: 标注目录路径
        """
        annotations_dir = Path(annotations_dir)
        if not annotations_dir.exists():
            logger.error(f"标注目录不存在: {annotations_dir}")
            return
        
        # 获取所有标注文件
        annotation_files = list(annotations_dir.glob("*_annotation.json"))
        
        logger.info(f"开始验证 {len(annotation_files)} 个标注文件")
        
        valid_count = 0
        invalid_count = 0
        
        for annotation_file in annotation_files:
            if self.validate_annotation(annotation_file):
                valid_count += 1
            else:
                invalid_count += 1
        
        logger.info(f"验证完成: 有效文件 {valid_count} 个，无效文件 {invalid_count} 个")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="教师风格画像分析系统 - 数据收集和标注工具")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 创建单个标注文件命令
    create_parser = subparsers.add_parser("create-annotation", help="为单个视频创建标注文件")
    create_parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    
    # 批量创建标注文件命令
    batch_parser = subparsers.add_parser("batch-create", help="批量为目录中的视频创建标注文件")
    batch_parser.add_argument("--dir", type=str, required=True, help="视频目录路径")
    
    # 生成数据收集指南命令
    guide_parser = subparsers.add_parser("generate-guide", help="生成数据收集和标注指南")
    
    # 验证标注文件命令
    validate_parser = subparsers.add_parser("validate", help="验证标注文件格式")
    validate_parser.add_argument("--file", type=str, help="单个标注文件路径")
    validate_parser.add_argument("--dir", type=str, help="标注目录路径")
    
    args = parser.parse_args()
    
    tool = DataCollectionTool()
    
    if args.command == "create-annotation":
        tool.create_annotation_file(args.video)
    elif args.command == "batch-create":
        tool.batch_create_annotation_files(args.dir)
    elif args.command == "generate-guide":
        tool.generate_data_collection_guide()
    elif args.command == "validate":
        if args.file:
            tool.validate_annotation(args.file)
        elif args.dir:
            tool.validate_annotations_dir(args.dir)
        else:
            logger.error("请指定--file或--dir参数")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
