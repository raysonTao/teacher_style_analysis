"""
TBU 数据集转换脚本
将 TBU 数据集转换为标注所需的格式
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TBUConverter:
    """TBU 数据集转换器"""

    def __init__(self, tbu_data_path: str):
        """
        初始化转换器

        Args:
            tbu_data_path: TBU 数据集根目录
        """
        self.tbu_path = Path(tbu_data_path)
        if not self.tbu_path.exists():
            raise ValueError(f"TBU 数据路径不存在: {tbu_data_path}")

        logger.info(f"TBU 数据路径: {self.tbu_path}")

    def convert_to_annotation_format(self,
                                    output_path: str,
                                    max_samples: int = None,
                                    include_images: bool = True) -> List[Dict]:
        """
        转换为标注格式

        Args:
            output_path: 输出路径
            max_samples: 最大样本数（None表示全部）
            include_images: 是否包含图片路径

        Returns:
            转换后的样本列表
        """
        logger.info("开始转换 TBU 数据集...")

        # 查找所有视频/数据
        samples = self._find_all_samples()

        logger.info(f"找到 {len(samples)} 个样本")

        # 限制样本数
        if max_samples:
            samples = samples[:max_samples]
            logger.info(f"限制为 {max_samples} 个样本")

        # 转换每个样本
        converted_samples = []
        for idx, sample in enumerate(samples):
            try:
                converted = self._convert_single_sample(
                    sample,
                    sample_id=f'tbu_{idx:05d}',
                    include_images=include_images
                )
                converted_samples.append(converted)

                if (idx + 1) % 100 == 0:
                    logger.info(f"已转换 {idx + 1}/{len(samples)} 个样本")

            except Exception as e:
                logger.error(f"转换样本 {idx} 失败: {str(e)}")
                continue

        # 保存结果
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_samples, f, ensure_ascii=False, indent=2)

        logger.info(f"转换完成！共 {len(converted_samples)} 个样本")
        logger.info(f"输出文件: {output_file}")

        return converted_samples

    def _find_all_samples(self) -> List[Dict]:
        """查找所有样本"""
        samples = []

        # TBU 数据集结构可能是：
        # - annotations/ (标注文件)
        # - videos/ (视频文件)
        # - frames/ (提取的帧)

        # 方法1: 查找标注文件
        annotation_files = list(self.tbu_path.rglob("*.json"))
        if annotation_files:
            logger.info(f"找到 {len(annotation_files)} 个标注文件")
            for ann_file in annotation_files:
                try:
                    with open(ann_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 根据实际 TBU 格式解析
                    if isinstance(data, list):
                        # 列表格式
                        for item in data:
                            samples.append({
                                'annotation_file': str(ann_file),
                                'data': item
                            })
                    elif isinstance(data, dict):
                        # 字典格式
                        samples.append({
                            'annotation_file': str(ann_file),
                            'data': data
                        })

                except Exception as e:
                    logger.warning(f"读取标注文件失败 {ann_file}: {str(e)}")

        # 方法2: 如果没有标注文件，查找视频文件
        if not samples:
            video_files = list(self.tbu_path.rglob("*.mp4"))
            video_files.extend(self.tbu_path.rglob("*.avi"))
            logger.info(f"找到 {len(video_files)} 个视频文件")

            for video_file in video_files:
                samples.append({
                    'video_file': str(video_file),
                    'data': None
                })

        return samples

    def _convert_single_sample(self,
                              sample: Dict,
                              sample_id: str,
                              include_images: bool = True) -> Dict:
        """转换单个样本"""

        # 提取行为序列
        behavior_sequence = self._extract_behaviors(sample)

        # 提取行为时长
        behavior_durations = self._extract_durations(sample)

        # 查找关键帧
        video_frames = []
        if include_images:
            video_frames = self._find_video_frames(sample)

        # 提取文本（如果有）
        lecture_text = self._extract_text(sample)

        # 提取元数据
        metadata = self._extract_metadata(sample)

        return {
            'sample_id': sample_id,
            'behavior_sequence': behavior_sequence,
            'behavior_durations': behavior_durations,
            'video_frames': video_frames,
            'lecture_text': lecture_text,
            'audio_transcript': None,  # TBU 可能没有语音
            'metadata': metadata,
            'source': 'TBU'
        }

    def _extract_behaviors(self, sample: Dict) -> List[str]:
        """提取行为序列"""
        data = sample.get('data', {})

        # TBU 可能的字段名
        possible_keys = ['behaviors', 'actions', 'annotations', 'labels', 'categories']

        for key in possible_keys:
            if key in data:
                behaviors = data[key]
                if isinstance(behaviors, list):
                    return [str(b) for b in behaviors]
                elif isinstance(behaviors, str):
                    return [behaviors]

        # 如果没有找到，返回空列表
        return []

    def _extract_durations(self, sample: Dict) -> Dict[str, float]:
        """提取行为时长"""
        data = sample.get('data', {})

        durations = {}

        # 尝试从时间戳计算
        if 'timestamps' in data or 'temporal_segments' in data:
            # 解析时间段
            pass

        # 如果没有时长信息，根据行为序列估算
        behaviors = self._extract_behaviors(sample)
        if behaviors and not durations:
            # 假设每个行为平均5秒
            from collections import Counter
            behavior_counts = Counter(behaviors)
            for behavior, count in behavior_counts.items():
                durations[behavior] = count * 5.0

        return durations

    def _find_video_frames(self, sample: Dict) -> List[str]:
        """查找视频关键帧"""
        frames = []

        # 方法1: 从标注文件中查找
        if 'annotation_file' in sample:
            ann_file = Path(sample['annotation_file'])
            # 查找同名文件夹下的帧
            frame_dir = ann_file.parent / ann_file.stem
            if frame_dir.exists():
                frame_files = sorted(frame_dir.glob("*.jpg"))
                frame_files.extend(sorted(frame_dir.glob("*.png")))
                frames = [str(f) for f in frame_files[:10]]  # 最多10帧

        # 方法2: 从视频文件推断
        if not frames and 'video_file' in sample:
            video_file = Path(sample['video_file'])
            # 查找对应的帧目录
            frame_dir = video_file.parent / 'frames' / video_file.stem
            if frame_dir.exists():
                frame_files = sorted(frame_dir.glob("*.jpg"))
                frames = [str(f) for f in frame_files[:10]]

        return frames

    def _extract_text(self, sample: Dict) -> str:
        """提取讲课文本"""
        data = sample.get('data', {})

        # 可能的文本字段
        text_keys = ['text', 'transcript', 'lecture', 'content', 'description']

        for key in text_keys:
            if key in data and data[key]:
                return str(data[key])

        return ""

    def _extract_metadata(self, sample: Dict) -> Dict:
        """提取元数据"""
        data = sample.get('data', {})

        metadata = {}

        # 提取常见字段
        common_keys = ['discipline', 'subject', 'grade', 'duration',
                      'teacher_id', 'school', 'date']

        for key in common_keys:
            if key in data:
                metadata[key] = data[key]

        return metadata


def convert_tbu_dataset(tbu_path: str,
                       output_path: str,
                       max_samples: int = None):
    """
    转换 TBU 数据集（便捷函数）

    Args:
        tbu_path: TBU 数据集路径
        output_path: 输出路径
        max_samples: 最大样本数
    """
    converter = TBUConverter(tbu_path)
    samples = converter.convert_to_annotation_format(
        output_path=output_path,
        max_samples=max_samples,
        include_images=True
    )

    # 打印统计
    print(f"\n转换统计:")
    print(f"总样本数: {len(samples)}")

    # 行为统计
    from collections import Counter
    all_behaviors = []
    for s in samples:
        all_behaviors.extend(s['behavior_sequence'])

    if all_behaviors:
        behavior_counts = Counter(all_behaviors)
        print(f"\n行为类型分布:")
        for behavior, count in behavior_counts.most_common(10):
            print(f"  {behavior}: {count}")

    print(f"\n输出文件: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='转换 TBU 数据集')
    parser.add_argument('--tbu_path', type=str, required=True,
                       help='TBU 数据集路径')
    parser.add_argument('--output', type=str,
                       default='data/tbu/tbu_for_annotation.json',
                       help='输出文件路径')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数（用于测试）')

    args = parser.parse_args()

    convert_tbu_dataset(
        tbu_path=args.tbu_path,
        output_path=args.output,
        max_samples=args.max_samples
    )
