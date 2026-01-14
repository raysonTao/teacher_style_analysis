"""
视觉大模型教学风格标注器
使用 Claude API 进行教学风格标注
"""

import os
import sys
import json
import base64
import time
from pathlib import Path
from typing import List, Dict, Optional
import anthropic
from tqdm import tqdm
import logging

# 导入FeatureEncoder用于特征解读
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features.feature_encoder import FeatureEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLMStyleAnnotator:
    """使用视觉大模型进行教学风格标注"""

    # 7种教学风格定义
    TEACHING_STYLES = {
        '理论讲授型': '以系统讲解理论知识为主，注重概念阐述和知识传授',
        '启发引导型': '通过提问和引导促进学生思考，培养分析能力',
        '互动导向型': '强调师生互动和课堂参与，鼓励讨论交流',
        '逻辑推导型': '注重逻辑严密的推理过程，强调因果关系和论证',
        '题目驱动型': '以解题为核心组织教学，通过练习巩固知识',
        '情感表达型': '善用肢体语言和情感投入，感染和激励学生',
        '耐心细致型': '讲解细致入微，关注每个细节，重复强调重点'
    }

    def __init__(self,
                 api_key: str,
                 base_url: str = "https://aidev.deyecloud.com/api",
                 model: str = "claude-opus-4-5-20251101",
                 max_retries: int = 3,
                 retry_delay: int = 5):
        """
        初始化标注器

        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 使用的模型
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 创建特征编码器实例（用于生成特征解读）
        self.feature_encoder = FeatureEncoder()

        logger.info(f"VLM标注器初始化完成")
        logger.info(f"使用模型: {model}")
        logger.info(f"API地址: {base_url}")

    def annotate_single_sample(self,
                               video_frames: List[str] = None,
                               behavior_sequence: List[str] = None,
                               behavior_durations: Dict[str, float] = None,
                               audio_transcript: str = None,
                               lecture_text: str = None,
                               metadata: Dict = None,
                               raw_features: Dict = None) -> Dict:
        """
        标注单个样本的教学风格

        Args:
            video_frames: 视频关键帧图片路径列表
            behavior_sequence: 行为序列
            behavior_durations: 行为持续时间
            audio_transcript: 语音转文本
            lecture_text: 讲课文本
            metadata: 其他元数据
            raw_features: 原始提取的特征（用于生成量化解读）

        Returns:
            标注结果
        """
        for attempt in range(self.max_retries):
            try:
                # 构建提示词
                prompt = self._build_prompt(
                    behavior_sequence=behavior_sequence,
                    behavior_durations=behavior_durations,
                    audio_transcript=audio_transcript,
                    lecture_text=lecture_text,
                    metadata=metadata,
                    raw_features=raw_features
                )

                # 准备消息内容
                content = []

                # 添加图片（如果有）
                if video_frames:
                    for frame_path in video_frames[:10]:  # 最多10帧
                        if os.path.exists(frame_path):
                            with open(frame_path, 'rb') as f:
                                image_data = base64.b64encode(f.read()).decode('utf-8')
                                content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_data
                                    }
                                })

                # 添加文本提示
                content.append({
                    "type": "text",
                    "text": prompt
                })

                # 调用API
                logger.debug(f"调用API，尝试次数: {attempt + 1}/{self.max_retries}")
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=0.3,  # 降低温度以获得更稳定的输出
                    messages=[{
                        "role": "user",
                        "content": content
                    }]
                )

                # 解析结果
                result_text = response.content[0].text
                result = self._parse_response(result_text)

                logger.debug(f"标注成功: {result['style']} (置信度: {result['confidence']})")
                return result

            except Exception as e:
                logger.error(f"标注失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")

                if attempt < self.max_retries - 1:
                    logger.info(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("达到最大重试次数，返回错误结果")
                    return {
                        'style': None,
                        'confidence': 0.0,
                        'reasoning': f'标注失败: {str(e)}',
                        'error': True
                    }

    def _build_prompt(self,
                     behavior_sequence: List[str] = None,
                     behavior_durations: Dict[str, float] = None,
                     audio_transcript: str = None,
                     lecture_text: str = None,
                     metadata: Dict = None,
                     raw_features: Dict = None) -> str:
        """构建提示词"""

        prompt = """# 教学风格分类任务

你是一位资深的教育专家，请分析提供的教学数据，判断教师的主要教学风格。

## 教学风格类别定义：

"""

        # 添加风格定义
        for idx, (style, description) in enumerate(self.TEACHING_STYLES.items(), 1):
            prompt += f"{idx}. **{style}**: {description}\n"

        prompt += "\n## 观察到的数据：\n\n"

        # 添加量化特征解读（如果有raw_features）
        if raw_features:
            try:
                # 使用FeatureEncoder编码并生成人类可读解读
                encoded = self.feature_encoder.encode_all(raw_features)
                feature_interpretation = self.feature_encoder.get_feature_interpretation(encoded)
                prompt += feature_interpretation + "\n\n"
            except Exception as e:
                logger.warning(f"生成特征解读失败: {e}")

        # 添加行为序列
        if behavior_sequence:
            prompt += f"### 教师行为序列：\n"
            prompt += f"{', '.join(behavior_sequence)}\n\n"

            # 添加行为时长统计
            if behavior_durations:
                total_duration = sum(behavior_durations.values())
                prompt += f"### 行为时长统计（总时长: {total_duration:.1f}秒）：\n"
                for behavior, duration in sorted(behavior_durations.items(),
                                                key=lambda x: x[1], reverse=True):
                    ratio = duration / total_duration * 100
                    prompt += f"- {behavior}: {duration:.1f}秒 ({ratio:.1f}%)\n"
                prompt += "\n"

        # 添加讲课文本
        if lecture_text:
            prompt += f"### 教师讲课内容：\n{lecture_text[:1000]}"
            if len(lecture_text) > 1000:
                prompt += "...\n\n"
            else:
                prompt += "\n\n"

        # 添加语音转文本
        if audio_transcript:
            prompt += f"### 语音转文本：\n{audio_transcript[:800]}"
            if len(audio_transcript) > 800:
                prompt += "...\n\n"
            else:
                prompt += "\n\n"

        # 添加元数据
        if metadata:
            prompt += f"### 其他信息：\n"
            if 'discipline' in metadata:
                prompt += f"- 学科: {metadata['discipline']}\n"
            if 'grade' in metadata:
                prompt += f"- 年级: {metadata['grade']}\n"
            if 'duration' in metadata:
                prompt += f"- 课程时长: {metadata['duration']}分钟\n"
            prompt += "\n"

        prompt += """## 分析任务：

请综合分析上述所有信息，完成以下任务：

1. **识别关键特征**: 分析教师的教学行为模式、语言表达、互动方式等
2. **判断主要风格**: 从7种风格中选择**最匹配**的1种
3. **评估置信度**: 给出0-1之间的置信度分数
4. **说明理由**: 用2-3句话解释为什么是这种风格（引用具体证据）

## 输出格式：

请严格按照以下JSON格式输出（不要包含```json标记）：

{
    "style": "教学风格名称（必须是上述7种之一）",
    "confidence": 0.85,
    "reasoning": "判断理由，包含具体证据...",
    "secondary_style": "次要风格（如果有）",
    "key_features": ["特征1", "特征2", "特征3"]
}

现在请开始分析并输出JSON结果：
"""

        return prompt

    def _parse_response(self, response_text: str) -> Dict:
        """解析API响应"""
        import re

        # 尝试提取JSON
        # 方法1: 查找```json...```
        json_match = re.search(r'```json\s*(.*?)\s*```',
                              response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 方法2: 查找 {...}
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text

        try:
            result = json.loads(json_str)

            # 验证必需字段
            if 'style' not in result:
                raise ValueError("缺少 'style' 字段")
            if 'confidence' not in result:
                result['confidence'] = 0.0
            if 'reasoning' not in result:
                result['reasoning'] = "未提供理由"

            # 确保 style 是有效的
            if result['style'] not in self.TEACHING_STYLES:
                raise ValueError(f"无效的风格: {result['style']}")

            # 确保 confidence 在 0-1 之间
            result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))

            return result

        except Exception as e:
            logger.error(f"解析响应失败: {str(e)}")
            logger.debug(f"原始响应: {response_text[:500]}")

            # 返回错误结果
            return {
                'style': None,
                'confidence': 0.0,
                'reasoning': f'解析失败: {str(e)}',
                'error': True,
                'raw_response': response_text[:200]
            }

    def batch_annotate(self,
                      samples: List[Dict],
                      output_path: str,
                      resume_from: int = 0,
                      save_interval: int = 10) -> List[Dict]:
        """
        批量标注

        Args:
            samples: 样本列表
            output_path: 输出路径
            resume_from: 从第几个样本开始（用于断点续传）
            save_interval: 保存间隔

        Returns:
            标注结果列表
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 尝试加载已有结果
        annotated_samples = []
        if output_file.exists() and resume_from > 0:
            logger.info(f"加载已有结果: {output_file}")
            with open(output_file, 'r', encoding='utf-8') as f:
                annotated_samples = json.load(f)
            logger.info(f"已加载 {len(annotated_samples)} 个标注结果")

        # 开始标注
        logger.info(f"开始批量标注，共 {len(samples)} 个样本")
        logger.info(f"从第 {resume_from} 个样本开始")

        for idx in tqdm(range(resume_from, len(samples)), desc="标注进度"):
            sample = samples[idx]

            try:
                # 标注单个样本
                result = self.annotate_single_sample(
                    video_frames=sample.get('video_frames'),
                    behavior_sequence=sample.get('behavior_sequence'),
                    behavior_durations=sample.get('behavior_durations'),
                    audio_transcript=sample.get('audio_transcript'),
                    lecture_text=sample.get('lecture_text'),
                    metadata=sample.get('metadata'),
                    raw_features=sample.get('raw_features')  # 传递原始特征
                )

                # 添加样本ID和原始数据
                annotated_sample = {
                    'sample_id': sample.get('sample_id', f'sample_{idx:04d}'),
                    'annotation': result,
                    'source_data': sample
                }

                annotated_samples.append(annotated_sample)

                # 定期保存
                if (idx + 1) % save_interval == 0:
                    self._save_results(annotated_samples, output_file)
                    logger.info(f"已保存 {len(annotated_samples)} 个结果")

                # 短暂延迟，避免API限流
                time.sleep(0.5)

            except KeyboardInterrupt:
                logger.warning("用户中断，保存当前结果...")
                self._save_results(annotated_samples, output_file)
                raise

            except Exception as e:
                logger.error(f"处理样本 {idx} 时出错: {str(e)}")
                # 继续处理下一个样本
                continue

        # 最终保存
        self._save_results(annotated_samples, output_file)
        logger.info(f"批量标注完成！共标注 {len(annotated_samples)} 个样本")
        logger.info(f"结果已保存至: {output_file}")

        return annotated_samples

    def _save_results(self, results: List[Dict], output_path: Path):
        """保存结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def get_annotation_statistics(self, annotated_samples: List[Dict]) -> Dict:
        """获取标注统计信息"""
        from collections import Counter

        styles = [s['annotation']['style'] for s in annotated_samples]
        confidences = [s['annotation']['confidence'] for s in annotated_samples]

        style_counts = Counter(styles)

        stats = {
            'total_samples': len(annotated_samples),
            'style_distribution': dict(style_counts),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'high_confidence_count': sum(1 for c in confidences if c > 0.8),
            'low_confidence_count': sum(1 for c in confidences if c < 0.5),
        }

        return stats


def test_annotator():
    """测试标注器"""

    # 从环境变量读取配置
    api_key = os.environ.get('ANTHROPIC_AUTH_TOKEN')
    base_url = os.environ.get('ANTHROPIC_BASE_URL', 'https://aidev.deyecloud.com/api')

    if not api_key:
        print("错误: 请设置 ANTHROPIC_AUTH_TOKEN 环境变量")
        return

    # 创建标注器
    annotator = VLMStyleAnnotator(
        api_key=api_key,
        base_url=base_url
    )

    # 测试样本
    test_sample = {
        'behavior_sequence': ['Writing', 'Explaining', 'Pointing', 'Writing', 'Explaining'],
        'behavior_durations': {
            'Writing': 120.5,
            'Explaining': 280.3,
            'Pointing': 45.2
        },
        'lecture_text': '今天我们来学习一元二次方程的求解方法。首先，我们回顾一下什么是一元二次方程...',
        'metadata': {
            'discipline': '数学',
            'grade': '初中',
            'duration': 40
        }
    }

    print("开始测试标注器...")
    result = annotator.annotate_single_sample(**test_sample)

    print("\n标注结果:")
    print(f"预测风格: {result['style']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"理由: {result['reasoning']}")
    if 'key_features' in result:
        print(f"关键特征: {', '.join(result['key_features'])}")

    print("\n测试成功！")


if __name__ == '__main__':
    test_annotator()
