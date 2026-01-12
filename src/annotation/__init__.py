"""
教学风格标注模块
支持使用视觉大模型 (VLM) 进行自动标注
"""

from .vlm_annotator import VLMStyleAnnotator

__all__ = [
    'VLMStyleAnnotator'
]
