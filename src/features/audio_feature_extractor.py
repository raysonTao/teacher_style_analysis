"""音频特征提取模块，负责提取音频中的特征"""
import os
import sys
import tempfile
from typing import List, Dict, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
from collections import defaultdict
import torch

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import BASE_DIR, AUDIO_CONFIG, logger

class AudioFeatureExtractor:
    """音频特征提取类，用于提取音频中的特征"""

    # 单例模式实现
    _instance = None

    def __new__(cls):
        """控制实例创建，确保只创建一个实例"""
        if cls._instance is None:
            cls._instance = super(AudioFeatureExtractor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化音频特征提取器"""
        # 确保模型只加载一次
        if not hasattr(self, 'whisper_model'):
            self.whisper_model = None
            self._load_model()

    def _load_model(self):
        """加载Whisper模型，强制使用float32精度"""
        try:
            import whisper
            logger.info("初始化Whisper模型...")

            # 使用绝对路径加载Whisper模型
            whisper_path = os.path.join(BASE_DIR, AUDIO_CONFIG['whisper_model_path'])
            logger.debug(f"Whisper模型路径: {whisper_path}")
            logger.debug(f"文件是否存在: {os.path.exists(whisper_path)}")

            if os.path.exists(whisper_path):
                logger.debug(f"文件大小: {os.path.getsize(whisper_path)} 字节")
                # 使用本地模型文件，显式指定CPU设备
                self.whisper_model = whisper.load_model(whisper_path, device="cpu")
                logger.info(f"Whisper模型加载成功，使用本地文件: {whisper_path}")
            else:
                # 回退到默认加载方式，显式指定CPU设备
                logger.warning("本地模型文件不存在，尝试从网络加载...")
                self.whisper_model = whisper.load_model(AUDIO_CONFIG['whisper_model_size'], device="cpu")
                logger.info(f"Whisper模型加载成功，大小: {AUDIO_CONFIG['whisper_model_size']}")

            logger.debug(f"模型类型: {type(self.whisper_model)}")
            logger.debug(f"模型设备: {self.whisper_model.device}")

            # 强制转换为float32精度（CPU不支持half精度）
            logger.info("将模型转换为float32精度...")
            self.whisper_model = self.whisper_model.float()

            # 验证所有参数是否为float32
            for name, param in self.whisper_model.named_parameters():
                if param.dtype != torch.float32:
                    logger.warning(f"参数 {name} 仍为 {param.dtype}，强制转换...")
                    param.data = param.data.float()

            logger.info(f"模型设备: {self.whisper_model.device}")
            logger.info("模型数据类型已验证: float32")

        except Exception as e:
            logger.error(f"Whisper模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.whisper_model = None
    
    def extract_features(self, audio_path: str) -> Dict:
        """
        提取音频特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频特征字典
        """
        features = {
            "volume": [],
            "pitch": [],
            "transcription": "",
            "sentiment": {
                "score": 0.0,
                "label": "neutral"
            },
            "voice_activity": [],
            "audio_duration": 0.0
        }
        
        try:
            # 加载音频文件
            y, sr = librosa.load(audio_path, sr=None)
            features["audio_duration"] = len(y) / sr
            
            # 计算音量
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
            features["volume"] = rms[0].tolist()
            
            # 计算语调
            try:
                pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, n_fft=2048, hop_length=512)
                
                # 获取每个帧的主要频率
                for i in range(pitches.shape[1]):
                    index = magnitudes[:, i].argmax()
                    pitch = pitches[index, i]
                    if pitch > 0:
                        features["pitch"].append(float(pitch))
                    else:
                        features["pitch"].append(0.0)
            except Exception as e:
                logger.warning(f"语调计算失败: {e}")
                features["pitch"] = [0.0] * len(features["volume"])
            
            # 语音活动检测
            energy = np.array(features["volume"])
            threshold = np.mean(energy) * 0.5
            voice_activity = [1 if e > threshold else 0 for e in energy]
            features["voice_activity"] = voice_activity
            
            # 语音识别
            if self.whisper_model is not None:
                logger.info("使用Whisper模型进行语音识别...")
                try:
                    # 使用高级transcribe API（自动处理dtype/device）
                    result = self.whisper_model.transcribe(
                        audio_path,
                        language="zh",
                        fp16=False  # 关键：禁用FP16，强制使用float32
                    )

                    transcription = result['text']
                    features["transcription"] = transcription
                    logger.info(f"语音识别结果: {transcription}")

                except Exception as e:
                    logger.error(f"Whisper语音识别失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 模拟情绪分数（实际项目中可以使用专业的情绪分析模型）
            avg_volume = np.mean(features["volume"])
            if avg_volume > 0.1:
                sentiment_score = 0.7
                sentiment_label = "positive"
            elif avg_volume < 0.05:
                sentiment_score = 0.3
                sentiment_label = "negative"
            else:
                sentiment_score = 0.5
                sentiment_label = "neutral"
            
            features["sentiment"] = {
                "score": sentiment_score,
                "label": sentiment_label
            }
            
        except Exception as e:
            logger.error(f"音频特征提取失败: {e}")
            import traceback
            traceback.print_exc()
        
        return features
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """
        从视频文件中提取音频
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            临时音频文件路径
        """
        try:
            import moviepy.editor as mp
            
            # 创建当前目录下的tmp文件夹
            tmp_dir = os.path.join(os.getcwd(), "tmp")
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
                logger.info(f"创建临时目录: {tmp_dir}")
            
            # 加载视频
            video = mp.VideoFileClip(video_path)
            
            # 创建临时音频文件在当前目录的tmp文件夹下
            temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tmp_dir)
            temp_audio_path = temp_audio_file.name
            temp_audio_file.close()
            
            # 提取音频并保存
            video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
            
            return temp_audio_path
            
        except Exception as e:
            logger.error(f"从视频提取音频失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.whisper_model is not None
