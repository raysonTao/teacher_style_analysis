"""音频特征提取模块，负责提取音频中的特征"""
import os
import sys
import tempfile
from typing import List, Dict, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
from collections import defaultdict

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import BASE_DIR, AUDIO_CONFIG, logger

# 全局导入whisper（避免作用域问题）
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper模块未安装，语音识别功能将不可用")

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
        """加载Whisper模型，支持自动下载和本地缓存"""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper模块不可用，跳过模型加载")
            self.whisper_model = None
            return

        try:
            logger.info("初始化Whisper模型...")

            # 设置Whisper缓存目录到项目内
            whisper_cache_dir = os.path.join(BASE_DIR, 'models', 'weights', 'whisper_cache')
            os.makedirs(whisper_cache_dir, exist_ok=True)

            # 设置环境变量，让whisper使用我们的缓存目录
            os.environ['XDG_CACHE_HOME'] = os.path.join(BASE_DIR, 'models', 'weights')

            # 方案1: 尝试从本地路径加载
            local_model_path = os.path.join(BASE_DIR, AUDIO_CONFIG['whisper_model_path'])

            if os.path.exists(local_model_path):
                logger.info(f"找到本地Whisper模型: {local_model_path}")
                logger.info(f"模型文件大小: {os.path.getsize(local_model_path) / (1024*1024):.2f} MB")
                try:
                    # 直接加载本地.pt文件
                    self.whisper_model = whisper.load_model(local_model_path, device="cpu")
                    logger.info(f"✓ Whisper模型加载成功（使用本地文件）")
                    return
                except Exception as e:
                    logger.warning(f"本地模型加载失败: {e}，尝试其他方式...")

            # 方案2: 使用模型名称加载（会自动下载到缓存）
            model_size = AUDIO_CONFIG['whisper_model_size']
            logger.info(f"尝试加载Whisper模型: {model_size}")
            logger.info(f"模型将缓存到: {whisper_cache_dir}")

            self.whisper_model = whisper.load_model(model_size, device="cpu", download_root=whisper_cache_dir)
            logger.info(f"✓ Whisper模型加载成功（模型: {model_size}）")

            # 保存模型信息以便下次快速加载
            model_info_path = os.path.join(whisper_cache_dir, f"{model_size}_info.txt")
            with open(model_info_path, 'w') as f:
                f.write(f"Model: {model_size}\n")
                f.write(f"Loaded at: {os.path.getctime(model_info_path)}\n")

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
            if WHISPER_AVAILABLE and self.whisper_model is not None:
                logger.info("使用Whisper模型进行语音识别（完整音频）...")
                logger.info(f"音频时长: {features.get('audio_duration', 0):.2f} 秒")
                try:
                    # 使用transcribe方法处理完整音频（自动分段）
                    result = whisper.transcribe(
                        self.whisper_model,
                        audio_path,
                        language="zh",
                        fp16=False,  # CPU不支持fp16
                        verbose=False  # 关闭详细输出
                    )

                    transcription = result["text"]
                    features["transcription"] = transcription
                    logger.info(f"✓ 语音识别成功！")
                    logger.info(f"  转录文本长度: {len(transcription)} 字")
                    logger.info(f"  识别片段数: {len(result.get('segments', []))} 段")
                    logger.info(f"  文本预览: {transcription[:150]}...")

                except Exception as e:
                    logger.error(f"Whisper语音识别失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                if not WHISPER_AVAILABLE:
                    logger.warning("Whisper模块不可用，跳过语音识别")
                else:
                    logger.warning("Whisper模型未加载，跳过语音识别")
            
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
            
            # 加载视频
            video = mp.VideoFileClip(video_path)
            
            # 创建临时音频文件
            temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
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
