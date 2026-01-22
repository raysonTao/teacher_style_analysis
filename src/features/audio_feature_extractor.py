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
            self.wav2vec2_model = None
            self.wav2vec2_emotion_model = None
            self.wav2vec2_processor = None
            self.wav2vec2_device = None
            self._load_model()

    def _load_model(self):
        """加载模型（Whisper与Wav2Vec2），支持自动下载和本地缓存"""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper模块不可用，跳过模型加载")
            self.whisper_model = None
            # Whisper不可用也不影响Wav2Vec2

        self._load_whisper_model()
        self._load_wav2vec2_models()

    def _load_whisper_model(self):
        """加载Whisper模型"""
        if not WHISPER_AVAILABLE:
            return

        try:
            logger.info("初始化Whisper模型...")

            # 检测GPU可用性
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {device}")

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
                    # 直接加载本地.pt文件，自动检测设备
                    self.whisper_model = whisper.load_model(local_model_path, device=device)
                    logger.info(f"✓ Whisper模型加载成功（使用本地文件，设备: {device}）")
                    return
                except Exception as e:
                    logger.warning(f"本地模型加载失败: {e}，尝试其他方式...")

            # 方案2: 使用模型名称加载（会自动下载到缓存）
            model_size = AUDIO_CONFIG['whisper_model_size']
            logger.info(f"尝试加载Whisper模型: {model_size}")
            logger.info(f"模型将缓存到: {whisper_cache_dir}")

            self.whisper_model = whisper.load_model(model_size, device=device, download_root=whisper_cache_dir)
            logger.info(f"✓ Whisper模型加载成功（模型: {model_size}，设备: {device}）")

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

    def _load_wav2vec2_models(self):
        """加载Wav2Vec2模型（声学表征 + 情感识别）"""
        try:
            from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification
            import torch

            logger.info("初始化Wav2Vec2模型...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.wav2vec2_device = device

            cache_dir = AUDIO_CONFIG.get('wav2vec2_cache_dir')
            base_name = AUDIO_CONFIG.get('wav2vec2_model_name')
            emotion_name = AUDIO_CONFIG.get('wav2vec2_emotion_model_name')

            self.wav2vec2_processor = AutoFeatureExtractor.from_pretrained(
                base_name,
                cache_dir=cache_dir
            )
            self.wav2vec2_model = AutoModel.from_pretrained(
                base_name,
                cache_dir=cache_dir
            ).to(device).eval()

            self.wav2vec2_emotion_model = AutoModelForAudioClassification.from_pretrained(
                emotion_name,
                cache_dir=cache_dir
            ).to(device).eval()

            logger.info(f"Wav2Vec2表征模型加载成功: {base_name}")
            logger.info(f"Wav2Vec2情感模型加载成功: {emotion_name}")

        except Exception as e:
            logger.warning(f"Wav2Vec2模型加载失败: {e}")
            self.wav2vec2_model = None
            self.wav2vec2_emotion_model = None
            self.wav2vec2_processor = None
            self.wav2vec2_device = None
    
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
            "wav2vec2_embedding": [],
            "wav2vec2_embedding_dim": 0,
            "emotion_label": "neutral",
            "emotion_logits": [],
            "sentiment": {
                "score": 0.0,
                "label": "neutral"
            },
            "sentiment_score": 0.0,
            "voice_activity": [],
            "audio_duration": 0.0,
            "speech_rate": 0.0,
            "pitch_variation": 0.0,
            "volume_level": 0.0,
            "silence_ratio": 1.0,
            "emotion_scores": {},
            "volume_statistics": {},
            "pitch_statistics": {},
            "voice_activity_ratio": 0.0,
            "audio_representation": {},
            "error": None
        }
        
        try:
            if not audio_path or not os.path.exists(audio_path):
                features["error"] = f"音频文件不存在: {audio_path}"
                logger.error(features["error"])
                return features

            # 加载音频文件
            y, sr = librosa.load(audio_path, sr=None)
            features["audio_duration"] = len(y) / sr
            
            # 计算音量
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
            features["volume"] = rms[0].tolist()
            if features["volume"]:
                volume_array = np.array(features["volume"], dtype=float)
                features["volume_level"] = float(np.mean(volume_array))
                features["volume_statistics"] = {
                    "mean": float(np.mean(volume_array)),
                    "std": float(np.std(volume_array)),
                    "min": float(np.min(volume_array)),
                    "max": float(np.max(volume_array))
                }
            
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

            if features["pitch"]:
                valid_pitch = [p for p in features["pitch"] if p > 0]
                if valid_pitch:
                    pitch_array = np.array(valid_pitch, dtype=float)
                    pitch_std = float(np.std(pitch_array))
                    features["pitch_variation"] = float(np.tanh(pitch_std / 50.0))
                    features["pitch_statistics"] = {
                        "mean": float(np.mean(pitch_array)),
                        "std": pitch_std,
                        "min": float(np.min(pitch_array)),
                        "max": float(np.max(pitch_array))
                    }
            
            # 语音活动检测
            energy = np.array(features["volume"])
            if energy.size > 0:
                intervals = librosa.effects.split(y, top_db=25)
                frame_times = librosa.frames_to_time(
                    np.arange(len(energy)), sr=sr, hop_length=512
                )
                active_mask = np.zeros_like(frame_times, dtype=bool)
                for start, end in intervals:
                    start_time = start / sr
                    end_time = end / sr
                    active_mask |= (frame_times >= start_time) & (frame_times <= end_time)
                voice_activity = active_mask.astype(int).tolist()
                features["voice_activity"] = voice_activity
                features["voice_activity_ratio"] = float(np.mean(voice_activity))
                features["silence_ratio"] = float(1.0 - features["voice_activity_ratio"])
            else:
                features["voice_activity"] = []
                features["voice_activity_ratio"] = 0.0
                features["silence_ratio"] = 1.0
            
            # Wav2Vec2深度声学表征与情感识别
            wav2vec2_inputs = self._prepare_wav2vec2_inputs(y, sr)
            if wav2vec2_inputs is not None:
                wav2vec2_outputs = self._extract_wav2vec2_representation(wav2vec2_inputs)
                features.update(wav2vec2_outputs.get("representation", {}))
                emotion_outputs = self._extract_wav2vec2_emotion(wav2vec2_inputs)
                features.update(emotion_outputs)
            else:
                features["error"] = "Wav2Vec2输入准备失败"

            # 语音识别
            if WHISPER_AVAILABLE and self.whisper_model is not None:
                logger.info("使用Whisper模型进行语音识别（完整音频）...")
                logger.info(f"音频时长: {features.get('audio_duration', 0):.2f} 秒")
                try:
                    # 检测设备类型，GPU支持fp16加速
                    import torch
                    use_fp16 = torch.cuda.is_available()
                    device_type = "CUDA" if use_fp16 else "CPU"
                    logger.info(f"使用设备: {device_type}, FP16: {use_fp16}")

                    # 使用transcribe方法处理完整音频（自动分段）
                    result = whisper.transcribe(
                        self.whisper_model,
                        audio_path,
                        language="zh",
                        fp16=use_fp16,  # GPU支持fp16，CPU不支持
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

            # 语速估计（字符/分钟）
            if features["transcription"] and features["audio_duration"] > 0:
                char_count = len(features["transcription"])
                minutes = features["audio_duration"] / 60.0
                features["speech_rate"] = float(char_count / minutes) if minutes > 0 else 0.0
            
            # 基于Wav2Vec2情感结果估计情感强度（不使用韵律降级）
            if features["emotion_scores"]:
                sentiment_score, sentiment_label = self._derive_sentiment(features["emotion_scores"])
                features["sentiment"] = {
                    "score": sentiment_score,
                    "label": sentiment_label
                }
                features["sentiment_score"] = sentiment_score
            else:
                features["emotion_label"] = "unknown"
                features["sentiment"] = {
                    "score": None,
                    "label": "unknown"
                }
                features["sentiment_score"] = None
                if features["error"] is None:
                    features["error"] = "Wav2Vec2情感模型未产生结果"
            
        except Exception as e:
            logger.error(f"音频特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            features["error"] = str(e)
        
        return features

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _prepare_wav2vec2_inputs(self, y: np.ndarray, sr: int):
        """准备Wav2Vec2输入，处理采样率和格式"""
        if self.wav2vec2_processor is None:
            return None

        try:
            target_sr = AUDIO_CONFIG.get('sample_rate', 16000)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            if y.ndim > 1:
                y = np.mean(y, axis=0)

            inputs = self.wav2vec2_processor(
                y,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True
            )
            return inputs
        except Exception as e:
            logger.warning(f"Wav2Vec2输入准备失败: {e}")
            return None

    def _extract_wav2vec2_representation(self, inputs) -> Dict:
        """提取Wav2Vec2深度声学表征"""
        if self.wav2vec2_model is None:
            return {}

        try:
            import torch
            device = self.wav2vec2_device or torch.device('cpu')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.wav2vec2_model(**inputs)

            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
            embedding = pooled.tolist()

            return {
                "representation": {
                    "wav2vec2_embedding": embedding,
                    "wav2vec2_embedding_dim": len(embedding),
                    "audio_representation": {
                        "wav2vec2_mean": embedding
                    }
                }
            }
        except Exception as e:
            logger.warning(f"Wav2Vec2表征提取失败: {e}")
            return {}

    def _extract_wav2vec2_emotion(self, inputs) -> Dict:
        """提取Wav2Vec2情感识别结果"""
        if self.wav2vec2_emotion_model is None:
            return {}

        try:
            import torch
            import torch.nn.functional as F

            device = self.wav2vec2_device or torch.device('cpu')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.wav2vec2_emotion_model(**inputs)

            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            id2label = getattr(self.wav2vec2_emotion_model.config, "id2label", None) or {}
            label_map = AUDIO_CONFIG.get('emotion_label_map', {})

            scores = {}
            for idx, score in enumerate(probs.tolist()):
                raw_label = id2label.get(idx, f"emotion_{idx}")
                mapped_label = label_map.get(raw_label, raw_label)
                scores[mapped_label] = float(score)

            if scores:
                emotion_label = max(scores.items(), key=lambda x: x[1])[0]
            else:
                emotion_label = "neutral"

            return {
                "emotion_scores": scores,
                "emotion_label": emotion_label,
                "emotion_logits": logits.squeeze(0).cpu().tolist()
            }
        except Exception as e:
            logger.warning(f"Wav2Vec2情感识别失败: {e}")
            return {}

    def _derive_sentiment(self, emotion_scores: Dict) -> Tuple[float, str]:
        """将情感分布映射为极性分数"""
        positive_labels = {"happy", "surprise", "excited", "joy"}
        negative_labels = {"sad", "angry", "fear", "disgust"}

        positive = sum(score for label, score in emotion_scores.items() if label in positive_labels)
        negative = sum(score for label, score in emotion_scores.items() if label in negative_labels)

        sentiment_score = self._clip01(0.5 + 0.5 * (positive - negative))
        if sentiment_score > 0.55:
            sentiment_label = "positive"
        elif sentiment_score < 0.45:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        return sentiment_score, sentiment_label
    
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
        return self.whisper_model is not None or self.wav2vec2_model is not None
