"""特征提取模块，负责从视频、音频和文本中提取多模态特征"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    MODEL_CONFIG, SYSTEM_CONFIG, 
    VIDEO_DIR, AUDIO_DIR, TEXT_DIR, FEATURES_DIR
)


class FeatureExtractor:
    """多模态特征提取器"""
    
    def __init__(self):
        self._init_models()
    
    def _init_models(self):
        """初始化各种特征提取模型"""
        try:
            # 初始化特征提取模型
            print("初始化特征提取模型...")
            
            # YOLOv8动作检测模型
            from ultralytics import YOLO
            self.yolo_model = YOLO(MODEL_CONFIG['yolo_model_path'])
            
            # OpenPose姿态估计模型
            try:
                import pyopenpose as op
                
                # 配置OpenPose参数
                params = {
                    "model_folder": MODEL_CONFIG['openpose_model_dir'],
                    "net_resolution": "-1x368",
                    "number_people_max": 1,  # 课堂场景通常只有一位教师
                    "model_pose": "BODY_25",  # 使用BODY_25模型
                    "render_threshold": 0.1,
                    "disable_blending": False,
                }
                
                # 初始化OpenPose对象
                self.openpose_wrapper = op.WrapperPython()
                self.openpose_wrapper.configure(params)
                self.openpose_wrapper.start()
                
                # 创建数据结构
                self.datum = op.Datum()
                self.openpose_model = self.openpose_wrapper
                print("OpenPose模型加载成功")
            except ImportError:
                print("pyopenpose库未安装，无法加载OpenPose模型")
                self.openpose_model = None
            except Exception as e:
                print(f"OpenPose模型加载失败: {e}")
                self.openpose_model = None
            
            # Whisper语音识别模型
            try:
                import whisper
                self.whisper_model = whisper.load_model(MODEL_CONFIG['whisper_model'])
                print("Whisper模型加载成功")
            except ImportError:
                print("whisper库未安装，无法加载Whisper模型")
                self.whisper_model = None
            except Exception as e:
                print(f"Whisper模型加载失败: {e}")
                self.whisper_model = None
            
            # BERT文本分析模型
            try:
                from transformers import BertModel, BertTokenizer
                self.bert_tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG['bert_model'])
                self.bert_model = BertModel.from_pretrained(MODEL_CONFIG['bert_model'])
                print("BERT模型加载成功")
            except ImportError:
                print("transformers库或BERT模型未安装，无法加载BERT模型")
                self.bert_model = None
                self.bert_tokenizer = None
            except Exception as e:
                print(f"BERT模型加载失败: {e}")
                self.bert_model = None
                self.bert_tokenizer = None
            
            print("模型初始化完成")
            
        except Exception as e:
            print(f"模型初始化失败: {e}")
    
    def get_status(self) -> Dict:
        """
        获取特征提取器状态
        
        Returns:
            包含状态信息的字典
        """
        return {
            'yolo_model_loaded': self.yolo_model is not None,
            'openpose_model_loaded': self.openpose_model is not None,
            'whisper_model_loaded': self.whisper_model is not None,
            'bert_model_loaded': self.bert_model is not None,
            'status': 'ready' if all([
                self.yolo_model is not None,
                self.openpose_model is not None,
                self.whisper_model is not None,
                self.bert_model is not None
            ]) else 'partially_loaded'
        }
    
    def extract_video_features(self, video_path: str) -> Dict:
        """
        提取视频特征
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含视频特征的字典
        """
        print(f"提取视频特征: {video_path}")
        
        # 导入必要的库
        import cv2
        import numpy as np
        
        features = {
            'action_sequence': [],  # 动作序列
            'pose_estimation': [],  # 姿态估计数据
            'motion_energy': [],  # 动作能量
            'spatial_distribution': {
                'front': 0.0,  # 讲台前区域
                'middle': 0.0,  # 教室中间区域
                'side': 0.0     # 教室两侧区域
            },
            'behavior_frequency': {
                'standing': 0.0,  # 站立比例
                'walking': 0.0,  # 行走比例
                'gesturing': 0.0,  # 手势比例
                'writing': 0.0,  # 书写比例
                'pointing': 0.0   # 指向比例
            },
            'total_frames': 0,
            'detected_frames': 0,
            'fps': 0,
            'video_duration': 0.0,
            'average_pose_confidence': 0.0
        }
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            # 返回空特征
            return features
        
        # 获取视频基本信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / fps if fps > 0 else 0.0
        
        features['total_frames'] = total_frames
        features['fps'] = fps
        features['video_duration'] = video_duration
        
        # 动作计数
        action_counts = {
            'standing': 0,
            'walking': 0,
            'gesturing': 0,
            'writing': 0,
            'pointing': 0
        }
        
        # 空间区域计数
        spatial_counts = {
            'front': 0,
            'middle': 0,
            'side': 0
        }
        
        # 姿态置信度累计
        pose_confidence_sum = 0.0
        pose_count = 0
        
        # 运动能量计算
        prev_frame_gray = None
        motion_energy_list = []
        
        # 每N帧处理一次，减少计算量
        skip_frames = 5
        frame_count = 0
        detected_frames = 0
        
        # 设置视频处理的时间窗口（每秒处理一帧）
        target_fps = 1
        processing_interval = max(1, int(fps / target_fps))
        
        print(f"视频信息: 总帧数={total_frames}, FPS={fps:.2f}, 时长={video_duration:.2f}秒")
        print(f"处理间隔: 每{processing_interval}帧处理一次")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 跳过部分帧，控制处理速度
            if frame_count % processing_interval != 0:
                continue
            
            try:
                # 计算运动能量
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_frame_gray is not None:
                    # 计算帧差
                    frame_delta = cv2.absdiff(prev_frame_gray, gray)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    
                    # 计算运动能量
                    motion_energy = np.sum(thresh) / 255.0
                    motion_energy_list.append(motion_energy)
                
                prev_frame_gray = gray
                
                # 使用YOLO模型检测
                results = self.yolo_model(frame)
                detected_frames += 1
                
                # 解析检测结果
                has_detections = False
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # 获取检测类别
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # 检查置信度
                        if conf < 0.5:
                            continue
                        
                        has_detections = True
                        
                        # 获取类别名称
                        cls_name = result.names[cls]
                        
                        # 更新动作计数
                        if cls_name in action_counts:
                            action_counts[cls_name] += 1
                        
                        # 计算物体中心点
                        x1, y1, x2, y2 = box.xyxy[0]
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # 确定空间区域（根据视频画面分区）
                        frame_height, frame_width = frame.shape[:2]
                        
                        # 计算区域边界
                        front_boundary = int(frame_height * 0.4)
                        middle_boundary = int(frame_height * 0.7)
                        
                        if center_y < front_boundary:
                            spatial_counts['front'] += 1
                        elif center_y < middle_boundary:
                            spatial_counts['middle'] += 1
                        else:
                            spatial_counts['side'] += 1
                        
                        # 保存动作序列
                        features['action_sequence'].append({
                            'frame': frame_count,
                            'timestamp': frame_count / fps,
                            'action': cls_name,
                            'confidence': conf,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'center': [float(center_x), float(center_y)]
                        })
                        
                        # 使用OpenPose进行姿态估计（如果模型可用）
                        if self.openpose_model is not None:
                            try:
                                import pyopenpose as op
                                
                                # 设置图像
                                self.datum.cvInputData = frame
                                
                                # 处理图像
                                self.openpose_model.emplaceAndPop([self.datum])
                                
                                # 获取姿态数据
                                if self.datum.poseKeypoints is not None:
                                    pose_keypoints = self.datum.poseKeypoints[0]
                                    keypoints = pose_keypoints.tolist()
                                    
                                    # 计算姿态置信度
                                    pose_confidence = np.mean(pose_keypoints[:, 2])
                                    pose_confidence_sum += pose_confidence
                                    pose_count += 1
                                    
                                    features['pose_estimation'].append({
                                        'frame': frame_count,
                                        'timestamp': frame_count / fps,
                                        'keypoints': keypoints,
                                        'confidence': float(pose_confidence),
                                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                                    })
                            except Exception as e:
                                print(f"OpenPose姿态估计失败: {e}")
                                continue
                
                # 如果没有检测到任何物体，记录为空白帧
                if not has_detections:
                    features['action_sequence'].append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'action': 'none',
                        'confidence': 0.0,
                        'bbox': [],
                        'center': []
                    })
            
            except Exception as e:
                print(f"检测帧 {frame_count} 时出错: {e}")
                continue
        
        # 释放视频捕获
        cap.release()
        
        # 计算行为频率
        if detected_frames > 0:
            for action in action_counts:
                features['behavior_frequency'][action] = action_counts[action] / detected_frames
        
        # 计算空间分布
        spatial_total = sum(spatial_counts.values())
        if spatial_total > 0:
            for region in spatial_counts:
                features['spatial_distribution'][region] = spatial_counts[region] / spatial_total
        
        # 计算平均姿态置信度
        if pose_count > 0:
            features['average_pose_confidence'] = pose_confidence_sum / pose_count
        
        # 计算运动能量
        if motion_energy_list:
            features['motion_energy'] = {
                'mean': np.mean(motion_energy_list),
                'std': np.std(motion_energy_list),
                'max': np.max(motion_energy_list),
                'min': np.min(motion_energy_list),
                'median': np.median(motion_energy_list)
            }
        
        features['detected_frames'] = detected_frames
        
        print(f"视频特征提取完成: 处理帧数={detected_frames}, 检测到姿态={pose_count}次")
        print(f"平均姿态置信度: {features['average_pose_confidence']:.3f}")
        print(f"运动能量均值: {features['motion_energy']['mean']:.2f} (像素)")
        
        return features
    
    def fuse_multimodal_features(self, video_features: Dict, audio_features: Dict, text_features: Dict) -> Dict:
        """
        融合多模态特征
        
        Args:
            video_features: 视频特征
            audio_features: 音频特征
            text_features: 文本特征
            
        Returns:
            融合后的特征字典
        """
        print("融合多模态特征...")
        
        # 计算教学风格指标
        fusion = {
            'interaction_level': 0.0,  # 互动水平
            'explanation_clarity': 0.0,  # 讲解清晰度
            'emotional_engagement': 0.0,  # 情感投入度
            'logical_structure': 0.0,  # 逻辑结构
            'teaching_style_metrics': {
                'lecturing': 0.0,  # 讲授型
                'guiding': 0.0,  # 引导型
                'interactive': 0.0,  # 互动型
                'logical': 0.0,  # 逻辑型
                'problem_driven': 0.0,  # 题目驱动型
                'emotional': 0.0,  # 情感型
                'patient': 0.0  # 耐心型
            }
        }
        
        # 计算互动水平（基于提问频率、手势频率等）
        question_freq = text_features.get('question_frequency', 0.0)
        gesture_freq = video_features.get('behavior_frequency', {}).get('gesturing', 0.0)
        fusion['interaction_level'] = min(1.0, (question_freq * 5 + gesture_freq * 2) / 2)
        
        # 计算讲解清晰度（基于语速、词汇丰富度、逻辑结构）
        speech_rate = audio_features.get('speech_rate', 120)
        vocab_richness = text_features.get('vocabulary_richness', 0.0)
        logical_indicators = sum(text_features.get('logical_indicators', {}).values())
        
        # 假设120字/分钟为最佳语速
        speech_quality = 1.0 - abs(speech_rate - 120) / 120
        fusion['explanation_clarity'] = (speech_quality * 0.4 + vocab_richness * 0.3 + logical_indicators * 0.3)
        
        # 计算情感投入度（基于情绪分数、语调变化、手势频率）
        emotion_scores = audio_features.get('emotion_scores', {})
        positive_emotion = emotion_scores.get('happy', 0.0) * 0.7 + emotion_scores.get('neutral', 0.0) * 0.3
        pitch_variation = audio_features.get('pitch_variation', 0.0)
        
        fusion['emotional_engagement'] = (positive_emotion * 0.5 + pitch_variation * 0.3 + gesture_freq * 0.2)
        
        # 计算逻辑结构（基于逻辑指示词、句子复杂度）
        sentence_complexity = text_features.get('sentence_complexity', 0.0)
        fusion['logical_structure'] = (logical_indicators * 0.6 + sentence_complexity * 0.4)
        
        # 计算各教学风格指标
        metrics = fusion['teaching_style_metrics']
        
        # 讲授型：高语速、高词汇丰富度、低互动
        metrics['lecturing'] = (speech_quality * 0.4 + vocab_richness * 0.3 + 
                               (1 - fusion['interaction_level']) * 0.3)
        
        # 引导型：高互动、中语速、高提问频率
        metrics['guiding'] = (fusion['interaction_level'] * 0.4 + speech_quality * 0.3 + 
                             question_freq * 0.3)
        
        # 互动型：高互动、高提问频率、高情感投入
        metrics['interactive'] = (fusion['interaction_level'] * 0.5 + question_freq * 0.3 + 
                               fusion['emotional_engagement'] * 0.2)
        
        # 逻辑型：高逻辑结构、中语速、高讲解清晰度
        metrics['logical'] = (fusion['logical_structure'] * 0.4 + speech_quality * 0.3 + 
                           fusion['explanation_clarity'] * 0.3)
        
        # 题目驱动型：高提问频率、高互动、中讲解清晰度
        metrics['problem_driven'] = (question_freq * 0.5 + fusion['interaction_level'] * 0.3 + 
                                    fusion['explanation_clarity'] * 0.2)
        
        # 情感型：高情感投入、高语调变化、高手势频率
        metrics['emotional'] = (fusion['emotional_engagement'] * 0.5 + pitch_variation * 0.3 + 
                               gesture_freq * 0.2)
        
        # 耐心型：低语速、高情感投入、中互动
        slow_speech_bonus = max(0, (140 - speech_rate) / 140)  # 语速越慢越耐心
        metrics['patient'] = (slow_speech_bonus * 0.4 + fusion['emotional_engagement'] * 0.4 + 
                           fusion['interaction_level'] * 0.2)
        
        # 确保所有指标都在0-1范围内
        for key in metrics:
            metrics[key] = min(1.0, max(0.0, metrics[key]))
        
        return {
            'video': video_features,
            'audio': audio_features,
            'text': text_features,
            'fusion': fusion
        }
    
    def extract_audio_features(self, audio_path: str) -> Dict:
        """
        提取音频特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            包含音频特征的字典
        """
        print(f"提取音频特征: {audio_path}")
        
        features = {
            'speech_rate': 0.0,  # 语速
            'pitch_variation': 0.0,  # 语调变化
            'emotion_scores': {},  # 情绪分数
            'volume_level': 0.0,  # 音量水平
            'silence_ratio': 0.0,  # 沉默比例
            'transcript': "",  # 语音转写文本
            'segments': [],  # 语音片段
            'language': "zh",  # 检测到的语言
        }
        
        try:
            # 检查音频路径是否为视频文件
            if audio_path.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.flv')):
                # 从视频中提取音频
                audio_file_path = self._extract_audio_from_video(audio_path)
            else:
                audio_file_path = audio_path
            
            # 使用librosa读取音频文件
            y, sr = None, None
            try:
                import librosa
                import numpy as np
                
                # 读取音频文件
                y, sr = librosa.load(audio_file_path, sr=16000)
                
                # 音频降噪处理
                try:
                    # 使用谱减法进行降噪
                    # 提取噪声样本（假设前0.5秒为噪声）
                    noise_sample = y[:int(sr * 0.5)]
                    # 计算噪声功率谱
                    noise_stft = np.abs(librosa.stft(noise_sample))
                    noise_power = np.mean(noise_stft, axis=1)
                    
                    # 计算信号功率谱
                    stft = librosa.stft(y)
                    stft_magnitude = np.abs(stft)
                    stft_phase = np.angle(stft)
                    
                    # 谱减法降噪
                    stft_magnitude_noise_reduced = np.maximum(stft_magnitude - noise_power[:, np.newaxis], 0)
                    
                    # 重建音频
                    y = librosa.istft(stft_magnitude_noise_reduced * np.exp(1j * stft_phase))
                    print("音频降噪完成")
                except Exception as e:
                    print(f"音频降噪失败: {e}")
                    # 继续使用原始音频
                    pass
                
                # 语音活动检测（VAD）
                try:
                    import webrtcvad
                    import struct
                    
                    # 将音频转换为webrtcvad所需的格式（16kHz, 16位PCM）
                    vad = webrtcvad.Vad()
                    vad.set_mode(3)  # 最敏感模式
                    
                    # 转换为16位PCM
                    audio_float32 = y.astype(np.float32)
                    max_val = np.max(np.abs(audio_float32))
                    if max_val > 0:
                        audio_float32 = audio_float32 / max_val
                    audio_int16 = (audio_float32 * 32767).astype(np.int16)
                    
                    # 转换为字节流
                    audio_bytes = struct.pack("<" + "h" * len(audio_int16), *audio_int16)
                    
                    # 每10ms处理一次
                    frame_duration = 10  # 10ms
                    frame_size = int(sr * frame_duration / 1000)  # 160 samples per 10ms
                    
                    # 检测语音帧
                    speech_frames = []
                    for i in range(0, len(audio_bytes), frame_size * 2):  # 2 bytes per sample
                        frame = audio_bytes[i:i+frame_size*2]
                        if len(frame) < frame_size * 2:
                            break
                        
                        is_speech = vad.is_speech(frame, sr)
                        if is_speech:
                            start_ms = (i // (frame_size * 2)) * frame_duration
                            end_ms = start_ms + frame_duration
                            speech_frames.append((start_ms, end_ms))
                    
                    # 计算语音活动比例
                    if speech_frames:
                        total_speech_duration = sum(end - start for start, end in speech_frames) / 1000
                        total_duration = len(y) / sr
                        features['speech_ratio'] = total_speech_duration / total_duration if total_duration > 0 else 0.0
                        features['silence_ratio'] = 1.0 - features['speech_ratio']
                    print("语音活动检测完成")
                except ImportError:
                    print("webrtcvad库未安装，无法进行语音活动检测")
                except Exception as e:
                    print(f"语音活动检测失败: {e}")
                    # 继续使用默认沉默比例计算
                    pass
                
                # 计算音量水平
                rms = librosa.feature.rms(y=y)
                features['volume_level'] = float(np.mean(rms)) if rms.size > 0 else 0.0
                
                # 计算语调变化（音高变化）
                if len(y) > sr * 0.05:  # 确保音频长度足够
                    # 提取基频（F0）
                    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                    
                    # 计算基频变化
                    if f0 is not None and np.any(voiced_flag):
                        valid_f0 = f0[voiced_flag]
                        if len(valid_f0) > 0:
                            features['pitch_variation'] = float(np.std(valid_f0) / np.mean(valid_f0) if np.mean(valid_f0) > 0 else 0.0)
            except ImportError:
                print("librosa库未安装，无法提取详细音频特征")
            except Exception as e:
                print(f"音频特征提取失败: {e}")
            
            # 使用Whisper模型进行语音识别
            if self.whisper_model is not None:
                try:
                    # 准备输入音频
                    if y is not None:
                        # 保存降噪后的音频用于转录
                        import tempfile
                        import soundfile as sf
                        
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            temp_audio_path = temp_file.name
                        
                        # 保存降噪后的音频
                        sf.write(temp_audio_path, y, sr)
                        
                        # 使用降噪后的音频进行转录
                        result = self.whisper_model.transcribe(temp_audio_path)
                        
                        # 删除临时文件
                        import os
                        os.unlink(temp_audio_path)
                    else:
                        # 如果无法加载音频，使用原始音频文件
                        result = self.whisper_model.transcribe(audio_file_path)
                    
                    features['transcript'] = result.get('text', '')
                    features['segments'] = result.get('segments', [])
                    features['language'] = result.get('language', 'zh')
                    
                    # 计算语速（字/分钟）
                    total_words = len(features['transcript'])
                    total_duration = sum(segment['end'] - segment['start'] for segment in features['segments']) if features['segments'] else 0
                    if total_duration > 0:
                        features['speech_rate'] = (total_words / total_duration) * 60
                    
                    # 如果没有通过VAD计算沉默比例，使用Whisper转录结果计算
                    if 'silence_ratio' not in features or features['silence_ratio'] == 0.0:
                        if features['segments'] and total_duration > 0:
                            # 假设两个片段之间的间隔大于0.5秒为沉默
                            silence_duration = 0.0
                            previous_end = 0.0
                            for segment in features['segments']:
                                silence = segment['start'] - previous_end
                                if silence > 0.5:
                                    silence_duration += silence
                                previous_end = segment['end']
                            features['silence_ratio'] = silence_duration / total_duration
                    
                except Exception as e:
                    print(f"Whisper语音识别失败: {e}")
            
            # 模拟情绪分数（实际应用中需要使用专门的情绪识别模型）
            features['emotion_scores'] = {
                'happy': 0.3,
                'neutral': 0.5,
                'serious': 0.2
            }
            
        except Exception as e:
            print(f"提取音频特征时出错: {e}")
            
            # 如果出现错误，使用默认值
            features['speech_rate'] = 120
            features['pitch_variation'] = 0.45
            features['emotion_scores'] = {
                'happy': 0.3,
                'neutral': 0.5,
                'serious': 0.2
            }
            features['volume_level'] = 0.75
            features['silence_ratio'] = 0.15
        
        return features
        
    def _extract_audio_from_video(self, video_path: str) -> str:
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            提取的音频文件路径
        """
        try:
            import os
            import tempfile
            from moviepy.editor import VideoFileClip
            
            # 创建临时音频文件
            temp_audio_file = tempfile.mktemp(suffix='.wav')
            
            # 使用moviepy提取音频
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(temp_audio_file, codec='pcm_s16le', ffmpeg_params=['-ar', '16000'])
            audio.close()
            video.close()
            
            print(f"已从视频中提取音频: {temp_audio_file}")
            return temp_audio_file
            
        except ImportError:
            print("moviepy库未安装，无法从视频中提取音频")
            return video_path
        except Exception as e:
            print(f"从视频中提取音频失败: {e}")
            return video_path
    
    def extract_text_features(self, transcript_path: str) -> Dict:
        """
        提取文本特征
        
        Args:
            transcript_path: 文本转录文件路径
            
        Returns:
            包含文本特征的字典
        """
        print(f"提取文本特征: {transcript_path}")
        
        # 读取转录文本
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
        except Exception as e:
            print(f"读取转录文本失败: {e}")
            # 使用默认空文本
            transcript = ""
        
        # 如果转录文本为空，返回默认特征
        if not transcript:
            return {
                'vocabulary_richness': 0.0,
                'sentence_complexity': 0.0,
                'question_frequency': 0.0,
                'keyword_density': {},
                'logical_indicators': {},
                'semantic_features': np.zeros(768),  # BERT模型输出维度
                'sentiment_score': 0.0,
                'teaching_terms': [],
                'transcript': ""
            }
        
        features = {
            'vocabulary_richness': 0.0,  # 词汇丰富度
            'sentence_complexity': 0.0,  # 句子复杂度
            'question_frequency': 0.0,  # 提问频率
            'keyword_density': {},  # 关键词密度
            'logical_indicators': {},  # 逻辑指示词统计
            'semantic_features': np.zeros(768),  # BERT模型输出维度
            'sentiment_score': 0.0,  # 情感分数
            'teaching_terms': [],  # 教学术语
            'transcript': transcript
        }
        
        # 基础文本特征计算
        try:
            # 计算词汇丰富度（不同词汇数/总词汇数）
            words = transcript.split()
            if words:
                unique_words = set(words)
                features['vocabulary_richness'] = len(unique_words) / len(words)
            
            # 计算句子复杂度（平均句长）
            sentences = transcript.split('。')
            sentences = [s for s in sentences if s.strip()]
            if sentences:
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                # 归一化到0-1范围
                features['sentence_complexity'] = min(1.0, avg_sentence_length / 30)  # 假设30个词为复杂句
            
            # 计算提问频率（问号数/总字数）
            question_count = transcript.count('？')
            total_chars = len(transcript)
            if total_chars > 0:
                features['question_frequency'] = question_count / total_chars
            
            # 教学关键词列表
            teaching_keywords = ['概念', '原理', '例子', '例题', '总结', '思考', '分析', '推导', '证明', '练习']
            
            # 计算关键词密度
            keyword_counts = {}
            for keyword in teaching_keywords:
                count = transcript.count(keyword)
                if count > 0:
                    keyword_counts[keyword] = count / total_chars if total_chars > 0 else 0
            features['keyword_density'] = keyword_counts
            
            # 逻辑指示词列表
            logical_words = ['因为', '所以', '因此', '首先', '其次', '最后', '但是', '然而', '另外', '例如', '比如']
            
            # 计算逻辑指示词频率
            logical_counts = {}
            for word in logical_words:
                count = transcript.count(word)
                if count > 0:
                    logical_counts[word] = count / total_chars if total_chars > 0 else 0
            features['logical_indicators'] = logical_counts
            
            # 提取教学术语
            features['teaching_terms'] = [kw for kw in teaching_keywords if kw in transcript]
            
        except Exception as e:
            print(f"基础文本特征计算失败: {e}")
        
        # 使用BERT模型提取语义特征
        if self.bert_model is not None and self.bert_tokenizer is not None:
            try:
                import torch
                
                # 处理长文本，分段编码
                max_length = 512
                chunks = []
                current_chunk = []
                current_length = 0
                
                # 按句子分割文本
                sentences = transcript.split('。')
                sentences = [s + '。' for s in sentences if s.strip()]
                
                for sentence in sentences:
                    # 计算句子的token长度
                    token_length = len(self.bert_tokenizer.encode(sentence, add_special_tokens=False))
                    
                    # 如果当前chunk加上这个句子超过max_length，就添加到chunks中
                    if current_length + token_length > max_length - 2:  # 减去特殊token的长度
                        if current_chunk:
                            chunks.append(''.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                    
                    # 添加句子到当前chunk
                    current_chunk.append(sentence)
                    current_length += token_length
                
                # 添加最后一个chunk
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                
                # 对每个chunk进行编码
                all_features = []
                for chunk in chunks:
                    # 编码文本
                    inputs = self.bert_tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                    
                    # 获取BERT输出
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                    
                    # 使用[CLS]标记的输出作为语义特征
                    cls_output = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    all_features.append(cls_output)
                
                # 平均所有chunk的特征
                if all_features:
                    features['semantic_features'] = np.mean(all_features, axis=0)
                    
            except Exception as e:
                print(f"BERT语义特征提取失败: {e}")
        
        # 情感分析（简化版本）
        try:
            # 简单的情感词典
            positive_words = ['好', '优秀', '正确', '棒', '精彩', '很好', '不错', '准确', '清晰', '明白']
            negative_words = ['错误', '不对', '差', '糟糕', '模糊', '混乱', '复杂', '困难', '麻烦', '问题']
            
            # 计算情感分数
            positive_count = sum(transcript.count(word) for word in positive_words)
            negative_count = sum(transcript.count(word) for word in negative_words)
            
            # 归一化到-1到1范围
            total_sentiment = positive_count - negative_count
            if positive_count + negative_count > 0:
                features['sentiment_score'] = total_sentiment / (positive_count + negative_count)
        except Exception as e:
            print(f"情感分析失败: {e}")
        
        print(f"文本特征提取完成，词汇丰富度: {features['vocabulary_richness']:.2f}")
        return features
    
    def fuse_features(self, video_features: Dict, audio_features: Dict, 
                     text_features: Dict) -> Dict:
        """
        融合多模态特征
        
        Args:
            video_features: 视频特征
            audio_features: 音频特征
            text_features: 文本特征
            
        Returns:
            融合后的特征字典
        """
        print("融合多模态特征")
        
        # 创建融合特征字典
        fused_features = {
            'video': video_features,
            'audio': audio_features,
            'text': text_features,
            'fusion': {
                'teaching_style_metrics': {},  # 教学风格指标
                'interaction_level': 0.0,  # 互动水平
                'explanation_clarity': 0.0,  # 讲解清晰度
                'emotional_engagement': 0.0,  # 情感投入度
                'logical_structure': 0.0,  # 逻辑结构
            }
        }
        
        # 计算一些融合指标（模拟）
        # 互动水平 = 文本提问频率 + 视频手势频率 + 音频情绪变化
        interaction_level = (
            text_features.get('question_frequency', 0) * 2.0 +
            video_features.get('behavior_frequency', {}).get('gesturing', 0) * 1.5 +
            audio_features.get('pitch_variation', 0) * 1.0
        ) / 4.5
        
        # 讲解清晰度 = 文本逻辑指示词密度 + 语速适中度
        logical_indicators = sum(text_features.get('logical_indicators', {}).values())
        speech_rate_factor = 1.0 - abs(audio_features.get('speech_rate', 0) - 120) / 120  # 语速适中度
        explanation_clarity = (logical_indicators * 10 + speech_rate_factor) / 2
        
        # 情感投入度 = 音频情绪积极度 + 语调变化
        positive_emotion = sum(audio_features.get('emotion_scores', {}).values()) * 0.5
        emotional_engagement = (positive_emotion + audio_features.get('pitch_variation', 0)) / 2
        
        # 逻辑结构 = 文本逻辑指示词 + 词汇丰富度
        logical_structure = (logical_indicators * 10 + text_features.get('vocabulary_richness', 0)) / 2
        
        # 更新融合特征
        fused_features['fusion']['interaction_level'] = float(interaction_level)
        fused_features['fusion']['explanation_clarity'] = float(explanation_clarity)
        fused_features['fusion']['emotional_engagement'] = float(emotional_engagement)
        fused_features['fusion']['logical_structure'] = float(logical_structure)
        
        # 计算教学风格指标
        fused_features['fusion']['teaching_style_metrics'] = {
            'lecturing': 0.0,
            'guiding': 0.0,
            'interactive': 0.0,
            'logical': 0.0,
            'problem_driven': 0.0,
            'emotional': 0.0,
            'patient': 0.0
        }
        
        # 填充风格指标（模拟计算）
        metrics = fused_features['fusion']['teaching_style_metrics']
        metrics['lecturing'] = 0.6 - interaction_level * 0.5  # 讲授型与互动负相关
        metrics['guiding'] = interaction_level * 0.6 + explanation_clarity * 0.4
        metrics['interactive'] = interaction_level
        metrics['logical'] = logical_structure * 0.8 + explanation_clarity * 0.2
        metrics['problem_driven'] = text_features.get('question_frequency', 0) * 3.0
        metrics['emotional'] = emotional_engagement
        metrics['patient'] = audio_features.get('silence_ratio', 0) * 2.0 + (1.0 - audio_features.get('speech_rate', 0) / 150)
        
        # 确保所有值在0-1范围内
        for key in metrics:
            metrics[key] = max(0, min(1, metrics[key]))
        
        return fused_features
    
    def process_video(self, video_path: str) -> Dict:
        """
        处理视频文件，提取所有模态的特征并融合
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含所有特征的字典
        """
        print(f"处理视频文件: {video_path}")
        
        # 提取视频特征
        video_features = self.extract_video_features(video_path)
        
        # 提取音频特征（自动处理视频到音频的转换）
        audio_features = self.extract_audio_features(video_path)
        
        # 检查是否有转录文本
        transcript = audio_features.get('transcript', '')
        transcript_path = None
        
        # 如果有转录文本，保存到文件
        if transcript:
            # 生成转录文件路径
            transcript_path = video_path.replace('.mp4', '_transcript.txt').replace('.avi', '_transcript.txt')
            try:
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                print(f"转录文本已保存到: {transcript_path}")
            except Exception as e:
                print(f"保存转录文本失败: {e}")
                transcript_path = None
        
        # 提取文本特征
        text_features = self.extract_text_features(transcript_path if transcript_path else video_path)
        
        # 如果没有从文件读取到转录文本，使用音频特征中的转录
        if not text_features.get('transcript', '') and transcript:
            text_features['transcript'] = transcript
        
        # 融合特征
        fused_features = self.fuse_multimodal_features(video_features, audio_features, text_features)
        
        # 返回完整的特征结果
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'text_features': text_features,
            'fused_features': fused_features
        }
    
    def extract_and_save_features(self, video_id: str) -> str:
        """
        提取并保存所有特征
        
        Args:
            video_id: 视频ID
            
        Returns:
            特征文件保存路径
        """
        try:
            # 构建文件路径
            video_path = str(VIDEO_DIR / f"{video_id}_*.mp4")  # 简化处理
            audio_path = str(AUDIO_DIR / f"{video_id}.wav")
            transcript_path = str(TEXT_DIR / f"{video_id}_transcript.txt")
            
            # 提取各模态特征
            video_features = self.extract_video_features(video_path)
            audio_features = self.extract_audio_features(audio_path)
            text_features = self.extract_text_features(transcript_path)
            
            # 融合特征
            fused_features = self.fuse_features(video_features, audio_features, text_features)
            
            # 保存特征到文件
            features_file = FEATURES_DIR / f"{video_id}_features.json"
            with open(features_file, 'w', encoding='utf-8') as f:
                json.dump(fused_features, f, ensure_ascii=False, indent=2)
            
            print(f"特征已保存到: {features_file}")
            return str(features_file)
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            raise


# 创建特征提取器实例
feature_extractor = FeatureExtractor()