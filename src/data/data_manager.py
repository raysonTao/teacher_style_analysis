"""数据管理模块，负责视频、音频和文本数据的上传、存储和索引"""
import os
import uuid
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    VIDEO_DIR, AUDIO_DIR, TEXT_DIR, DATA_DIR
)


class DataManager:
    """数据管理器类 - 使用文件系统存储"""
    
    def __init__(self):
        self.metadata_file = DATA_DIR / 'metadata.json'
        self._init_metadata()
        
        # 添加学科标准属性，用于API配置
        self.discipline_standards = {
            '数学': '理论推导',
            '语文': '情感表达',
            '英语': '互动导向',
            '物理': '逻辑推导',
            '化学': '实验探究',
            '生物': '观察分析',
            '历史': '史料分析',
            '地理': '空间思维',
            '政治': '思辨论证'
        }
        
        # 添加年级标准属性
        self.grade_standards = {
            '初中': '基础理解',
            '高中': '深化应用',
            '大学': '研究创新'
        }
        
        # 添加风格标签属性
        self.style_labels = {
            '理论讲授型': '知识传授',
            '启发引导型': '思维启发',
            '互动导向型': '交流互动',
            '逻辑推导型': '逻辑推理',
            '题目驱动型': '练习巩固',
            '情感表达型': '情感共鸣',
            '耐心细致型': '细致讲解'
        }
    
    def get_status(self) -> Dict:
        """
        获取数据管理器状态
        
        Returns:
            包含状态信息的字典
        """
        return {
            'data_dir_exists': self.data_dir.exists(),
            'videos_count': len(list(self.videos_dir.glob('*'))),
            'audio_count': len(list(self.audio_dir.glob('*'))),
            'transcripts_count': len(list(self.transcripts_dir.glob('*'))),
            'features_count': len(list(self.features_dir.glob('*'))),
            'results_count': len(list(self.results_dir.glob('*'))),
            'metadata_loaded': self.metadata is not None,
            'status': 'ready' if self.metadata is not None else 'not_loaded'
        }
    
    def _init_metadata(self):
        """初始化元数据文件"""
        if not self.metadata_file.exists():
            self.metadata = {'videos': {}, 'audios': {}, 'transcripts': {}}
            self._save_metadata()
        else:
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {'videos': {}, 'audios': {}, 'transcripts': {}}
    
    def _save_metadata(self):
        """保存元数据到文件"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def upload_video(self, file_path: str, teacher_id: Optional[str] = None, 
                    course_id: Optional[str] = None) -> Dict:
        """
        上传视频文件 - 使用文件系统存储
        
        Args:
            file_path: 视频文件路径
            teacher_id: 教师ID
            course_id: 课程ID
            
        Returns:
            包含视频信息的字典
        """
        try:
            # 生成唯一ID
            video_id = str(uuid.uuid4())
            filename = os.path.basename(file_path)
            
            # 复制文件到存储目录
            dest_path = VIDEO_DIR / f"{video_id}_{filename}"
            shutil.copy2(file_path, dest_path)
            
            # 保存元数据
            video_info = {
                'id': video_id,
                'filename': filename,
                'filepath': str(dest_path),
                'upload_time': datetime.now().isoformat(),
                'status': 'uploaded',
                'teacher_id': teacher_id,
                'course_id': course_id
            }
            
            self.metadata['videos'][video_id] = video_info
            self._save_metadata()
            
            return video_info
            
        except Exception as e:
            print(f"视频上传失败: {e}")
            raise
    
    def save_audio(self, video_id: str, audio_path: str) -> Dict:
        """
        保存从视频中提取的音频文件
        
        Args:
            video_id: 关联的视频ID
            audio_path: 音频文件路径
            
        Returns:
            包含音频信息的字典
        """
        try:
            audio_id = str(uuid.uuid4())
            filename = f"{video_id}.wav"
            dest_path = AUDIO_DIR / filename
            
            shutil.copy2(audio_path, dest_path)
            
            return {
                'id': audio_id,
                'video_id': video_id,
                'filepath': str(dest_path)
            }
            
        except Exception as e:
            print(f"音频保存失败: {e}")
            raise
    
    def save_video_info(self, video_info: Dict) -> str:
        """保存视频信息"""
        video_id = video_info.get('video_id', str(uuid.uuid4()))
        self.metadata['videos'][video_id] = video_info
        self._save_metadata()
        return video_id
    
    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """获取视频信息"""
        return self.metadata['videos'].get(video_id)
    
    def update_video_status(self, video_id: str, status: str, error_info: str = None):
        """更新视频状态"""
        if video_id in self.metadata['videos']:
            self.metadata['videos'][video_id]['status'] = status
            if error_info:
                self.metadata['videos'][video_id]['error'] = error_info
            self.metadata['videos'][video_id]['updated_time'] = datetime.now().isoformat()
            self._save_metadata()
    
    def list_videos(self, teacher_id: Optional[str] = None, 
                   discipline: Optional[str] = None,
                   status: Optional[str] = None,
                   page: int = 1, page_size: int = 10) -> Dict:
        """列出视频"""
        videos = list(self.metadata['videos'].values())
        
        # 过滤
        if teacher_id:
            videos = [v for v in videos if v.get('teacher_id') == teacher_id]
        if discipline:
            videos = [v for v in videos if v.get('discipline') == discipline]
        if status:
            videos = [v for v in videos if v.get('status') == status]
        
        # 分页
        total = len(videos)
        start = (page - 1) * page_size
        end = start + page_size
        videos_page = videos[start:end]
        
        return {
            'videos': videos_page,
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': (total + page_size - 1) // page_size
        }
    
    def save_transcript(self, video_id: str, transcript_content: str) -> Dict:
        """
        保存转录文本
        
        Args:
            video_id: 关联的视频ID
            transcript_content: 转录文本内容
            
        Returns:
            包含文本信息的字典
        """
        try:
            transcript_id = str(uuid.uuid4())
            filepath = TEXT_DIR / f"{video_id}_transcript.txt"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(transcript_content)
            
            return {
                'id': transcript_id,
                'video_id': video_id,
                'filepath': str(filepath)
            }
            
        except Exception as e:
            print(f"转录文本保存失败: {e}")
            raise
    
    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """
        获取视频信息
        
        Args:
            video_id: 视频ID
            
        Returns:
            视频信息字典，如果不存在则返回None
        """
        return self.metadata['videos'].get(video_id)
    
    def update_video_status(self, video_id: str, status: str, error_info: str = None):
        """
        更新视频状态
        
        Args:
            video_id: 视频ID
            status: 新状态
            error_info: 错误信息（可选）
        """
        if video_id in self.metadata['videos']:
            self.metadata['videos'][video_id]['status'] = status
            if error_info:
                self.metadata['videos'][video_id]['error'] = error_info
            self.metadata['videos'][video_id]['updated_time'] = datetime.now().isoformat()
            self._save_metadata()
            return True
        return False
    



# 创建数据管理器实例
data_manager = DataManager()