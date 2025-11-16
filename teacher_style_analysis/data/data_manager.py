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
            
            if self.db_conn:
                query = """
                INSERT INTO audios (id, video_id, filepath)
                VALUES (%s, %s, %s)
                """
                values = (audio_id, video_id, str(dest_path))
                self.db_cursor.execute(query, values)
                self.db_conn.commit()
            
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
            
            if self.db_conn:
                query = """
                INSERT INTO transcripts (id, video_id, filepath)
                VALUES (%s, %s, %s)
                """
                values = (transcript_id, video_id, str(filepath))
                self.db_cursor.execute(query, values)
                self.db_conn.commit()
            
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
        # 先从Redis缓存获取
        if self.redis_client:
            cached_path = self.redis_client.get(f"video:{video_id}")
            if cached_path:
                return {
                    'id': video_id,
                    'filepath': cached_path.decode('utf-8')
                }
        
        # 从数据库获取
        if self.db_conn:
            query = "SELECT * FROM videos WHERE id = %s"
            self.db_cursor.execute(query, (video_id,))
            result = self.db_cursor.fetchone()
            
            # 更新缓存
            if result and self.redis_client:
                self.redis_client.setex(
                    f"video:{video_id}",
                    3600,
                    result['filepath']
                )
            
            return result
        
        return None
    
    def update_video_status(self, video_id: str, status: str) -> bool:
        """
        更新视频处理状态
        
        Args:
            video_id: 视频ID
            status: 新状态
            
        Returns:
            是否更新成功
        """
        try:
            if self.db_conn:
                query = "UPDATE videos SET status = %s WHERE id = %s"
                self.db_cursor.execute(query, (status, video_id))
                self.db_conn.commit()
                return True
            return False
        except Exception as e:
            print(f"更新视频状态失败: {e}")
            return False
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'db_cursor') and self.db_cursor:
            self.db_cursor.close()
        if hasattr(self, 'db_conn') and self.db_conn:
            self.db_conn.close()


# 创建数据管理器实例
data_manager = DataManager()